# forecast-24h: YOUR model forecasts Connecticut electricity load for the next 24 hours, every hour.
#
# Every hour (:20, after the weather scrapers at :05 and the ISO-NE scraper at :15) this function:
#   1. loads your live model:          gs://<bucket>/energy/models/model.joblib (+ model_card.json)
#   2. picks the next 24 FULL hours:   if it runs at 2:20 p.m., it forecasts 3 p.m. today ... 2 p.m. tomorrow
#   3. builds the features with make_features(), the SAME code as the Lab 5 notebook:
#        weather  -> the newest NWS forecast for each hour (bdl_weather_forecast_master.csv)
#        load     -> the load at the same hour yesterday (ct_load_5min_hourly_master.csv, live five-minute load)
#   4. adds a low and high value (an 80% range: 10th and 90th percentiles) if your model card says how (Lab 6):
#        "range_method": "quantile_models"  -> two more models, energy/models/model_low.joblib and model_high.joblib
#        "range_method": "offsets_by_hour"  -> "range_offsets_mw": {"low": [24 MW], "high": [24 MW]}, by Connecticut hour
#        "range_method": "hrrr_qrf_pooled"  -> Lab 6C: energy/models/model_qrf.joblib (a quantile forest) on EVERY recent HRRR
#                                              run that covers each hour (weather uncertainty), pooled into one low,
#                                              expected and high. HRRR runs come from Dr. Dave's class file (HRRR_RUNS_URL);
#                                              if it is missing or stale, the NWS forecast is used instead
#        no "range_method" (Lab 5)          -> expected value only (the low and high columns stay empty)
#   5. writes 24 rows (target_hour_utc, predicted_low_mw, predicted_load_mw, predicted_high_mw, ...):
#        gs://<bucket>/energy/forecasts/latest.csv                       <- replaced every hour (Lab 6 reads this one)
#        gs://<bucket>/energy/forecasts/history/<issued time>.csv        <- one file per hour, never overwritten
#        gs://<bucket>/energy/datasets/forecasts_master.csv              <- every forecast you ever made, in one table
# No model yet? It politely does nothing ("status": "no_model"). No ISO-NE data yet? "status": "no_load_data".
# HTTP entrypoint: forecast_24h_http

import io
import json
import logging
import os
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
HOURS_AHEAD = int(os.getenv("HOURS_AHEAD", "24"))

MODEL_KEY = f"{ENERGY_PREFIX}/models/model.joblib"
CARD_KEY = f"{ENERGY_PREFIX}/models/model_card.json"
LOW_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_low.joblib"
HIGH_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_high.joblib"
QRF_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_qrf.joblib"
# Lab 6C: the class HRRR file, rewritten every hour at :05 by Dr. Dave's hrrr-bdl job (8 newest runs, 0-48 hours ahead)
HRRR_RUNS_URL = os.getenv("HRRR_RUNS_URL", "https://storage.googleapis.com/opim5512-leaderboard-f26/weather/bdl_hrrr_runs.csv")
HRRR_MAX_AGE_HOURS = 14        # newest run older than this = something is wrong upstream: use the NWS forecast instead
# 20 percentiles, each the middle of a 5% slice of the chances (2.5%, 7.5%, ..., 97.5%), same as the Lab 6C notebook
POOL_QUANTILES = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725,
                  0.775, 0.825, 0.875, 0.925, 0.975]
WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
LOAD_KEY = f"{ENERGY_PREFIX}/datasets/ct_load_5min_hourly_master.csv"
LATEST_KEY = f"{ENERGY_PREFIX}/forecasts/latest.csv"
HISTORY_PREFIX = f"{ENERGY_PREFIX}/forecasts/history"
MASTER_KEY = f"{ENERGY_PREFIX}/datasets/forecasts_master.csv"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------------------------------------------------------------------------------------------
# make_features: copied word for word from the Lab 5 notebook (Part 3). Change it there, change it here.
# ----------------------------------------------------------------------------------------------------------------
WEATHER = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct"]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]
FEATURES = CALENDAR + WEATHER + ["load_yesterday_mw"]


def make_features(hours_utc, weather, load):
    """One row of model inputs per hour.
    hours_utc: the hours to describe (UTC).  weather: table indexed by UTC hour with the WEATHER columns.
    load: Connecticut load in MW, indexed by UTC hour (only hours that are already over)."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    local = hours_utc.tz_convert("America/New_York")           # people use electricity on local time
    out = pd.DataFrame(index=hours_utc)
    out["hour"] = local.hour
    out["day_of_week"] = local.dayofweek                       # 0 = Monday ... 6 = Sunday
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(start="2020-01-01", end="2030-12-31")
    out["is_holiday"] = local.tz_localize(None).normalize().isin(holidays).astype(int)

    for column in WEATHER:
        out[column] = weather[column].reindex(hours_utc).values

    # Same hour yesterday. If that hour isn't over yet (only the last hour of a live forecast),
    # use the hour just before it.
    yesterday = load.reindex(hours_utc - pd.Timedelta(hours=24)).values
    hour_before = load.reindex(hours_utc - pd.Timedelta(hours=25)).values
    out["load_yesterday_mw"] = np.where(np.isnan(yesterday), hour_before, yesterday)
    return out[FEATURES]
# ----------------------------------------------------------------------------------------------------------------


def read_csv(bucket, key):
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def latest_weather(forecasts: pd.DataFrame) -> pd.DataFrame:
    """For every target hour, the NEWEST forecast that covers it."""
    forecasts["scraped_hour_utc"] = pd.to_datetime(forecasts["scraped_hour_utc"], utc=True)
    forecasts["target_hour_utc"] = pd.to_datetime(forecasts["target_hour_utc"], utc=True)
    forecasts = forecasts.sort_values(["target_hour_utc", "scraped_hour_utc"])
    newest = forecasts.drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")
    return newest


def forecast_range(bucket, card, X, preds, targets):
    """The low and high ends of your 80% range for each hour, plus how they were made.
    Returns (low, high, method), or (None, None, "none") when your model card asks for no range."""
    method = str(card.get("range_method", "none"))
    if method == "quantile_models":
        if not bucket.blob(LOW_MODEL_KEY).exists() or not bucket.blob(HIGH_MODEL_KEY).exists():
            raise ValueError("Your model card says quantile_models, but model_low.joblib or model_high.joblib is missing.")
        low_model = joblib.load(io.BytesIO(bucket.blob(LOW_MODEL_KEY).download_as_bytes()))
        high_model = joblib.load(io.BytesIO(bucket.blob(HIGH_MODEL_KEY).download_as_bytes()))
        low = low_model.predict(X[FEATURES])
        high = high_model.predict(X[FEATURES])
    elif method == "offsets_by_hour":
        offsets = card["range_offsets_mw"]
        if len(offsets["low"]) != 24 or len(offsets["high"]) != 24:
            raise ValueError("range_offsets_mw needs 24 low and 24 high offsets, one per Connecticut hour (0 = midnight).")
        local_hours = pd.DatetimeIndex(targets).tz_convert("America/New_York").hour
        low = preds + np.array(offsets["low"], dtype=float)[local_hours]
        high = preds + np.array(offsets["high"], dtype=float)[local_hours]
    else:
        return None, None, "none"

    # "range_widen": 1.3 stretches the range 30% around the expected value, no retraining (Lab 6: if the leaderboard
    # says only 65% of actual loads landed inside your 80% range, your range is too narrow)
    widen = float(card.get("range_widen", 1.0))
    low = preds - widen * (preds - low)
    high = preds + widen * (high - preds)
    # The expected value always sits inside its own range
    low = np.minimum(low, preds)
    high = np.maximum(high, preds)
    return low, high, method


def fill_inputs(X, load):
    """Honest gap filling (rare): a missing weather value takes its neighbor; a missing yesterday load the newest load."""
    X[WEATHER] = X[WEATHER].interpolate(limit_direction="both")
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    return X


def hrrr_run_tables(card, now):
    """The recent HRRR runs from the class file, newest last: a list of weather tables indexed by UTC hour.
    Returns an empty list when the file is missing or stale."""
    try:
        runs = pd.read_csv(HRRR_RUNS_URL)
    except Exception as err:
        logging.warning("HRRR runs not available: %s", err)
        return []
    runs["init_utc"] = pd.to_datetime(runs["init_utc"], utc=True)
    runs["hour_utc"] = pd.to_datetime(runs["hour_utc"], utc=True)
    inits = sorted(runs["init_utc"].unique())
    if len(inits) == 0 or pd.Timestamp(now) - pd.Timestamp(inits[-1]) > pd.Timedelta(hours=HRRR_MAX_AGE_HOURS):
        return []
    n_runs = int(card.get("hrrr_runs", 8))
    tables = []
    for init in inits[-n_runs:]:
        tables.append(runs[runs["init_utc"] == init].set_index("hour_utc")[WEATHER])
    return tables


def pooled_range(bucket, card, targets, load, nws_weather, now):
    """Lab 6C: the quantile forest's 20 percentiles (2.5%, 7.5%, ..., 97.5%) for every HRRR run that covers an hour, pooled
    into one pile per hour; low, expected and high are the 10th, 50th and 90th percentile of the pile ("hazen": each
    value is the middle of its 5% slice, so a single run gives back exactly its own 10th, 50th and 90th).
    The model's uncertainty is inside each run's percentiles; the weather's is in how much the runs disagree.
    Returns (low, expected, high, info)."""
    if not bucket.blob(QRF_MODEL_KEY).exists():
        raise ValueError("Your model card says hrrr_qrf_pooled, but energy/models/model_qrf.joblib is missing.")
    qrf = joblib.load(io.BytesIO(bucket.blob(QRF_MODEL_KEY).download_as_bytes()))
    tables = hrrr_run_tables(card, now)
    weather_source = "hrrr"
    if len(tables) == 0:
        tables = [nws_weather]
        weather_source = "nws_fallback"
    piles = []
    for i in range(len(targets)):
        piles.append([])
    for table in tables:
        covered = np.asarray(targets.isin(table.index))
        if not covered.any():
            continue
        X = fill_inputs(make_features(targets[covered], table, load), load)
        if X[WEATHER].isna().any().any():
            continue
        q = qrf.predict(X[FEATURES], quantiles=POOL_QUANTILES)
        positions = np.where(covered)[0]
        for j in range(len(positions)):
            piles[positions[j]].extend(q[j])
    # an hour no HRRR run covers (should not happen) gets the NWS forecast
    empty = []
    for i in range(len(targets)):
        if len(piles[i]) == 0:
            empty.append(i)
    if len(empty) > 0:
        X = fill_inputs(make_features(targets[empty], nws_weather, load), load)
        q = qrf.predict(X[FEATURES], quantiles=POOL_QUANTILES)
        for j in range(len(empty)):
            piles[empty[j]].extend(q[j])
    low = np.zeros(len(targets))
    expected = np.zeros(len(targets))
    high = np.zeros(len(targets))
    runs_per_hour = []
    for i in range(len(targets)):
        low[i], expected[i], high[i] = np.quantile(piles[i], [0.1, 0.5, 0.9], method="hazen")
        runs_per_hour.append(len(piles[i]) // len(POOL_QUANTILES))
    widen = float(card.get("range_widen", 1.0))
    low = expected - widen * (expected - low)
    high = expected + widen * (high - expected)
    info = {"weather_source": weather_source, "hrrr_runs_read": len(tables) if weather_source == "hrrr" else 0,
            "runs_per_hour_min": int(min(runs_per_hour)), "runs_per_hour_max": int(max(runs_per_hour))}
    return low, expected, high, info


def run_once(dry_run: bool = False) -> dict:
    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    if not bucket.blob(MODEL_KEY).exists():
        return {"status": "no_model", "message": f"Upload your model to gs://{GCS_BUCKET}/{MODEL_KEY} (Lab 5, Part 7)."}
    model = joblib.load(io.BytesIO(bucket.blob(MODEL_KEY).download_as_bytes()))
    card = {}
    if bucket.blob(CARD_KEY).exists():
        card = json.loads(bucket.blob(CARD_KEY).download_as_text())
    model_version = str(card.get("trained_at", "unknown"))

    forecasts = read_csv(bucket, WEATHER_KEY)
    if forecasts is None or len(forecasts) == 0:
        return {"status": "no_weather_data", "message": "scraper-weather-forecast has not written its master yet."}
    load_table = read_csv(bucket, LOAD_KEY)
    if load_table is None or len(load_table) == 0:
        return {"status": "no_load_data", "message": "No ISO-NE load yet: add the ISO_NE_USERNAME and ISO_NE_PASSWORD "
                "repository secrets, re-run 'Deploy scraper-isone', and wait for its next run (:15)."}
    load_table["hour_utc"] = pd.to_datetime(load_table["hour_utc"], utc=True)
    load = load_table.set_index("hour_utc")["load_mw_5min_avg"].sort_index()

    now = datetime.now(timezone.utc)
    this_hour = pd.Timestamp(now).floor("h")
    targets = pd.date_range(this_hour + pd.Timedelta(hours=1), periods=HOURS_AHEAD, freq="h")
    weather = latest_weather(forecasts)
    X = make_features(targets, weather, load)

    # Honest gap filling (rare): a missing weather value takes its neighbor; a missing yesterday load takes the newest load
    filled_weather = int(X[WEATHER].isna().sum().sum())
    X[WEATHER] = X[WEATHER].interpolate(limit_direction="both")
    filled_load = int(X["load_yesterday_mw"].isna().sum())
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    if X[WEATHER].isna().any().any():
        return {"status": "no_weather_data", "message": "The weather forecast does not cover the next 24 hours yet."}

    preds = model.predict(X[FEATURES])
    range_error = ""
    pool_info = {}
    try:                         # a broken range never costs you your expected-value forecast
        if str(card.get("range_method", "none")) == "hrrr_qrf_pooled":
            low, pooled_expected, high, pool_info = pooled_range(bucket, card, targets, load, weather, now)
            preds = pooled_expected            # the expected value is the middle of the pooled pile
            range_method = "hrrr_qrf_pooled"
            if pool_info["weather_source"] != "hrrr":
                range_method = "hrrr_qrf_pooled_nws"
        else:
            low, high, range_method = forecast_range(bucket, card, X, preds, targets)
    except Exception as err:
        low, high, range_method = None, None, "error"
        range_error = str(err)[:300]
    if low is None:
        low = np.full(len(preds), np.nan)
        high = np.full(len(preds), np.nan)
    out = pd.DataFrame({"issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "target_hour_utc": targets.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lead_hours": np.arange(1, HOURS_AHEAD + 1),
                        "predicted_low_mw": np.round(low, 1),
                        "predicted_load_mw": np.round(preds, 1),
                        "predicted_high_mw": np.round(high, 1),
                        "model_version": model_version,
                        "range_method": range_method})
    summary = {"status": "ok", "model_version": model_version, "first_target_utc": out["target_hour_utc"].iloc[0],
               "last_target_utc": out["target_hour_utc"].iloc[-1], "newest_load_hour_utc": load.index.max().isoformat(),
               "min_mw": float(out["predicted_load_mw"].min()), "max_mw": float(out["predicted_load_mw"].max()),
               "range_method": range_method,
               "avg_range_width_mw": None if low is None or np.isnan(low).all() else round(float(np.mean(high - low)), 1),
               "range_error": range_error,
               "filled_weather_values": filled_weather, "filled_load_values": filled_load}
    summary.update(pool_info)
    if dry_run:
        summary["dry_run"] = True
        return summary

    csv_text = out.to_csv(index=False)
    bucket.blob(LATEST_KEY).upload_from_string(csv_text, content_type="text/csv")
    bucket.blob(f"{HISTORY_PREFIX}/{now.strftime('%Y%m%dT%H%M%SZ')}.csv").upload_from_string(csv_text, content_type="text/csv")
    old = read_csv(bucket, MASTER_KEY)
    master = out if old is None else pd.concat([old, out], ignore_index=True)
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index=False), content_type="text/csv")
    summary["latest_key"] = LATEST_KEY
    summary["master_rows"] = int(len(master))
    return summary


def forecast_24h_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
