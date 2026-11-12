# forecast-24h-dl: YOUR deep learning model (OPIM 5509 final project) forecasts Connecticut electricity load for the next
# 24 hours, every hour. It replaces forecast-24h for 5509 students: deploy one or the other, not both (they write the
# same forecast files).
#
# Every hour (:20, after the weather scrapers at :05 and the ISO-NE scraper at :15) this function:
#   1. loads your live model:          gs://<bucket>/energy/models/model.keras (+ model_card.json)
#   2. picks the next 24 FULL hours:   if it runs at 2:20 p.m., it forecasts 3 p.m. today ... 2 p.m. tomorrow
#   3. builds the inputs, the SAME code as the final project notebooks: make_window() for a one-input model (FP1's
#      Sequential LSTM: one 192-row table, past week + next 24 hours), make_inputs() for a two-input model (past, future):
#        past   -> the last 168 hours of live load (ct_load_5min_hourly_master.csv) + hour of day
#        future -> the weather for the next 24 hours + the calendar. Weather = the newest HRRR runs from Dr. Dave's
#                  class file (HRRR_RUNS_URL) that reach a full day ahead; if that file is missing or stale, the newest
#                  NWS forecast (bdl_weather_forecast_master.csv)
#   4. makes the low, expected and high (an 80% range: 10th, 50th and 90th percentiles) the way your card says:
#        "range_method": "none"                   -> FP1: one forecast, newest weather run, no range
#        "range_method": "lstm_hrrr_runs"         -> FP2 option A: the model on each of the newest runs; the range is how
#                                                    much the runs' forecasts disagree
#        "range_method": "lstm_quantiles"         -> FP2 option B: a percentile model (several outputs per hour, listed in
#                                                    "quantiles") on the newest run
#        "range_method": "lstm_quantiles_pooled"  -> FP2 option C: the percentile model on each of the newest runs, every
#                                                    percentile pooled into one pile per hour
#      Low, expected and high are always the 10th, 50th and 90th percentile of each hour's pile, np.quantile(...,
#      method="hazen"), exactly like the notebooks. "range_widen": 1.3 (optional) stretches the range 30%.
#   5. writes 24 rows (target_hour_utc, predicted_low_mw, predicted_load_mw, predicted_high_mw, ...):
#        gs://<bucket>/energy/forecasts/latest.csv                       <- replaced every hour
#        gs://<bucket>/energy/forecasts/history/<issued time>.csv        <- one file per hour, never overwritten
#        gs://<bucket>/energy/datasets/forecasts_master.csv              <- every forecast you ever made, in one table
#      plus gs://<bucket>/energy/forecasts/run_summary.json (this function, the Keras version and your network's layers),
#      which the class leaderboard checks: OPIM 5509 forecasts must come from a Keras network you trained.
# No model yet? It politely does nothing ("status": "no_model"). No ISO-NE data yet? "status": "no_load_data".
# HTTP entrypoint: forecast_24h_dl_http

import io
import json
import logging
import os
import tempfile
import traceback
from datetime import datetime, timezone

import keras
import numpy as np
import pandas as pd
from google.cloud import storage
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")

MODEL_KEY = f"{ENERGY_PREFIX}/models/model.keras"
CARD_KEY = f"{ENERGY_PREFIX}/models/model_card.json"
# the class HRRR file, rewritten every hour at :05 by Dr. Dave's hrrr-bdl job (8 newest runs, 0-48 hours ahead)
HRRR_RUNS_URL = os.getenv("HRRR_RUNS_URL", "https://storage.googleapis.com/opim5512-leaderboard-f26/weather/bdl_hrrr_runs.csv")
HRRR_MAX_AGE_HOURS = 14        # newest run older than this = something is wrong upstream: use the NWS forecast instead
MAX_FILLED_LOAD_HOURS = 24     # more missing hours than this in the past week = the load data is broken, do not guess
WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
LOAD_KEY = f"{ENERGY_PREFIX}/datasets/ct_load_5min_hourly_master.csv"
LATEST_KEY = f"{ENERGY_PREFIX}/forecasts/latest.csv"
HISTORY_PREFIX = f"{ENERGY_PREFIX}/forecasts/history"
MASTER_KEY = f"{ENERGY_PREFIX}/datasets/forecasts_master.csv"
RANGE_METHODS = ["none", "lstm_hrrr_runs", "lstm_quantiles", "lstm_quantiles_pooled"]
# What made this hour's forecast (this function, the Keras version, the network's layers): the class collector reads it
# to check that every 5509 forecast comes from a Keras network
RUN_SUMMARY_KEY = f"{ENERGY_PREFIX}/forecasts/run_summary.json"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------------------------------------------------------------------------------------------
# calendar_features, make_inputs and make_window: copied word for word from the final project notebooks. Change them there,
# change them here.
# ----------------------------------------------------------------------------------------------------------------
WEATHER = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct"]
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2020-01-01", end="2030-12-31")


def calendar_features(hours_utc):
    """Hour of day, day of week and time of year as sines and cosines, plus weekend and holiday flags, in Connecticut time."""
    local = pd.DatetimeIndex(hours_utc).tz_convert("America/New_York")
    out = pd.DataFrame(index=pd.DatetimeIndex(hours_utc))
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    out["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    out["is_weekend"] = (local.dayofweek >= 5).astype(float)
    out["is_holiday"] = local.tz_localize(None).normalize().isin(HOLIDAYS).astype(float)
    return out


def make_inputs(issue_hour_utc, load, weather, card):
    """The model's two inputs for ONE forecast issued at issue_hour_utc.
    load: Connecticut load in MW indexed by UTC hour (only hours that are over).
    weather: the WEATHER columns indexed by UTC hour, covering the next 24 hours.
    card: the model card (look-back, horizon and the scaling numbers).
    Returns past (lookback, 3) and future (horizon, 13), both scaled."""
    issue = pd.Timestamp(issue_hour_utc)
    past_hours = pd.date_range(issue - pd.Timedelta(hours=card["lookback_hours"]), periods=card["lookback_hours"], freq="h")
    future_hours = pd.date_range(issue + pd.Timedelta(hours=1), periods=card["horizon_hours"], freq="h")
    past_load = (load.reindex(past_hours).values - card["load_mean"]) / card["load_std"]
    past_cal = calendar_features(past_hours)[["hour_sin", "hour_cos"]].values
    past = np.column_stack([past_load, past_cal])
    future_weather = weather[WEATHER].reindex(future_hours)
    scaled_weather = []
    for column in WEATHER:
        scaled_weather.append((future_weather[column].values - card["weather_mean"][column]) / card["weather_std"][column])
    future = np.column_stack(scaled_weather + [calendar_features(future_hours).values])
    return past.astype("float32"), future.astype("float32")


def make_window(issue_hour_utc, load, weather, card):
    """The model's ONE input for ONE forecast issued at issue_hour_utc (FP1's Sequential LSTM): a table with one row
    per hour and the columns in card["columns"].
    Rows 1-168: the past week. The load is known (load_known = 1); the weather columns are 0.
    Rows 169-192: the next 24 hours. The weather forecast and the calendar are filled in; the load is hidden
    (load_known = 0, load_mw = 0), because that is what the model predicts."""
    issue = pd.Timestamp(issue_hour_utc)
    past_hours = pd.date_range(issue - pd.Timedelta(hours=card["lookback_hours"]), periods=card["lookback_hours"], freq="h")
    future_hours = pd.date_range(issue + pd.Timedelta(hours=1), periods=card["horizon_hours"], freq="h")

    past = calendar_features(past_hours)
    for column in WEATHER:
        past[column] = 0.0
    past["load_known"] = 1.0
    past["load_mw"] = (load.reindex(past_hours).values - card["load_mean"]) / card["load_std"]

    future = calendar_features(future_hours)
    for column in WEATHER:
        future[column] = (weather[column].reindex(future_hours).values - card["weather_mean"][column]) / card["weather_std"][column]
    future["load_known"] = 0.0
    future["load_mw"] = 0.0

    window = pd.concat([past, future])
    return window[card["columns"]].values.astype("float32")
# ----------------------------------------------------------------------------------------------------------------


def read_csv(bucket, key):
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def latest_weather(forecasts: pd.DataFrame) -> pd.DataFrame:
    """For every target hour, the NEWEST NWS forecast that covers it."""
    forecasts["scraped_hour_utc"] = pd.to_datetime(forecasts["scraped_hour_utc"], utc=True)
    forecasts["target_hour_utc"] = pd.to_datetime(forecasts["target_hour_utc"], utc=True)
    forecasts = forecasts.sort_values(["target_hour_utc", "scraped_hour_utc"])
    newest = forecasts.drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")
    return newest


def hrrr_run_tables(targets, n_runs, now):
    """The newest n_runs HRRR runs from the class file that cover ALL the target hours (an LSTM needs the whole day),
    newest last: a list of weather tables indexed by UTC hour. Empty when the file is missing or stale."""
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
    tables = []
    for init in inits:
        table = runs[runs["init_utc"] == init].set_index("hour_utc")[WEATHER]
        if targets.isin(table.index).all() and not table.reindex(targets).isna().any().any():
            tables.append(table)
    return tables[-n_runs:]


def predict_scaled(model, card, this_hour, past_load, tables):
    """Run the model once per weather run (one table per run). A one-input model (FP1's Sequential LSTM) gets one
    make_window table per run; a two-input model (past, future) gets make_inputs. Returns the scaled predictions."""
    if len(model.inputs) == 1:
        windows = []
        for table in tables:
            windows.append(make_window(this_hour, past_load, table, card))
        return np.asarray(model.predict(np.stack(windows), verbose=0))
    futures = []
    for table in tables:
        past, future = make_inputs(this_hour, past_load, table, card)
        futures.append(future)
    past_batch = np.repeat(past[np.newaxis], len(futures), axis=0)
    return np.asarray(model.predict([past_batch, np.stack(futures)], verbose=0))


def predict_piles(card, scaled, medians_only):
    """One pile per hour: (24, values), where values = runs (a one-number model, or each run's 50th percentile when
    medians_only) or runs x percentiles (a percentile model, sorted so they never cross)."""
    mw = scaled * card["load_std"] + card["load_mean"]
    if mw.ndim == 2:                   # (runs, 24): one forecast per hour
        return mw.T
    mw = np.sort(mw, axis=-1)          # (runs, 24, percentiles)
    if medians_only:
        return np.quantile(mw, 0.5, axis=-1, method="hazen").T
    return np.moveaxis(mw, 0, 1).reshape(mw.shape[1], -1)


def low_expected_high(pile_by_hour):
    """10th, 50th and 90th percentile of each hour's pile ("hazen": each value is the middle of its slice, so a pile of
    20 percentiles gives back the model's own 10th, 50th and 90th)."""
    out = np.quantile(pile_by_hour, [0.1, 0.5, 0.9], axis=-1, method="hazen")
    return out[0], out[1], out[2]


def network_layers(model):
    """The network's layers by type, for example ['LSTM', 'RepeatVector', 'Concatenate', 'LSTM',
    'TimeDistributed(Dense)']. A wrapper shows the layer inside it, and a model inside the model (an ensemble) lists its
    own layers in brackets."""
    names = []
    for layer in model.layers:
        name = layer.__class__.__name__
        if name == "InputLayer":
            continue
        inner = getattr(layer, "layer", None)
        if inner is not None:
            name = name + "(" + inner.__class__.__name__ + ")"
        if isinstance(layer, keras.Model):
            name = name + "[" + ", ".join(network_layers(layer)) + "]"
        names.append(name)
    return names


def run_once(dry_run: bool = False) -> dict:
    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    if not bucket.blob(MODEL_KEY).exists() or not bucket.blob(CARD_KEY).exists():
        return {"status": "no_model", "message": f"Upload model.keras and model_card.json to gs://{GCS_BUCKET}/"
                f"{ENERGY_PREFIX}/models/ (final project notebook, Part 7)."}
    card = json.loads(bucket.blob(CARD_KEY).download_as_text())
    with tempfile.TemporaryDirectory() as folder:          # Keras loads .keras files from disk
        path = os.path.join(folder, "model.keras")
        bucket.blob(MODEL_KEY).download_to_filename(path)
        model = keras.models.load_model(path, compile=False)     # the cloud never trains, so no loss function needed
    layers = network_layers(model)
    model_version = str(card.get("trained_at", "unknown"))
    method = str(card.get("range_method", "none"))
    if method not in RANGE_METHODS:
        return {"status": "bad_card", "message": f"range_method '{method}' is not one of {RANGE_METHODS}."}
    quantile_model = len(model.output_shape) == 3
    if method in ["lstm_quantiles", "lstm_quantiles_pooled"] and not quantile_model:
        return {"status": "bad_card", "message": f"range_method '{method}' needs a percentile model (FP2), but this "
                "model makes one number per hour. Use 'none' or 'lstm_hrrr_runs', or upload the FP2 model."}

    load_table = read_csv(bucket, LOAD_KEY)
    if load_table is None or len(load_table) == 0:
        return {"status": "no_load_data", "message": "No ISO-NE load yet: add the ISO_NE_USERNAME and ISO_NE_PASSWORD "
                "repository secrets, re-run 'Deploy scraper-isone', and wait for its next run (:15)."}
    load_table["hour_utc"] = pd.to_datetime(load_table["hour_utc"], utc=True)
    load = load_table.set_index("hour_utc")["load_mw_5min_avg"].sort_index()

    now = datetime.now(timezone.utc)
    this_hour = pd.Timestamp(now).floor("h")
    horizon = int(card["horizon_hours"])
    targets = pd.date_range(this_hour + pd.Timedelta(hours=1), periods=horizon, freq="h")

    # Honest gap filling (rare): a missing hour in the past week takes its neighbors; too many missing = stop
    past_hours = pd.date_range(this_hour - pd.Timedelta(hours=int(card["lookback_hours"])),
                               periods=int(card["lookback_hours"]), freq="h")
    past_load = load.reindex(past_hours)
    filled_load = int(past_load.isna().sum())
    if filled_load > MAX_FILLED_LOAD_HOURS:
        return {"status": "no_load_data", "message": f"{filled_load} of the last {len(past_hours)} hours of load are "
                "missing. The LSTM needs the past week: check scraper-isone."}
    past_load = past_load.interpolate(limit_direction="both")

    # Weather: the newest HRRR runs that reach a full day ahead (one run for "none" and "lstm_quantiles")
    n_runs = 1
    if method in ["lstm_hrrr_runs", "lstm_quantiles_pooled"]:
        n_runs = int(card.get("hrrr_runs", 3))
    tables = hrrr_run_tables(targets, n_runs, now)
    weather_source = "hrrr"
    filled_weather = 0
    if len(tables) == 0:
        forecasts = read_csv(bucket, WEATHER_KEY)
        if forecasts is None or len(forecasts) == 0:
            return {"status": "no_weather_data", "message": "No HRRR runs and no NWS forecast (scraper-weather-forecast)."}
        nws = latest_weather(forecasts)[WEATHER].reindex(targets)
        filled_weather = int(nws.isna().sum().sum())
        nws = nws.interpolate(limit_direction="both")
        if nws.isna().any().any():
            return {"status": "no_weather_data", "message": "The weather forecast does not cover the next 24 hours yet."}
        tables = [nws]
        weather_source = "nws_fallback"

    if len(model.inputs) == 1 and "columns" not in card:
        return {"status": "bad_card", "message": "This model has one input, so model_card.json needs the \"columns\" list "
                "(FP1 Part 7 writes it)."}
    scaled = predict_scaled(model, card, this_hour, past_load, tables)
    piles = predict_piles(card, scaled, medians_only=(method == "lstm_hrrr_runs"))
    low, expected, high = low_expected_high(piles)
    if method == "none":
        low = np.full(horizon, np.nan)
        high = np.full(horizon, np.nan)
    else:
        # "range_widen": 1.3 stretches the range 30% around the expected value, no retraining
        widen = float(card.get("range_widen", 1.0))
        low = expected - widen * (expected - low)
        high = expected + widen * (high - expected)
    range_method = method
    if weather_source != "hrrr":
        range_method = method + "_nws"

    out = pd.DataFrame({"issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "target_hour_utc": targets.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lead_hours": np.arange(1, horizon + 1),
                        "predicted_low_mw": np.round(low, 1),
                        "predicted_load_mw": np.round(expected, 1),
                        "predicted_high_mw": np.round(high, 1),
                        "model_version": model_version,
                        "range_method": range_method})
    summary = {"status": "ok", "model_version": model_version, "first_target_utc": out["target_hour_utc"].iloc[0],
               "last_target_utc": out["target_hour_utc"].iloc[-1], "newest_load_hour_utc": load.index.max().isoformat(),
               "min_mw": float(out["predicted_load_mw"].min()), "max_mw": float(out["predicted_load_mw"].max()),
               "range_method": range_method,
               "avg_range_width_mw": None if np.isnan(low).all() else round(float(np.mean(high - low)), 1),
               "weather_source": weather_source, "weather_runs_used": len(tables),
               "filled_weather_values": filled_weather, "filled_load_values": filled_load,
               "network": ", ".join(layers), "params": int(model.count_params())}
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
    run_summary = {"function": "forecast-24h-dl", "keras_version": keras.__version__,
                   "issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "model_version": model_version,
                   "range_method": range_method, "layers": layers, "params": int(model.count_params())}
    bucket.blob(RUN_SUMMARY_KEY).upload_from_string(json.dumps(run_summary, indent=1), content_type="application/json")
    return summary


def forecast_24h_dl_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
