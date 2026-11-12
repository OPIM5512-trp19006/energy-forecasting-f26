"""
Your own energy forecast dashboard (OPTIONAL, no points).

A public Streamlit page that shows what your forecast-24h Cloud Function is doing right now:
  - your next 24 hours (low, expected and high) next to ISO New England's own forecast
  - how your past forecasts did against the real load, scored the same way as the class leaderboard
  - the Bradley Airport temperature forecast your model is reading

It only READS the files your energy project already writes to gs://<your bucket>/energy/. It never writes anything.

Run it three ways (see dashboard/README.md):
  1. Cloud Shell:  streamlit run dashboard/app.py --server.port 8080   then Web Preview -> port 8080
  2. Your laptop:  copy your energy/ folder down, then pick "A folder on this computer" in the sidebar
  3. Streamlit Community Cloud (free, public link): a read-only service account key goes in the app's Secrets box,
     NEVER in your GitHub repo.
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------------------------------------------
# Settings: change these here, not inside the code below
# ---------------------------------------------------------------------------------------------------------------
SETTINGS = {
    "energy_prefix": "energy",
    "files": {
        "forecasts": "datasets/forecasts_master.csv",            # every forecast your model ever made
        "latest": "forecasts/latest.csv",                        # your newest 24 hours
        "load": "datasets/ct_load_5min_hourly_master.csv",       # the real load (live 5-minute data, hourly average)
        "isone": "datasets/isone_ct_forecast_master.csv",        # ISO New England's own forecast
        "weather": "datasets/bdl_weather_forecast_master.csv",   # NWS forecast for Bradley Airport
    },
    "local_timezone": "America/New_York",
    "interval_alpha": 0.2,                                       # low/high are an 80% range (10th and 90th percentiles)
    "lead_blocks": [(1, 6), (7, 12), (13, 18), (19, 24)],        # same blocks as the class leaderboard
    "default_days": 7,
    "refresh_minutes": 10,
    "colors": {"expected": "#1f4e9c", "range": "rgba(31, 78, 156, 0.18)", "actual": "#111111",
               "isone": "#d9822b", "baseline": "#8a8a8a", "temp": "#c0392b"},
}

st.set_page_config(page_title="My Connecticut load forecast", layout="wide")


# ---------------------------------------------------------------------------------------------------------------
# Reading your files
# ---------------------------------------------------------------------------------------------------------------
def read_secret(name):
    """A value from Streamlit Secrets, or None when there is no secrets file (Cloud Shell, your laptop)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner=False)
def gcs_bucket(bucket_name):
    """Your bucket, read with the service account key in Secrets, or with your own login in Cloud Shell."""
    from google.cloud import storage
    key = read_secret("gcp_service_account")
    if key is not None:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(dict(key))
        client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        try:
            client = storage.Client()
        except Exception:
            client = storage.Client(project="read-only")    # reading a file does not need a project
    return client.bucket(bucket_name)


@st.cache_data(ttl=SETTINGS["refresh_minutes"] * 60, show_spinner=False)
def read_table(kind, where, name):
    """One of your CSV files as a table, or None if it is not there yet."""
    path = SETTINGS["energy_prefix"] + "/" + SETTINGS["files"][name]
    if kind == "folder":
        full_path = os.path.join(where, path)
        if not os.path.exists(full_path):
            return None
        return pd.read_csv(full_path)
    blob = gcs_bucket(where).blob(path)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def to_utc(column):
    return pd.to_datetime(column, utc=True)


def to_local(column):
    return column.dt.tz_convert(SETTINGS["local_timezone"])


# ---------------------------------------------------------------------------------------------------------------
# Scoring: the same rules as the class leaderboard
#   point only:        |actual - expected|
#   with low and high: (0.5 * |actual - expected| + (alpha / 2) * interval score) / 1.5
#                      interval score = width + (2 / alpha) * how far the actual fell outside the range
#   skill = 1 - your score / same-hour-yesterday error, for each block of lead times, then averaged
# ---------------------------------------------------------------------------------------------------------------
def score_forecasts(forecasts, actual, isone):
    pairs = forecasts.copy()
    pairs["issued"] = to_utc(pairs["issued_at_utc"])
    pairs["target"] = to_utc(pairs["target_hour_utc"])
    pairs["actual_mw"] = actual.reindex(pairs["target"]).values
    pairs = pairs[pairs["actual_mw"].notna() & pairs["predicted_load_mw"].notna()].copy()
    if len(pairs) == 0:
        return pairs

    # Same hour yesterday, using only load that was already over when the forecast was made
    source = pairs["target"] - pd.Timedelta(hours=24)
    too_new = source + pd.Timedelta(hours=1) > pairs["issued"]
    source = source.where(~too_new, source - pd.Timedelta(hours=24))
    pairs["same_hour_yesterday_mw"] = actual.reindex(source).values

    # ISO New England: its newest forecast published before yours
    pairs["isone_forecast_mw"] = np.nan
    if isone is not None and len(isone) > 0:
        iso = pd.DataFrame({"iso_issued": to_utc(isone["issued_at_utc"]),
                            "iso_target": to_utc(isone["target_hour_utc"]),
                            "iso_mw": isone["isone_forecast_mw"]})
        iso = iso.dropna().sort_values("iso_issued")
        left = pairs[["issued", "target"]].reset_index().sort_values("issued")
        matched = pd.merge_asof(left, iso, left_on="issued", right_on="iso_issued", left_by="target",
                                right_by="iso_target", direction="backward")
        pairs["isone_forecast_mw"] = matched.set_index("index")["iso_mw"]

    alpha = SETTINGS["interval_alpha"]
    pairs["abs_error_mw"] = (pairs["predicted_load_mw"] - pairs["actual_mw"]).abs()
    pairs["baseline_abs_error_mw"] = (pairs["same_hour_yesterday_mw"] - pairs["actual_mw"]).abs()
    pairs["isone_abs_error_mw"] = (pairs["isone_forecast_mw"] - pairs["actual_mw"]).abs()
    pairs["score_mw"] = pairs["abs_error_mw"]
    pairs["inside_range"] = np.nan
    pairs["range_width_mw"] = np.nan
    has_range = pairs["predicted_low_mw"].notna() & pairs["predicted_high_mw"].notna()
    low = np.minimum(pairs["predicted_low_mw"], pairs["predicted_high_mw"])
    high = np.maximum(pairs["predicted_low_mw"], pairs["predicted_high_mw"])
    y = pairs["actual_mw"]
    width = high - low
    miss = (low - y).clip(lower=0) + (y - high).clip(lower=0)
    interval_score = width + (2 / alpha) * miss
    weighted = (0.5 * pairs["abs_error_mw"] + (alpha / 2) * interval_score) / 1.5
    pairs.loc[has_range, "score_mw"] = weighted[has_range]
    pairs.loc[has_range, "inside_range"] = ((y >= low) & (y <= high))[has_range].astype(float)
    pairs.loc[has_range, "range_width_mw"] = width[has_range]
    pairs["lead_hours"] = pairs["lead_hours"].astype(int)
    return pairs


def skill_pct(rows):
    skills = []
    for first, last in SETTINGS["lead_blocks"]:
        block = rows[(rows["lead_hours"] >= first) & (rows["lead_hours"] <= last)]
        block = block[block["baseline_abs_error_mw"].notna()]
        if len(block) > 0 and block["baseline_abs_error_mw"].mean() > 0:
            skills.append(1 - block["score_mw"].mean() / block["baseline_abs_error_mw"].mean())
    if len(skills) == 0:
        return np.nan
    return 100 * float(np.mean(skills))


def show_number(value, unit, digits=1):
    if value is None or pd.isna(value):
        return "not yet"
    return f"{value:,.{digits}f}{unit}"


# ---------------------------------------------------------------------------------------------------------------
# Sidebar: where your files are
# ---------------------------------------------------------------------------------------------------------------
nickname = read_secret("nickname") or os.getenv("DASHBOARD_NICKNAME") or "My forecast"
bucket_from_secrets = read_secret("bucket_name") or os.getenv("ENERGY_BUCKET")

with st.sidebar:
    st.header("Settings")
    if bucket_from_secrets:
        kind = "bucket"
        where = bucket_from_secrets
        st.caption("Reading your energy project on Google Cloud.")
    else:
        choice = st.radio("Where are your files?", ["My Google Cloud bucket", "A folder on this computer"])
        if choice == "My Google Cloud bucket":
            kind = "bucket"
            where = st.text_input("Bucket name (no gs://)", placeholder="your-energy-bucket").strip()
        else:
            kind = "folder"
            where = st.text_input("Folder that holds energy/", value=os.getenv("ENERGY_DATA_DIR", "")).strip()
    days = st.slider("Days of history to score", min_value=1, max_value=30, value=SETTINGS["default_days"])
    lead = st.slider("Chart: forecasts made this many hours ahead", min_value=1, max_value=24, value=1)
    show_isone = st.checkbox("Show ISO New England's forecast", value=True)
    if st.button("Refresh now"):
        st.cache_data.clear()

st.title(f"{nickname}: Connecticut's electricity load, next 24 hours")
st.caption("A forecast made every hour by my own model running on Google Cloud. Times are Connecticut time. "
           "Scored the same way as the OPIM 5512 class leaderboard.")

if not where:
    st.info("Type your bucket name (or a folder) in the sidebar to start.")
    st.stop()

try:
    forecasts = read_table(kind, where, "forecasts")
    latest = read_table(kind, where, "latest")
    load = read_table(kind, where, "load")
    isone = read_table(kind, where, "isone")
    weather = read_table(kind, where, "weather")
except Exception as err:
    st.error("Could not read your files. Check the bucket name and that the dashboard's service account can read it "
             f"(Storage Object Viewer).\n\n{type(err).__name__}: {str(err)[:300]}")
    st.stop()

if forecasts is None or load is None:
    st.warning("Did not find energy/datasets/forecasts_master.csv and energy/datasets/ct_load_5min_hourly_master.csv. "
               "Is forecast-24h running every hour?")
    st.stop()

actual = load.copy()
actual["hour"] = to_utc(actual["hour_utc"])
actual = actual.dropna(subset=["load_mw_5min_avg"]).drop_duplicates("hour", keep="last")
actual = actual.set_index("hour")["load_mw_5min_avg"].sort_index()

for column in ["predicted_low_mw", "predicted_high_mw"]:
    if column not in forecasts.columns:
        forecasts[column] = np.nan

# ---------------------------------------------------------------------------------------------------------------
# Headline numbers for the last N days
# ---------------------------------------------------------------------------------------------------------------
pairs = score_forecasts(forecasts, actual, isone)
now = pd.Timestamp.now(tz="UTC")
recent = pairs
if len(pairs) > 0:
    recent = pairs[pairs["target"] >= now - pd.Timedelta(days=days)]

st.subheader(f"How I did over the last {days} days")
cols = st.columns(6)
if len(recent) == 0:
    st.info("No scored hours yet: the real load for your forecast hours has not come in.")
else:
    cols[0].metric("CRPS skill vs same hour yesterday", show_number(skill_pct(recent), "%"),
                   help="Higher is better. 0% = no better than copying yesterday's load.")
    cols[1].metric("CRPS (MW)", show_number(recent["score_mw"].mean(), ""),
                   help="The class leaderboard score: lower is better. Equals the average error if you have no range.")
    cols[2].metric("Average error (MW)", show_number(recent["abs_error_mw"].mean(), ""))
    cols[3].metric("Inside my 80% range", show_number(100 * recent["inside_range"].mean(), "%"),
                   help="Aim for about 80%. Much lower = range too narrow; much higher = range too wide.")
    cols[4].metric("Average range width (MW)", show_number(recent["range_width_mw"].mean(), ""))
    cols[5].metric("ISO-NE average error, same hours (MW)", show_number(recent["isone_abs_error_mw"].mean(), ""))

# ---------------------------------------------------------------------------------------------------------------
# Chart 1: the next 24 hours
# ---------------------------------------------------------------------------------------------------------------
st.subheader("My next 24 hours")
if latest is None or len(latest) == 0:
    st.info("No energy/forecasts/latest.csv yet.")
else:
    nxt = latest.copy()
    nxt["when"] = to_local(to_utc(nxt["target_hour_utc"]))
    figure = go.Figure()
    if "predicted_high_mw" in nxt.columns and nxt["predicted_high_mw"].notna().any():
        figure.add_trace(go.Scatter(x=nxt["when"], y=nxt["predicted_high_mw"], mode="lines", line={"width": 0},
                                    showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=nxt["when"], y=nxt["predicted_low_mw"], mode="lines", line={"width": 0},
                                    fill="tonexty", fillcolor=SETTINGS["colors"]["range"], name="My 80% range"))
    figure.add_trace(go.Scatter(x=nxt["when"], y=nxt["predicted_load_mw"], mode="lines+markers",
                                line={"color": SETTINGS["colors"]["expected"], "width": 3}, name="My expected load"))
    if show_isone and isone is not None and len(isone) > 0:
        iso = isone.copy()
        iso["issued"] = to_utc(iso["issued_at_utc"])
        iso["target"] = to_utc(iso["target_hour_utc"])
        iso = iso.sort_values("issued").drop_duplicates("target", keep="last")
        iso = iso[iso["target"].isin(to_utc(nxt["target_hour_utc"]))]
        if len(iso) > 0:
            figure.add_trace(go.Scatter(x=to_local(iso["target"]), y=iso["isone_forecast_mw"], mode="lines",
                                        line={"color": SETTINGS["colors"]["isone"], "dash": "dash"},
                                        name="ISO New England's forecast"))
    issued = to_local(to_utc(nxt["issued_at_utc"])).iloc[0] if "issued_at_utc" in nxt.columns else None
    figure.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="MW",
                         legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(figure, width="stretch")
    if issued is not None:
        st.caption(f"Made {issued.strftime('%a %b %d, %I:%M %p')} Connecticut time.")

# Weather the model is reading
if weather is not None and len(weather) > 0 and "temp_f" in weather.columns:
    snap = weather.copy()
    snap["scraped"] = to_utc(snap["scraped_hour_utc"])
    snap = snap[snap["scraped"] == snap["scraped"].max()]
    snap["when"] = to_local(to_utc(snap["target_hour_utc"]))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=snap["when"], y=snap["temp_f"], mode="lines+markers",
                                line={"color": SETTINGS["colors"]["temp"]}, name="Temperature (F)"))
    figure.update_layout(height=220, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="°F",
                         title={"text": "Bradley Airport temperature forecast (National Weather Service)", "font": {"size": 14}})
    st.plotly_chart(figure, width="stretch")

# ---------------------------------------------------------------------------------------------------------------
# Chart 2: past forecasts vs what really happened
# ---------------------------------------------------------------------------------------------------------------
st.subheader(f"My forecasts made {lead} hour{'s' if lead > 1 else ''} ahead vs the real load")
start = now - pd.Timedelta(days=days)
past = forecasts.copy()
past["target"] = to_utc(past["target_hour_utc"])
past = past[(past["lead_hours"].astype(int) == lead) & (past["target"] >= start)].sort_values("target")
past = past.drop_duplicates("target", keep="last")
seen = actual[actual.index >= start]
figure = go.Figure()
if past["predicted_high_mw"].notna().any():
    figure.add_trace(go.Scatter(x=to_local(past["target"]), y=past["predicted_high_mw"], mode="lines",
                                line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=to_local(past["target"]), y=past["predicted_low_mw"], mode="lines",
                                line={"width": 0}, fill="tonexty", fillcolor=SETTINGS["colors"]["range"],
                                name="My 80% range"))
figure.add_trace(go.Scatter(x=to_local(past["target"]), y=past["predicted_load_mw"], mode="lines",
                            line={"color": SETTINGS["colors"]["expected"]}, name="My expected load"))
figure.add_trace(go.Scatter(x=seen.index.tz_convert(SETTINGS["local_timezone"]), y=seen.values, mode="lines",
                            line={"color": SETTINGS["colors"]["actual"], "width": 2}, name="Real load"))
figure.update_layout(height=400, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="MW",
                     legend={"orientation": "h", "y": 1.12})
st.plotly_chart(figure, width="stretch")

# ---------------------------------------------------------------------------------------------------------------
# Chart 3: error by how far ahead
# ---------------------------------------------------------------------------------------------------------------
if len(recent) > 0:
    st.subheader("Average error by how far ahead the forecast was made")
    by_lead = recent.groupby("lead_hours").agg(mine=("abs_error_mw", "mean"),
                                               crps=("score_mw", "mean"),
                                               yesterday=("baseline_abs_error_mw", "mean"),
                                               isone=("isone_abs_error_mw", "mean")).reset_index()
    figure = go.Figure()
    figure.add_trace(go.Bar(x=by_lead["lead_hours"], y=by_lead["mine"], name="My average error",
                            marker_color=SETTINGS["colors"]["expected"]))
    figure.add_trace(go.Scatter(x=by_lead["lead_hours"], y=by_lead["yesterday"], mode="lines+markers",
                                line={"color": SETTINGS["colors"]["baseline"]}, name="Same hour yesterday"))
    if show_isone and by_lead["isone"].notna().any():
        figure.add_trace(go.Scatter(x=by_lead["lead_hours"], y=by_lead["isone"], mode="lines+markers",
                                    line={"color": SETTINGS["colors"]["isone"], "dash": "dash"},
                                    name="ISO New England"))
    figure.update_layout(height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="Hours ahead",
                         yaxis_title="Average error (MW)", legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(figure, width="stretch")

    with st.expander("The numbers behind the charts"):
        table = recent[["issued_at_utc", "target_hour_utc", "lead_hours", "predicted_low_mw", "predicted_load_mw",
                        "predicted_high_mw", "actual_mw", "same_hour_yesterday_mw", "isone_forecast_mw",
                        "abs_error_mw", "score_mw", "inside_range"]]
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button("Download as CSV", table.to_csv(index=False), file_name="my_scored_forecasts.csv")

newest = forecasts.tail(1)
footer = "Model: " + str(newest["model_version"].iloc[0]) if "model_version" in newest.columns else "Model: unknown"
if "range_method" in newest.columns:
    footer = footer + " · range: " + str(newest["range_method"].iloc[0])
st.caption(footer + " · Real load: ISO New England 5-minute data, averaged per hour.")
