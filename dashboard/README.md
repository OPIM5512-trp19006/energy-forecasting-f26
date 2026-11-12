# Your own forecast dashboard (optional, no points)

A public web page for **your** live energy forecast, built with [Streamlit](https://streamlit.io). It is optional and
worth no points: build it if you want something to show an employer or put on your resume. It only **reads** the files
your energy project already writes every hour; it never changes anything.

What it shows (Connecticut time):
- your next 24 hours (low, expected and high) next to ISO New England's own forecast
- the Bradley Airport temperature forecast your model is reading
- your past forecasts against the real load, for any lead time from 1 to 24 hours ahead
- your score over the last 1 to 30 days, worked out the same way as the class leaderboard (CRPS, CRPS skill vs same hour
  yesterday, average error, how often the real load landed inside your 80% range)

Everything to change (colors, files, time zone) is in the `SETTINGS` dictionary at the top of [`app.py`](app.py).
Make it yours: add a chart, change the colors, explain your model in a sentence or two.

## 1. Try it in Cloud Shell (5 minutes, no keys)

Open Cloud Shell **in your energy project**, then:

```bash
cd ~ && git clone https://github.com/YOUR-GITHUB-NAME/energy-forecasting.git 2>/dev/null; cd ~/energy-forecasting && git pull
python3 -m venv ~/dash-venv && ~/dash-venv/bin/pip install -q -r dashboard/requirements.txt
ENERGY_BUCKET=YOUR-BUCKET DASHBOARD_NICKNAME="Your nickname" ~/dash-venv/bin/streamlit run dashboard/app.py --server.port 8080
```

Click **Web Preview → Preview on port 8080**. Cloud Shell reads your bucket with your own login, so no key is needed.
Press Ctrl+C to stop it. (Get the newest dashboard any time with **Sync fork → Update branch** on GitHub.)

## 2. Put it on the internet (free, Streamlit Community Cloud)

The public app needs its own read-only login to your bucket: a **service account** that can only *view* files in that
one bucket. Its key goes in the app's **Secrets** box on Streamlit, **never in GitHub**.

**a. Make the read-only service account and its key** (Cloud Shell, in your energy project):

```bash
PROJECT_ID=$(gcloud config get-value project)
BUCKET=YOUR-BUCKET
gcloud iam service-accounts create dashboard-reader --display-name="Streamlit dashboard (read only)"
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member=serviceAccount:dashboard-reader@$PROJECT_ID.iam.gserviceaccount.com --role=roles/storage.objectViewer
gcloud iam service-accounts keys create ~/dashboard-key.json --iam-account=dashboard-reader@$PROJECT_ID.iam.gserviceaccount.com
python3 ~/energy-forecasting/dashboard/make_secrets.py ~/dashboard-key.json $BUCKET "Your nickname"
```

Copy everything the last command prints (it starts with `nickname =`), then delete the key file:

```bash
rm ~/dashboard-key.json
```

**b. Create the app.** Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub → **Create app** →
your `energy-forecasting` fork, branch `main`, main file path `dashboard/app.py`. Open **Advanced settings**, paste the
block into **Secrets**, and click **Deploy**. In a minute or two you have a public link like
`https://your-name-energy.streamlit.app`.

## Rules

- **Never commit a key.** The key belongs only in Streamlit's Secrets box. This repo's `.gitignore` already blocks
  `secrets.toml` and `*-key.json`, but check `git status` before every push. If a key ever lands on GitHub, delete it
  right away: `gcloud iam service-accounts keys list --iam-account=dashboard-reader@$PROJECT_ID.iam.gserviceaccount.com`,
  then `gcloud iam service-accounts keys delete KEY_ID --iam-account=...`.
- **Nickname only.** Use your leaderboard nickname on the page, not your name or NetID.
- The dashboard's real load is ISO New England's live 5-minute data, so its numbers can differ a little from the class
  leaderboard. The leaderboard is the official score.
- If `keys create` says key creation is disabled for your project, skip part 2 and use part 1 (Cloud Shell) for
  screenshots.

## Run it on your laptop instead

Copy your files down (`gcloud storage cp -r gs://YOUR-BUCKET/energy ./mydata/`), run
`streamlit run dashboard/app.py`, and choose **A folder on this computer** → `mydata` in the sidebar.
