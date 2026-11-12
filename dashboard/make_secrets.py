"""
Prints the Streamlit Secrets block for your dashboard, ready to paste into Streamlit Community Cloud.

    python3 dashboard/make_secrets.py ~/dashboard-key.json YOUR-BUCKET "Your leaderboard nickname"

Paste the output into your app's Settings -> Secrets box, then DELETE the key file (rm ~/dashboard-key.json).
Never commit the key or the output to GitHub.
"""
import json
import sys

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

key_path = sys.argv[1]
bucket_name = sys.argv[2]
nickname = sys.argv[3]

with open(key_path) as handle:
    key = json.load(handle)

print("nickname = " + json.dumps(nickname))
print("bucket_name = " + json.dumps(bucket_name))
print("")
print("[gcp_service_account]")
for name in key:
    print(name + " = " + json.dumps(key[name]))      # a JSON string is also a valid TOML string
