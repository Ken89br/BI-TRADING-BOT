#ml_utils.py
import os
import requests
import base64
import json
import pandas as pd


def download_model(url: str, dest: str = "model.pkl") -> bool:
    if not url:
        print("⚠️ No MODEL_URL provided.")
        return False
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"⬇️ Model downloaded from {url} to {dest}")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def upload_model(model_path: str, upload_url: str) -> bool:
    """
    Upload model file (model.pkl) to Firebase/S3/etc.
    """
    if not upload_url:
        print("⚠️ UPLOAD_MODEL_URL is not set.")
        return False

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False

    try:
        with open(model_path, "rb") as f:
            response = requests.put(upload_url, data=f)
        if response.status_code in (200, 201):
            print(f"✅ Model uploaded to {upload_url}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception during upload: {e}")
        return False


def upload_to_github(file_path, repo, path, token, commit_msg="Update model.pkl"):
    """
    Upload or update a file (model.pkl, CSV, etc.) in GitHub.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    try:
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return False

    headers = {"Authorization": f"token {token}"}
    sha = None
    r = requests.get(api_url, headers=headers)

    if r.status_code == 200:
        sha = r.json().get("sha")

    data = {
        "message": commit_msg,
        "content": content,
        "branch": "main"
    }
    if sha:
        data["sha"] = sha

    try:
        r = requests.put(api_url, headers=headers, data=json.dumps(data))
        if r.status_code in (200, 201):
            print(f"✅ File '{file_path}' uploaded to GitHub at {path}")
            return True
        else:
            print(f"❌ GitHub upload failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        print(f"❌ GitHub upload exception: {e}")
        return False


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common indicators to a price DataFrame.
    Requires: open, high, low, close, volume
    """
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df
    
