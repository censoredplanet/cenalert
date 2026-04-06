import os, json, mimetypes
from pathlib import Path
import requests
from datetime import datetime

SLACK_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_CHANNEL_ID = os.environ["SLACK_CHANNEL_ID"]
GET_UPLOAD_URL = "https://slack.com/api/files.getUploadURLExternal"
COMPLETE_UPLOAD = "https://slack.com/api/files.completeUploadExternal"
CHAT_POST = "https://slack.com/api/chat.postMessage"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
TEXT_EXTS = {".txt", ".md"}

def chat_post_text(text: str):
    headers = {"Authorization": f"Bearer {SLACK_TOKEN}",
               "Content-Type": "application/json;charset=utf-8"}
    body = {"channel": SLACK_CHANNEL_ID, "text": text}
    r = requests.post(CHAT_POST, headers=headers, data=json.dumps(body), timeout=20)
    r.raise_for_status()
    if not r.json().get("ok"):
        raise RuntimeError(f"chat.postMessage error: {r.text}")

def slack_get_upload_url(filename: str, length: int):
    headers = {"Authorization": f"Bearer {SLACK_TOKEN}"}
    data = {"filename": filename, "length": str(length)}
    r = requests.post(GET_UPLOAD_URL, headers=headers, data=data, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(f"getUploadURLExternal error: {payload}")
    return payload["upload_url"], payload["file_id"]

def slack_upload_file(upload_url: str, filepath: Path):
    ctype, _ = mimetypes.guess_type(filepath.name)
    if not ctype:
        ctype = "application/octet-stream"
    with open(filepath, "rb") as f:
        files = {"filename": (filepath.name, f, ctype)}
        r = requests.post(upload_url, files=files, timeout=60)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"Upload failed HTTP {r.status_code}: {r.text}")

def slack_complete_upload(file_id: str, initial_comment: str):
    headers = {"Authorization": f"Bearer {SLACK_TOKEN}",
               "Content-Type": "application/json;charset=utf-8"}
    body = {"files": [{"id": file_id}],
            "channel_id": SLACK_CHANNEL_ID,
            "initial_comment": initial_comment}
    r = requests.post(COMPLETE_UPLOAD, headers=headers, data=json.dumps(body), timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(f"completeUploadExternal error: {payload}")

def _pair_and_send_in_dir(ts_dir: Path):
    """Send pairs inside a single timestamp directory."""
    texts = sorted([p for p in ts_dir.iterdir() if p.is_file() and p.suffix.lower() in TEXT_EXTS])
    images = sorted([p for p in ts_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    others = sorted([p for p in ts_dir.iterdir()
                     if p.is_file() and p.suffix.lower() not in TEXT_EXTS | IMAGE_EXTS])

    # Pair text+image by index; if counts differ, leftover items get sent solo.
    pairs = min(len(texts), len(images))
    for idx in range(pairs):
        txt = texts[idx].read_text(encoding="utf-8")
        img = images[idx]
        upload_url, file_id = slack_get_upload_url(img.name, img.stat().st_size)
        slack_upload_file(upload_url, img)
        slack_complete_upload(file_id, txt)
        print(f"[OK] Sent pair from {ts_dir.name}: {texts[idx].name} + {img.name}")

    # Remaining texts (if any)
    for txt_path in texts[pairs:]:
        chat_post_text(txt_path.read_text(encoding="utf-8"))
        print(f"[OK] Sent text from {ts_dir.name}: {txt_path.name}")

    # Remaining images (if any)
    for img_path in images[pairs:]:
        upload_url, file_id = slack_get_upload_url(img_path.name, img_path.stat().st_size)
        slack_upload_file(upload_url, img_path)
        slack_complete_upload(file_id, img_path.name)
        print(f"[OK] Uploaded image from {ts_dir.name}: {img_path.name}")

    # Other files (if any)
    for other in others:
        upload_url, file_id = slack_get_upload_url(other.name, other.stat().st_size)
        slack_upload_file(upload_url, other)
        slack_complete_upload(file_id, other.name)
        print(f"[OK] Uploaded other from {ts_dir.name}: {other.name}")

def send_today_timestamp_dirs(base_folder: Path):
    today_tag = datetime.now().strftime("%Y%m%d")

    ts_dirs = sorted(
        [p for p in base_folder.iterdir() if p.is_dir() and p.name.startswith(today_tag + "_")],
        reverse=True
    )

    if not ts_dirs:
        print(f"[INFO] No timestamp dirs for today ({today_tag}) found in {base_folder}")
        return

    print(f"[INFO] Found {len(ts_dirs)} timestamp dirs for {today_tag}:")
    for d in ts_dirs:
        print(" -", d.name)

    for d in ts_dirs:
        try:
            _pair_and_send_in_dir(d)
        except Exception as e:
            # Log and continue with remaining dirs
            print(f"[ERROR] Failed to send dir {d.name}: {e}")

    
if __name__ == "__main__":
    import importlib.util, importlib.machinery

    loader = importlib.machinery.SourceFileLoader("bin/config", "./bin/config")
    spec = importlib.util.spec_from_loader("bin/config", loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)

    # Prefer new OUTPUT_DIR; fall back to legacy ALERTS for backward compatibility
    OUTPUT_DIR = getattr(config, "OUTPUT_DIR", None) or getattr(config, "ALERTS")
    TOPIC = getattr(config, "TOPIC")

    folder = Path(OUTPUT_DIR, TOPIC)
    send_today_timestamp_dirs(folder)
