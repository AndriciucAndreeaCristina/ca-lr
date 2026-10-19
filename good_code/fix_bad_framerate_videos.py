import os
import io
import json
import zipfile
import cv2
import shutil
import subprocess
import re
import unicodedata
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


SERVICE_ACCOUNT_FILE = 'credentials.json'
PARENT_FOLDER_ID = '1QNCf_ceT85DOb5RDbMwZKUMDeTShipLc' 
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TARGET_FPS = 25
LOG_FILE = 'wrongly_cut_log_new.json'


def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


def fix_encoding(filename):
    try:
        return filename.encode('cp437').decode('utf-8')
    except:
        return filename


def normalize_filename(name):
    """Normalize unicode characters for reliable matching"""
    return unicodedata.normalize('NFC', name).lower()


def list_all_zips(service, folder_id):
    query = f"'{folder_id}' in parents and name contains '.zip' and trashed = false"
    return service.files().list(q=query, fields="files(id, name)", pageSize=100).execute().get('files', [])


def download_zip(service, file_id, local_path):
    request = service.files().get_media(fileId=file_id)
    with open(local_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def analyze_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, 0

    try:
        meta_fps = cap.get(cv2.CAP_PROP_FPS)
        ts = []
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()

    if len(ts) < 2:
        return False, meta_fps

    effective_fps = 1000.0 / np.mean(np.diff(ts))

    meta_non_integer = abs(meta_fps - round(meta_fps)) > 0.01
    effective_non_integer = abs(effective_fps - round(effective_fps)) > 0.01

    return meta_non_integer or effective_non_integer, meta_fps


def safe_remove(path, retries=5, delay=0.5):
    import time
    for i in range(retries):
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            return True
        except (PermissionError, OSError) as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(f"  [WARN] Could not remove {path} after {retries} attempts: {e}")
                return False
    return False


def get_video_frame_count(video_path):
    """Get frame count from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def recut_video_with_trimming(raw_path, ann_path, output_path, target_fps=25):
    """
    Recut video while preserving the original segment boundaries from the annotated file.

    Ann file (corrupted FPS) defines the segment boundaries via frame count.
    Raw file (master) needs to be trimmed to match those boundaries, then resampled to target FPS.
    """
    ann_frame_count = get_video_frame_count(ann_path)

    if ann_frame_count <= 0:
        print(f"    [ERROR] Could not read frame count from {os.path.basename(ann_path)}")
        return False

    raw_cap = cv2.VideoCapture(raw_path)
    if not raw_cap.isOpened():
        print(f"    [ERROR] Could not open {os.path.basename(raw_path)}")
        return False

    try:
        raw_fps = raw_cap.get(cv2.CAP_PROP_FPS)
    finally:
        raw_cap.release()

    if raw_fps <= 0:
        print(f"    [ERROR] Could not get FPS from {os.path.basename(raw_path)}")
        return False

    # FIX: duration must be based on target_fps, not raw_fps.
    # This ensures output has exactly ann_frame_count frames at target_fps.
    # Old (buggy): duration_seconds = ann_frame_count / raw_fps
    #   → if raw_fps < target_fps the clip is too long → too many output frames
    #   → if raw_fps > target_fps the clip is too short → too few output frames
    # New (correct): duration_seconds = ann_frame_count / target_fps
    #   → output frames = duration_seconds * target_fps = ann_frame_count ✓
    duration_seconds = ann_frame_count / target_fps

    cmd = [
        'ffmpeg', '-y', '-i', raw_path,
        '-t', f'{duration_seconds:.4f}',        # Trim to exact output duration
        '-r', str(target_fps),                  # Force output framerate
        '-vframes', str(ann_frame_count),       # Hard cap: exactly ann_frame_count frames
        '-filter:v', 'format=yuv420p',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
        '-c:a', 'aac', '-b:a', '128k',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        out_frame_count = get_video_frame_count(output_path)
        print(f"    [TRIM] Annotated: {ann_frame_count}f → Output: {out_frame_count}f @ {target_fps} FPS")
        return True
    else:
        print(f"    [ERROR] FFmpeg failed: {result.stderr}")
        return False


def validate_output_video(path):
    """Validate that output video has proper framerate"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, 0, 0

    try:
        meta_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        is_valid = (meta_fps > 0 and
                   abs(meta_fps - TARGET_FPS) < 1.0 and
                   frame_count > 0)

        return is_valid, meta_fps, frame_count
    finally:
        cap.release()


def recut_video(raw_path, ann_path, output_path):
    """
    Main recutting function using segment-aware trimming
    """
    print(f"    Using segment-aware trimming...", end='', flush=True)
    success = recut_video_with_trimming(raw_path, ann_path, output_path, TARGET_FPS)

    if success:
        is_valid, meta_fps, frame_count = validate_output_video(output_path)
        if is_valid:
            print(f" ✓ (Output FPS: {meta_fps:.2f}, Frames: {frame_count})")
            return True
        else:
            print(f" ✗ Validation failed (Output FPS: {meta_fps:.2f}, Frames: {frame_count})")
            return False
    else:
        print(f" ✗ Trimming failed")
        return False


import subprocess
import librosa
from scipy import signal


def get_audio_offset(raw_path, ann_path, sr=22050):
    """Finds the time offset of ann_path relative to raw_path using cross-correlation."""
    y_raw, _ = librosa.load(raw_path, sr=sr, mono=True)
    y_ann, _ = librosa.load(ann_path, sr=sr, mono=True)

    correlation = signal.correlate(y_raw, y_ann, mode='valid', method='fft')
    offset_samples = np.argmax(correlation)
    offset_seconds = offset_samples / sr
    return offset_seconds


def recut_video_with_audio_sync(raw_path, ann_path, output_path, target_fps=25):
    ann_frame_count = get_video_frame_count(ann_path)

    if ann_frame_count <= 0:
        print(f"    [ERROR] Could not read frame count from {os.path.basename(ann_path)}")
        return False

    raw_cap = cv2.VideoCapture(raw_path)
    if not raw_cap.isOpened():
        return False
    raw_fps = raw_cap.get(cv2.CAP_PROP_FPS)
    raw_cap.release()

    try:
        start_offset = get_audio_offset(raw_path, ann_path)
    except Exception as e:
        print(f"[ERROR] Audio sync failed: {e}")
        return False

    # FIX: duration must be based on target_fps, not raw_fps.
    # This ensures the output has exactly ann_frame_count frames at target_fps.
    # Old (buggy): duration_seconds = ann_frame_count / raw_fps
    #   → when raw_fps < target_fps (e.g. 20 fps raw → 25 fps out), the clip
    #     was too long and FFmpeg produced more frames than ann_frame_count.
    # New (correct): duration_seconds = ann_frame_count / target_fps
    #   → output frames = duration_seconds * target_fps = ann_frame_count ✓
    duration_seconds = ann_frame_count / target_fps

    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_offset:.4f}',       # Seek to aligned start in raw master
        '-t',  f'{duration_seconds:.4f}',   # Extract exactly the right duration
        '-i', raw_path,
        '-r', str(target_fps),              # Force output framerate to 25 fps
        '-vframes', str(ann_frame_count),   # Hard cap: exactly ann_frame_count frames
        '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
        '-filter:v', 'format=yuv420p',
        '-c:a', 'aac', '-b:a', '128k',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        out_count = get_video_frame_count(output_path)
        print(f"[SYNC] Offset: {start_offset:.3f}s | Annotated: {ann_frame_count}f → Output: {out_count}f @ {target_fps} FPS")
        return True

    print(f"[ERROR] FFmpeg error: {result.stderr}")
    return False


def find_source_video(ann_filename, raw_dir, debug=False):
    """
    Find source video using multiple matching strategies with fallbacks.
    Prioritizes matching by SPEAKER to avoid cross-speaker matches.
    Handles special characters in Romanian filenames.
    Returns path to source video or None.
    """
    if debug:
        print(f"    [DEBUG] Looking for source of: {ann_filename}")

    parts = ann_filename.split('_')
    segment_idx = next((i for i, p in enumerate(parts) if p == 'segment'), None)

    if segment_idx is None or segment_idx == 0:
        if debug:
            print(f"    [DEBUG] Could not extract speaker from: {ann_filename}")
            print(f"    [DEBUG] Split parts: {parts}")
        return None

    speaker = '_'.join(parts[:segment_idx])
    segment_num = parts[segment_idx + 1] if segment_idx + 1 < len(parts) else None

    if not segment_num or not segment_num.isdigit():
        if debug:
            print(f"    [DEBUG] Could not extract segment number from: {ann_filename}")
        return None

    word = parts[-2] if len(parts) > 1 else None

    if debug:
        print(f"    [DEBUG] Extracted: speaker={speaker}, segment={segment_num}, word={word}")

    # Strategy 1: Exact match
    base_name = ann_filename.rsplit('_', 1)[0]
    master_file = f"{base_name}.mp4"
    if debug:
        print(f"    [DEBUG] Strategy 1 - Exact match: {master_file}")
    for r_root, _, r_files in os.walk(raw_dir):
        for r_file in r_files:
            if r_file == master_file:
                if debug:
                    print(f"    [DEBUG] ✓ Found: {r_file}")
                return os.path.join(r_root, r_file)

    # Strategy 2: Case-insensitive exact match
    if debug:
        print(f"    [DEBUG] Strategy 2 - Case-insensitive exact match")
    for r_root, _, r_files in os.walk(raw_dir):
        for r_file in r_files:
            if r_file.lower() == master_file.lower():
                if debug:
                    print(f"    [DEBUG] ✓ Found (case-insensitive): {r_file}")
                return os.path.join(r_root, r_file)

    # Strategy 3: Speaker + Segment (exact segment number)
    if debug:
        print(f"    [DEBUG] Strategy 3 - Speaker + Segment: {speaker} + segment_{segment_num}")
    for r_root, _, r_files in os.walk(raw_dir):
        for r_file in r_files:
            if r_file.lower().endswith('.mp4'):
                if r_file.lower().startswith(speaker.lower()):
                    if re.search(rf'segment_{segment_num}(?:_|\.|$)', r_file):
                        if debug:
                            print(f"    [DEBUG] ✓ Found (speaker+segment): {r_file}")
                        return os.path.join(r_root, r_file)

    # Strategy 4: Speaker + Word
    if debug:
        print(f"    [DEBUG] Strategy 4 - Speaker + Word: {speaker} + {word}")
    if word:
        for r_root, _, r_files in os.walk(raw_dir):
            for r_file in r_files:
                if r_file.lower().endswith('.mp4'):
                    if r_file.lower().startswith(speaker.lower()) and word.lower() in r_file.lower():
                        if debug:
                            print(f"    [DEBUG] ✓ Found (speaker+word): {r_file}")
                        return os.path.join(r_root, r_file)

    # Strategy 5: Speaker only (last resort)
    if debug:
        print(f"    [DEBUG] Strategy 5 - Speaker only (fallback)")
    for r_root, _, r_files in os.walk(raw_dir):
        for r_file in r_files:
            if r_file.lower().endswith('.mp4'):
                if r_file.lower().startswith(speaker.lower()):
                    if debug:
                        print(f"    [DEBUG] ✓ Found (speaker only): {r_file}")
                    return os.path.join(r_root, r_file)

    if debug:
        print(f"    [DEBUG] ✗ No match found after all strategies")
    return None


def process_recursive(service, adnotate_folder_id, raw_root_id, log_data):
    """
    Process zips directly in adnotate and raw folders (no subfolders)
    """
    zips = list_all_zips(service, adnotate_folder_id)
    raw_zips = list_all_zips(service, raw_root_id)

    if zips:
        for z in zips:
            ann_dir, raw_dir = "temp_ann", "temp_raw"
            ann_zip_path = f"ann_{z['name']}"
            raw_zip_name = z['name'].replace('_adnotate.zip', '.zip')
            raw_zip_path = f"raw_{raw_zip_name}"

            print(f"--- Processing Zip: {z['name']} ---")
            print(f"  Cleaning temporary directories...")
            for temp_dir in [ann_dir, raw_dir]:
                if os.path.exists(temp_dir):
                    safe_remove(temp_dir)

            os.makedirs(ann_dir, exist_ok=True)
            os.makedirs(raw_dir, exist_ok=True)

            print(f"  Downloading annotated zip...")
            download_zip(service, z['id'], ann_zip_path)

            print(f"  Extracting annotated zip...")
            with zipfile.ZipFile(ann_zip_path, 'r') as zf:
                for m in zf.infolist():
                    m.filename = fix_encoding(m.filename)
                    zf.extract(m, ann_dir)

            raw_zip_normalized = normalize_filename(raw_zip_name)
            rz_id = next((rz['id'] for rz in raw_zips if normalize_filename(rz['name']) == raw_zip_normalized), None)
            if rz_id:
                print(f"  ✓ Found matching raw zip: {raw_zip_name}")
                print(f"  Downloading raw zip...")
                download_zip(service, rz_id, raw_zip_path)

                print(f"  Extracting raw zip...")
                with zipfile.ZipFile(raw_zip_path, 'r') as zf:
                    for m in zf.infolist():
                        m.filename = fix_encoding(m.filename)
                        zf.extract(m, raw_dir)
            else:
                print(f"  ✗ Could not find matching raw zip: {raw_zip_name}")
                continue

            for root, _, files in os.walk(ann_dir):
                for file in files:
                    if file.lower().endswith('.mp4'):
                        path = os.path.join(root, file)
                        is_bad, meta_fps = analyze_video(path)

                        if is_bad:
                            print(f"  [ANOMALY] {file} (Meta FPS: {meta_fps:.2f})")
                            parts = file.split('_')
                            class_name = parts[-2] if len(parts) > 1 else "unknown"

                            source_path = find_source_video(file, raw_dir, debug=True)

                            if source_path:
                                out_dir = os.path.join("recut_fixed_v2", class_name)
                                os.makedirs(out_dir, exist_ok=True)
                                out_path = os.path.join(out_dir, file)

                                source_filename = os.path.basename(source_path)
                                print(f"  [FIXING] {file} from {source_filename}")

                                if recut_video_with_audio_sync(source_path, path, out_path):
                                    log_data.append({
                                        "original_path": path,
                                        "raw_path": source_path,
                                        "new_saved_path": out_path,
                                        "status": "success"
                                    })
                                else:
                                    print(f"    [FAILED] Could not fix {file}")
                                    log_data.append({
                                        "original_path": path,
                                        "raw_path": source_path,
                                        "new_saved_path": out_path,
                                        "status": "failed"
                                    })
                            else:
                                print(f"  [WARN] Could not find matching source video")
                                log_data.append({
                                    "original_path": path,
                                    "raw_path": None,
                                    "new_saved_path": None,
                                    "status": "source_not_found"
                                })

            print(f"  Cleaning up temporary files...")
            for p in [ann_zip_path, raw_zip_path]:
                if os.path.exists(p):
                    safe_remove(p)

            for temp_dir in [ann_dir, raw_dir]:
                if os.path.exists(temp_dir):
                    safe_remove(temp_dir)


def main():
    service = get_drive_service()
    main_folders = service.files().list(
        q=f"'{PARENT_FOLDER_ID}' in parents and mimeType = 'application/vnd.google-apps.folder'"
    ).execute().get('files', [])

    adnotate_id = next((f['id'] for f in main_folders if f['name'].lower() == 'adnotate'), None)
    raw_root_id = next((f['id'] for f in main_folders if f['name'].lower() == 'raw'), None)

    if not adnotate_id or not raw_root_id:
        print("Could not find 'adnotate' or 'raw' folders. Check PARENT_FOLDER_ID.")
        return

    print(f"Starting scan...")
    print(f"  Adnotate folder: {adnotate_id}")
    print(f"  Raw folder: {raw_root_id}")
    print()

    log_data = []
    process_recursive(service, adnotate_id, raw_root_id, log_data)

    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\nProcess complete. Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
