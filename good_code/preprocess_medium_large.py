import subprocess
import time
from pathlib import Path
import unicodedata
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_ROOT      = Path(r"C:\videos\LiRoPod")
SYNC_OUT      = Path(r"C:\videos\LiRoPod_sync")
RUN_PIPELINE  = Path(r"D:\masters\disertation\ca-lr\multivsr\preprocess\run_pipeline.py")
CWD           = RUN_PIPELINE.parent
LOG_DIR       = SYNC_OUT / "logs"

MIN_TRACK     = 5
MIN_FACE_SIZE = 10
VALID_SPLITS  = {"train_m", "val_m", "test_m"}

# ── GPU WARNING ────────────────────────────────────────────────────────────────
# Face detection (S3FD) runs on the GPU inside run_pipeline.py.
# With a single GPU, running more than 1 worker in parallel will cause
# CUDA out-of-memory errors or severe slowdowns because all workers share
# the same GPU memory.
#
#   Single GPU  → NUM_WORKERS = 1  (safe, recommended)
#   Two GPUs    → NUM_WORKERS = 2  (requires passing --device per worker,
#                                   see note below)
# ──────────────────────────────────────────────────────────────────────────────
NUM_WORKERS = 3


def sanitize(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^\w\-]", "_", ascii_name)


def process_single_video(vid_path, word_name, split_name, t_start, processed_counter, total_videos):
    reference = f"{word_name}_{split_name}_{sanitize(vid_path.stem)}"
    crop_dir  = SYNC_OUT / "pycrop" / reference
    log_file  = LOG_DIR / f"{reference}.log"

    # Skip if already processed
    if crop_dir.exists() and any(crop_dir.glob("*.avi")):
        return reference, 0.0, "SKIPPED"

    cmd = [
        "python", str(RUN_PIPELINE),
        "--videofile",      str(vid_path),
        "--reference",      reference,
        "--data_dir",       str(SYNC_OUT),
        "--min_track",      str(MIN_TRACK),
        "--min_face_size",  str(MIN_FACE_SIZE),
    ]

    t0 = time.time()
    with open(log_file, "w") as log:
        res = subprocess.run(cmd, cwd=str(CWD), stdout=log, stderr=log)
    elapsed = time.time() - t0

    if res.returncode != 0:
        return reference, elapsed, "FAILED"
    return reference, elapsed, "OK"


if __name__ == "__main__":
    assert RUN_PIPELINE.exists(), f"Not found: {RUN_PIPELINE}"
    SYNC_OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Gather tasks
    tasks = []
    for word_dir in sorted(RAW_ROOT.iterdir()):
        if not word_dir.is_dir(): continue
        word_name = sanitize(word_dir.name)
        for split_dir in sorted(word_dir.iterdir()):
            if not split_dir.is_dir(): continue
            split_name = split_dir.name
            if split_name not in VALID_SPLITS: continue
            for vid_path in sorted(split_dir.glob("*.mp4")):
                tasks.append((vid_path, word_name, split_name))

    total_videos = len(tasks)
    print(f"Found {total_videos} videos to process with {NUM_WORKERS} worker(s).\n")

    processed  = 0
    failed     = []
    t_start    = time.time()

    # ThreadPoolExecutor is correct here — work happens in subprocesses,
    # so the GIL is irrelevant and thread overhead is lower than process overhead.
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_single_video, *task, t_start, processed, total_videos): task
            for task in tasks
        }

        for future in as_completed(futures):
            processed += 1
            reference, elapsed, status = future.result()

            # ETA based on actual wall-clock time, not summed per-job time
            wall_elapsed = time.time() - t_start
            avg_wall     = wall_elapsed / processed
            eta_sec      = avg_wall * (total_videos - processed)

            if status == "FAILED":
                failed.append(reference)
                print(
                    f"[{processed}/{total_videos}] FAILED  {reference} "
                    f"({elapsed:.1f}s) — see logs/{reference}.log"
                )
            elif status == "SKIPPED":
                print(f"[{processed}/{total_videos}] SKIP    {reference}")
            else:
                print(
                    f"[{processed}/{total_videos}] OK      {reference} "
                    f"| {elapsed:.1f}s | ETA: {eta_sec/60:.1f} min"
                )

    wall_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done. {processed}/{total_videos} videos processed.")
    print(f"Wall-clock time: {wall_total/60:.1f} min")
    if failed:
        print(f"\n{len(failed)} failed jobs (check logs/ for details):")
        for ref in failed:
            print(f"  {ref}")
            
#             python vtp/extract_feats.py \
#   --videos_root "C:\videos\LiRoPod_sync\pycrop" \
#   --file_list "**/*.avi" \
#   --feats_root "C:\videos\LiRoPod_vtp_features" \
#   --ckpt_path "D:\masters\disertation\ca-lr\feature_extractor.pth" \
#   --num_parts 1 --part 0 --builder vtp24x24