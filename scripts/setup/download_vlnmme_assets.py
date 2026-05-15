#!/usr/bin/env python3
"""Download VLN-MME datasets and selected Hugging Face model weights.

The script is intended to run inside the vlnmme container with the ``vlnmme``
conda environment active. Hugging Face Hub downloads are resumable; marked
observation images are also skipped one PNG at a time so interrupted runs can
be resumed safely.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse


DATASET_REPOS = (
    ("annotations", "VLN-MME/VLN_annotations", "dataset"),
    ("mp3d", "VLN-MME/MP3D", "dataset"),
)

MODEL_REPOS = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_5_9b": "Qwen/Qwen3.5-9B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "internvl3_2b": "OpenGVLab/InternVL3-2B",
}

DEFAULT_MODELS = ("qwen2_5_vl", "qwen2_5_vl_3b", "qwen3_vl_4b", "internvl3_2b")
DIRECTIONS = ("left", "front", "right", "back")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    env_data = os.environ.get("VLNMME_UPSTREAM_DATA_DIR") or os.environ.get("VLNMME_DATA_DIR")
    if env_data:
        return Path(env_data)
    return repo_root_from_script() / "third_party" / "VLN-MME" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all VLN-MME runtime data and selected model weights."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="VLN-MME data directory. Default: VLNMME_UPSTREAM_DATA_DIR or third_party/VLN-MME/data.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=[*MODEL_REPOS.keys(), "all"],
        help=(
            "Model weights to cache in HF_HOME. Default: "
            f"{' '.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override HF_ENDPOINT for this run, e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache dir. Default: HF_HOME / hub default.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel workers for snapshot downloads. Default: 2.",
    )
    parser.add_argument("--skip-data", action="store_true", help="Skip annotations and MP3D files.")
    parser.add_argument("--skip-marked-obs", action="store_true", help="Skip marked observation PNGs.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model weight downloads.")
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Download marked observations for only the first N scans. Useful for smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading.")
    parser.add_argument(
        "--no-network-check",
        action="store_true",
        help="Skip the quick TCP check for the Hugging Face endpoint.",
    )
    return parser.parse_args()


def require_hf_packages(args: argparse.Namespace) -> None:
    required_modules = ["huggingface_hub"]
    if not args.skip_marked_obs:
        required_modules.extend(["datasets", "tqdm"])

    missing: list[str] = []
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            missing.append(exc.name)

    if missing:
        for name in dict.fromkeys(missing):
            print(f"[error] Missing Python package: {name}", file=sys.stderr)
        print(
            "[error] Activate an environment with the required packages, then rerun this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def configure_endpoint(endpoint: str | None) -> str:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def configure_cache_environment(cache_dir: Path | None, dry_run: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if cache_dir is None or os.environ.get("HF_XET_CACHE"):
        return

    xet_cache = cache_dir / "xet"
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    if not dry_run:
        xet_cache.mkdir(parents=True, exist_ok=True)


def ensure_writable_dir(path: Path, label: str, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".write-test.", dir=path, delete=True):
            pass
    except OSError as exc:
        raise SystemExit(
            f"[error] {label} is not writable: {path}\n"
            f"[error] {exc}\n"
            "[error] Choose a writable --cache-dir or set HF_XET_CACHE to a writable path."
        ) from exc


def check_network(endpoint: str, timeout: int = 8) -> None:
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if not host:
        raise SystemExit(f"[error] Invalid HF endpoint: {endpoint}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    print(f"[network] HF_ENDPOINT={endpoint}")
    print(f"[network] checking {host}:{port} ...")
    start = time.time()
    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            print(f"[network] OK in {time.time() - start:.2f}s")
    except OSError as exc:
        print(f"[network] FAILED: {exc}", file=sys.stderr)
        print("[network] Check proxy/VPN/HF_ENDPOINT and rerun.", file=sys.stderr)
        sys.exit(1)


def selected_models(tokens: list[str]) -> list[tuple[str, str]]:
    if "all" in tokens:
        names = list(MODEL_REPOS)
    else:
        names = tokens

    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        result.append((name, MODEL_REPOS[name]))
    return result


def snapshot_download_compat(**kwargs) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("max_workers", None)
        return snapshot_download(**kwargs)


def download_snapshot(
    *,
    stage: str,
    repo_id: str,
    repo_type: str | None,
    local_dir: Path | None,
    cache_dir: Path | None,
    max_workers: int,
    dry_run: bool,
) -> None:
    print()
    print(f"[stage] {stage}")
    print(f"  repo: {repo_id}")
    if repo_type:
        print(f"  repo_type: {repo_type}")
    if local_dir:
        print(f"  local_dir: {local_dir}")
    else:
        print(f"  cache: {cache_dir or os.environ.get('HF_HOME') or 'huggingface default'}")

    if dry_run:
        return

    kwargs = {
        "repo_id": repo_id,
        "max_workers": max_workers,
    }
    if repo_type:
        kwargs["repo_type"] = repo_type
    if local_dir:
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)

    result = snapshot_download_compat(**kwargs)
    print(f"[done] {stage}: {result}")


def ensure_scans_file(data_dir: Path, dry_run: bool) -> Path:
    scans_file = data_dir / "MP3D" / "scans.txt"
    if scans_file.exists():
        return scans_file
    print(f"[marked_obs] missing {scans_file}; downloading MP3D metadata first.")
    if dry_run:
        return scans_file
    download_snapshot(
        stage="mp3d",
        repo_id="VLN-MME/MP3D",
        repo_type="dataset",
        local_dir=data_dir / "MP3D",
        cache_dir=None,
        max_workers=8,
        dry_run=False,
    )
    return scans_file


def read_scans(scans_file: Path) -> list[str]:
    return [line.strip() for line in scans_file.read_text().splitlines() if line.strip()]


def is_complete_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def save_png_atomic(image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def download_marked_observations(data_dir: Path, max_scans: int | None, dry_run: bool) -> None:
    from datasets import load_dataset
    from tqdm import tqdm

    save_dir = data_dir / "marked_obs"
    scans_file = ensure_scans_file(data_dir, dry_run=dry_run)

    print()
    print("[stage] marked_obs")
    print(f"  repo: VLN-MME/MP3D_marked_obs")
    print(f"  output: {save_dir}")
    print("  resume: skip existing non-empty PNG files")

    if dry_run:
        return

    scans = read_scans(scans_file)
    if max_scans is not None:
        scans = scans[:max_scans]

    save_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0
    for index, scan_id in enumerate(scans, start=1):
        print()
        print(f"[marked_obs] scan {index}/{len(scans)}: {scan_id}")
        dataset = load_dataset("VLN-MME/MP3D_marked_obs", split=scan_id, streaming=True)
        saved = 0
        skipped = 0

        progress = tqdm(dataset, desc=f"{scan_id}", unit="viewpoint")
        for item in progress:
            viewpoint_id = item["viewpoint_id"]
            for direction in DIRECTIONS:
                target = save_dir / f"{scan_id}_{viewpoint_id}_{direction}.png"
                if is_complete_file(target):
                    skipped += 1
                    continue
                save_png_atomic(item[direction], target)
                saved += 1
            progress.set_postfix(saved=saved, skipped=skipped)

        total_saved += saved
        total_skipped += skipped
        print(f"[marked_obs] {scan_id}: saved={saved}, skipped={skipped}")

    print()
    print(f"[done] marked_obs: saved={total_saved}, skipped={total_skipped}, output={save_dir}")


def print_auth_status() -> None:
    from huggingface_hub import HfApi

    try:
        whoami = HfApi().whoami()
        user = whoami.get("name") or whoami.get("fullname") or "unknown"
        print(f"[auth] logged in as: {user}")
    except Exception as exc:
        print(f"[auth] warning: not logged in or whoami failed: {exc}")
        print("[auth] VLN-MME/MP3D is gated; run `hf auth login` if MP3D download fails.")


def main() -> None:
    args = parse_args()
    endpoint = configure_endpoint(args.endpoint)

    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    configure_cache_environment(cache_dir, dry_run=args.dry_run)
    if cache_dir is not None:
        ensure_writable_dir(cache_dir, "--cache-dir", dry_run=args.dry_run)
    if os.environ.get("HF_XET_CACHE"):
        ensure_writable_dir(Path(os.environ["HF_XET_CACHE"]), "HF_XET_CACHE", dry_run=args.dry_run)

    require_hf_packages(args)

    print("[plan] VLN-MME asset download")
    print(f"  data_dir: {data_dir}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME') or 'unset'}")
    print(f"  HF_XET_CACHE: {os.environ.get('HF_XET_CACHE') or 'unset'}")
    print(f"  HF_HUB_DISABLE_XET: {os.environ.get('HF_HUB_DISABLE_XET') or 'unset'}")
    print(f"  HF_ENDPOINT: {endpoint}")
    print(f"  models: {', '.join(name for name, _ in selected_models(args.models))}")
    print(f"  max_workers: {args.max_workers}")
    print(f"  dry_run: {args.dry_run}")

    if not args.no_network_check and not args.dry_run:
        check_network(endpoint)
    print_auth_status()

    if not args.skip_data:
        download_snapshot(
            stage="annotations",
            repo_id="VLN-MME/VLN_annotations",
            repo_type="dataset",
            local_dir=data_dir,
            cache_dir=cache_dir,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
        )
        download_snapshot(
            stage="mp3d",
            repo_id="VLN-MME/MP3D",
            repo_type="dataset",
            local_dir=data_dir / "MP3D",
            cache_dir=cache_dir,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
        )

    if not args.skip_marked_obs:
        download_marked_observations(data_dir, max_scans=args.max_scans, dry_run=args.dry_run)

    if not args.skip_models:
        for name, repo_id in selected_models(args.models):
            download_snapshot(
                stage=f"model:{name}",
                repo_id=repo_id,
                repo_type=None,
                local_dir=None,
                cache_dir=cache_dir,
                max_workers=args.max_workers,
                dry_run=args.dry_run,
            )

    print()
    print("[done] All requested VLN-MME assets are downloaded or already present.")


if __name__ == "__main__":
    main()
