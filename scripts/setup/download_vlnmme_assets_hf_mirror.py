#!/usr/bin/env python3
"""Download VLN-MME assets through hf-mirror.com using hfd.sh.

This mirror-oriented downloader keeps the model/data selection surface close to
``download_vlnmme_assets.py`` while avoiding ``huggingface_hub.snapshot_download``.
The official Hub client performs strict per-file metadata checks that can fail
against hf-mirror.com for Xet-backed large model files. hfd.sh uses direct
resolver URLs instead, which is the mirror site's recommended path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DATASET_REPOS = (
    ("annotations", "VLN-MME/VLN_annotations", "dataset"),
    ("mp3d", "VLN-MME/MP3D", "dataset"),
)

MODEL_REPOS = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3_vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_5_0_8b": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_9b": "Qwen/Qwen3.5-9B",
    "qwen3_5_4b": "Qwen/Qwen3.5-4B",
    "internvl3_2b": "OpenGVLab/InternVL3-2B",
}

DEFAULT_MODELS = ("qwen2_5_vl", "qwen2_5_vl_3b", "qwen3_vl_4b", "internvl3_2b")
DIRECTIONS = ("left", "front", "right", "back")
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def log(message: str = "") -> None:
    print(message, flush=True)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    env_data = os.environ.get("VLNMME_UPSTREAM_DATA_DIR") or os.environ.get("VLNMME_DATA_DIR")
    if env_data:
        return Path(env_data)
    return repo_root_from_script() / "third_party" / "VLN-MME" / "data"


def default_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VLN-MME data and selected model weights through hf-mirror.com via hfd.sh."
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
            "Model weights to cache in a Hugging Face-compatible cache layout. Default: "
            f"{' '.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"HF mirror endpoint. Default: {DEFAULT_ENDPOINT}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="Hugging Face hub cache directory. Default: HF_HUB_CACHE, HF_HOME/hub, or ~/.cache/huggingface/hub.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model/dataset revision passed to hfd.sh. Default: main.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent file downloads for hfd.sh aria2c mode. Default: 2.",
    )
    parser.add_argument(
        "--hfd-threads",
        type=int,
        default=4,
        help="Threads per file for hfd.sh aria2c mode. Default: 4.",
    )
    parser.add_argument(
        "--hfd-tool",
        choices=("auto", "aria2c", "wget"),
        default="auto",
        help="Downloader backend passed to hfd.sh. Default: auto, preferring aria2c then wget.",
    )
    parser.add_argument(
        "--hfd-path",
        type=Path,
        default=None,
        help="Path to hfd.sh. Default: HFD_PATH, hfd, or hfd.sh found on PATH.",
    )
    parser.add_argument(
        "--hf-username",
        default=os.environ.get("HF_USERNAME", ""),
        help="Optional Hugging Face username passed to hfd.sh for gated repos.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "",
        help="Optional Hugging Face token passed to hfd.sh for gated repos.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="Optional hfd include patterns, mainly for smoke tests.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Optional hfd exclude patterns.",
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
        help="Skip the quick TCP check for the mirror endpoint.",
    )
    return parser.parse_args()


def selected_models(tokens: list[str]) -> list[tuple[str, str]]:
    names = list(MODEL_REPOS) if "all" in tokens else tokens
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        result.append((name, MODEL_REPOS[name]))
    return result


def check_network(endpoint: str, timeout: int = 8) -> None:
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if not host:
        raise SystemExit(f"[error] Invalid endpoint: {endpoint}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    log(f"[network] HF_ENDPOINT={endpoint}")
    log(f"[network] checking {host}:{port} ...")
    start = time.time()
    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            log(f"[network] OK in {time.time() - start:.2f}s")
    except OSError as exc:
        print(f"[network] FAILED: {exc}", file=sys.stderr)
        print("[network] Check proxy/VPN/HF_ENDPOINT and rerun.", file=sys.stderr)
        sys.exit(1)


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
            "[error] Choose a writable --cache-dir or --data-dir."
        ) from exc


def resolve_hfd_path(explicit_path: Path | None, dry_run: bool) -> Path | None:
    candidates: list[str | Path] = []
    if explicit_path:
        candidates.append(explicit_path)
    if os.environ.get("HFD_PATH"):
        candidates.append(os.environ["HFD_PATH"])
    candidates.extend(["hfd", "hfd.sh"])

    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_file():
            return candidate_path.resolve()
        found = shutil.which(str(candidate))
        if found:
            return Path(found).resolve()

    if dry_run:
        return None
    raise SystemExit(
        "[error] hfd.sh was not found. Install it and either put it on PATH, "
        "set HFD_PATH, or pass --hfd-path /path/to/hfd.sh."
    )


def choose_hfd_tool(requested: str, dry_run: bool) -> str:
    if requested != "auto":
        if not dry_run and not shutil.which(requested):
            raise SystemExit(f"[error] {requested} is not installed; install it or use --hfd-tool auto.")
        return requested
    if shutil.which("aria2c"):
        return "aria2c"
    if shutil.which("wget"):
        return "wget"
    if dry_run:
        return "auto"
    raise SystemExit("[error] Neither aria2c nor wget is installed. Install one of them first.")


def require_marked_obs_packages(args: argparse.Namespace) -> None:
    if args.skip_marked_obs or args.dry_run:
        return
    missing: list[str] = []
    for module_name in ("datasets", "tqdm"):
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            missing.append(exc.name)
    if missing:
        for name in dict.fromkeys(missing):
            print(f"[error] Missing Python package: {name}", file=sys.stderr)
        print("[error] Install these packages or rerun with --skip-marked-obs.", file=sys.stderr)
        sys.exit(1)


def repo_cache_name(repo_id: str, repo_type: str | None) -> str:
    prefix = "datasets" if repo_type == "dataset" else "models"
    return f"{prefix}--{repo_id.replace('/', '--')}"


def api_repo_kind(repo_type: str | None) -> str:
    return "datasets" if repo_type == "dataset" else "models"


def fetch_repo_info(
    *,
    endpoint: str,
    repo_id: str,
    repo_type: str | None,
    revision: str,
    token: str,
) -> dict:
    quoted_repo = quote(repo_id, safe="/")
    api_path = f"{api_repo_kind(repo_type)}/{quoted_repo}"
    if revision != "main":
        api_path = f"{api_path}/revision/{quote(revision, safe='')}"
    url = f"{endpoint.rstrip('/')}/api/{api_path}"
    try:
        if shutil.which("curl"):
            cmd = ["curl", "-fsSL", url]
            if token:
                cmd.extend(["-H", f"Authorization: Bearer {token}"])
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads(completed.stdout)

        request = Request(url, headers={"User-Agent": "vln-lab-hf-mirror-downloader/1.0"})
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"[error] Failed to fetch repo metadata for {repo_id}: {exc}") from exc


def run_hfd(
    *,
    hfd_path: Path | None,
    repo_id: str,
    repo_type: str | None,
    local_dir: Path,
    revision: str,
    endpoint: str,
    tool: str,
    max_workers: int,
    hfd_threads: int,
    hf_username: str,
    hf_token: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    dry_run: bool,
) -> None:
    cmd = [str(hfd_path or "hfd.sh"), repo_id, "--local-dir", str(local_dir), "--revision", revision, "--tool", tool]
    if repo_type == "dataset":
        cmd.append("--dataset")
    if tool == "aria2c":
        cmd.extend(["-j", str(max_workers), "-x", str(hfd_threads)])
    if hf_username:
        cmd.extend(["--hf_username", hf_username])
    if hf_token:
        cmd.extend(["--hf_token", hf_token])
    if include_patterns:
        cmd.append("--include")
        cmd.extend(include_patterns)
    if exclude_patterns:
        cmd.append("--exclude")
        cmd.extend(exclude_patterns)

    redacted_cmd = ["***" if part == hf_token and hf_token else part for part in cmd]
    log(f"  command: {' '.join(redacted_cmd)}")
    if dry_run:
        return

    env = os.environ.copy()
    env["HF_ENDPOINT"] = endpoint.rstrip("/")
    subprocess.run(cmd, env=env, check=True)


def write_ref(cache_dir: Path, repo_id: str, repo_type: str | None, revision: str, sha: str) -> None:
    refs_dir = cache_dir / repo_cache_name(repo_id, repo_type) / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / revision).write_text(sha)


def download_hfd_snapshot(
    *,
    stage: str,
    repo_id: str,
    repo_type: str | None,
    local_dir: Path | None,
    cache_dir: Path,
    args: argparse.Namespace,
    hfd_path: Path | None,
    hfd_tool: str,
) -> None:
    log()
    log(f"[stage] {stage}")
    log(f"  repo: {repo_id}")
    if repo_type:
        log(f"  repo_type: {repo_type}")

    target_dir = local_dir
    sha = None
    if target_dir is None:
        if args.dry_run:
            cache_repo = cache_dir / repo_cache_name(repo_id, repo_type)
            log(f"  cache_repo: {cache_repo}")
            log("  snapshot: resolved from mirror metadata at runtime")
            return
        info = fetch_repo_info(
            endpoint=args.endpoint,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=args.revision,
            token=args.hf_token,
        )
        sha = info.get("sha")
        if not sha:
            raise SystemExit(f"[error] Missing sha in repo metadata for {repo_id}.")
        target_dir = cache_dir / repo_cache_name(repo_id, repo_type) / "snapshots" / sha
        log(f"  cache_snapshot: {target_dir}")
    else:
        log(f"  local_dir: {target_dir}")

    run_hfd(
        hfd_path=hfd_path,
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=target_dir,
        revision=args.revision,
        endpoint=args.endpoint,
        tool=hfd_tool,
        max_workers=args.max_workers,
        hfd_threads=args.hfd_threads,
        hf_username=args.hf_username,
        hf_token=args.hf_token,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        dry_run=args.dry_run,
    )
    if sha is not None:
        write_ref(cache_dir, repo_id, repo_type, args.revision, sha)
    log(f"[done] {stage}: {target_dir}")


def ensure_scans_file(
    data_dir: Path,
    args: argparse.Namespace,
    hfd_path: Path | None,
    hfd_tool: str,
) -> Path:
    scans_file = data_dir / "MP3D" / "scans.txt"
    if scans_file.exists():
        return scans_file
    log(f"[marked_obs] missing {scans_file}; downloading MP3D metadata first.")
    if args.dry_run:
        return scans_file
    download_hfd_snapshot(
        stage="mp3d",
        repo_id="VLN-MME/MP3D",
        repo_type="dataset",
        local_dir=data_dir / "MP3D",
        cache_dir=args.cache_dir.resolve(),
        args=args,
        hfd_path=hfd_path,
        hfd_tool=hfd_tool,
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


def download_marked_observations(
    data_dir: Path,
    args: argparse.Namespace,
    hfd_path: Path | None,
    hfd_tool: str,
) -> None:
    save_dir = data_dir / "marked_obs"
    scans_file = ensure_scans_file(data_dir, args=args, hfd_path=hfd_path, hfd_tool=hfd_tool)

    log()
    log("[stage] marked_obs")
    log("  repo: VLN-MME/MP3D_marked_obs")
    log(f"  output: {save_dir}")
    log("  resume: skip existing non-empty PNG files")

    if args.dry_run:
        return

    from datasets import load_dataset
    from tqdm import tqdm

    scans = read_scans(scans_file)
    if args.max_scans is not None:
        scans = scans[: args.max_scans]

    save_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0
    for index, scan_id in enumerate(scans, start=1):
        log()
        log(f"[marked_obs] scan {index}/{len(scans)}: {scan_id}")
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
        log(f"[marked_obs] {scan_id}: saved={saved}, skipped={skipped}")

    log()
    log(f"[done] marked_obs: saved={total_saved}, skipped={total_skipped}, output={save_dir}")


def main() -> None:
    args = parse_args()
    args.endpoint = args.endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = args.endpoint

    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    hfd_path = resolve_hfd_path(args.hfd_path, dry_run=args.dry_run)
    hfd_tool = choose_hfd_tool(args.hfd_tool, dry_run=args.dry_run)

    ensure_writable_dir(cache_dir, "--cache-dir", dry_run=args.dry_run)
    if not args.skip_data:
        ensure_writable_dir(data_dir, "--data-dir", dry_run=args.dry_run)
    require_marked_obs_packages(args)

    log("[plan] VLN-MME hf-mirror asset download")
    log(f"  data_dir: {data_dir}")
    log(f"  cache_dir: {cache_dir}")
    log(f"  HF_ENDPOINT: {args.endpoint}")
    log(f"  hfd_path: {hfd_path or 'not checked in dry-run'}")
    log(f"  hfd_tool: {hfd_tool}")
    log(f"  revision: {args.revision}")
    log(f"  models: {', '.join(name for name, _ in selected_models(args.models))}")
    log(f"  max_workers: {args.max_workers}")
    log(f"  dry_run: {args.dry_run}")

    if not args.no_network_check and not args.dry_run:
        check_network(args.endpoint)
    if args.hf_token:
        log("[auth] HF token: set")
    else:
        log("[auth] HF token: unset; gated repos such as VLN-MME/MP3D may fail")

    if not args.skip_data:
        download_hfd_snapshot(
            stage="annotations",
            repo_id="VLN-MME/VLN_annotations",
            repo_type="dataset",
            local_dir=data_dir,
            cache_dir=cache_dir,
            args=args,
            hfd_path=hfd_path,
            hfd_tool=hfd_tool,
        )
        download_hfd_snapshot(
            stage="mp3d",
            repo_id="VLN-MME/MP3D",
            repo_type="dataset",
            local_dir=data_dir / "MP3D",
            cache_dir=cache_dir,
            args=args,
            hfd_path=hfd_path,
            hfd_tool=hfd_tool,
        )

    if not args.skip_marked_obs:
        download_marked_observations(data_dir, args=args, hfd_path=hfd_path, hfd_tool=hfd_tool)

    if not args.skip_models:
        for name, repo_id in selected_models(args.models):
            download_hfd_snapshot(
                stage=f"model:{name}",
                repo_id=repo_id,
                repo_type=None,
                local_dir=None,
                cache_dir=cache_dir,
                args=args,
                hfd_path=hfd_path,
                hfd_tool=hfd_tool,
            )

    log()
    log("[done] All requested hf-mirror assets are downloaded or already present.")


if __name__ == "__main__":
    main()
