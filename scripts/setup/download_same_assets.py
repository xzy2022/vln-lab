#!/usr/bin/env python3
"""Download SAME annotations, runtime assets, and model weights.

This downloader populates the repository-level ``data/same/`` directory using
the layout expected by ``third_party/SAME`` while letting the caller choose:

- which datasets to download
- which annotation levels to download (train / val / test / aug)
- whether to include runtime assets (MatterSim-only or full)
- whether to include model weights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
SAME_MODEL_REPO = "ZGZzz/SAME"
VERSNAV_DATASET_REPO = "ZGZzz/VersNav"
DEFAULT_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

BASE_DATASETS = (
    "R2R",
    "REVERIE",
    "RXR-EN",
    "CVDN",
    "SOON",
    "OBJNAV_MP3D",
)

DATASET_ROOTS = {
    "R2R": "R2R",
    "REVERIE": "REVERIE",
    "RXR-EN": "RXR-EN",
    "CVDN": "CVDN",
    "SOON": "SOON",
    "OBJNAV_MP3D": "MP3D",
}

LEVEL_CHOICES = ("train", "val", "test", "aug", "all")
RUNTIME_CHOICES = ("none", "mattersim", "full")
MODEL_CHOICES = ("none", "eval", "all")

# Sizes are used for planning and skip checks.
MODEL_ITEMS = {
    "eval": [
        ("model", "ckpt/SAME.pt", 2_589_569_179),
        ("model", "pretrain/Attnq_pretrained_ckpt.pt", 817_587_718),
    ],
    "all": [
        ("model", "ckpt/SAME.pt", 2_589_569_179),
        ("model", "pretrain/Attnq_pretrained_ckpt.pt", 817_587_718),
        ("model", "pretrain/Attnkv_pretrained_ckpt.pt", 817_600_439),
        ("model", "pretrain/FFN_pretrained_ckpt.pt", 817_607_588),
    ],
}

MATTERSIM_RUNTIME_ITEMS = [
    (
        "dataset",
        "features/img_features/clip_vit-b16_mp3d_hm3d_gibson.hdf5",
        11_586_333_694,
    ),
    ("dataset", "simulator/mp3d_scanvp_candidates.json", 10_848_675),
    ("dataset", "simulator/mp3d_connectivity_graphs.json", 1_330_190),
]


@dataclass(frozen=True)
class DownloadItem:
    kind: str
    relpath: str
    size: int | None


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def resolve_endpoint(args: argparse.Namespace) -> str:
    if args.endpoint:
        return args.endpoint.rstrip("/")
    env_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if env_endpoint:
        return env_endpoint.rstrip("/")
    if args.source == "hf-mirror":
        return HF_MIRROR_ENDPOINT
    return OFFICIAL_ENDPOINT


def auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_resolve_url(endpoint: str, kind: str, relpath: str) -> str:
    if kind == "dataset":
        return f"{endpoint}/datasets/{VERSNAV_DATASET_REPO}/resolve/main/{relpath}"
    if kind == "model":
        return f"{endpoint}/{SAME_MODEL_REPO}/resolve/main/{relpath}"
    raise ValueError(f"Unsupported item kind: {kind}")


def build_tree_api_url(endpoint: str, kind: str, root: str) -> str:
    if kind == "dataset":
        return f"{endpoint}/api/datasets/{VERSNAV_DATASET_REPO}/tree/main/{root}?recursive=1&expand=1"
    if kind == "model":
        return f"{endpoint}/api/models/{SAME_MODEL_REPO}/tree/main/{root}?recursive=1&expand=1"
    raise ValueError(f"Unsupported tree kind: {kind}")


def extract_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        left = part.find("<")
        right = part.find(">", left + 1)
        if left != -1 and right != -1:
            return part[left + 1 : right]
    return None


def normalize_next_url(current_url: str, next_link: str | None) -> str | None:
    if not next_link:
        return None

    next_url = urljoin(current_url, next_link)
    current_parts = urlparse(current_url)
    next_parts = urlparse(next_url)
    official_parts = urlparse(OFFICIAL_ENDPOINT)

    # hf-mirror may return absolute pagination links pointing back to the
    # official host. Keep pagination on the currently selected endpoint.
    if current_parts.netloc != next_parts.netloc and next_parts.netloc == official_parts.netloc:
        next_parts = next_parts._replace(scheme=current_parts.scheme, netloc=current_parts.netloc)
        return next_parts.geturl()

    return next_url


def api_get_json_page(url: str, timeout: int) -> tuple[list[dict], str | None]:
    request = Request(url, headers={"Accept": "application/json", **auth_headers()})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        next_link = normalize_next_url(url, extract_next_link(response.headers.get("Link")))
        return payload, next_link


def fetch_tree_items(endpoint: str, kind: str, root: str, timeout: int) -> list[DownloadItem]:
    primary = build_tree_api_url(endpoint, kind, root)
    urls = [primary]
    if endpoint != OFFICIAL_ENDPOINT:
        urls.append(build_tree_api_url(OFFICIAL_ENDPOINT, kind, root))

    last_error: Exception | None = None
    for api_url in urls:
        try:
            items: list[DownloadItem] = []
            seen_paths: set[str] = set()
            page_idx = 1
            next_url: str | None = api_url

            while next_url is not None:
                print(f"Fetching manifest page {page_idx}: {next_url}")
                payload, next_url = api_get_json_page(next_url, timeout=timeout)
                for item in payload:
                    if item.get("type") != "file":
                        continue
                    path = item["path"]
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    items.append(DownloadItem(kind, path, int(item.get("size", 0))))
                page_idx += 1
            return items
        except Exception as exc:  # pragma: no cover - network fallback
            print(f"  manifest fetch failed: {exc}")
            last_error = exc

    raise RuntimeError(f"Failed to fetch the {root} manifest from Hugging Face.") from last_error


def expand_datasets(dataset_tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in dataset_tokens:
        if token == "all":
            expanded.extend(BASE_DATASETS)
        else:
            expanded.append(token)
    seen: set[str] = set()
    ordered: list[str] = []
    for dataset_name in expanded:
        if dataset_name not in seen:
            seen.add(dataset_name)
            ordered.append(dataset_name)
    return ordered


def expand_levels(level_tokens: list[str]) -> set[str]:
    expanded: set[str] = set()
    for token in level_tokens:
        if token == "all":
            expanded.update({"train", "val", "test", "aug"})
        else:
            expanded.add(token)
    return expanded


def classify_dataset_annotation(dataset_name: str, relpath: str) -> str | None:
    path = PurePosixPath(relpath)
    name = path.name

    if dataset_name == "R2R":
        if name == "R2R_train_mergesim_enc.json":
            return "train"
        if name in {
            "R2R_val_train_seen_enc.json",
            "R2R_val_seen_enc.json",
            "R2R_val_unseen_enc.json",
        }:
            return "val"
        if name == "R2R_test_enc.json":
            return "test"
        if name in {
            "R2R_prevalent_aug_train_enc.json",
            "R2R_scalevln_aug_train_enc.json",
        }:
            return "aug"
        return None

    if dataset_name == "REVERIE":
        if name == "BBoxes.json":
            return "support"
        if name == "REVERIE_train_enc.json":
            return "train"
        if name in {
            "REVERIE_val_train_seen_enc.json",
            "REVERIE_val_seen_enc.json",
            "REVERIE_val_unseen_enc.json",
        }:
            return "val"
        if name == "REVERIE_test_enc.json":
            return "test"
        if name in {
            "REVERIE_scalevln_aug_train_enc.jsonl",
            "REVERIE_speaker_aug_enc.jsonl",
        }:
            return "aug"
        return None

    if dataset_name == "RXR-EN":
        if name == "RXR-EN_train_enc.json":
            return "train"
        if name in {"RXR-EN_val_seen_enc.json", "RXR-EN_val_unseen_enc.json"}:
            return "val"
        return None

    if dataset_name == "CVDN":
        if name == "train.json":
            return "train"
        if name in {"val_seen.json", "val_unseen.json"}:
            return "val"
        if name == "test_cleaned.json":
            return "test"
        return None

    if dataset_name == "SOON":
        if name == "train_enc_pseudo_obj_ade30k_label.jsonl":
            return "train"
        if name in {
            "val_unseen_instrs_enc_pseudo_obj_ade30k_label.jsonl",
            "val_unseen_house_enc_pseudo_obj_ade30k_label.jsonl",
        }:
            return "val"
        if name in {"test_enc.jsonl", "test_v2_enc.jsonl"}:
            return "test"
        return None

    if dataset_name == "OBJNAV_MP3D":
        rel = relpath.replace("\\", "/")
        if rel.startswith("MP3D/habitatweb/train/"):
            return "train"
        if rel.startswith("MP3D/habitatweb/val_train_seen/"):
            return "val"
        if rel.startswith("MP3D/v1/val/"):
            return "val"
        return None

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def collect_annotation_items(
    endpoint: str,
    datasets: list[str],
    levels: set[str],
    timeout: int,
) -> list[DownloadItem]:
    allowed_groups = set(levels)
    allowed_groups.add("support")
    collected: list[DownloadItem] = []

    for dataset_name in datasets:
        root = DATASET_ROOTS[dataset_name]
        tree_items = fetch_tree_items(endpoint, "dataset", root, timeout=timeout)
        matched = 0
        for item in tree_items:
            group = classify_dataset_annotation(dataset_name, item.relpath)
            if group is None or group not in allowed_groups:
                continue
            collected.append(item)
            matched += 1
        print(f"Selected {matched} annotation files for {dataset_name}")

    return collected


def collect_runtime_items(endpoint: str, runtime: str, timeout: int) -> list[DownloadItem]:
    if runtime == "none":
        return []
    if runtime == "mattersim":
        items = [DownloadItem(kind, relpath, size) for kind, relpath, size in MATTERSIM_RUNTIME_ITEMS]
        items.extend(fetch_tree_items(endpoint, "dataset", "simulator/connectivity", timeout=timeout))
        return items
    if runtime == "full":
        items = fetch_tree_items(endpoint, "dataset", "features", timeout=timeout)
        items.extend(fetch_tree_items(endpoint, "dataset", "simulator", timeout=timeout))
        return items
    raise ValueError(f"Unsupported runtime profile: {runtime}")


def collect_model_items(models: str) -> list[DownloadItem]:
    if models == "none":
        return []
    if models in MODEL_ITEMS:
        return [DownloadItem(kind, relpath, size) for kind, relpath, size in MODEL_ITEMS[models]]
    raise ValueError(f"Unsupported model profile: {models}")


def dedupe_items(items: Iterable[DownloadItem]) -> list[DownloadItem]:
    deduped: list[DownloadItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.kind, item.relpath)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_download_plan(args: argparse.Namespace, endpoint: str) -> list[DownloadItem]:
    plan: list[DownloadItem] = []

    if args.datasets:
        datasets = expand_datasets(args.datasets)
        levels = expand_levels(args.levels)
        plan.extend(collect_annotation_items(endpoint, datasets, levels, timeout=args.timeout))

    plan.extend(collect_runtime_items(endpoint, args.runtime, timeout=args.timeout))
    plan.extend(collect_model_items(args.models))
    return dedupe_items(plan)


def existing_size_matches(path: Path, expected_size: int | None) -> bool:
    return path.exists() and path.is_file() and expected_size is not None and path.stat().st_size == expected_size


def print_plan(items: Iterable[DownloadItem]) -> int:
    total = 0
    for item in items:
        if item.size is not None:
            total += item.size
        print(f"- {item.relpath} ({human_size(item.size)})")
    print(f"\nPlanned total: {human_size(total)} ({total} bytes)")
    return total


def download_file(
    url: str,
    destination: Path,
    expected_size: int | None,
    timeout: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    request = Request(url, headers=auth_headers())
    with urlopen(request, timeout=timeout) as response, temp_path.open("wb") as file_obj:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else expected_size
        downloaded = 0
        last_reported = -1
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file_obj.write(chunk)
            downloaded += len(chunk)

            if total_bytes:
                percent = int(downloaded * 100 / total_bytes)
                if percent != last_reported:
                    last_reported = percent
                    sys.stdout.write(
                        f"\r  {destination.name}: {percent:3d}% "
                        f"({human_size(downloaded)} / {human_size(total_bytes)})"
                    )
                    sys.stdout.flush()

    if total_bytes:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if expected_size is not None and temp_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {destination}: "
            f"expected {expected_size}, got {temp_path.stat().st_size}"
        )

    temp_path.replace(destination)


def run_download(
    plan: list[DownloadItem],
    endpoint: str,
    data_root: Path,
    retries: int,
    timeout: int,
) -> None:
    total_bytes = sum(item.size or 0 for item in plan)
    downloaded_bytes = 0
    skipped = 0

    print(f"Downloading into: {data_root}")
    print(f"Source endpoint: {endpoint}")
    print(f"Planned total size: {human_size(total_bytes)}")

    for index, item in enumerate(plan, start=1):
        destination = data_root / item.relpath
        url = build_resolve_url(endpoint, item.kind, item.relpath)

        if existing_size_matches(destination, item.size):
            skipped += 1
            downloaded_bytes += item.size or 0
            print(
                f"[{index}/{len(plan)}] Skip existing {item.relpath} "
                f"({human_size(item.size)})"
            )
            continue

        print(f"[{index}/{len(plan)}] Download {item.relpath}")

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                download_file(url, destination, item.size, timeout=timeout)
                downloaded_bytes += item.size or destination.stat().st_size
                last_error = None
                break
            except (HTTPError, URLError, RuntimeError) as exc:
                last_error = exc
                part_file = destination.with_suffix(destination.suffix + ".part")
                if part_file.exists():
                    part_file.unlink()
                if attempt == retries:
                    raise RuntimeError(
                        f"Failed to download {item.relpath} after {retries} attempts"
                    ) from exc
                wait_seconds = min(5 * attempt, 15)
                print(f"  attempt {attempt} failed: {exc}. retry in {wait_seconds}s")
                time.sleep(wait_seconds)

        if last_error is None:
            print(f"  saved to {destination} ({human_size(destination.stat().st_size)})")

    print(
        f"\nDone. {len(plan) - skipped} files downloaded, {skipped} files reused. "
        f"Current accounted size: {human_size(downloaded_bytes)}"
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data" / "same"

    parser = argparse.ArgumentParser(
        description="Download SAME annotations, runtime assets, and model weights into data/same/.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/setup/download_same_assets.py \\\n"
            "    --datasets R2R REVERIE CVDN SOON --levels val test \\\n"
            "    --runtime mattersim --models eval\n\n"
            "  python scripts/setup/download_same_assets.py \\\n"
            "    --datasets all --levels all --runtime full --models all \\\n"
            "    --source hf-mirror\n"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Target SAME data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--source",
        choices=["official", "hf-mirror"],
        default="official",
        help="Download source. HF_ENDPOINT overrides this if set.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Custom Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*BASE_DATASETS, "all"],
        help="Datasets to download annotations for. Use 'all' to select every SAME dataset.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVEL_CHOICES,
        help="Annotation levels to download: train / val / test / aug / all.",
    )
    parser.add_argument(
        "--runtime",
        choices=RUNTIME_CHOICES,
        default="none",
        help="Runtime assets to download: none / mattersim / full. Default: %(default)s",
    )
    parser.add_argument(
        "--models",
        choices=MODEL_CHOICES,
        default="none",
        help="Model weights to download: none / eval / all. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned files and total size without downloading.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download retries per file. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Socket timeout in seconds for manifest and file requests. Default: %(default)s",
    )
    args = parser.parse_args()

    if args.datasets and not args.levels:
        parser.error("--levels is required when --datasets is provided.")
    if args.levels and not args.datasets:
        parser.error("--datasets is required when --levels is provided.")
    if not args.datasets and args.runtime == "none" and args.models == "none":
        parser.error("Nothing selected. Choose dataset annotations, runtime assets, and/or model weights.")

    return args


def main() -> int:
    args = parse_args()
    endpoint = resolve_endpoint(args)
    data_root = args.data_root.expanduser().resolve()
    plan = build_download_plan(args, endpoint)

    if not plan:
        print("Nothing matched the requested selection.")
        return 0

    print_plan(plan)

    if args.dry_run:
        return 0

    data_root.mkdir(parents=True, exist_ok=True)
    run_download(
        plan,
        endpoint,
        data_root,
        retries=max(1, args.retries),
        timeout=max(1, args.timeout),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
