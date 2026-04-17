#!/usr/bin/env python3
"""Download the minimal SAME assets needed for R2R evaluation.

The script populates the repository-level `data/` directory using the layout
that SAME expects, while avoiding the full `download.py --data` payload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
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

# These files are the fixed minimum for R2R evaluation. Sizes are current at the
# time this script was written and are only used for planning/skip checks.
FIXED_ITEMS = [
    ("model", "ckpt/SAME.pt", 2_589_569_179),
    ("model", "pretrain/Attnq_pretrained_ckpt.pt", 817_587_718),
    (
        "dataset",
        "features/img_features/clip_vit-b16_mp3d_hm3d_gibson.hdf5",
        11_586_333_694,
    ),
    ("dataset", "simulator/mp3d_scanvp_candidates.json", 10_848_675),
    ("dataset", "simulator/mp3d_connectivity_graphs.json", 1_330_190),
    ("dataset", "R2R/R2R_val_train_seen_enc.json", 1_115_567),
    ("dataset", "R2R/R2R_val_unseen_enc.json", 1_749_355),
    ("dataset", "R2R/R2R_test_enc.json", 2_812_078),
]

EXTRA_TEST_ITEMS = [
    ("dataset", "REVERIE/BBoxes.json", 7_757_728),
    ("dataset", "REVERIE/REVERIE_test_enc.json", 1_699_626),
    ("dataset", "CVDN/test_cleaned.json", 2_645_768),
    ("dataset", "SOON/test_v2_enc.jsonl", 17_372_890),
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


def api_get_json_page(url: str, timeout: int) -> tuple[list[dict], str | None]:
    request = Request(url, headers={"Accept": "application/json", **auth_headers()})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        next_link = extract_next_link(response.headers.get("Link"))
        if next_link:
            next_link = urljoin(url, next_link)
        return payload, next_link


def fetch_connectivity_items(endpoint: str, timeout: int) -> list[DownloadItem]:
    api_url = (
        f"{endpoint}/api/datasets/{VERSNAV_DATASET_REPO}/tree/main/"
        "simulator/connectivity?recursive=1&expand=1"
    )
    fallback_url = (
        f"{OFFICIAL_ENDPOINT}/api/datasets/{VERSNAV_DATASET_REPO}/tree/main/"
        "simulator/connectivity?recursive=1&expand=1"
    )
    last_error: Exception | None = None

    for url in (api_url, fallback_url):
        try:
            items: list[DownloadItem] = []
            seen_paths: set[str] = set()
            page_idx = 1
            next_url: str | None = url

            while next_url is not None:
                print(f"Fetching connectivity manifest page {page_idx}: {next_url}")
                payload, next_url = api_get_json_page(next_url, timeout=timeout)
                for item in payload:
                    if item.get("type") != "file":
                        continue
                    path = item["path"]
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    items.append(
                        DownloadItem("dataset", path, int(item.get("size", 0)))
                    )
                page_idx += 1

            return items
        except Exception as exc:  # pragma: no cover - network fallback
            print(f"  manifest fetch failed: {exc}")
            last_error = exc

    raise RuntimeError(
        "Failed to fetch the SAME connectivity manifest from Hugging Face."
    ) from last_error


def build_download_plan(args: argparse.Namespace, endpoint: str) -> list[DownloadItem]:
    plan: list[DownloadItem] = []
    include_pretrain = not args.no_pretrain
    include_test_split = not args.no_test_split
    include_extra_test_datasets = args.include_extra_test_datasets

    for kind, relpath, size in FIXED_ITEMS:
        if relpath == "pretrain/Attnq_pretrained_ckpt.pt" and not include_pretrain:
            continue
        if relpath == "R2R/R2R_test_enc.json" and not include_test_split:
            continue
        plan.append(DownloadItem(kind, relpath, size))

    if include_extra_test_datasets:
        for kind, relpath, size in EXTRA_TEST_ITEMS:
            plan.append(DownloadItem(kind, relpath, size))

    plan.extend(fetch_connectivity_items(endpoint, timeout=args.timeout))
    return plan


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
                break
            except (HTTPError, URLError, RuntimeError) as exc:
                last_error = exc
                if destination.with_suffix(destination.suffix + ".part").exists():
                    destination.with_suffix(destination.suffix + ".part").unlink()
                if attempt == retries:
                    raise RuntimeError(
                        f"Failed to download {item.relpath} after {retries} attempts"
                    ) from exc
                wait_seconds = min(5 * attempt, 15)
                print(f"  attempt {attempt} failed: {exc}. retry in {wait_seconds}s")
                time.sleep(wait_seconds)

        if last_error is None:
            print(
                f"  saved to {destination} "
                f"({human_size(destination.stat().st_size)})"
            )

    print(
        f"\nDone. {len(plan) - skipped} files downloaded, {skipped} files reused. "
        f"Current accounted size: {human_size(downloaded_bytes)}"
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data" / "same"

    parser = argparse.ArgumentParser(
        description="Download the minimal SAME R2R evaluation assets into the repository-level data/same/ directory."
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
        "--no-pretrain",
        action="store_true",
        help="Do not download Attnq_pretrained_ckpt.pt.",
    )
    parser.add_argument(
        "--no-test-split",
        action="store_true",
        help="Do not download R2R_test_enc.json.",
    )
    parser.add_argument(
        "--include-extra-test-datasets",
        action="store_true",
        help="Also download the minimal REVERIE/CVDN/SOON files used by SAME test runs.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = resolve_endpoint(args)
    data_root = args.data_root.expanduser().resolve()
    plan = build_download_plan(args, endpoint)

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
