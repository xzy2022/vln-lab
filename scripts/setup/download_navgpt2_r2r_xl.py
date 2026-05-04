#!/usr/bin/env python3
from pathlib import Path
import sys
import socket
import time
import os
from urllib.parse import urlparse
from contextlib import closing

from huggingface_hub import snapshot_download
from tqdm import tqdm


ASSET_ROOT = Path("/data/E/NavGPT-2")
DATA_DIR = ASSET_ROOT / "datasets"
QFORMER_DIR = ASSET_ROOT / "map_nav_src/models/lavis/output"

NETWORK_TIMEOUT = 8


def get_hf_host_and_port():
    """
    优先使用 HF_ENDPOINT。
    如果没有设置，则默认检测 huggingface.co。
    """
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    parsed = urlparse(endpoint)

    host = parsed.hostname
    scheme = parsed.scheme

    if host is None:
        print(f"[network] invalid HF_ENDPOINT: {endpoint}")
        sys.exit(1)

    if parsed.port is not None:
        port = parsed.port
    elif scheme == "https":
        port = 443
    elif scheme == "http":
        port = 80
    else:
        port = 443

    return endpoint, host, port


def check_network(timeout: int = NETWORK_TIMEOUT) -> None:
    endpoint, host, port = get_hf_host_and_port()

    print(f"[network] HF_ENDPOINT = {endpoint}")
    print(f"[network] checking connection to {host}:{port} ...")

    start = time.time()
    try:
        with closing(socket.create_connection((host, port), timeout=timeout)):
            elapsed = time.time() - start
            print(f"[network] OK, connected in {elapsed:.2f}s")
    except OSError as e:
        print(f"[network] FAILED: cannot connect to {host}:{port}")
        print(f"[network] error: {e}")
        print()
        print("Please check:")
        print("  1. Internet connection")
        print("  2. HF_ENDPOINT setting")
        print("  3. Proxy / VPN settings")
        print()
        print("Current endpoint:")
        print(f"  HF_ENDPOINT={endpoint}")
        print()
        print("For hf-mirror, try:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        print()
        print("If you need a proxy, try:")
        print("  export HTTP_PROXY=http://127.0.0.1:7890")
        print("  export HTTPS_PROXY=http://127.0.0.1:7890")
        sys.exit(1)


def download_with_stage(
    stage_name: str,
    repo_id: str,
    repo_type: str,
    allow_patterns,
    local_dir: Path,
):
    print()
    print(f"[{stage_name}] repo: {repo_id}")
    print(f"[{stage_name}] target: {local_dir}")

    local_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=1, desc=stage_name, unit="stage") as pbar:
        result = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        pbar.update(1)

    print(f"[{stage_name}] finished: {result}")
    return result


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QFORMER_DIR.mkdir(parents=True, exist_ok=True)

    check_network()

    download_with_stage(
        stage_name="data",
        repo_id="ZGZzz/NavGPT-R2R",
        repo_type="dataset",
        allow_patterns="*.zip.*",
        local_dir=DATA_DIR,
    )

    download_with_stage(
        stage_name="policy-checkpoint",
        repo_id="ZGZzz/NavGPT2-FlanT5-XL",
        repo_type="model",
        allow_patterns="best_val_unseen_xl",
        local_dir=DATA_DIR / "R2R/trained_models",
    )

    download_with_stage(
        stage_name="q-former",
        repo_id="ZGZzz/NavGPT2-FlanT5-XL",
        repo_type="model",
        allow_patterns="*.pth",
        local_dir=QFORMER_DIR,
    )

    print()
    print("[done] downloads finished")
    print()
    print("Next unzip R2R:")
    print(f"  cd {DATA_DIR}")
    print(
        "  cat R2R.zip.001 R2R.zip.002 R2R.zip.003 R2R.zip.004 R2R.zip.005 "
        "R2R.zip.006 R2R.zip.007 R2R.zip.008 R2R.zip.009 R2R.zip.010 > R2R.zip"
    )
    print("  unzip R2R.zip")


if __name__ == "__main__":
    main()