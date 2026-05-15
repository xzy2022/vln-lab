#!/usr/bin/env python3
"""Parse local path mounts for container startup.

Output format:
  project_root<TAB><container project root>
  mount<TAB><key><TAB><host><TAB><container><TAB><mode>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


VALID_MODES = {"ro", "rw"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse configs/global/local/paths.yaml into Docker mount rows."
    )
    parser.add_argument("--config", required=True, help="Path to paths.yaml.")
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Repository root used to resolve relative host paths.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def resolve_host_path(raw_host: object, repo_root: Path) -> str:
    host = os.path.expandvars(os.path.expanduser(str(raw_host)))
    path = Path(host)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve(strict=False))


def resolve_container_path(raw_container: object, project_root: str) -> str:
    container = str(raw_container)
    if container.startswith("/"):
        return container
    return str(Path(project_root) / container)


def load_paths(config_path: Path) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        fail(f"配置文件不存在: {config_path}")
    except yaml.YAMLError as exc:
        fail(f"无法解析 YAML: {config_path}: {exc}")

    if not isinstance(data, dict):
        fail(f"{config_path} 顶层必须是 YAML mapping。")
    return data


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve(strict=False)
    config_path = Path(args.config).expanduser().resolve(strict=False)
    paths = load_paths(config_path)

    project_root = paths.get("project_root")
    if not project_root:
        fail(f"{config_path} 缺少 project_root。")
    project_root = str(project_root)

    mounts = paths.get("mounts")
    if not isinstance(mounts, dict):
        fail(f"{config_path} 缺少 mounts mapping。")

    print(f"project_root\t{project_root}")
    for key, mount in mounts.items():
        if not isinstance(mount, dict):
            fail(f"mounts.{key} 必须是 mapping。")

        host = mount.get("host")
        container = mount.get("container")
        mode = str(mount.get("mode", "rw"))
        if not host or not container:
            fail(f"mounts.{key} 需要 host 和 container。")
        if mode not in VALID_MODES:
            fail(f"mounts.{key}.mode 只能是 ro 或 rw。")

        host_path = resolve_host_path(host, repo_root)
        container_path = resolve_container_path(container, project_root)
        print(f"mount\t{key}\t{host_path}\t{container_path}\t{mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
