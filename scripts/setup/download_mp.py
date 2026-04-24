#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Matterport3D public data with retries, progress, and file locks."""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request

try:
    import fcntl
except ImportError:  # pragma: no cover - only used on non-POSIX platforms
    fcntl = None


BASE_URL = "https://kaldir.vc.in.tum.de/matterport"
RELEASE = "v1/scans"
RELEASE_TASKS = "v1/tasks"
RELEASE_SIZE = "1.3 TB"
TOS_URL = f"{BASE_URL}/MP_TOS.pdf"
USER_AGENT = "vln-lab-matterport-downloader/1.0"
CHUNK_SIZE_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRIES = 4
DEFAULT_STATUS_INTERVAL_SECONDS = 5.0
DEFAULT_LOCK_POLL_SECONDS = 5.0

FILETYPES = [
    "cameras",
    "matterport_camera_intrinsics",
    "matterport_camera_poses",
    "matterport_color_images",
    "matterport_depth_images",
    "matterport_hdr_images",
    "matterport_mesh",
    "matterport_skybox_images",
    "undistorted_camera_parameters",
    "undistorted_color_images",
    "undistorted_depth_images",
    "undistorted_normal_images",
    "house_segmentations",
    "region_segmentations",
    "image_overlap_data",
    "poisson_meshes",
    "sens",
]
TASK_FILES = {
    "keypoint_matching_data": ["keypoint_matching/data.zip"],
    "keypoint_matching_models": ["keypoint_matching/models.zip"],
    "surface_normal_data": ["surface_normal/data_list.zip"],
    "surface_normal_models": ["surface_normal/models.zip"],
    "region_classification_data": ["region_classification/data.zip"],
    "region_classification_models": ["region_classification/models.zip"],
    "semantic_voxel_label_data": ["semantic_voxel_label/data.zip"],
    "semantic_voxel_label_models": ["semantic_voxel_label/models.zip"],
    "minos": ["mp3d_minos.zip"],
    "gibson": ["mp3d_for_gibson.tar.gz"],
    "habitat": ["mp3d_habitat.zip"],
    "pixelsynth": ["mp3d_pixelsynth.zip"],
    "igibson": ["mp3d_for_igibson.zip"],
    "mp360": [
        "mp3d_360/data_00.zip",
        "mp3d_360/data_01.zip",
        "mp3d_360/data_02.zip",
        "mp3d_360/data_03.zip",
        "mp3d_360/data_04.zip",
        "mp3d_360/data_05.zip",
        "mp3d_360/data_06.zip",
    ],
}

_WARNED_LEGACY_TMP_DIRS: set[Path] = set()


@dataclass(frozen=True)
class DownloadConfig:
    timeout_seconds: float
    retries: int
    status_interval_seconds: float
    lock_poll_seconds: float
    assume_yes: bool
    probe_only: bool
    no_resume: bool
    overwrite: bool
    adopt_legacy_tmp: bool


@dataclass(frozen=True)
class RemoteFileInfo:
    url: str
    size_bytes: int | None
    accept_ranges: bool
    status_code: int


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def build_request(url: str, *, method: str = "GET", byte_range: str | None = None) -> request.Request:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = byte_range
    return request.Request(url, headers=headers, method=method)


def parse_content_length(headers) -> int | None:
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_total_size_from_content_range(headers) -> int | None:
    raw = headers.get("Content-Range")
    if not raw or "/" not in raw:
        return None
    tail = raw.rsplit("/", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def backoff_seconds(attempt: int) -> float:
    return min(2 ** (attempt - 1), 15)


def describe_exc(exc: BaseException) -> str:
    if isinstance(exc, error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def should_retry(exc: BaseException) -> bool:
    if isinstance(exc, error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, error.URLError):
        return True
    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno not in {28}  # ENOSPC
    return False


def open_url_with_retries(url: str, config: DownloadConfig):
    last_error: BaseException | None = None
    for attempt in range(1, config.retries + 1):
        try:
            return request.urlopen(build_request(url), timeout=config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - network exceptions vary by platform
            last_error = exc
            if attempt >= config.retries or not should_retry(exc):
                break
            sleep_for = backoff_seconds(attempt)
            log(
                f"读取 {url} 失败，第 {attempt}/{config.retries} 次尝试: "
                f"{describe_exc(exc)}，{format_duration(sleep_for)} 后重试。"
            )
            time.sleep(sleep_for)
    assert last_error is not None
    raise RuntimeError(f"无法读取 {url}: {describe_exc(last_error)}") from last_error


def probe_remote_file(url: str, config: DownloadConfig) -> RemoteFileInfo:
    last_error: BaseException | None = None
    for attempt in range(1, config.retries + 1):
        try:
            with request.urlopen(build_request(url, method="HEAD"), timeout=config.timeout_seconds) as response:
                return RemoteFileInfo(
                    url=url,
                    size_bytes=parse_content_length(response.headers),
                    accept_ranges="bytes" in response.headers.get("Accept-Ranges", "").lower(),
                    status_code=response.getcode(),
                )
        except error.HTTPError as exc:
            last_error = exc
            if exc.code not in {405, 501}:
                if attempt >= config.retries or not should_retry(exc):
                    break
            else:
                try:
                    with request.urlopen(
                        build_request(url, byte_range="bytes=0-0"),
                        timeout=config.timeout_seconds,
                    ) as response:
                        size_bytes = (
                            parse_total_size_from_content_range(response.headers)
                            or parse_content_length(response.headers)
                        )
                        return RemoteFileInfo(
                            url=url,
                            size_bytes=size_bytes,
                            accept_ranges="bytes" in response.headers.get("Accept-Ranges", "").lower(),
                            status_code=response.getcode(),
                        )
                except Exception as fallback_exc:  # noqa: BLE001
                    last_error = fallback_exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        assert last_error is not None
        if attempt >= config.retries or not should_retry(last_error):
            break
        sleep_for = backoff_seconds(attempt)
        log(
            f"探测远端文件失败，第 {attempt}/{config.retries} 次尝试: "
            f"{describe_exc(last_error)}，{format_duration(sleep_for)} 后重试。"
        )
        time.sleep(sleep_for)

    assert last_error is not None
    raise RuntimeError(f"无法探测远端文件 {url}: {describe_exc(last_error)}") from last_error


def get_release_scans(release_file: str, config: DownloadConfig) -> list[str]:
    with open_url_with_retries(release_file, config) as scan_lines:
        scans = [line.decode("utf-8").strip() for line in scan_lines if line.strip()]
    log(f"已加载 {len(scans)} 个 scan id。")
    return scans


def build_progress_message(
    destination: Path,
    downloaded_bytes: int,
    total_bytes: int | None,
    session_downloaded_bytes: int,
    session_start: float,
) -> str:
    elapsed = max(time.monotonic() - session_start, 1e-6)
    speed = session_downloaded_bytes / elapsed
    parts = [f"{destination.name}: {format_bytes(downloaded_bytes)}"]
    if total_bytes is not None and total_bytes > 0:
        percent = downloaded_bytes / total_bytes * 100
        parts[0] = (
            f"{destination.name}: {format_bytes(downloaded_bytes)} / "
            f"{format_bytes(total_bytes)} ({percent:.1f}%)"
        )
        if speed > 0:
            eta = (total_bytes - downloaded_bytes) / speed
            parts.append(f"ETA {format_duration(eta)}")
    if speed > 0:
        parts.append(f"{format_bytes(int(speed))}/s")
    return "下载进度 " + ", ".join(parts)


def warn_for_legacy_temp_files(out_dir: Path) -> None:
    if out_dir in _WARNED_LEGACY_TMP_DIRS or not out_dir.exists():
        return
    legacy = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name.startswith("tmp"):
            legacy.append(f"{path.name} ({format_bytes(path.stat().st_size)})")
    if legacy:
        log(
            f"发现旧版脚本遗留的临时文件，当前会忽略它们: {', '.join(legacy)}。"
            " 如果确认没有旧进程在写，可以手动删除。"
        )
    _WARNED_LEGACY_TMP_DIRS.add(out_dir)


def legacy_tmp_candidates(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    return [path for path in sorted(out_dir.iterdir()) if path.is_file() and path.name.startswith("tmp")]


def adopt_legacy_tmp_file(
    *,
    out_file: Path,
    part_file: Path,
    remote: RemoteFileInfo,
    config: DownloadConfig,
) -> None:
    if not config.adopt_legacy_tmp or out_file.exists() or part_file.exists():
        return

    candidates: list[tuple[int, Path]] = []
    for path in legacy_tmp_candidates(out_file.parent):
        size = path.stat().st_size
        if remote.size_bytes is not None and size > remote.size_bytes:
            continue
        candidates.append((size, path))

    if not candidates:
        return

    size, chosen = max(candidates, key=lambda item: item[0])
    target = out_file if remote.size_bytes is not None and size == remote.size_bytes else part_file
    log(
        f"按 --adopt-legacy-tmp 接管旧临时文件 {chosen.name} "
        f"({format_bytes(size)}) -> {target.name}"
    )
    os.replace(chosen, target)


class FileLock(contextlib.AbstractContextManager):
    def __init__(self, lock_path: Path, target_path: Path, poll_seconds: float):
        self._lock_path = lock_path
        self._target_path = target_path
        self._poll_seconds = poll_seconds
        self._handle = None

    def __enter__(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._lock_path.open("a+")
        if fcntl is None:
            return self
        wait_started = time.monotonic()
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle.seek(0)
                self._handle.truncate()
                self._handle.write(f"pid={os.getpid()} target={self._target_path}\n")
                self._handle.flush()
                return self
            except BlockingIOError:
                waited = time.monotonic() - wait_started
                log(
                    f"另一个进程正在处理 {self._target_path.name}，"
                    f"继续等待锁中（已等待 {format_duration(waited)}）。"
                )
                time.sleep(self._poll_seconds)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def partial_path_for(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def lock_path_for(destination: Path) -> Path:
    return destination.with_name(destination.name + ".lock")


def download_file(url: str, out_file: Path, config: DownloadConfig) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    warn_for_legacy_temp_files(out_file.parent)

    if out_file.exists() and not config.overwrite:
        log(f"已存在，跳过: {out_file}")
        return

    remote = probe_remote_file(url, config)
    range_support = "支持 Range 续传" if remote.accept_ranges else "未声明 Range 续传"
    log(f"远端可访问: HTTP {remote.status_code}, 大小 {format_bytes(remote.size_bytes)}, {range_support}")
    if config.probe_only:
        part_file = partial_path_for(out_file)
        if part_file.exists():
            log(f"本地存在部分下载: {part_file} ({format_bytes(part_file.stat().st_size)})")
        return

    with FileLock(lock_path_for(out_file), out_file, config.lock_poll_seconds):
        if out_file.exists() and not config.overwrite:
            log(f"等待锁结束后发现文件已完成，跳过: {out_file}")
            return

        if config.overwrite and out_file.exists():
            out_file.unlink()

        part_file = partial_path_for(out_file)
        adopt_legacy_tmp_file(
            out_file=out_file,
            part_file=part_file,
            remote=remote,
            config=config,
        )
        if config.no_resume and part_file.exists():
            log(f"按要求丢弃旧的部分下载: {part_file}")
            part_file.unlink()

        if part_file.exists() and remote.size_bytes is not None:
            part_size = part_file.stat().st_size
            if part_size == remote.size_bytes:
                os.replace(part_file, out_file)
                log(f"检测到完整的 .part 文件，已直接恢复为最终文件: {out_file}")
                return
            if part_size > remote.size_bytes:
                log(
                    f"部分下载大小 {format_bytes(part_size)} 超过远端文件大小 "
                    f"{format_bytes(remote.size_bytes)}，将重新开始下载。"
                )
                part_file.unlink()

        last_error: BaseException | None = None
        for attempt in range(1, config.retries + 1):
            resume_from = 0
            if part_file.exists() and not config.no_resume:
                resume_from = part_file.stat().st_size
                if resume_from and not remote.accept_ranges:
                    log("远端未声明支持续传，已有 .part 文件将从头重新下载。")
                    part_file.unlink()
                    resume_from = 0

            try:
                _download_once(
                    url=url,
                    out_file=out_file,
                    part_file=part_file,
                    remote=remote,
                    config=config,
                    attempt=attempt,
                    resume_from=resume_from,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= config.retries or not should_retry(exc):
                    break
                sleep_for = backoff_seconds(attempt)
                log(
                    f"下载失败，第 {attempt}/{config.retries} 次尝试: "
                    f"{describe_exc(exc)}，将于 {format_duration(sleep_for)} 后重试。"
                )
                time.sleep(sleep_for)

        assert last_error is not None
        raise RuntimeError(f"下载失败: {out_file}，原因: {describe_exc(last_error)}") from last_error


def _download_once(
    *,
    url: str,
    out_file: Path,
    part_file: Path,
    remote: RemoteFileInfo,
    config: DownloadConfig,
    attempt: int,
    resume_from: int,
) -> None:
    byte_range = f"bytes={resume_from}-" if resume_from else None
    log(
        f"开始下载 {out_file.name}，第 {attempt}/{config.retries} 次尝试，"
        f"超时 {config.timeout_seconds:.0f}s。"
    )
    if resume_from:
        log(f"检测到未完成下载，将从 {format_bytes(resume_from)} 继续。")

    req = build_request(url, byte_range=byte_range)
    with request.urlopen(req, timeout=config.timeout_seconds) as response:
        status_code = response.getcode()
        if resume_from and status_code != 206:
            log(f"服务端未接受续传请求（HTTP {status_code}），本次改为从头重新下载。")
            part_file.unlink(missing_ok=True)
            _download_once(
                url=url,
                out_file=out_file,
                part_file=part_file,
                remote=remote,
                config=config,
                attempt=attempt,
                resume_from=0,
            )
            return

        total_size = remote.size_bytes
        if total_size is None:
            total_size = parse_total_size_from_content_range(response.headers)
        if total_size is None:
            content_length = parse_content_length(response.headers)
            if content_length is not None:
                total_size = content_length + resume_from

        write_mode = "ab" if resume_from else "wb"
        bytes_downloaded = resume_from
        session_downloaded = 0
        session_started = time.monotonic()
        last_reported = session_started - config.status_interval_seconds

        with part_file.open(write_mode) as sink:
            while True:
                chunk = response.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                bytes_downloaded += len(chunk)
                session_downloaded += len(chunk)

                now = time.monotonic()
                if now - last_reported >= config.status_interval_seconds:
                    log(
                        build_progress_message(
                            out_file,
                            downloaded_bytes=bytes_downloaded,
                            total_bytes=total_size,
                            session_downloaded_bytes=session_downloaded,
                            session_start=session_started,
                        )
                    )
                    last_reported = now

        if total_size is not None and bytes_downloaded != total_size:
            raise IOError(
                f"下载不完整，期望 {total_size} 字节，实际 {bytes_downloaded} 字节"
            )

        os.replace(part_file, out_file)
        log(f"下载完成: {out_file} ({format_bytes(bytes_downloaded)})")


def download_scan(scan_id: str, out_dir: Path, file_types: list[str], config: DownloadConfig) -> None:
    action = "探测" if config.probe_only else "下载"
    log(f"开始{action} scan {scan_id} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_type in file_types:
        url = f"{BASE_URL}/{RELEASE}/{scan_id}/{file_type}.zip"
        out_file = out_dir / f"{file_type}.zip"
        download_file(url, out_file, config)
    log(f"scan {scan_id} 处理完成。")


def download_release(release_scans: list[str], out_dir: Path, file_types: list[str], config: DownloadConfig) -> None:
    action = "探测" if config.probe_only else "下载"
    log(f"开始{action}完整 MP release -> {out_dir}")
    for scan_id in release_scans:
        scan_out_dir = out_dir / scan_id
        download_scan(scan_id, scan_out_dir, file_types, config)
    log("完整 MP release 处理完成。")


def download_task_data(task_data: list[str], out_dir: Path, config: DownloadConfig) -> None:
    action = "探测" if config.probe_only else "下载"
    log(f"开始{action}任务数据: {', '.join(task_data)}")
    for task_data_id in task_data:
        for relative_path in TASK_FILES[task_data_id]:
            url = f"{BASE_URL}/{RELEASE_TASKS}/{relative_path}"
            local_path = out_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, local_path, config)
        log(f"任务数据下载完成: {task_data_id}")


def confirm_or_exit(prompt: str, config: DownloadConfig) -> None:
    if config.assume_yes:
        log("使用 --assume_yes，自动继续。")
        return
    if not sys.stdin.isatty():
        raise RuntimeError("当前不是交互式终端，请加上 --assume_yes 以跳过确认。")
    input(prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downloads MP public data release.\n"
            "Example invocation:\n"
            "  python download_mp.py -o base_dir --id ALL --type matterport_skybox_images\n"
            "  python download_mp.py -o base_dir --id vyrNrziPKCB --type matterport_skybox_images --probe\n"
            "\n"
            "下载完成后，scan 数据位于 base_dir/v1/scans/<scan_id>/，"
            "task 数据位于 base_dir/v1/tasks/。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-o", "--out_dir", required=True, help="directory in which to download")
    parser.add_argument(
        "--task_data",
        default=[],
        nargs="+",
        help="task data files to download. Any of: " + ",".join(TASK_FILES.keys()),
    )
    parser.add_argument("--id", default="ALL", help="specific scan id to download or ALL to download entire dataset")
    parser.add_argument(
        "--type",
        nargs="+",
        help="specific file types to download. Any of: " + ",".join(FILETYPES),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"network timeout in seconds, default {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"max retry count per file, default {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL_SECONDS,
        help=f"seconds between progress logs, default {DEFAULT_STATUS_INTERVAL_SECONDS}",
    )
    parser.add_argument(
        "--lock-poll-interval",
        type=float,
        default=DEFAULT_LOCK_POLL_SECONDS,
        help=f"seconds between lock wait logs, default {DEFAULT_LOCK_POLL_SECONDS}",
    )
    parser.add_argument("--probe", action="store_true", help="only probe remote files, do not download")
    parser.add_argument("--assume_yes", action="store_true", help="skip interactive confirmations")
    parser.add_argument("--no-resume", action="store_true", help="do not resume from existing .part files")
    parser.add_argument("--overwrite", action="store_true", help="redownload even if the destination file exists")
    parser.add_argument(
        "--adopt-legacy-tmp",
        action="store_true",
        help="adopt the largest legacy tmp* file from the old downloader as a .part file",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.retries <= 0:
        parser.error("--retries must be > 0")
    if args.status_interval <= 0:
        parser.error("--status-interval must be > 0")
    if args.lock_poll_interval <= 0:
        parser.error("--lock-poll-interval must be > 0")
    if args.no_resume and args.adopt_legacy_tmp:
        parser.error("--no-resume and --adopt-legacy-tmp cannot be used together")

    if args.task_data:
        invalid_task_data = sorted(set(args.task_data) - set(TASK_FILES))
        if invalid_task_data:
            parser.error("Unrecognized task data id(s): " + ", ".join(invalid_task_data))

    if args.type:
        invalid_types = sorted(set(args.type) - set(FILETYPES))
        if invalid_types:
            parser.error("Invalid file type(s): " + ", ".join(invalid_types))

    return args


def main() -> None:
    args = parse_args()
    config = DownloadConfig(
        timeout_seconds=args.timeout,
        retries=args.retries,
        status_interval_seconds=args.status_interval,
        lock_poll_seconds=args.lock_poll_interval,
        assume_yes=args.assume_yes,
        probe_only=args.probe,
        no_resume=args.no_resume,
        overwrite=args.overwrite,
        adopt_legacy_tmp=args.adopt_legacy_tmp,
    )

    log(f"TOS: {TOS_URL}")
    release_file = f"{BASE_URL}/{RELEASE}.txt"
    release_scans = get_release_scans(release_file, config)
    file_types = args.type or FILETYPES

    if args.task_data:
        out_dir = Path(args.out_dir) / RELEASE_TASKS
        download_task_data(args.task_data, out_dir, config)
        log(f"任务数据处理完成: {args.task_data}")
        if not config.probe_only:
            confirm_or_exit("按回车继续下载主数据，或 Ctrl-C 退出。", config)

    requested_id = args.id.strip()
    if requested_id.lower() != "all":
        if requested_id not in release_scans:
            raise SystemExit(f"ERROR: Invalid scan id: {requested_id}")
        out_dir = Path(args.out_dir) / RELEASE / requested_id
        download_scan(requested_id, out_dir, file_types, config)
        return

    if "minos" not in args.task_data:
        if len(file_types) == len(FILETYPES):
            log(f"警告：你将下载完整 MP release，预计需要 {RELEASE_SIZE} 空间。")
        else:
            log(f"警告：你将下载所有 scan 的类型: {', '.join(file_types)}")
        log("现有文件会被跳过；若想重下某个文件，可删除对应文件或加 --overwrite。")
        if not config.probe_only:
            confirm_or_exit("按回车继续，或 Ctrl-C 退出。", config)
        out_dir = Path(args.out_dir) / RELEASE
        download_release(release_scans, out_dir, file_types, config)


if __name__ == "__main__":
    main()
