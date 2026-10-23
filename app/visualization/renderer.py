"""
Isolated Manim rendering.

Manim never runs in the API process. A render is a short-lived subprocess with: a scrubbed environment (no provider keys or
tokens), a private temporary working directory, a hard wall-clock deadline with process-TREE kill, a watched output-size
cap, and validation of the produced file. The scene source is generated only from approved templates
(`manim_templates`); the model's text never reaches the interpreter as code.

Honest limits: on a plain host this is process isolation, not a sandbox. Network and filesystem confinement need a
container/namespace (see docs/VISUALIZATIONS.md for the deployment recipe); the code does not claim them.
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.visualization.manim_templates import ValidatedSpec, build_scene_source
from app.visualization.policy import FailureCategory

_JOB_ID = re.compile(r"^mj_[0-9a-f]{32}$")
_ENV_KEEP = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL")
MIN_MP4_BYTES = 1024


@dataclass
class RenderResult:
    ok: bool
    category: FailureCategory | None = None
    path: Path | None = None
    detail: str = ""            # short, internal-only (never shown to users or logged with source)
    thumbnail: Path | None = None   # optional PNG preview (last frame); a temporary copy the CALLER must move or delete


class MediaStore:
    """Filesystem store keyed by job id (a hash, so it can never contain a path component)."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def path_for(self, job_id: str) -> Path | None:
        if not _JOB_ID.match(job_id or ""):
            return None
        return self.root / f"{job_id}.mp4"

    def thumb_path_for(self, job_id: str) -> Path | None:
        p = self.path_for(job_id)
        return p.with_suffix(".png") if p else None

    def save_thumbnail(self, job_id: str, source: Path) -> None:
        dest = self.thumb_path_for(job_id)
        if dest is None:
            raise ValueError("invalid job id")
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)

    def save(self, job_id: str, source: Path) -> str:
        dest = self.path_for(job_id)
        if dest is None:
            raise ValueError("invalid job id")
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)          # atomic: readers never see a partial file
        return job_id

    def exists(self, job_id: str) -> bool:
        p = self.path_for(job_id)
        return bool(p and p.is_file())

    def delete(self, job_id: str) -> None:
        for p in (self.path_for(job_id), self.thumb_path_for(job_id)):
            if p and p.exists():
                p.unlink(missing_ok=True)

    def purge_older_than(self, hours: float) -> int:
        if not self.root.exists():
            return 0
        cutoff, n = time.time() - hours * 3600, 0
        for f in [*self.root.glob("mj_*.mp4"), *self.root.glob("mj_*.png")]:
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                n += 1
        return n


def is_valid_mp4(path: Path, max_bytes: int) -> bool:
    try:
        size = path.stat().st_size
        if not MIN_MP4_BYTES <= size <= max_bytes:
            return False
        with open(path, "rb") as f:
            return f.read(12)[4:8] == b"ftyp"
    except OSError:
        return False


def _scrubbed_env(tmp: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env.update({"TEMP": tmp, "TMP": tmp, "TMPDIR": tmp, "HOME": tmp, "USERPROFILE": tmp, "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


MAX_THUMB_BYTES = 2 * 1024 * 1024
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


async def _capture_frame(args_base: list[str], workdir: str, scene: Path, media: Path, timeout_seconds: float) -> Path | None:
    """Best-effort preview: render only the final frame (`manim -s`). Any failure just means "no thumbnail"; it can
    never fail or delay the video beyond its own short deadline."""
    args = [*args_base, "-s", "--media_dir", str(media), str(scene), "GeneratedScene"]
    kwargs: dict = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=workdir, env=_scrubbed_env(workdir), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, **kwargs)
    except (OSError, ValueError):
        return None
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        _kill_tree(proc)
        await proc.wait()
        return None
    if proc.returncode != 0:
        return None
    for png in sorted(media.rglob("GeneratedScene*.png")):
        try:
            if png.stat().st_size <= MAX_THUMB_BYTES and open(png, "rb").read(8) == _PNG_MAGIC:
                fd, name = tempfile.mkstemp(prefix="vizthumb_", suffix=".png")
                os.close(fd)
                shutil.copyfile(png, name)
                return Path(name)
        except OSError:
            continue
    return None


def _classify_stderr(text: str) -> FailureCategory:
    low = text.lower()
    if "memoryerror" in low or "out of memory" in low:
        return FailureCategory.RESOURCE_LIMIT
    return FailureCategory.RENDER_ERROR


async def render_spec(
    spec: ValidatedSpec,
    *,
    python: str = "",
    quality: str = "l",
    timeout_seconds: int = 90,
    max_output_bytes: int = 20 * 1024 * 1024,
    cancelled=lambda: False,
) -> RenderResult:
    """Render a validated spec to an mp4. The throwaway working directory is always removed; on success `result.path` is a
    temporary copy of the video that the CALLER must move into the MediaStore and delete."""
    try:
        source = build_scene_source(spec)
    except Exception:
        return RenderResult(False, FailureCategory.INVALID_SPEC, detail="scene build failed")

    workdir = tempfile.mkdtemp(prefix="viz_")
    try:
        scene = Path(workdir) / "scene.py"
        scene.write_text(source, encoding="utf-8")
        media = Path(workdir) / "media"
        args = [python or sys.executable, "-I", "-m", "manim", "render", f"-q{quality}", "--disable_caching",
                "--format", "mp4", "--media_dir", str(media), str(scene), "GeneratedScene"]
        kwargs: dict = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=workdir, env=_scrubbed_env(workdir), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, **kwargs)
        except (OSError, ValueError):
            return RenderResult(False, FailureCategory.WORKER_UNAVAILABLE, detail="could not start renderer")

        stderr_task = asyncio.create_task(proc.stderr.read(20000))      # bounded capture
        deadline = time.monotonic() + timeout_seconds
        while proc.returncode is None:
            if cancelled():
                _kill_tree(proc)
                await proc.wait()
                return RenderResult(False, FailureCategory.CANCELLED)
            if time.monotonic() > deadline:
                _kill_tree(proc)
                await proc.wait()
                return RenderResult(False, FailureCategory.TIMEOUT, detail=f"exceeded {timeout_seconds}s")
            if await asyncio.to_thread(_dir_size, workdir) > max_output_bytes * 4:
                _kill_tree(proc)
                await proc.wait()
                return RenderResult(False, FailureCategory.RESOURCE_LIMIT, detail="working directory too large")
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
        stderr = (await stderr_task).decode("utf-8", "replace")
        if proc.returncode != 0:
            return RenderResult(False, _classify_stderr(stderr), detail=stderr[-300:])
        videos = sorted(media.rglob("GeneratedScene.mp4"))
        if not videos or not is_valid_mp4(videos[0], max_output_bytes):
            size = videos[0].stat().st_size if videos else 0
            cat = FailureCategory.RESOURCE_LIMIT if size > max_output_bytes else FailureCategory.INVALID_OUTPUT
            return RenderResult(False, cat, detail="no usable video")
        # keep the workdir alive for the caller: move the file to a sibling temp name that survives cleanup
        fd, name = tempfile.mkstemp(prefix="vizout_", suffix=".mp4")
        os.close(fd)
        keep = Path(name)
        shutil.copyfile(videos[0], keep)
        thumb = await _capture_frame(args[: args.index("--disable_caching") + 1], workdir, scene, media, min(30, timeout_seconds))
        return RenderResult(True, path=keep, thumbnail=thumb)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
