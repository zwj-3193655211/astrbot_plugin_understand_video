"""
本地音视频处理：路径识别、ffmpeg 探测/提取/转 WAV
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

VIDEO_EXT = {".mp4", ".flv", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aiff"}


def is_local_video(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.suffix.lower() in VIDEO_EXT


def is_local_audio(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.suffix.lower() in AUDIO_EXT


def find_ffmpeg() -> Optional[str]:
    """查找 ffmpeg：环境变量 FFMPEG_PATH → skill 目录 → PATH"""
    env = os.environ.get("FFMPEG_PATH")
    if env and os.path.exists(env):
        return env
    skill_dir = Path(__file__).parent
    for n in ("ffmpeg.exe", "ffmpeg"):
        c = skill_dir / n
        if c.exists():
            return str(c)
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for n in ("ffmpeg.exe", "ffmpeg"):
            c = Path(p) / n
            if c.exists():
                return str(c)
    return None


def probe_duration(path: str) -> float:
    """用 ffprobe（或 ffmpeg -i 兜底）探测时长（秒），失败返回 0.0"""
    ff = find_ffmpeg()
    if not ff:
        return 0.0
    # 优先 ffprobe
    probe = ff.replace("ffmpeg", "ffprobe")
    if os.path.exists(probe):
        try:
            r = subprocess.run(
                [probe, "-v", "quiet", "-print_format", "json", "-show_format", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(json.loads(r.stdout).get("format", {}).get("duration", 0.0))
        except Exception:
            pass
    # 兜底：ffmpeg -i 解析 stderr
    try:
        r = subprocess.run([ff, "-i", path, "-hide_banner"], capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})", r.stderr)
        if m:
            h, mn, s, cs = map(int, m.groups())
            return h * 3600 + mn * 60 + s + cs / 100
    except Exception:
        return 0.0
    return 0.0


def extract_audio(video_path: str, output_path: str,
                  log: Callable[[str], None] = lambda *_: None) -> Optional[str]:
    """从本地视频提取音频为 m4a（AAC）"""
    ff = find_ffmpeg()
    if not ff:
        log("❌ 未找到 ffmpeg，请先安装（https://ffmpeg.org/）并加入 PATH")
        return None
    try:
        cmd = [ff, "-y", "-i", video_path, "-vn", "-acodec", "aac", output_path]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        r = subprocess.run(cmd, capture_output=True, creationflags=flags)
        if r.returncode == 0 and Path(output_path).exists():
            log(f"✅ 音频已提取：{Path(output_path).name}")
            return output_path
        err = r.stderr.decode("utf-8", errors="ignore")[-300:] if r.stderr else ""
        log(f"❌ ffmpeg 提取失败：{err}")
        return None
    except Exception as e:
        log(f"❌ 提取失败：{e}")
        return None


def convert_to_wav(input_path: str, output_path: str,
                   log: Callable[[str], None] = lambda *_: None) -> Optional[str]:
    """转 16kHz mono WAV（FunASR 推荐格式）"""
    ff = find_ffmpeg()
    if not ff:
        log("❌ 未找到 ffmpeg")
        return None
    try:
        cmd = [ff, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        r = subprocess.run(cmd, capture_output=True, creationflags=flags)
        if r.returncode == 0 and Path(output_path).exists():
            return output_path
        err = r.stderr.decode("utf-8", errors="ignore")[-300:] if r.stderr else ""
        log(f"❌ WAV 转换失败：{err}")
        return None
    except Exception as e:
        log(f"❌ 转换失败：{e}")
        return None
