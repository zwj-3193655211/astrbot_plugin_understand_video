#!/usr/bin/env python3
"""
video-transcriber - skill 主入口（Agent Skills 标准，适配任意智能体）

处理 B 站链接 / 本地视频 / 本地音频，输出转写文本 + 句级时间戳。

Skill 调用：
    import video_transcriber
    result = video_transcriber.run("BV1xx411c7mD")
    info = video_transcriber.info()
    status = video_transcriber.check_status()
    video_transcriber.initialize()      # 下载缺失模型（run() 首次会自动调用）
    video_transcriber.clear_cache()      # 清空缓存

CLI：
    python video_transcriber.py "BV1xx411c7mD"
    python video_transcriber.py "C:\\path\\to\\video.mp4"
    python video_transcriber.py --setup
    python video_transcriber.py --init
    python video_transcriber.py --status
    python video_transcriber.py --clear-cache
    python video_transcriber.py --info
"""
import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import bilibili
import local_media
import recognizer
import douyin
from cache import CacheManager
from config import CONFIG_FILE, SKILL_DIR, load_config
import asr_backend

__skill__: Dict[str, Any] = {
    "name": "video-transcriber",
    "version": "3.0.0",
    "description": "转写 B 站视频或本地音视频为文本（FunASR Paraformer-Large + VAD + 标点）",
}


# ==================== 输入识别 ====================

def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def identify_input(text: str) -> Dict[str, Any]:
    """识别：B 站 / 本地视频 / 本地音频 / 未知"""
    text = (text or "").strip()
    if not text:
        return {"type": "unknown", "original": text}
    if local_media.is_local_video(text):
        return {"type": "local_video", "path": text, "original": text}
    if local_media.is_local_audio(text):
        return {"type": "local_audio", "path": text, "original": text}
    # 抖音（v.douyin.com / douyin.com / iesdouyin.com），支持整段分享文本
    low = text.lower()
    if "douyin.com" in low or "iesdouyin.com" in low:
        return {"type": "douyin", "url": text, "original": text}
    url = bilibili.parse_url(text)
    if url:
        return {"type": "bilibili", "url": url, "original": text}
    # 通用平台链接（YouTube/AcFun/微博等，走 yt-dlp）
    m = re.search(r"https?://[^\s，。、；;！!？?]+", text)
    if m:
        return {"type": "generic_url", "url": m.group(0), "original": text}
    return {"type": "unknown", "original": text}


# ==================== 处理流 ====================

async def _transcribe_audio(
    title: str,
    audio_path: str,
    src_type: str,
    meta: Dict[str, Any],
    duration: float,
    cfg: Dict[str, Any],
    cache: CacheManager,
) -> Dict[str, Any]:
    """统一：解析 ASR 后端 →（本地：转 WAV + 确保模型）→ 识别 → 存文本"""
    # 1. 解析后端（模型缺失且无 key 时给出两条方案）
    try:
        backend = asr_backend.resolve(cfg)
    except asr_backend.AsrConfigError as e:
        return {"status": "error", "error": "asr_unavailable", "message": str(e)}

    # 2. 本地后端：转 16kHz mono WAV + 确保模型就绪（skill 规范：首次自动下载）
    if backend == "local":
        wav_path = cache.audio_dir / f"{Path(audio_path).stem}.wav"
        if not wav_path.exists():
            ok = local_media.convert_to_wav(audio_path, str(wav_path))
            if not ok:
                return {"status": "error", "error": "convert_failed",
                        "message": "WAV 转换失败，请确认 ffmpeg 已安装"}
        ready, missing = recognizer.check_models(cfg)
        if not ready:
            if cfg.get("auto_download", True):
                init = initialize()
                if init["status"] != "success":
                    return {"status": "error", "error": "model_download_failed",
                            "message": "自动下载模型失败", "detail": init}
                ready, missing = recognizer.check_models(cfg)
            if not ready:
                return {
                    "status": "error",
                    "error": "model_missing",
                    "message": "ASR 模型未就绪",
                    "missing": missing,
                    "suggestion": "运行 `python setup.py` 或 `python video_transcriber.py --init`",
                }
        input_path = str(wav_path)
    else:
        # 云端：直接用音频文件（m4a/mp3/wav 均可）
        input_path = str(audio_path)

    # 3. 识别（FunASR / SiliconFlow 都是阻塞调用，放到线程池）
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, asr_backend.transcribe, input_path, cfg)

    # 4. 存文本
    text = result["text"]
    txt_path = cache.save_text(title, text)

    return {
        "status": "success",
        "type": src_type,
        "title": title,
        "transcription": text,
        "transcription_path": str(txt_path),
        "segments": result["segments"],
        "duration": round(duration or result["duration"], 2),
        **meta,
    }


async def _analyze_bilibili(url: str, cfg: Dict[str, Any], cache: CacheManager) -> Dict[str, Any]:
    custom_cookie = cfg.get("cookie", "")
    info = bilibili.get_audio(url, str(cache.audio_dir), custom_cookie=custom_cookie)
    if not info:
        return {"status": "error", "error": "fetch_failed",
                "message": "无法获取 B 站视频或下载音频"}

    duration = local_media.probe_duration(info["audio_path"])
    max_dur = cfg["max_duration_minutes"] * 60
    if max_dur > 0 and duration > max_dur:
        return {"status": "error", "error": "duration_exceeded",
                "message": f"时长 {duration / 60:.1f} 分钟超过限制 {cfg['max_duration_minutes']} 分钟"}

    return await _transcribe_audio(
        title=info["title"],
        audio_path=info["audio_path"],
        src_type="bilibili",
        meta={"bv_id": info["bv_id"], "url": url},
        duration=duration,
        cfg=cfg, cache=cache,
    )


async def _analyze_douyin(url: str, cfg: Dict[str, Any], cache: CacheManager) -> Dict[str, Any]:
    """抖音：下载音频（H5 → 浏览器 cookie → API 多级）→ 转写"""
    info = douyin.download_douyin(
        url, str(cache.audio_dir),
        log=lambda msg: print(msg),
        custom_cookie=cfg.get("cookie", ""),
    )
    if not info:
        return {"status": "error", "error": "fetch_failed",
                "message": "无法获取抖音视频（风控时可稍后重试，或浏览器登录抖音后复制 cookie 填入 config.json）"}

    duration = local_media.probe_duration(info["audio_path"])
    max_dur = cfg["max_duration_minutes"] * 60
    if max_dur > 0 and duration > max_dur:
        return {"status": "error", "error": "duration_exceeded",
                "message": f"时长 {duration / 60:.1f} 分钟超过限制 {cfg['max_duration_minutes']} 分钟"}

    return await _transcribe_audio(
        title=info["title"],
        audio_path=info["audio_path"],
        src_type="douyin",
        meta={"aweme_id": info["aweme_id"], "url": url},
        duration=duration,
        cfg=cfg, cache=cache,
    )


async def _analyze_generic_url(url: str, cfg: Dict[str, Any], cache: CacheManager) -> Dict[str, Any]:
    """通用平台（YouTube/AcFun/微博等）：yt-dlp 下载音频 → 转写"""
    info = douyin.download_via_ytdlp(
        url, str(cache.audio_dir),
        log=lambda msg: print(msg),
        custom_cookie=cfg.get("cookie", ""),
    )
    if not info:
        return {"status": "error", "error": "fetch_failed",
                "message": "无法下载该平台音频（需安装 yt-dlp；YouTube 需代理；部分平台需 cookie 填入 config.json）"}

    duration = local_media.probe_duration(info["audio_path"])
    max_dur = cfg["max_duration_minutes"] * 60
    if max_dur > 0 and duration > max_dur:
        return {"status": "error", "error": "duration_exceeded",
                "message": f"时长 {duration / 60:.1f} 分钟超过限制 {cfg['max_duration_minutes']} 分钟"}

    return await _transcribe_audio(
        title=info["title"],
        audio_path=info["audio_path"],
        src_type="generic",
        meta={"url": url},
        duration=duration,
        cfg=cfg, cache=cache,
    )


async def _analyze_local_video(path: str, cfg: Dict[str, Any], cache: CacheManager) -> Dict[str, Any]:
    p = Path(path).resolve()
    title = p.stem
    cache_key = _hash_key(str(p))
    audio_path = cache.audio_dir / f"{cache_key}.m4a"

    if not audio_path.exists():
        ok = local_media.extract_audio(path, str(audio_path))
        if not ok:
            return {"status": "error", "error": "extract_failed",
                    "message": "ffmpeg 提取音频失败，请确认 ffmpeg 已安装且视频可访问"}

    duration = local_media.probe_duration(str(audio_path))
    max_dur = cfg["max_duration_minutes"] * 60
    if max_dur > 0 and duration > max_dur:
        return {"status": "error", "error": "duration_exceeded",
                "message": f"时长 {duration / 60:.1f} 分钟超过限制 {cfg['max_duration_minutes']} 分钟"}

    return await _transcribe_audio(
        title=title,
        audio_path=str(audio_path),
        src_type="local_video",
        meta={"path": str(p)},
        duration=duration,
        cfg=cfg, cache=cache,
    )


async def _analyze_local_audio(path: str, cfg: Dict[str, Any], cache: CacheManager) -> Dict[str, Any]:
    p = Path(path).resolve()
    duration = local_media.probe_duration(str(p))
    max_dur = cfg["max_duration_minutes"] * 60
    if max_dur > 0 and duration > max_dur:
        return {"status": "error", "error": "duration_exceeded",
                "message": f"时长 {duration / 60:.1f} 分钟超过限制 {cfg['max_duration_minutes']} 分钟"}

    return await _transcribe_audio(
        title=p.stem,
        audio_path=str(p),
        src_type="local_audio",
        meta={"path": str(p)},
        duration=duration,
        cfg=cfg, cache=cache,
    )


# ==================== Skill 公开接口 ====================

def run(input_text: str, **kwargs) -> Dict[str, Any]:
    """
    Skill 主入口。
    
    Args:
        input_text: B 站链接 / BV 号 / b23.tv 短链 / 本地音视频路径
        **kwargs: 预留（暂未使用）
    Returns:
        {"status": "success"|"error", "transcription": ..., "segments": ..., ...}
    """
    if not input_text or not input_text.strip():
        return {"status": "error", "error": "invalid_input",
                "message": "输入为空，请提供 B 站链接、BV 号或本地音视频路径"}

    # 短命令
    cmd = input_text.strip().lower()
    if cmd in ("--help", "help", "-h", "?"):
        return {"status": "help", "message": _help_text()}
    if cmd in ("--status", "status"):
        return check_status()
    if cmd in ("--clear-cache", "clear-cache"):
        return clear_cache()
    if cmd in ("--init", "init", "--initialize", "initialize"):
        return initialize()
    if cmd in ("--setup", "setup"):
        return setup()
    if cmd in ("--info", "info"):
        return info()

    # 业务逻辑
    info_input = identify_input(input_text)
    if info_input["type"] == "unknown":
        return {
            "status": "error",
            "error": "invalid_input",
            "message": f"无法识别输入：{input_text[:120]}",
            "suggestion": "支持：B 站链接（BV 号 / bilibili.com URL / b23.tv 短链）、抖音链接（v.douyin.com / douyin.com）、通用平台链接（YouTube/AcFun 等，需 yt-dlp）、本地 mp4 / mkv / mp3 / wav / m4a 等",
        }

    cfg = load_config()
    cache = CacheManager(cfg["cache_dir"])
    try:
        if info_input["type"] == "bilibili":
            result = asyncio.run(_analyze_bilibili(info_input["url"], cfg, cache))
        elif info_input["type"] == "douyin":
            result = asyncio.run(_analyze_douyin(info_input["url"], cfg, cache))
        elif info_input["type"] == "generic_url":
            result = asyncio.run(_analyze_generic_url(info_input["url"], cfg, cache))
        elif info_input["type"] == "local_video":
            result = asyncio.run(_analyze_local_video(info_input["path"], cfg, cache))
        elif info_input["type"] == "local_audio":
            result = asyncio.run(_analyze_local_audio(info_input["path"], cfg, cache))
        else:
            return {"status": "error", "error": "invalid_input"}
    except Exception as e:
        return {"status": "error", "error": "execution_failed", "message": str(e)}

    result["_skill"] = {"name": __skill__["name"], "version": __skill__["version"]}
    return result


def info() -> Dict[str, Any]:
    """返回 skill 元数据"""
    return {"status": "success", "skill": __skill__}


def check_status() -> Dict[str, Any]:
    """检查模型、缓存、ffmpeg 状态"""
    cfg = load_config()
    cache = CacheManager(cfg["cache_dir"])
    ready, missing = recognizer.check_models(cfg)
    try:
        backend = asr_backend.resolve(cfg)
    except asr_backend.AsrConfigError:
        backend = None
    return {
        "status": "ready" if (ready or backend == "cloud") else "need_init",
        "asr_backend": backend or "none",
        "asr_provider": asr_backend.provider_label(cfg) if backend == "cloud" else "local",
        "models_ready": ready,
        "missing_models": missing,
        "api_key_set": bool(asr_backend.api_key(cfg)),
        "model_root": cfg["model_root"],
        "ffmpeg": local_media.find_ffmpeg() or "missing",
        "config_path": str(CONFIG_FILE),
        "config": {k: v for k, v in cfg.items() if k != "cache_dir"},
        "cache": cache.info(),
    }


def clear_cache() -> Dict[str, Any]:
    """清理所有缓存（音频 + 转写文本）"""
    cfg = load_config()
    cache = CacheManager(cfg["cache_dir"])
    return cache.clear_all()


def initialize() -> Dict[str, Any]:
    """触发模型检查/下载（模型缺失时 run() 会自动调用）"""
    cfg = load_config()
    root = Path(cfg["model_root"])
    root.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, str(SKILL_DIR / "download.py"), "--model-root", str(root)],
    )
    return {
        "status": "success" if r.returncode == 0 else "error",
        "model_root": str(root),
        "returncode": r.returncode,
    }


def setup() -> Dict[str, Any]:
    """一键安装向导：环境检查 + 下载模型 + 写配置（等价于 python setup.py）"""
    r = subprocess.run([sys.executable, str(SKILL_DIR / "setup.py")])
    return {
        "status": "success" if r.returncode == 0 else "error",
        "returncode": r.returncode,
        "detail": "运行 python setup.py 完成环境检查 / 模型下载 / 配置写入",
    }


def _help_text() -> str:
    cfg = load_config()
    return (
        f"video-transcriber v{__skill__['version']}\n"
        f"{__skill__['description']}\n\n"
        "用法：\n"
        '    skill.run("<B站/抖音链接 / BV号 / 本地音视频路径>")\n\n'
        "示例：\n"
        '    skill.run("BV1xx411c7mD")\n'
        '    skill.run("https://www.bilibili.com/video/BV1xx411c7mD")\n'
        '    skill.run("https://v.douyin.com/xxx/")\n'
        '    skill.run("C:\\\\Videos\\\\lecture.mp4")\n'
        '    skill.run("D:/audios/interview.wav")\n\n'
        "命令：\n"
        "    --setup          一键安装（环境检查 + 下载模型 + 写配置）\n"
        "    --status         检查模型 / 缓存 / ffmpeg 状态\n"
        "    --clear-cache    清理所有缓存\n"
        "    --init           下载缺失的 ASR 模型\n"
        "    --info           显示 skill 元数据\n"
        "    --help           显示此帮助\n\n"
        "ASR 后端（config.json 的 asr_backend / asr_provider）：\n"
        "    auto         本地模型就绪用 FunASR，否则有 key 用云端（默认）\n"
        "    local        FunASR Paraformer-Large（需 ~1.1GB 模型）\n"
        "    cloud        云端转写，供应商可插拔：\n"
        "                   siliconflow（默认）：api.siliconflow.cn\n"
        "                   openai：任意 OpenAI 兼容端点，如火山方舟\n"
        "                     asr_base_url=https://ark.cn-beijing.volces.com/api/v3\n"
        "                     asr_api_key=<key>  asr_model=<豆包语音模型>\n\n"
        "配置：\n"
        f"    配置文件：{CONFIG_FILE}\n"
        f"    模型目录：{cfg['model_root']}\n"
        f"    缓存目录：{cfg['cache_dir']}\n"
        "\n"
        "首次使用：模型缺失时 run() 会自动下载；也可手动 `python setup.py`。\n"
        "不想下载模型？设环境变量 ASR_API_KEY 走云端转写。\n"
        "复用已有模型：在 config.json 设置 model_root 指向\n"
        "已下完 Paraformer 三件套的目录即可（无需重新下载）。\n"
    )


# ==================== CLI ====================

def main():
    p = argparse.ArgumentParser(
        description=f"video-transcriber v{__skill__['version']}: B 站/本地视频转文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="B 站链接 / BV 号 / 本地音视频路径")
    p.add_argument("--setup", action="store_true", help="一键安装向导")
    p.add_argument("--init", action="store_true", help="下载 ASR 模型")
    p.add_argument("--status", action="store_true", help="检查状态")
    p.add_argument("--clear-cache", action="store_true", help="清理缓存")
    p.add_argument("--info", action="store_true", help="显示元数据")
    args = p.parse_args()

    if args.setup:
        r = setup()
    elif args.init:
        r = initialize()
    elif args.status:
        r = check_status()
    elif args.clear_cache:
        r = clear_cache()
    elif args.info:
        r = info()
    elif args.input:
        r = run(args.input)
    else:
        print(_help_text())
        return
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
