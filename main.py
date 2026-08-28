"""
AstrBot 视频理解插件 v2.0

底层使用 video-transcriber skill（嵌入在 core/ 目录），支持：
- B 站（BV / bilibili.com / b23.tv）
- 抖音（v.douyin.com / douyin.com）
- 通用平台（YouTube / AcFun / 微博等，依赖 yt-dlp）
- 本地音视频（mp4 / mkv / mp3 / wav / m4a 等）

两种 ASR 后端：
- local  : FunASR Paraformer-Large（GPU 最佳，需 ~1.1GB 模型）
- cloud  : SiliconFlow / 火山方舟 / OpenAI 兼容端点（按用量计费，零模型）
- auto   : 本地优先，否则云端

数据存储规范：所有缓存（音频 + 转写文本）和模型放在 AstrBot 数据目录下，
不污染插件目录本身（AstrBot 升级 / 重装插件时不会丢数据）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==================== 第三方库 ====================
# 抑制第三方库的噪音日志（FunASR / modelscope 启动时会刷一堆 INFO）
logging.getLogger().setLevel(logging.WARNING)
for _name in ("websockets", "httpcore", "httpx", "openai", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ==================== AstrBot SDK ====================
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Plain
from astrbot.api.event import MessageChain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api import logger

# ==================== 内嵌 skill 导入 ====================
# 把 core/ 加进 sys.path，让 skill 模块可以 import
_PLUGIN_DIR = Path(__file__).resolve().parent
# 注意：_PLUGIN_DATA_DIR 在 __init__ 里通过 StarTools.get_data_dir 设置
# 这里先用 _PLUGIN_DIR/../.. 拼相对路径作为兜底（plugin 部署后通常是 Astrbot/data/plugin_data/<name>）
_PLUGIN_DATA_DIR_FALLBACK = _PLUGIN_DIR.parent.parent / "plugin_data" / "astrbot_plugin_understand_video"
_PLUGIN_DATA_DIR = _PLUGIN_DATA_DIR_FALLBACK  # 启动时被 __init__ 覆盖
_CORE_DIR = _PLUGIN_DIR / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

# 这些 import 必须在 sys.path 调整之后
import config as _skill_config      # noqa: E402
import video_transcriber as _vt      # noqa: E402
import recognizer as _skill_recognizer      # noqa: E402
import cache as _skill_cache         # noqa: E402
import local_media as _skill_local   # noqa: E402
import asr_backend as _skill_asr     # noqa: E402
import bilibili as _skill_bilibili   # noqa: E402
import douyin as _skill_douyin       # noqa: E402


# ==================== 常量 ====================

# skill 模块的路径常量
SKILL_DIR = _CORE_DIR

# 插件默认配置
DEFAULT_MAX_DURATION_MINUTES = 70
DEFAULT_AUTO_CLEAR_DAYS = 7
DEFAULT_ASR_BACKEND = "auto"
DEFAULT_ASR_PROVIDER = "siliconflow"
DEFAULT_ASR_MODEL = "FunAudioLLM/SenseVoiceSmall"

# 总结提示词默认值（与老版本兼容）
SUMMARY_PROMPT_DEFAULT = """请对以下视频的转写内容进行总结，提取关键信息和要点：

视频标题：{title}

转写内容：
{content}

请按以下格式输出：
1. 视频主要内容概述
2. 关键信息点（列表形式）
3. 核心结论或观点"""


# ==================== 工具函数 ====================

def _format_duration(seconds: float) -> str:
    """把秒数格式化成「x分y秒」"""
    seconds = int(seconds or 0)
    return f"{seconds // 60}分{seconds % 60}秒"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024 / 1024 / 1024:.2f} GB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _check_ffmpeg() -> Optional[str]:
    """多路径查找 ffmpeg（避免用户 PATH 还没刷新导致找不到）

    查找顺序：
    1. FFMPEG_PATH 环境变量（用户显式指定，最优先）
    2. 系统 PATH（shutil.which）
    3. 插件 data 目录（ffmpeg.exe/ffmpeg，用户手动放这里最稳）
    4. AstrBot 安装目录（ffmpeg.exe/ffmpeg）
    5. 兜底：插件目录（历史兼容）

    推荐用法：用户把 ffmpeg.exe 复制到
    C:\\Users\\<name>\\Astrbot\\data\\plugin_data\\astrbot_plugin_understand_video\\
    即使 PATH 没刷新也能用。
    """
    # 1. 环境变量
    env_path = os.environ.get("FFMPEG_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    # 2. 系统 PATH
    ffmpeg = shutil_which("ffmpeg") or shutil_which("ffmpeg.exe")
    if ffmpeg:
        return ffmpeg

    # 3. 插件 data 目录（推荐用法 - 不依赖 PATH）
    data_dir = _PLUGIN_DATA_DIR if '_PLUGIN_DATA_DIR' in globals() else _PLUGIN_DATA_DIR_FALLBACK
    for name in ("ffmpeg.exe", "ffmpeg"):
        cand = data_dir / name
        if cand.exists():
            return str(cand)

    # 4. AstrBot 安装目录
    astrbot_root = Path(__file__).resolve().parent.parent.parent  # plugin/..  → Astrbot/
    for name in ("ffmpeg.exe", "ffmpeg"):
        cand = astrbot_root / name
        if cand.exists():
            return str(cand)

    # 5. 插件目录（历史兼容）
    for name in ("ffmpeg.exe", "ffmpeg"):
        cand = _PLUGIN_DIR / name
        if cand.exists():
            return str(cand)

    return None


def shutil_which(name: str) -> Optional[str]:
    """shutil.which 的本地实现（避免 import 路径差异）"""
    import shutil as _shutil
    return _shutil.which(name)


def _resolve_input(text: str) -> str:
    """
    从用户消息中提取 URL / 本地路径（B 站 / 抖音 / 通用平台 / 本地文件）。
    skill 的 identify_input 会自己处理整段消息，但这里我们提前做一个简单剥离，
    把指令前缀（/video 等）去掉。
    """
    t = (text or "").strip()
    # 去掉可能的命令前缀（/video 之类）
    for prefix in ("/video", "/understand", "/transcribe", "/video "):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].lstrip()
            break
    return t


# 视频链接自动识别正则（用于自然语言触发，不需 /video 前缀）
# 覆盖：B 站（b23.tv / bilibili.com / m.bilibili.com）/ 抖音（v.douyin / douyin.com）
#       YouTube（youtu.be / youtube.com）/ AcFun / 西瓜视频 / 微博
# 注：前缀用 [a-z0-9.-]* 兼容 www. / m. 等子域
# 字符类放宽：允许 URL 常见字符（含 `?` `=` `&` 等 query 字符）
VIDEO_URL_PATTERN = re.compile(
    r"https?://[a-zA-Z0-9.\-]*"
    r"(?:"
    r"b23\.tv"
    r"|bili(?:bili)?\.com"
    r"|v\.douyin\.com|iesdouyin\.com|douyin\.com"
    r"|youtu\.?be|youtube\.com"
    r"|acfun\.cn"
    r"|ixigua\.com"
    r"|weibo\.(?:com|cn)"
    r")"
    r"(?:[/?&=#%\w\-.~:+\u4e00-\u9fff]*)?",
    re.IGNORECASE,
)


# ==================== 插件主类 ====================

@register(
    "astrbot_plugin_understand_video",
    "zwj-3193655211",
    "视频理解助手 - B站/抖音/本地音视频转文字 + AI 总结（基于 video-transcriber skill）",
    "2.1.0",
    "https://github.com/zwj-3193655211/astrbot_plugin_understand_video",
)
class UnderstandVideoPlugin(Star):
    """AstrBot 视频理解插件 - 包装 video-transcriber skill"""

    def __init__(self, context: Context, config):
        super().__init__(context)
        self.config = config

        # 设置全局 _PLUGIN_DATA_DIR（_check_ffmpeg 用）
        global _PLUGIN_DATA_DIR
        _PLUGIN_DATA_DIR = StarTools.get_data_dir("astrbot_plugin_understand_video")

        # ----- 路径规划 -----
        # 严格遵循 AstrBot 规范：模型/缓存/用户配置都放 data_dir（plugin_data_dir），
        # 插件目录只放代码。
        self.plugin_dir = _PLUGIN_DIR
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_understand_video")
        self.cache_dir = self.data_dir / "cache"      # skill 的 cache_dir 指向这里
        self.model_dir = self.data_dir / "model"      # skill 的 model_root 指向这里
        self.config_path = self.data_dir / "config.json"  # skill 的 config.json 写在这里

        # skill 模块的路径常量：让 skill 读写我们指定的 config.json
        _skill_config.CONFIG_FILE = self.config_path

        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "text").mkdir(parents=True, exist_ok=True)

        # ----- 同步 AstrBot 配置 → skill 配置 -----
        self._sync_to_skill_config()

        # 任务状态
        self._is_processing: bool = False
        self._stop_flag: bool = False

        # ----- 启动时环境自检 + 自动装缺失 Python 包 -----
        self._startup_warnings: List[str] = []
        self._auto_install_on_load()

        logger.info(
            f"视频理解插件 v2.1.0 已加载\n"
            f"  插件目录：{self.plugin_dir}\n"
            f"  数据目录：{self.data_dir}\n"
            f"  缓存目录：{self.cache_dir}\n"
            f"  模型目录：{self.model_dir}"
        )
        if self._startup_warnings:
            logger.warning("首次启动提示：\n" + "\n".join(f"  • {w}" for w in self._startup_warnings))

        # 启动时记录 ffmpeg 探测结果（用户调试用）
        ffmpeg_path = _check_ffmpeg()
        if ffmpeg_path:
            logger.info(f"ffmpeg OK: {ffmpeg_path}")
        else:
            # PATH 里没有，但插件 data 目录里可能也没有——给用户最具体的提示
            data_dir = StarTools.get_data_dir("astrbot_plugin_understand_video")
            logger.warning(
                f"ffmpeg 找不到。把 ffmpeg.exe 放到 {data_dir} 即可（无需重启）"
            )

    # ==================== 启动自检 ====================

    def _auto_install_on_load(self) -> None:
        """
        启动时自检：自动安装缺失的 Python 包（funasr / modelscope），
        ffmpeg 系统级包无法自动装，会在启动日志 + _check_env 里给友好提示。

        不会下载 ASR 模型（~1.1GB，太大，不应自动）—— 用户主动跑 /video_install
        或 /video_init 才下。
        """
        # 1. ffmpeg 检查（仅提示，无法自动装）
        if not _check_ffmpeg():
            sysname = platform.system()
            if sysname == "Windows":
                tip = "ffmpeg 未安装。Windows 装法：winget install ffmpeg，或从 https://www.gyan.dev/ffmpeg/builds/ 下载"
            elif sysname == "Darwin":
                tip = "ffmpeg 未安装。macOS 装法：brew install ffmpeg"
            else:
                tip = "ffmpeg 未安装。Linux 装法：sudo apt install ffmpeg（Debian/Ubuntu）或 sudo yum install ffmpeg（CentOS/RHEL）"
            self._startup_warnings.append(tip)

        # 2. 自动装本地 ASR 需要的 Python 包（funasr / modelscope）
        cfg = _skill_config.load_config()
        backend = cfg.get("asr_backend", "auto")
        api_key = (cfg.get("asr_api_key") or cfg.get("siliconflow_api_key") or "").strip()
        need_local_deps = backend in ("local", "auto") and not api_key

        if not need_local_deps:
            return

        missing_pkgs: List[Tuple[str, str]] = []
        try:
            import funasr  # noqa: F401
        except ImportError:
            missing_pkgs.append(("funasr", "本地 ASR（Paraformer）"))
        try:
            import modelscope  # noqa: F401
        except ImportError:
            missing_pkgs.append(("modelscope", "下载 ASR 模型"))

        if not missing_pkgs:
            return

        # 真的缺，自动装
        names = [name for name, _ in missing_pkgs]
        logger.info(f"检测到缺失依赖 {names}，自动 pip install...")
        try:
            import subprocess as _sp
            mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
            result = _sp.run(
                [sys.executable, "-m", "pip", "install", *names, "-i", mirror],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info(f"自动安装 {names} 成功")
            else:
                self._startup_warnings.append(
                    f"自动安装 {'/'.join(names)} 失败。请手动：pip install {' '.join(names)} -i https://pypi.tuna.tsinghua.edu.cn/simple"
                )
        except Exception as e:
            self._startup_warnings.append(
                f"自动安装 {'/'.join(names)} 异常：{e}。请手动：pip install {' '.join(names)} -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )

    # ==================== 配置桥接 ====================

    def _sync_to_skill_config(self) -> None:
        """
        把 AstrBot 的 plugin_config 同步到 skill 的 config.json。

        字段映射：
        - max_duration_minutes     → max_duration_minutes
        - asr_backend              → asr_backend
        - asr_provider             → asr_provider
        - asr_api_key              → asr_api_key
        - asr_base_url             → asr_base_url
        - asr_model                → asr_model
        - asr_verbose_json         → asr_verbose_json
        - use_punc                 → use_punc
        - device                   → device
        - cookie                   → cookie
        - model_root               → model_root
        - auto_clear_days          → （仅插件本地用，不进 skill）
        - summary_model / summary_prompt → （仅插件本地用，AI 总结走 AstrBot LLM）
        """
        # 加载 skill 现有的配置（用户可能手动编辑过）
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                cfg = {}
        except Exception as e:
            logger.warning(f"读取 skill 配置失败，使用空配置：{e}")
            cfg = {}

        # 用 AstrBot 的当前值覆盖（source of truth 是 AstrBot 配置面板）
        cfg["max_duration_minutes"] = int(self.config.get("max_duration_minutes", DEFAULT_MAX_DURATION_MINUTES))
        cfg["asr_backend"] = self.config.get("asr_backend", DEFAULT_ASR_BACKEND)
        cfg["asr_provider"] = self.config.get("asr_provider", DEFAULT_ASR_PROVIDER)
        cfg["asr_api_key"] = (self.config.get("asr_api_key") or "").strip()
        cfg["asr_base_url"] = (self.config.get("asr_base_url") or "").strip()
        cfg["asr_model"] = (self.config.get("asr_model") or DEFAULT_ASR_MODEL).strip()
        cfg["asr_verbose_json"] = bool(self.config.get("asr_verbose_json", False))
        cfg["use_punc"] = bool(self.config.get("use_punc", True))
        cfg["device"] = self.config.get("device", "auto")
        cfg["cookie"] = (self.config.get("cookie") or "").strip()
        # 模型目录默认放插件 data_dir，但允许用户指向其他位置
        user_model_root = (self.config.get("model_root") or "").strip()
        cfg["model_root"] = user_model_root or str(self.model_dir)
        # 缓存目录始终在 data_dir
        cfg["cache_dir"] = str(self.cache_dir)

        # 兼容旧字段（plugin_config 里有就带过去）
        if self.config.get("siliconflow_api_key"):
            cfg["siliconflow_api_key"] = self.config["siliconflow_api_key"].strip()
        if self.config.get("siliconflow_model"):
            cfg["siliconflow_model"] = self.config["siliconflow_model"].strip()

        # 写回
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"写入 skill 配置失败：{e}")

    # ==================== 环境检查 ====================

    def _check_env(self) -> Tuple[bool, List[str]]:
        """检查运行环境（Python / ffmpeg / 后端依赖）"""
        missing: List[str] = []

        # 1. Python 版本
        py = sys.version_info
        if py.major != 3 or py.minor >= 13 or py.minor < 10:
            missing.append(
                f"Python 版本不兼容（当前 {py.major}.{py.minor}，需要 3.10-3.12）"
            )

        # 2. ffmpeg（本地音视频 / 模型就绪性检查需要）
        if not _check_ffmpeg():
            sysname = platform.system()
            if sysname == "Windows":
                tip = "ffmpeg 未安装（必装）。Windows: `winget install ffmpeg` 或下载 https://www.gyan.dev/ffmpeg/builds/ 后解压到 PATH"
            elif sysname == "Darwin":
                tip = "ffmpeg 未安装（必装）。macOS: `brew install ffmpeg`"
            else:
                tip = "ffmpeg 未安装（必装）。Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) 或 `sudo yum install ffmpeg` (CentOS/RHEL)"
            missing.append(tip)

        # 3. 后端依赖
        cfg = _skill_config.load_config()
        backend = cfg.get("asr_backend", "auto")
        if backend == "local" or backend == "auto":
            try:
                import funasr  # noqa: F401
            except ImportError:
                missing.append("funasr（本地 ASR 必需，云端模式不需要）")
            try:
                import modelscope  # noqa: F401
            except ImportError:
                missing.append("modelscope（本地 ASR 必需，云端模式不需要）")

        return len(missing) == 0, missing

    # ==================== 自然语言链接触发（不需 /video 前缀）====================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def auto_detect_video(self, event: AstrMessageEvent):
        """
        自动识别消息中包含的视频链接（B站/抖音/YouTube/AcFun/西瓜/微博），
        命中后 stop_event() 阻止 LLM 处理，由插件自己接管。

        关键设计：钩子里**不**用 yield event.plain_result()。
        因为 AstrBot 钩子 await 只执行一次，async generator 后续 yield 会被丢弃。
        改为：钩子只检测 + stop_event + 启动 asyncio.create_task，
        task 里用 event.send() 直接发消息（不依赖 respond stage）。

        这样 /video 命令路径仍走 cmd_video 的 yield 模式（兼容 AstrBot 命令系统），
        自然语言路径用独立 task 走 event.send 模式（避免钩子 yield 丢失）。
        """
        try:
            msg = (event.message_str or "").strip()
        except Exception:
            return

        if not msg:
            return
        # 命令前缀的让 cmd_video 处理，不重复触发
        if msg.lower().lstrip().startswith(("/video", "/understand", "/transcribe", "/video_status", "/video_init", "/video_clear", "/video_stop", "/video_open", "/video_help")):
            return
        # 已经处理中
        if getattr(self, "_is_processing", False):
            return

        match = VIDEO_URL_PATTERN.search(msg)
        if not match:
            return

        url = match.group(0)
        logger.info(f"[video] auto-detected URL: {url}")

        # 阻止 LLM 处理（这条消息我们自己接管）
        try:
            event.stop_event()
        except Exception:
            pass

        # 启动后台 task 处理（不在钩子里 yield，避免 yield 丢失）
        asyncio.create_task(self._run_video_in_task(event, url))

    async def _run_video_in_task(self, event: AstrMessageEvent, url: str):
        """
        独立 task 跑 video 处理：用 event.send 直接发，不依赖 respond stage。
        复制 cmd_video 的核心逻辑（但去 yield 化）。
        """
        from datetime import datetime as _dt
        start = _dt.now()
        try:
            # 1. 环境检查
            ready, missing = self._check_env()
            if not ready:
                await event.send(MessageChain([Plain(
                    f"❌ 环境未就绪，缺少：\n"
                    + "\n".join(f"  • {m}" for m in missing)
                    + "\n\n请先运行 /video_install 初始化环境"
                )]))
                return

            # 2. 同步配置
            self._sync_to_skill_config()

            # 3. 类型预判
            info = _vt.identify_input(url)
            type_hint = {
                "bilibili": "B 站", "douyin": "抖音",
                "local_video": "本地视频", "local_audio": "本地音频",
                "generic_url": "通用平台", "unknown": "未知",
            }.get(info["type"], "未知")
            await event.send(MessageChain([Plain(
                f"🎬 开始处理（{type_hint}）\n"
                f"  输入：{url[:100]}{'...' if len(url) > 100 else ''}"
            )]))

            # 4. 跑转写（不阻塞事件循环）
            self._is_processing = True
            self._stop_flag = False
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, _vt.run, url)
            finally:
                self._is_processing = False
                self._stop_flag = False

            # 5. 处理结果
            if not isinstance(result, dict):
                await event.send(MessageChain([Plain(f"❌ 异常返回：{result}")]))
                return

            status = result.get("status")
            if status != "success":
                await event.send(MessageChain([Plain(_format_error(result))]))
                return

            title = result.get("title", "未知标题")
            transcription = result.get("transcription", "")
            duration = result.get("duration", 0)
            elapsed = (_dt.now() - start).seconds
            txt_path = result.get("transcription_path", "")

            await event.send(MessageChain([Plain(
                f"✅ 转写完成\n"
                f"  📌 标题：{title}\n"
                f"  📁 来源：{type_hint}\n"
                f"  ⏱️ 时长：{_format_duration(duration)}\n"
                f"  📝 字符数：{len(transcription)}\n"
                f"  🕒 耗时：{elapsed // 60}分{elapsed % 60}秒\n"
                f"  💾 已保存：{txt_path}"
            )]))

            # 6. AI 总结
            if not transcription:
                return

            await event.send(MessageChain([Plain("🤖 正在生成 AI 总结...")]))
            summary = await self._summarize_with_ai(transcription, title)
            if not summary:
                await event.send(MessageChain([Plain(
                    "⚠️ AI 总结失败（请检查 AstrBot 是否配置了 LLM provider）\n"
                    f"转写文本已保存到：{txt_path}"
                )]))
                return

            await event.send(MessageChain([Plain(
                f"📺 视频理解结果\n"
                f"📌 标题：{title}\n\n"
                f"🤖 AI 总结：\n{summary}\n\n"
                f"⏱️ 总耗时：{elapsed // 60}分{elapsed % 60}秒"
            )]))

        except Exception as e:
            logger.exception(f"[video] auto-detect processing failed: {e}")
            try:
                await event.send(MessageChain([Plain(f"❌ 处理异常：{e}")]))
            except Exception:
                pass

    # ==================== 命令处理 ====================

    @filter.command("video")
    async def cmd_video(self, event: AstrMessageEvent):
        """处理视频：B 站 / 抖音 / 通用平台 / 本地音视频"""
        # 1. 解析输入
        message = event.message_str.strip()
        url_or_path = _resolve_input(message)
        if not url_or_path or url_or_path.lower() in ("help", "-h", "--help", "帮助", "?"):
            yield event.plain_result(_HELP_TEXT)
            return

        # 2. 防止并发
        if self._is_processing:
            yield event.plain_result(
                "⏳ 当前有任务正在处理中，请等待完成或使用 /video_stop 取消"
            )
            return

        # 3. 环境检查
        ready, missing = self._check_env()
        if not ready:
            yield event.plain_result(
                f"❌ 环境未就绪，缺少：\n"
                + "\n".join(f"  • {m}" for m in missing)
                + "\n\n请先运行 /video_init 初始化环境（或安装 ffmpeg）"
            )
            return

        # 4. 同步最新配置
        self._sync_to_skill_config()

        # 5. 类型预判（给用户更准确的提示）
        info = _vt.identify_input(url_or_path)
        type_hint = {
            "bilibili": "B 站",
            "douyin": "抖音",
            "local_video": "本地视频",
            "local_audio": "本地音频",
            "generic_url": "通用平台",
            "unknown": "未知",
        }.get(info["type"], "未知")
        yield event.plain_result(
            f"🎬 开始处理（{type_hint}）\n"
            f"  输入：{url_or_path[:100]}{'...' if len(url_or_path) > 100 else ''}"
        )

        # 6. 执行（放到线程池，不阻塞 AstrBot 事件循环）
        self._is_processing = True
        self._stop_flag = False
        start = datetime.now()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _vt.run, url_or_path)
        except Exception as e:
            logger.exception("skill.run 异常")
            yield event.plain_result(f"❌ 处理异常：{e}")
            return
        finally:
            self._is_processing = False
            self._stop_flag = False

        # 7. 处理结果
        if not isinstance(result, dict):
            yield event.plain_result(f"❌ 异常返回：{result}")
            return

        status = result.get("status")
        if status != "success":
            yield event.plain_result(_format_error(result))
            return

        # 转写成功
        title = result.get("title", "未知标题")
        transcription = result.get("transcription", "")
        src_type = result.get("type", "")
        duration = result.get("duration", 0)
        elapsed = (datetime.now() - start).seconds
        txt_path = result.get("transcription_path", "")

        yield event.plain_result(
            f"✅ 转写完成\n"
            f"  📌 标题：{title}\n"
            f"  📁 来源：{type_hint}\n"
            f"  ⏱️ 时长：{_format_duration(duration)}\n"
            f"  📝 字符数：{len(transcription)}\n"
            f"  🕒 耗时：{elapsed // 60}分{elapsed % 60}秒\n"
            f"  💾 已保存：{txt_path}"
        )

        # 8. AI 总结
        if not transcription:
            return

        yield event.plain_result("🤖 正在生成 AI 总结...")
        summary = await self._summarize_with_ai(transcription, title)
        if not summary:
            yield event.plain_result(
                "⚠️ AI 总结失败（请检查 AstrBot 是否配置了 LLM provider）\n"
                f"转写文本已保存到：{txt_path}"
            )
            return

        yield event.plain_result(
            f"📺 视频理解结果\n"
            f"📌 标题：{title}\n\n"
            f"🤖 AI 总结：\n{summary}\n\n"
            f"⏱️ 总耗时：{elapsed // 60}分{elapsed % 60}秒"
        )

    @filter.command("video_init")
    async def cmd_init(self, event: AstrMessageEvent):
        """初始化环境：检查并按需安装依赖、下载模型"""
        yield event.plain_result("🔧 开始初始化环境...")

        # 1. Python 版本检查
        py = sys.version_info
        if py.major != 3 or py.minor >= 13 or py.minor < 10:
            yield event.plain_result(
                f"❌ Python 版本不兼容（当前 {py.major}.{py.minor}，需要 3.10-3.12）\n\n"
                "解决方案：\n"
                "1. 安装 Python 3.12（推荐）或 3.11\n"
                "2. 重建 AstrBot 虚拟环境：\n"
                "   cd AstrBot目录\n"
                "   rmdir /s /q venv\n"
                "   C:\\Python\\Python312\\python.exe -m venv venv\n"
                "   venv\\Scripts\\activate\n"
                "   pip install AstrBot -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                "3. 重启 AstrBot 后再次运行 /video_init"
            )
            return

        # 2. ffmpeg 检查
        if not _check_ffmpeg():
            yield event.plain_result(
                "⚠️ 未检测到 ffmpeg（本地音视频功能必需）\n"
                "  Windows: winget install ffmpeg\n"
                "  Linux:   sudo apt install ffmpeg\n"
                "  macOS:   brew install ffmpeg\n"
                "安装后重启 AstrBot 即可。\n\n"
                "（云端 ASR + 纯链接场景其实可以跳过 ffmpeg，先继续...）"
            )
        else:
            yield event.plain_result(f"✅ ffmpeg 已就绪：{_check_ffmpeg()}")

        # 3. 同步配置
        self._sync_to_skill_config()
        cfg = _skill_config.load_config()
        backend = cfg.get("asr_backend", "auto")
        api_key = cfg.get("asr_api_key") or cfg.get("siliconflow_api_key") or ""

        # 4. 按后端决定要不要装本地依赖
        need_local_deps = backend in ("local", "auto") and not api_key
        if need_local_deps:
            yield event.plain_result(
                "📦 当前后端需要本地 FunASR（auto 模式但未配置 API key）\n"
                "正在安装 Python 依赖（约 1-3 分钟）..."
            )
            ok, msg = await self._install_local_deps()
            if not ok:
                yield event.plain_result(
                    f"❌ 依赖安装失败：{msg}\n\n"
                    "手动安装：\n"
                    "  1. 激活 AstrBot 虚拟环境\n"
                    "  2. pip install funasr modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
                )
                return
            yield event.plain_result("✅ Python 依赖已就绪")

        # 5. 决定要不要下模型
        ready, missing = _skill_recognizer.check_models(cfg)
        if ready:
            yield event.plain_result("✅ ASR 模型已就绪")
        else:
            if backend == "cloud":
                yield event.plain_result(
                    "ℹ️ 当前使用云端 ASR，不需要下载本地模型\n"
                    f"  缺失本地模型（如需切换本地模式可下载）：{len(missing)} 个"
                )
            else:
                yield event.plain_result(
                    f"📥 正在下载 ASR 模型（约 1.1GB，请耐心等待）..."
                )
                init_result = await asyncio.get_event_loop().run_in_executor(
                    None, _vt.initialize
                )
                if init_result.get("status") == "success":
                    yield event.plain_result("✅ ASR 模型下载完成")
                else:
                    yield event.plain_result(
                        f"❌ 模型下载失败，请检查网络后重试\n"
                        f"  详情：{init_result}"
                    )
                    return

        # 6. 最终状态
        ready, missing = self._check_env()
        if ready:
            yield event.plain_result(
                "🎉 环境初始化完成！\n\n"
                "可用命令：\n"
                "  /video <链接/路径>  - 处理 B 站 / 抖音 / 本地音视频\n"
                "  /video_status       - 查看环境状态\n"
                "  /video_clear        - 清空缓存\n"
                "  /video_open         - 打开缓存目录\n"
                "  /video_help         - 完整帮助"
            )
        else:
            yield event.plain_result(
                f"⚠️ 仍有缺失项：\n" + "\n".join(f"  • {m}" for m in missing)
            )

    @filter.command("video_install")
    async def cmd_install(self, event: AstrMessageEvent):
        """一键装所有本地依赖：ffmpeg 检测 + pip 装包 + 下载 1.1GB ASR 模型"""
        yield event.plain_result("🚀 一键安装 - 检查并安装本地依赖\n")

        # 1. ffmpeg 检测
        if not _check_ffmpeg():
            sysname = platform.system()
            if sysname == "Windows":
                install_tip = "Windows: `winget install ffmpeg` 或下载 https://www.gyan.dev/ffmpeg/builds/"
            elif sysname == "Darwin":
                install_tip = "macOS: `brew install ffmpeg`"
            else:
                install_tip = "Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) 或 `sudo yum install ffmpeg` (CentOS)"
            yield event.plain_result(
                f"❌ ffmpeg 未安装（必装；本地音视频需要）\n"
                f"  {install_tip}\n\n"
                f"装好后重启 AstrBot 即可。"
            )
        else:
            yield event.plain_result(f"✅ ffmpeg 已就绪：{_check_ffmpeg()}")

        # 2. 同步配置
        self._sync_to_skill_config()
        cfg = _skill_config.load_config()
        backend = cfg.get("asr_backend", "auto")
        api_key = cfg.get("asr_api_key") or cfg.get("siliconflow_api_key") or ""

        # 3. 装 Python 包
        need_local_deps = backend in ("local", "auto") and not api_key
        if need_local_deps:
            yield event.plain_result("📦 检查 Python 依赖（funasr / modelscope）...")
            ok, msg = await self._install_local_deps()
            if not ok:
                yield event.plain_result(
                    f"❌ 依赖安装失败：{msg}\n\n"
                    f"手动装：\n"
                    f"  {sys.executable} -m pip install funasr modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
                )
                return
            yield event.plain_result("✅ Python 依赖已就绪")
        else:
            yield event.plain_result("ℹ️ 当前为云端模式，跳过本地 Python 依赖")

        # 4. 下模型
        ready, missing = _skill_recognizer.check_models(cfg)
        if ready:
            yield event.plain_result("✅ ASR 模型已就绪")
        else:
            if backend == "cloud":
                yield event.plain_result(
                    f"ℹ️ 当前使用云端 ASR，无需下载本地模型\n"
                    f"  （{len(missing)} 个本地模型缺失，要切本地模式再下）"
                )
            else:
                yield event.plain_result(
                    f"📥 正在下载 ASR 模型（约 1.1GB，请耐心等待 2-10 分钟）..."
                )
                init_result = await asyncio.get_event_loop().run_in_executor(
                    None, _vt.initialize
                )
                if init_result.get("status") == "success":
                    yield event.plain_result("✅ ASR 模型下载完成")
                else:
                    yield event.plain_result(
                        f"❌ 模型下载失败，请检查网络后重试\n"
                        f"  详情：{init_result}"
                    )
                    return

        # 5. 最终状态
        yield event.plain_result("\n📊 最终环境状态：")
        ready, missing = self._check_env()
        if ready:
            yield event.plain_result(
                "🎉 全部就绪！\n\n"
                "现在可以：\n"
                "  • 发 B 站/抖音链接（含 URL 的消息自动识别处理）\n"
                "  • 或用 /video <链接> 命令\n"
                "  • /video_status 看详细状态"
            )
        else:
            yield event.plain_result(
                f"⚠️ 仍有缺失项：\n" + "\n".join(f"  • {m}" for m in missing)
            )

    @filter.command("video_stop")
    async def cmd_stop(self, event: AstrMessageEvent):
        """停止当前任务（仅作语义占位 - 实际取消由 skill.run 阻塞决定）"""
        if not self._is_processing:
            yield event.plain_result("当前没有正在进行的任务")
            return
        self._stop_flag = True
        yield event.plain_result(
            "⚠️ 任务正在进行中，由于底层 ASR 是阻塞调用，无法立即中断。\n"
            "建议：等待当前视频处理完成；下一个任务会立即响应。"
        )

    @filter.command("video_clear")
    async def cmd_clear(self, event: AstrMessageEvent):
        """清除缓存（音频 + 转写文本）"""
        self._sync_to_skill_config()
        try:
            result = _vt.clear_cache()
        except Exception as e:
            yield event.plain_result(f"❌ 清理失败：{e}")
            return

        if result.get("status") == "success":
            yield event.plain_result(
                f"✅ 缓存已清理\n"
                f"  删除文件：{result.get('cleared_files', 0)} 个\n"
                f"  释放空间：{result.get('freed_mb', 0):.1f} MB"
            )
        else:
            yield event.plain_result(f"❌ 清理失败：{result}")

    @filter.command("video_status")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看状态：环境 / 后端 / 模型 / 缓存"""
        self._sync_to_skill_config()

        # 环境
        env_ok, env_missing = self._check_env()

        # 后端 / 模型
        try:
            status = _vt.check_status()
        except Exception as e:
            yield event.plain_result(f"❌ 状态检查失败：{e}")
            return

        asr_backend = status.get("asr_backend", "none")
        asr_provider = status.get("asr_provider", "n/a")
        models_ready = status.get("models_ready", False)
        missing_models = status.get("missing_models", [])
        api_key_set = status.get("api_key_set", False)
        ffmpeg = status.get("ffmpeg", "missing")
        model_root = status.get("model_root", "")

        # 缓存
        cache_info = status.get("cache", {}) or {}
        audio_info = cache_info.get("audio", {}) or {}
        text_info = cache_info.get("text", {}) or {}
        total_mb = cache_info.get("total_mb", 0)

        # 后端文字描述
        if asr_backend == "local":
            backend_desc = "🖥️ 本地 FunASR（Paraformer-Large）"
        elif asr_backend == "cloud":
            backend_desc = f"☁️ 云端 ASR（{asr_provider}）"
        else:
            backend_desc = "❌ 未配置（无本地模型 + 无 API key）"

        # 模型描述
        if models_ready:
            model_desc = f"✅ 已就绪（{model_root}）"
        else:
            model_desc = f"❌ 缺失 {len(missing_models)} 个（{model_root}）"

        # ffmpeg
        ffmpeg_desc = "✅ " + ffmpeg if ffmpeg and ffmpeg != "missing" else "❌ 未安装"

        # API key
        if api_key_set:
            api_desc = "✅ 已配置"
        else:
            api_desc = "❌ 未配置（云端模式必需）"

        # 处理状态
        proc = "🔄 正在处理" if self._is_processing else "💤 空闲"

        # 当前 ASR 后端配置提示
        cfg = _skill_config.load_config()
        cfg_backend = cfg.get("asr_backend", "auto")

        msg = (
            "📊 视频理解插件状态\n"
            "\n"
            f"🔧 环境：{'✅ 已就绪' if env_ok else '❌ 未就绪'}\n"
            f"🔄 当前状态：{proc}\n"
            f"⏱️ 最大时长：{cfg.get('max_duration_minutes', 70)} 分钟\n"
            f"\n"
            f"🎙️ ASR 后端：{backend_desc}\n"
            f"   配置：asr_backend={cfg_backend} / provider={cfg.get('asr_provider', 'siliconflow')}\n"
            f"   API Key：{api_desc}\n"
            f"   模型：{model_desc}\n"
            f"   ffmpeg：{ffmpeg_desc}\n"
            f"\n"
            f"💾 缓存：{total_mb:.1f} MB\n"
            f"   音频：{audio_info.get('files', 0)} 个 / {audio_info.get('size_mb', 0):.1f} MB\n"
            f"   文本：{text_info.get('files', 0)} 个 / {text_info.get('size_mb', 0):.1f} MB\n"
            f"   目录：{self.cache_dir}\n"
            f"\n"
            f"💡 命令列表：\n"
            f"  /video <链接/路径>  - 处理视频\n"
            f"  /video_init         - 初始化环境\n"
            f"  /video_stop         - 取消任务（语义）\n"
            f"  /video_clear        - 清除缓存\n"
            f"  /video_status       - 查看状态\n"
            f"  /video_open         - 打开缓存目录\n"
            f"  /video_help         - 完整帮助"
        )
        yield event.plain_result(msg)

    @filter.command("video_open")
    async def cmd_open_cache(self, event: AstrMessageEvent):
        """打开缓存目录（优先打开转写文本目录）"""
        text_dir = self.cache_dir / "text"
        target = text_dir if text_dir.exists() and any(text_dir.iterdir()) else self.cache_dir
        target.mkdir(parents=True, exist_ok=True)

        try:
            sysname = platform.system()
            if sysname == "Windows":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sysname == "Darwin":
                subprocess.run(["open", str(target)], check=False)
            else:
                subprocess.run(["xdg-open", str(target)], check=False)
            yield event.plain_result(
                f"📂 已打开缓存目录：\n{target}\n\n"
                f"转写文本位于 {text_dir} 子目录，可用记事本打开查看。"
            )
        except Exception as e:
            yield event.plain_result(f"⚠️ 无法自动打开，请手动访问：\n{target}\n\n错误：{e}")

    @filter.command("video_help")
    async def cmd_help(self, event: AstrMessageEvent):
        """显示完整帮助"""
        yield event.plain_result(_HELP_TEXT)

    # ==================== AI 总结 ====================

    async def _summarize_with_ai(self, text: str, title: str) -> Optional[str]:
        """调用 AstrBot 的 LLM provider 做总结"""
        try:
            provider = self.context.get_using_provider()
            if provider is None:
                logger.warning("未配置 LLM provider")
                return None

            prompt_template = self.config.get("summary_prompt", SUMMARY_PROMPT_DEFAULT)
            prompt = prompt_template.format(content=text, title=title)

            # 防止 prompt 过长撑爆上下文
            max_len = 8000
            if len(prompt) > max_len:
                prompt = prompt[:max_len] + "\n...(内容过长已截断)"

            custom_model = (self.config.get("summary_model") or "").strip()
            if custom_model:
                logger.info(f"使用自定义总结模型: {custom_model}")
                resp = await provider.text_chat(
                    prompt=prompt, session_id="video_summary", model=custom_model,
                )
            else:
                resp = await provider.text_chat(
                    prompt=prompt, session_id="video_summary",
                )

            if resp and resp.completion_text:
                return resp.completion_text.strip()
            logger.warning("LLM 返回为空")
            return None
        except Exception as e:
            logger.error(f"AI 总结失败：{e}")
            return None

    # ==================== 依赖安装 ====================

    async def _install_local_deps(self) -> Tuple[bool, str]:
        """安装本地 ASR 所需的 Python 依赖（funasr / modelscope）"""
        try:
            mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
            steps = [
                ([sys.executable, "-m", "pip", "install",
                  "funasr", "modelscope", "-i", mirror],
                 "FunASR + ModelScope"),
            ]
            for cmd, name in steps:
                logger.info(f"安装 {name}...")
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
                except asyncio.TimeoutError:
                    proc.kill()
                    return False, f"{name} 安装超时"
                if proc.returncode != 0:
                    err = stderr.decode("utf-8", errors="ignore")
                    return False, f"{name} 安装失败：{err[-300:]}"
            return True, "ok"
        except Exception as e:
            return False, str(e)

    # ==================== 生命周期 ====================

    async def terminate(self):
        self._stop_flag = True
        logger.info("视频理解插件已卸载")


# ==================== 帮助文本 ====================

_HELP_TEXT = """📺 视频理解助手 v2.0（基于 video-transcriber skill）

支持：B 站（BV/bilibili.com/b23.tv）· 抖音（v.douyin.com/douyin.com）· 通用平台（YouTube/AcFun/微博，需 yt-dlp）· 本地音视频（mp4/mkv/mp3/wav/m4a/...）

两种 ASR 后端：
  • 本地 FunASR Paraformer-Large：GPU 最佳，首次下 ~1.1GB 模型
  • 云端（SiliconFlow / 火山方舟豆包 / 任意 OpenAI 兼容端点）：按用量计费，零模型

命令：
  /video <链接/路径>  处理视频
  /video_init         初始化环境（装依赖 + 下模型 + 写配置）
  /video_status       查看环境/后端/模型/缓存状态
  /video_clear        清空缓存
  /video_stop         取消任务（语义，当前阻塞任务需等待完成）
  /video_open         打开缓存目录
  /video_help         本帮助

示例：
  /video BV1xx411c7mD
  /video https://www.bilibili.com/video/BV1xx411c7mD
  /video https://b23.tv/abc123
  /video https://v.douyin.com/xxxxx/
  /video C:\\Videos\\lecture.mp4
  /video D:\\audios\\interview.wav

配置面板（在 AstrBot WebUI）：
  • asr_backend      auto / local / cloud
  • asr_provider     siliconflow / openai（火山方舟等 OpenAI 兼容端点选 openai）
  • asr_api_key      云端 API key
  • asr_base_url     openai 模式必填（如 https://ark.cn-beijing.volces.com/api/v3）
  • asr_model        供应商支持的转写模型
  • cookie           B 站/抖音 cookie（可选，提高成功率）
  • model_root       已下完的模型目录（跳过 1.1GB 下载）
  • max_duration_minutes  单视频最大时长
  • summary_model / summary_prompt  AI 总结模型与提示词

故障排查：/video_status 看后端 / 模型 / ffmpeg 状态。
"""


# ==================== 错误格式化 ====================

def _format_error(result: Dict[str, Any]) -> str:
    """把 skill 的 error 字典转成可读消息"""
    err = result.get("error", "unknown")
    msg = result.get("message", "")
    suggestion = result.get("suggestion", "")

    if err == "invalid_input":
        return (
            f"❌ 无法识别输入\n"
            f"  {msg}\n\n"
            f"支持：B 站链接（BV / bilibili.com / b23.tv）、"
            f"抖音（v.douyin.com / douyin.com）、通用平台（yt-dlp）、"
            f"本地音视频路径（mp4/mkv/mp3/wav/m4a...）"
        )
    if err == "fetch_failed":
        return (
            f"❌ 下载失败\n  {msg}\n\n"
            f"建议：检查链接是否有效 / 装 yt-dlp 自动兜底 / "
            f"填 cookie 走登录态"
        )
    if err == "duration_exceeded":
        return f"⏱️ {msg}\n\n可在 AstrBot 配置面板调大 max_duration_minutes"
    if err == "asr_unavailable":
        return (
            f"❌ ASR 不可用\n  {msg}\n\n"
            f"两条路：\n"
            f"  A. 本地模式：/video_init 自动下载 ~1.1GB 模型\n"
            f"  B. 云端模式：在配置面板填 asr_api_key（SiliconFlow / 火山方舟 / OpenAI）"
        )
    if err == "model_missing":
        miss = result.get("missing", [])
        return (
            f"❌ ASR 模型未就绪\n"
            f"  缺失：{len(miss)} 个\n\n"
            f"请运行 /video_init 下载模型（或切换到云端模式）"
        )
    if err == "model_download_failed":
        return f"❌ 模型下载失败\n  {msg}\n  详情：{result.get('detail', '')}"
    if err == "convert_failed":
        return f"❌ {msg}\n  请确认 ffmpeg 已安装"
    if err == "extract_failed":
        return f"❌ {msg}"
    if err == "execution_failed":
        return f"❌ 执行失败：{msg}"

    return f"❌ 处理失败（{err}）：{msg}" + (f"\n建议：{suggestion}" if suggestion else "")
