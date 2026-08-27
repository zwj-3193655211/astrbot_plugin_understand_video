"""
video-transcriber 配置管理

默认配置 + 用户配置覆盖。模型目录默认在 skill/model/，但用户可改 model_root
复用其他位置（例如 AI-VedioToText 项目的 model/）已下好的模型。
"""
import json
from pathlib import Path
from typing import Any, Dict

SKILL_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL_DIR = SKILL_DIR / "model"
DEFAULT_CACHE_DIR = SKILL_DIR / "cache"
CONFIG_FILE = SKILL_DIR / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "max_duration_minutes": 70,
    "use_punc": True,
    "device": "auto",            # auto / cuda / cpu
    "auto_download": True,        # 本地模型缺失时自动下载（首次运行）
    "asr_backend": "auto",        # auto / local / cloud（云端可插拔供应商）
    "asr_provider": "siliconflow",  # siliconflow / openai（任意 OpenAI 兼容端点，如火山方舟）
    "asr_base_url": "",           # openai 模式必填，例：https://ark.cn-beijing.volces.com/api/v3
    "asr_api_key": "",            # 云端 API key（也可用环境变量 ASR_API_KEY；兼容旧字段 siliconflow_api_key）
    "asr_model": "FunAudioLLM/SenseVoiceSmall",  # 任选供应商支持的转写模型
    "asr_verbose_json": False,     # true: 请求 verbose_json 拿句级时间戳（OpenAI 兼容供应商）
    "siliconflow_api_key": "",    # 旧字段（仍兼容，推荐改用 asr_api_key）
    "siliconflow_model": "",      # 旧字段（仍兼容，推荐改用 asr_model）
    "model_root": str(DEFAULT_MODEL_DIR),
    "cache_dir": str(DEFAULT_CACHE_DIR),
    "cookie": "",                # 可选：自填 B 站 cookie 提高成功率
}


def load_config() -> Dict[str, Any]:
    """加载配置（用户 config.json 覆盖默认值），并确保缓存目录存在"""
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            cfg.update(user_cfg)
        except Exception as e:
            print(f"[config] 加载用户配置失败，使用默认：{e}")
    # 确保缓存目录
    cache_root = Path(cfg["cache_dir"])
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "audio").mkdir(parents=True, exist_ok=True)
    (cache_root / "text").mkdir(parents=True, exist_ok=True)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """保存当前配置到 config.json"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
