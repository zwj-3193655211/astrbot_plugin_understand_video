"""
ASR 后端抽象：local (FunASR) / cloud (可插拔供应商) / auto

- local  : FunASR Paraformer-Large + FSMN-VAD + CT-PUNC（本地模型 ~1.1GB，GPU 最佳）
- cloud  : 云端转写 API，供应商可插拔（当前内置两个）：
    1. siliconflow : https://api.siliconflow.cn/v1/audio/transcriptions
       模型任选官方支持的转写模型（OpenAPI enum 2025-08：
       FunAudioLLM/SenseVoiceSmall、TeleAI/TeleSpeechASR，官方会不定期上下架）
    2. openai      : 任意 OpenAI 兼容的 /audio/transcriptions 端点
       例：火山方舟 Ark → asr_base_url=https://ark.cn-beijing.volces.com/api/v3
                      + asr_api_key=<火山方舟 API Key> + asr_model=<豆包语音模型>
       例：OpenAI     → asr_base_url=https://api.openai.com/v1
       例：智谱       → asr_base_url=https://open.bigmodel.cn/api/paas/v4
       asr_verbose_json=true 时请求 response_format=verbose_json，
       支持句级时间戳的供应商（如火山方舟 seed-asr）会返回 segments。
- auto   : 本地模型就绪 → local；否则有 key → cloud；都没有 → 报错给两条方案

云端通用限制（以各家为准）：常见 音频≤1h、文件≤50MB。

config.json 相关字段：
  "asr_backend": "auto" | "local" | "cloud"
  "asr_provider": "siliconflow" | "openai"
  "asr_base_url": ""            # openai 模式必填（如火山方舟 https://ark.cn-beijing.volces.com/api/v3）
  "asr_api_key": ""             # 也可用环境变量 ASR_API_KEY；兼容旧字段 siliconflow_api_key / SILICONFLOW_API_KEY
  "asr_model": "FunAudioLLM/SenseVoiceSmall"
  "asr_verbose_json": false     # true: 请求 verbose_json 拿句级时间戳（OpenAI 兼容供应商）

设计原则：本模块除标准库外不导入任何第三方包；
funasr / recognizer 仅在 local 后端真正使用时才懒加载，
保证"云端路径"可以在只有 python3 的机器上直接跑。
"""
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from netutil import make_ssl_context

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
SILICONFLOW_DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"


class AsrConfigError(RuntimeError):
    """ASR 配置不可用（无模型也无 key / 显式指定但缺 key）"""


# ==================== 配置解析 ====================

def api_key(cfg: Dict[str, Any]) -> str:
    """key 来源优先级：asr_api_key → siliconflow_api_key(旧) → 环境变量 ASR_API_KEY / SILICONFLOW_API_KEY"""
    for k in ("asr_api_key", "siliconflow_api_key"):
        v = (cfg.get(k) or "").strip()
        if v:
            return v
    for env in ("ASR_API_KEY", "SILICONFLOW_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    return ""


def _provider(cfg: Dict[str, Any]) -> str:
    """解析供应商：asr_provider，兼容旧值 asr_backend=siliconflow"""
    provider = (cfg.get("asr_provider") or "").strip().lower()
    if not provider:
        backend_legacy = (cfg.get("asr_backend") or "auto").lower()
        if backend_legacy in ("siliconflow", "cloud", "sensevoice"):
            provider = "siliconflow" if backend_legacy == "siliconflow" else "siliconflow"
        else:
            provider = "siliconflow"
    return provider


def resolve(cfg: Dict[str, Any]) -> str:
    """返回实际生效的后端：'local' 或 'cloud'"""
    backend = (cfg.get("asr_backend") or "auto").lower()
    if backend in ("local", "funasr"):
        return "local"
    if backend in ("cloud", "siliconflow", "sensevoice", "openai"):
        if not api_key(cfg):
            raise AsrConfigError(
                f"asr_backend={backend} 但未配置 API key。\n"
                "  在 config.json 填 asr_api_key，或设置环境变量 ASR_API_KEY"
            )
        return "cloud"
    # auto：模型优先，其次云端
    from recognizer import check_models  # 只查文件，不导入 funasr
    ready, _ = check_models(cfg)
    if ready:
        return "local"
    if api_key(cfg):
        return "cloud"
    raise AsrConfigError(
        "未找到可用 ASR：本地模型未就绪，且未设置 API key。\n"
        "  方案 A（本地，~1.1GB）：python setup.py\n"
        "  方案 B（云端，0 下载）：设置环境变量 ASR_API_KEY=<你的 key>"
    )


def provider_label(cfg: Dict[str, Any]) -> str:
    """给用户看的供应商描述，如 'siliconflow' / 'openai@ark.cn-beijing.volces.com'"""
    p = _provider(cfg)
    if p == "openai":
        base = (cfg.get("asr_base_url") or "").strip().rstrip("/")
        return f"openai@{base}" if base else "openai(custom base_url)"
    return p


def transcribe(audio_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一转写入口。
    返回 {"text", "segments": [{start,end,text}], "duration", ...}
    """
    backend = resolve(cfg)
    if backend == "cloud":
        return transcribe_cloud(audio_path, cfg)
    from recognizer import transcribe as _local  # 懒加载 funasr
    return _local(audio_path, cfg)


# ==================== 云端后端（纯标准库，可插拔供应商） ====================

def transcribe_cloud(audio_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """按 asr_provider 分发到对应云端转写 API"""
    provider = _provider(cfg)
    if provider == "openai":
        return transcribe_openai_compatible(audio_path, cfg)
    return transcribe_siliconflow(audio_path, cfg)


def _check_file_limits(audio_path: str) -> None:
    size_mb = Path(audio_path).stat().st_size / 1024 / 1024
    if size_mb > 50:
        raise RuntimeError(
            f"云端转写普遍限制音频 ≤ 50MB（当前 {size_mb:.1f}MB），"
            "建议改用本地 ASR 或先裁剪音频"
        )


def _parse_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 verbose_json 响应解析句级时间戳（OpenAI 兼容格式：segments[].start/end/text）"""
    segments: List[Dict[str, Any]] = []
    for seg in data.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return segments


def transcribe_siliconflow(audio_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """SiliconFlow /audio/transcriptions。官方 OpenAPI 仅 file+model，无 verbose_json"""
    key = api_key(cfg)
    if not key:
        raise AsrConfigError("SiliconFlow API key 未设置")
    model = cfg.get("asr_model") or cfg.get("siliconflow_model") or SILICONFLOW_DEFAULT_MODEL
    _check_file_limits(audio_path)

    body, content_type = _multipart_body(
        fields={"model": model},
        file_field="file",
        file_path=audio_path,
        file_content_type="audio/mpeg",
    )
    data = _post_json(SILICONFLOW_URL, body, content_type, key, "SiliconFlow")

    full_text = (data.get("text") or "").strip()
    if not full_text:
        raise RuntimeError("SiliconFlow 未返回文本，请检查音频是否有效")
    # 该接口无句级时间戳，segments 留空（duration 由调用方探测）
    return {
        "text": full_text,
        "segments": [],
        "duration": 0.0,
        "asr": "cloud",
        "provider": "siliconflow",
        "model": model,
    }


def transcribe_openai_compatible(audio_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    任意 OpenAI 兼容的 /audio/transcriptions 端点（火山方舟 Ark / OpenAI / 智谱 / Groq 等）。
    asr_verbose_json=true 时请求 verbose_json，能返回 segments 的供应商给句级时间戳。
    """
    key = api_key(cfg)
    if not key:
        raise AsrConfigError("API key 未设置（asr_api_key 或环境变量 ASR_API_KEY）")
    base_url = (cfg.get("asr_base_url") or "").strip().rstrip("/")
    if not base_url:
        raise AsrConfigError("asr_provider=openai 但未设置 asr_base_url（如 https://ark.cn-beijing.volces.com/api/v3）")
    model = cfg.get("asr_model") or ""
    if not model:
        raise AsrConfigError("asr_provider=openai 但未设置 asr_model（如 doubao-seed-asr-1）")
    verbose = bool(cfg.get("asr_verbose_json", False))
    _check_file_limits(audio_path)

    fields: Dict[str, str] = {"model": model}
    if verbose:
        fields["response_format"] = "verbose_json"
    body, content_type = _multipart_body(
        fields=fields,
        file_field="file",
        file_path=audio_path,
        file_content_type="audio/mpeg",
    )
    url = f"{base_url}/audio/transcriptions"
    data = _post_json(url, body, content_type, key, base_url)

    full_text = (data.get("text") or "").strip()
    if not full_text and verbose:
        # 部分实现不返回顶层 text，从 segments 拼接
        segments = _parse_segments(data)
        full_text = "".join(s["text"] for s in segments).strip()
    if not full_text:
        raise RuntimeError(f"转写响应未包含文本: {json.dumps(data, ensure_ascii=False)[:300]}")

    segments = _parse_segments(data) if verbose else []
    return {
        "text": full_text,
        "segments": segments,
        "duration": segments[-1]["end"] if segments else 0.0,
        "asr": "cloud",
        "provider": "openai",
        "model": model,
        "verbose_json": verbose,
    }


def _post_json(url: str, body: bytes, content_type: str, key: str, label: str) -> Dict[str, Any]:
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=900, context=make_ssl_context()) as resp:
            response_text = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} 转写失败 HTTP {e.code}: {detail[:300]}") from e
    except URLError as e:
        raise RuntimeError(f"{label} 网络错误: {e}") from e
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{label} 返回非 JSON: {response_text[:300]}") from e


def _multipart_body(
    *,
    fields: Dict[str, str],
    file_field: str,
    file_path: str,
    file_content_type: str,
):
    """手写 multipart/form-data（纯标准库，不依赖 requests）"""
    boundary = f"----video-transcriber-{uuid.uuid4().hex}"
    parts: list = []
    for name, value in fields.items():
        parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    parts.extend([
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{Path(file_path).name}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"),
        Path(file_path).read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
