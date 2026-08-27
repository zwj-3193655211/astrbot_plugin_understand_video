"""
FunASR Paraformer-Large 识别（VAD + PUNC + 字级时间戳）

基石模型：Paraformer-Large 离线中文 ASR + FSMN-VAD + CT-PUNC 标点恢复。
- 模型路径由 config.model_root 决定（默认 skill/model/，可指向其他位置复用）
- 模型全局缓存，第一次调用后驻留内存
- 识别返回全文 + 标点切句的 segments（带起止秒数）
"""
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import load_config

_lock = threading.Lock()
_model = None
_model_root_signature: Optional[str] = None


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(cfg["model_root"])
    return {
        "vad": root / "vad" / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": root / "punc" / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "offline": (
            root / "paraformer" / "paraformer-offline" / "iic"
            / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        ),
    }


def check_models(cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """检查基石模型文件是否齐全（每个目录需有 model.pt）"""
    cfg = cfg or load_config()
    paths = _resolve_paths(cfg)
    missing: List[str] = []
    if not (paths["vad"] / "model.pt").exists():
        missing.append(f"VAD: {paths['vad']}")
    if not (paths["offline"] / "model.pt").exists():
        missing.append(f"Paraformer: {paths['offline']}")
    if cfg.get("use_punc", True) and not (paths["punc"] / "model.pt").exists():
        missing.append(f"PUNC: {paths['punc']}")
    return len(missing) == 0, missing


def _resolve_device(cfg: Dict[str, Any]) -> str:
    device = cfg.get("device", "auto")
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _get_model(cfg: Dict[str, Any]):
    """全局单例加载 FunASR AutoModel"""
    global _model, _model_root_signature
    signature = str(Path(cfg["model_root"]).resolve())
    with _lock:
        if _model is not None and _model_root_signature == signature:
            return _model
        from funasr import AutoModel

        paths = _resolve_paths(cfg)
        device = _resolve_device(cfg)
        use_punc = cfg.get("use_punc", True)

        kwargs: Dict[str, Any] = dict(
            model=str(paths["offline"]),
            vad_model=str(paths["vad"]),
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
        if use_punc:
            kwargs["punc_model"] = str(paths["punc"])

        suffix = " + CT-PUNC" if use_punc else ""
        print(f"[ASR] 加载 Paraformer-Large + FSMN-VAD{suffix} @ {device}")
        _model = AutoModel(**kwargs)
        _model_root_signature = signature
        return _model


def _build_segments(text: str, ts: List[List[float]]) -> List[Dict[str, Any]]:
    """根据字级时间戳和标点切句，返回 [{start, end, text}]"""
    if not text or not ts:
        return []
    try:
        parts = re.split(r"(?<=[。？！；.!?；])", text)
        parts = [p for p in parts if p.strip()]
        n_chars = max(len(text), 1)
        n_ts = len(ts)
        char_cursor = 0
        out: List[Dict[str, Any]] = []
        for sent in parts:
            L = len(sent)
            if L <= 0 or char_cursor >= n_chars:
                continue
            end_char = min(char_cursor + L, n_chars)
            i0 = int((char_cursor / n_chars) * (n_ts - 1))
            i1 = int((end_char / n_chars) * (n_ts - 1))
            i1 = max(i1, i0 + 1)
            i1 = min(i1, n_ts - 1)
            out.append({
                "start": round(float(ts[i0][0]) / 1000.0, 2),
                "end": round(float(ts[i1][1]) / 1000.0, 2),
                "text": sent.strip(),
            })
            char_cursor = end_char
        return out
    except Exception:
        return []


def transcribe(audio_path: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    转写音频 → {"text": str, "segments": [{start, end, text}], "duration": float}
    """
    cfg = cfg or load_config()
    audio_path = str(audio_path)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"音频不存在：{audio_path}")

    model = _get_model(cfg)
    result = model.generate(
        input=audio_path,
        cache={},
        batch_size_s=60,
        pred_timestamp=True,
    )

    raw = (result[0].get("text", "") if result else "") or ""
    text = re.sub(r"<\|[^|]+\|>", "", raw).strip()
    text = re.sub(r"\s+", " ", text).strip()

    ts = result[0].get("timestamp") if result else None
    segments: List[Dict[str, Any]] = []
    if ts and isinstance(ts, list) and all(
        isinstance(x, (list, tuple)) and len(x) == 2 for x in ts
    ):
        segments = _build_segments(text, ts)

    duration = segments[-1]["end"] if segments else 0.0
    return {"text": text, "segments": segments, "duration": duration}
