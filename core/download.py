"""
video-transcriber ASR 模型下载/检查

基石模型（3 个，对齐 AI-VedioToText 项目的下载清单）：
  1. VAD:  iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
  2. PUNC: iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
  3. Paraformer-Large (中文 ASR): iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

默认下载到 skill 根目录的 model/ 子目录（约 1.1 GB）。
复用已下载模型：在 config.json 把 model_root 指向其他位置
（例：D:\\Desktop_Archive\\AI-VedioToText\\model），然后跑 --check。

用法：
  python download.py                       # 检查并补齐
  python download.py --check               # 只检查
  python download.py --force               # 强制重下
  python download.py --model-root <path>   # 自定义模型目录

环境变量（可选）：
  MODELSCOPE_CACHE    指定模型缓存目录（默认用户目录）
  HF_ENDPOINT         代理镜像（如 https://hf-mirror.com）
"""
import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modelscope import snapshot_download

DEFAULT_MODEL_ROOT = Path(__file__).parent / "model"

# (显示名, ModelScope ID, 父目录, 模型目录名)
MODELS = [
    (
        "VAD 语音活动检测",
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "vad",
        "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    ),
    (
        "PUNC 标点恢复",
        "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "punc",
        "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    ),
    (
        "Paraformer-Large 离线中文 ASR",
        "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "paraformer/paraformer-offline/iic",
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    ),
]

MARKER = "model.pt"


def model_ready(model_dir: Path) -> bool:
    return (model_dir / MARKER).exists()


def _size_mb(model_dir: Path) -> float:
    try:
        return sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1024 / 1024
    except Exception:
        return 0.0


def check_one(name: str, model_dir: Path) -> bool:
    ready = model_ready(model_dir)
    status = "✅" if ready else "❌"
    size = f" ({_size_mb(model_dir):.0f} MB)" if ready else ""
    print(f"  {status} {name}{size}")
    print(f"      {model_dir}")
    return ready


def download_one(name: str, model_id: str, model_dir: Path, force: bool = False) -> bool:
    if model_ready(model_dir) and not force:
        check_one(name, model_dir)
        return True
    print(f"\n⏳ 正在下载 {name} ...")
    print(f"   模型 ID: {model_id}")
    # 断点续传式重试（modelscope 支持本地续传）
    for attempt in range(1, 4):
        try:
            snapshot_download(model_id, local_dir=str(model_dir), revision="master")
            if model_ready(model_dir):
                print(f"✅ {name} 下载完成 ({_size_mb(model_dir):.0f} MB)")
                return True
            print(f"⚠️  {name} 下载后未找到 {MARKER}，请检查目录")
            return False
        except Exception as e:
            print(f"❌ {name} 下载失败 (第 {attempt}/3 次): {e}")
            if attempt < 3:
                print("   5 秒后重试...")
                time.sleep(5)
    return False


def main():
    p = argparse.ArgumentParser(description="下载/检查 ASR 模型")
    p.add_argument("--check", action="store_true", help="只检查不下载")
    p.add_argument("--force", action="store_true", help="强制重新下载")
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT),
                   help=f"模型根目录（默认 {DEFAULT_MODEL_ROOT}）")
    args = p.parse_args()

    root = Path(args.model_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 缓存目录提示
    cache_env = os.environ.get("MODELSCOPE_CACHE", "系统默认 (~/.cache/modelscope)")
    print("=" * 60)
    print(f"video-transcriber 模型检查/下载")
    print(f"  root: {root}")
    print(f"  modelscope cache: {cache_env}")
    print("=" * 60)

    all_ready = True
    for name, model_id, parent, dirname in MODELS:
        target = root / parent / dirname
        if args.check:
            ready = check_one(name, target)
        else:
            ready = download_one(name, model_id, target, force=args.force)
        all_ready = all_ready and ready

    print("=" * 60)
    if all_ready:
        print("✅ 所有模型已就绪")
    else:
        print("⚠️ 存在缺失，请重试或检查网络")
    print("=" * 60)
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
