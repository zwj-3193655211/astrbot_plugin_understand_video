"""
缓存管理 - 音频（m4a / wav）和转写文本（txt）
"""
import re
from pathlib import Path
from typing import Any, Dict


class CacheManager:
    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir)
        self.audio_dir = self.root / "audio"
        self.text_dir = self.root / "text"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def audio_path(self, key: str, suffix: str = ".m4a") -> Path:
        return self.audio_dir / f"{key}{suffix}"

    def text_path(self, title: str) -> Path:
        safe = re.sub(r'[\\/:*?"<>|]', '_', title).strip() or "untitled"
        return self.text_dir / f"{safe}.txt"

    def save_text(self, title: str, text: str) -> Path:
        path = self.text_path(title)
        path.write_text(text, encoding="utf-8")
        return path

    def clear_all(self) -> Dict[str, Any]:
        cleared = 0
        freed = 0
        for d in (self.audio_dir, self.text_dir):
            if not d.exists():
                continue
            for f in d.glob("*"):
                if not f.is_file():
                    continue
                try:
                    freed += f.stat().st_size
                    f.unlink()
                    cleared += 1
                except Exception:
                    pass
        return {
            "status": "success",
            "cleared_files": cleared,
            "freed_mb": round(freed / 1024 / 1024, 2),
        }

    def info(self) -> Dict[str, Any]:
        def stat(d: Path) -> Dict[str, Any]:
            if not d.exists():
                return {"files": 0, "size_mb": 0.0}
            files = [f for f in d.glob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            return {"files": len(files), "size_mb": round(size / 1024 / 1024, 2)}
        return {
            "audio": stat(self.audio_dir),
            "text": stat(self.text_dir),
            "total_mb": round(
                (sum(f.stat().st_size for f in self.audio_dir.glob("*") if f.is_file())
                 + sum(f.stat().st_size for f in self.text_dir.glob("*") if f.is_file()))
                / 1024 / 1024, 2
            ) if (self.audio_dir.exists() and self.text_dir.exists()) else 0.0,
        }
