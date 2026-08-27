"""
抖音视频下载：多级 fallback（H5 SSR → 浏览器 cookie → API CDN）

方案（与 AI-VedioToText 的 GetDouyinVideo.py 同款，纯标准库 + 可选 selenium）：
  L1 H5 分享页 SSR：分享短链 → 解析 window._ROUTER_DATA → videoInfoRes
     → 无水印端点 aweme/v1/play 下载（抖音动态放量，可能被 Argus 风控拦截）
  L2 Selenium（可选）：headless Edge/Chrome 打开视频页自动通过验证生成 cookie
     → 调 web detail API → 拿 CDN 地址（365yg.com，无需任何头直接可下）
  L3 手动 cookie：浏览器复制 Cookie 字符串 → 直接调 web detail API

返回 {"title", "audio_path", "aweme_id"}，失败返回 None。
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from netutil import make_ssl_context

MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
    "Mobile/15E148 Safari/604.1"
)
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
H5_HEADERS = {"User-Agent": MOBILE_UA, "Referer": "https://www.iesdouyin.com/"}
API_HEADERS = {"User-Agent": DESKTOP_UA, "Referer": "https://www.douyin.com/",
               "Accept": "application/json"}
DEFAULT_RATIO = "1080p"

# 从分享文本中提取 URL（用户常整段粘贴：emoji + 文案 + 链接 + 复制提示）
URL_RE = re.compile(r"https?://[^\s，。、；;！!？?]+")

_CTX = None


def _ctx():
    global _CTX
    if _CTX is None:
        _CTX = make_ssl_context()
    return _CTX


def _http_get(url: str, headers: dict, timeout: int = 30) -> bytes:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout, context=_ctx()) as resp:
        return resp.read()


def _extract_url(text: str) -> str:
    """从混合文本中提取第一个 http(s) 链接；没有则原样返回"""
    if not text:
        return ""
    m = URL_RE.search(text)
    if m:
        return m.group(0)
    return text.strip()


def _safe_filename(name: str) -> str:
    stem = re.sub(r"\s+", " ", name).strip()[:60] or "douyin-video"
    return re.sub(r'[\\/:*?"<>|#]+', "-", stem).strip(" .-") or "douyin-video"


def _download_binary(url: str, output_path: str, headers: dict,
                     log: Callable[[str], None]) -> bool:
    """分块流式下载（纯标准库）"""
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=180, context=_ctx()) as resp:
            with open(output_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        if Path(output_path).stat().st_size < 1024:
            log("⚠️ 下载内容异常（小于 1KB），可能被风控拦截")
            Path(output_path).unlink(missing_ok=True)
            return False
        return True
    except Exception as e:
        log(f"❌ 视频下载失败：{e}")
        return False


def _extract_audio(video_path: str, audio_path: str,
                   log: Callable[[str], None]) -> bool:
    """ffmpeg 提取音频（m4a AAC）"""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        log("❌ 未找到 ffmpeg")
        return False
    try:
        cmd = [ff, "-y", "-i", video_path, "-vn", "-acodec", "aac",
               "-b:a", "192k", audio_path]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              creationflags=flags, timeout=600)
        if proc.returncode != 0:
            log(f"❌ 音频提取失败：{proc.stderr.decode('utf-8', errors='ignore')[-200:]}")
            return False
        return True
    except Exception as e:
        log(f"❌ 音频提取异常：{e}")
        return False


# ==================== L1: H5 分享页 SSR ====================

def _fetch_h5(url: str, log: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    try:
        html = _http_get(url, H5_HEADERS, timeout=30).decode("utf-8", errors="ignore")
    except Exception as e:
        log(f"❌ 分享页请求失败：{e}")
        return None
    m = re.search(r"window\._ROUTER_DATA\s*=\s*(\{.*?\})</script>", html, re.S)
    if not m:
        return None
    try:
        router_data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    for key, value in (router_data.get("loaderData") or {}).items():
        if "video" not in key or not isinstance(value, dict):
            continue
        item_list = (value.get("videoInfoRes") or {}).get("item_list") or []
        if not item_list:
            continue
        item = item_list[0]
        url_list = ((item.get("video") or {}).get("play_addr") or {}).get("url_list") or []
        play_url = url_list[0] if url_list else None
        if not play_url:
            continue
        video_id = (parse_qs(urlparse(play_url).query).get("video_id") or [None])[0]
        if not video_id:
            continue
        return {
            "title": item.get("desc") or "",
            "aweme_id": str(item.get("aweme_id") or "unknown"),
            "video_url": (f"https://aweme.snssdk.com/aweme/v1/play/"
                          f"?video_id={video_id}&ratio={DEFAULT_RATIO}&line=0"),
            "headers": H5_HEADERS,
            "source": "h5",
        }
    return None


# ==================== L3/L2: web detail API ====================

def _fetch_api(aweme_id: str, cookie: str,
               log: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    url = f"https://www.douyin.com/aweme/v1/web/aweme/detail/?aweme_id={aweme_id}"
    headers = dict(API_HEADERS)
    if cookie:
        headers["Cookie"] = cookie
    try:
        body = _http_get(url, headers, timeout=30)
        if not body.strip():
            return None
        detail = json.loads(body.decode("utf-8", errors="ignore")).get("aweme_detail") or {}
        if not detail:
            return None
        url_list = ((detail.get("video") or {}).get("play_addr") or {}).get("url_list") or []
        if not url_list:
            return None
        return {
            "title": detail.get("desc") or "",
            "aweme_id": str(detail.get("aweme_id") or aweme_id),
            "video_url": url_list[0],
            "headers": API_HEADERS,
            "source": "api",
        }
    except Exception as e:
        log(f"❌ web detail API 请求失败：{e}")
        return None


def _cookie_from_selenium(aweme_id: str,
                          log: Callable[[str], None]) -> str:
    """L2：headless 浏览器拿 cookie（可选依赖 selenium）"""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.edge.service import Service as EdgeService
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        log("ℹ️ 未安装 selenium，跳过浏览器方案（pip install selenium）")
        return ""

    driver_path = os.environ.get("EDGE_DRIVER", "")
    candidates = [driver_path, r"D:\tools\chromedriver\msedgedriver.exe",
                  r"D:\tools\chromedriver\chromedriver.exe", "msedgedriver", "chromedriver"]
    driver = None
    for cand in candidates:
        if not cand:
            continue
        try:
            opts = EdgeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--no-sandbox")
            driver = webdriver.Edge(service=EdgeService(cand), options=opts)
            break
        except Exception:
            driver = None
    if driver is None:
        log("⚠️ 未能启动浏览器驱动（装 Edge 或设 EDGE_DRIVER 指定 msedgedriver 路径）")
        return ""
    try:
        log("🖥️ 启动浏览器获取抖音 cookie...")
        driver.get(f"https://www.douyin.com/video/{aweme_id}")
        WebDriverWait(driver, 30).until(lambda d: d.find_elements(By.TAG_NAME, "video"))
        cookies = [c for c in driver.get_cookies() if c.get("name")]
        if not cookies:
            log("⚠️ 浏览器未获得有效 cookie")
            return ""
        log(f"✅ 浏览器 cookie 获取成功（{len(cookies)} 个）")
        return "; ".join(f"{c['name']}={c['value']}" for c in cookies)
    except Exception as e:
        log(f"⚠️ 浏览器获取 cookie 失败：{e}")
        return ""
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def download_via_ytdlp(url: str, output_dir: str,
                          log: Callable[[str], None] = lambda *_: None,
                          custom_cookie: str = "") -> Optional[Dict[str, Any]]:
    """
    通用平台下载（yt-dlp 主路由）：任意 yt-dlp 支持的链接（YouTube/AcFun/微博/西瓜等）
    → 音频 m4a。需安装 yt-dlp；YouTube 需代理；部分平台需 cookie。

    返回 {"title", "audio_path", "url"} 或 None
    """
    import shutil
    import subprocess

    url = _extract_url(url)
    if not url:
        log("无效的链接")
        return None
    log(f"通用通道（yt-dlp）处理：{url}")

    yt = shutil.which("yt-dlp")
    cmd_prefix = None
    if yt:
        cmd_prefix = [yt]
    else:
        try:
            import yt_dlp  # noqa: F401
            cmd_prefix = [sys.executable, "-m", "yt_dlp"]
        except ImportError:
            log("未安装 yt-dlp，无法处理该平台（pip install yt-dlp）")
            return None

    os.makedirs(output_dir, exist_ok=True)
    cmd = cmd_prefix + [
        "--no-playlist",
        "-f", "ba/b",
        "-x", "--audio-format", "m4a",
        "-o", os.path.join(output_dir, "%(title)s.%(ext)s"),
    ]
    if custom_cookie:
        cmd += ["--add-header", f"Cookie: {custom_cookie}"]
    cmd += [url]
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        r = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=flags, timeout=1800)
        if r.returncode != 0:
            log(f"yt-dlp 失败：{(r.stderr or '')[-200:]}")
            return None
    except Exception as e:
        log(f"yt-dlp 异常：{e}")
        return None

    exts = (".m4a", ".mp3", ".webm", ".aac")
    files = [f for f in Path(output_dir).iterdir()
             if f.is_file() and f.suffix.lower() in exts]
    if not files:
        log("yt-dlp 未产出音频文件")
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    log(f"✅ 音频已下载：{latest.name}")
    return {"title": latest.stem, "audio_path": str(latest), "url": url}


# ==================== 对外接口 ====================

def download_douyin(url: str, output_dir: str,
                    log: Callable[[str], None] = lambda *_: None,
                    custom_cookie: str = "") -> Optional[Dict[str, Any]]:
    """
    下载抖音视频音频 → {"title", "audio_path", "aweme_id"} 或 None
    """
    def _resolve_aweme_id(u: str) -> str:
        m = re.search(r"/(?:video|share/video)/(\d+)", u)
        return m.group(1) if m else ""

    url = _extract_url(url)  # 支持整段粘贴分享文本
    if not url or ("douyin.com" not in url and "iesdouyin.com" not in url):
        log("无效的抖音链接")
        return None
    log(f"处理抖音链接：{url}")
    os.makedirs(output_dir, exist_ok=True)
    aweme_id = _resolve_aweme_id(url)

    # L3 手动 cookie → API
    info = None
    if custom_cookie:
        log("使用手动 cookie 调 web detail API...")
        if not aweme_id:
            try:
                _ = _http_get(url, H5_HEADERS, timeout=20)
            except Exception:
                pass
            # 短链重定向拿 aweme_id（urlopen 已跟随）
        if aweme_id:
            info = _fetch_api(aweme_id, custom_cookie, log)

    # L1 H5
    if not info:
        info = _fetch_h5(url, log)
        if info:
            log("✅ H5 分享页方案成功（无水印直连）")

    # L2 Selenium → API
    if not info:
        if not aweme_id:
            aweme_id = _resolve_aweme_id(url)
        if aweme_id:
            ck = _cookie_from_selenium(aweme_id, log)
            if ck:
                info = _fetch_api(aweme_id, ck, log)
                if info:
                    log("✅ 浏览器 cookie → API 方案成功")

    if not info:
        log("❌ 全部方案失败（抖音风控：稍后重试，或浏览器登录抖音后复制 cookie 传入）")
        return None

    title = _safe_filename(info["title"] or f"抖音视频-{info['aweme_id']}")
    tmp_video = os.path.join(output_dir, f"tmp_{info['aweme_id']}.mp4")
    ok = _download_binary(info["video_url"], tmp_video, info["headers"], log)
    # L1 下载端点被风控 → 回退 L2 API CDN
    if not ok and info["source"] == "h5":
        log("H5 下载端点被风控，回退浏览器方案获取 CDN 地址...")
        ck = _cookie_from_selenium(info["aweme_id"], log)
        if ck:
            api_info = _fetch_api(info["aweme_id"], ck, log)
            if api_info:
                ok = _download_binary(api_info["video_url"], tmp_video,
                                      api_info["headers"], log)
                if ok:
                    info = api_info
    if not ok:
        return None
    log(f"✅ 视频已下载：{os.path.getsize(tmp_video) / 1024 / 1024:.1f} MB")

    # 抽音频
    audio_path = os.path.join(output_dir, f"{title}.m4a")
    counter = 1
    while os.path.exists(audio_path):
        audio_path = os.path.join(output_dir, f"{title}_{counter}.m4a")
        counter += 1
    if not _extract_audio(tmp_video, audio_path, log):
        return None
    os.remove(tmp_video)
    log(f"✅ 音频已提取：{Path(audio_path).name}")

    return {"title": title, "audio_path": str(audio_path), "aweme_id": info["aweme_id"]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python douyin.py <抖音链接> [输出目录] [cookie]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "cache/audio"
    ck = sys.argv[3] if len(sys.argv) > 3 else ""
    r = download_douyin(sys.argv[1], out, log=print, custom_cookie=ck)
    print(json.dumps(r, ensure_ascii=False, indent=2) if r else "下载失败")
