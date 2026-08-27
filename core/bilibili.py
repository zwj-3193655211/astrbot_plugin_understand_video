"""
B 站视频爬虫：提取 BV 号、解析短链、下载 m4a 音频

纯标准库实现（urllib，不依赖 requests），失败时自动降级 yt-dlp 兜底：
- 主路由：直连页面抓 window.__playinfo__（DASH 音频流），自动 cookie 重抓
- 兜底：直连失败 / 音频下载失败时，若系统装了 yt-dlp 则自动切换

实现借鉴了 AI-VedioToText 项目的 GetBiliBiliVideo.py 与
video-downloader skill 的 yt-dlp 兜底模式。
"""
import gzip
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from netutil import make_ssl_context

BV_PATTERN = re.compile(r"BV[a-zA-Z0-9]{10,12}")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# 补齐浏览器全套头，降低被 B 站反爬拦截的概率
FULL_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip() or "untitled"


def _http_get(url: str, headers: Optional[dict] = None, timeout: int = 30):
    """GET 请求（纯标准库，自动解 gzip）。返回 (body, set_cookie_list, final_url)"""
    hdrs = dict(FULL_HEADERS)
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    with urlopen(req, timeout=timeout, context=make_ssl_context()) as resp:
        body = resp.read()
        content_encoding = resp.headers.get("Content-Encoding", "").lower()
        cookies = resp.headers.get_all("Set-Cookie") or []
        final_url = resp.geturl()
    if content_encoding == "gzip":
        body = gzip.decompress(body)
    return body, cookies, final_url


def _cookies_to_string(set_cookie_list) -> str:
    """把 Set-Cookie 响应头转成 'name=value; ...' 的 cookie 串"""
    parts = []
    for sc in set_cookie_list:
        nv = sc.split(";")[0].strip()
        if nv and "=" in nv:
            parts.append(nv)
    return "; ".join(parts)


def get_bilibili_cookie() -> Optional[str]:
    """从 B 站主页动态拿 cookie；失败返回 None。"""
    try:
        _, cookies, _ = _http_get(
            "https://www.bilibili.com/", headers={"User-Agent": USER_AGENT}, timeout=30
        )
        s = _cookies_to_string(cookies)
        return s or None
    except Exception:
        return None


def extract_bv(text: str) -> Optional[str]:
    """从混合文本提取第一个 BV 号"""
    m = BV_PATTERN.search(text)
    return m.group() if m else None


def parse_url(text: str) -> Optional[str]:
    """从混合文本提取 B 站标准 URL（自动展开 b23.tv 短链）"""
    text = text.strip()
    # 1. 直接 BV 号
    bv = extract_bv(text)
    if bv:
        return f"https://www.bilibili.com/video/{bv}"
    # 2. 短链
    m = re.search(r'(?:https?://)?b23\.tv/[a-zA-Z0-9]+', text)
    if m:
        short = m.group()
        if not short.startswith("http"):
            short = "https://" + short
        try:
            # urlopen 自动跟随重定向，geturl() 取最终地址
            _, _, final = _http_get(short, headers={"User-Agent": USER_AGENT}, timeout=15)
            bv = extract_bv(final)
            if bv:
                return f"https://www.bilibili.com/video/{bv}"
        except Exception:
            return None
    return None


# ==================== yt-dlp 兜底 ====================

def _download_with_ytdlp(url: str, output_dir: str,
                         log: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    """直连失败时用 yt-dlp 下载纯音频（ba/b）。未安装 yt-dlp 则返回 None。"""
    yt = shutil.which("yt-dlp")
    if not yt:
        return None
    log("⚠️ 切换 yt-dlp 兜底...")
    out = Path(output_dir)
    try:
        cmd = [yt, "--no-playlist", "-f", "ba/b",
               "-o", str(out / "%(title)s.%(ext)s"), url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            log(f"yt-dlp 失败：{(r.stderr or '')[-200:]}")
            return None
    except Exception as e:
        log(f"yt-dlp 异常：{e}")
        return None
    files = [f for f in out.iterdir() if f.suffix.lower() in (".m4a", ".mp3", ".webm", ".aac")]
    if not files:
        return None
    f = max(files, key=lambda x: x.stat().st_mtime)
    log(f"✅ yt-dlp 下载完成：{f.name}")
    return {"title": _safe_filename(f.stem), "audio_path": str(f), "bv_id": extract_bv(url) or ""}


# ==================== 主流程 ====================

def get_audio(url: str, output_dir: str,
              log: Callable[[str], None] = lambda *_: None,
              custom_cookie: str = "") -> Optional[Dict[str, Any]]:
    """
    下载 B 站视频音频到 output_dir（直连优先，yt-dlp 兜底）

    Returns:
        {"title": str, "audio_path": str, "bv_id": str} 或 None
    """
    bv = extract_bv(url)
    if not bv:
        log("❌ 无效的 B 站链接")
        return None
    page_url = f"https://www.bilibili.com/video/{bv}"
    log(f"处理视频：{page_url}")

    os.makedirs(output_dir, exist_ok=True)

    # 抓页面（带 cookie 重试一次）
    json_data: Optional[dict] = None
    title: Optional[str] = None
    cookie = custom_cookie or None

    for attempt in range(2):
        if not cookie:
            cookie = get_bilibili_cookie()
        headers = {"Referer": page_url, "User-Agent": USER_AGENT}
        if cookie:
            headers["Cookie"] = cookie
        try:
            body, _, _ = _http_get(page_url, headers=headers, timeout=30)
            html = body.decode("utf-8", errors="ignore")
        except Exception as e:
            log(f"网络请求失败：{e}")
            cookie = None  # 下次重抓
            continue

        m = re.findall(r'title="([^"]+)"', html)
        if m:
            title = _safe_filename(m[0])
        m = re.search(r'window\.__playinfo__\s*=\s*({.*?})\s*<', html, re.DOTALL)
        if m:
            try:
                json_data = json.loads(m.group(1))
                break
            except json.JSONDecodeError:
                pass
        # 第一次没拿到数据，强制重抓 cookie
        if attempt == 0:
            log("未拿到播放信息，刷新 cookie 重试...")
            cookie = None

    # 直连页面失败 → yt-dlp 兜底
    if not json_data or not title:
        log("直连未获取到播放信息，尝试 yt-dlp 兜底...")
        fb = _download_with_ytdlp(page_url, output_dir, log)
        if fb:
            return fb
        log("❌ 未能获取视频信息（可能需要登录或地区限制；可装 yt-dlp 增加成功率）")
        return None

    if 'data' not in json_data or 'dash' not in json_data['data']:
        log("❌ 视频数据格式异常（老 FLV 格式暂不支持），尝试 yt-dlp 兜底...")
        fb = _download_with_ytdlp(page_url, output_dir, log)
        if fb:
            return fb
        return None

    audio_streams = json_data['data']['dash'].get('audio', [])
    if not audio_streams:
        log("❌ 未找到音频流")
        return None

    audio_url = audio_streams[0]['baseUrl']
    headers = {"Referer": page_url, "User-Agent": USER_AGENT}
    if cookie:
        headers["Cookie"] = cookie

    # 下载音频（重试 3 次）
    content: Optional[bytes] = None
    for attempt in range(3):
        try:
            body, _, _ = _http_get(audio_url, headers=headers, timeout=60)
            content = body
            break
        except Exception as e:
            log(f"音频下载失败 (第 {attempt + 1}/3 次)：{e}")
    if content is None:
        log("直连音频下载失败，尝试 yt-dlp 兜底...")
        fb = _download_with_ytdlp(page_url, output_dir, log)
        if fb:
            return fb
        log("❌ 音频下载失败")
        return None

    # 保存（重名自增）
    out_path = Path(output_dir) / f"{title}.m4a"
    counter = 1
    while out_path.exists():
        out_path = Path(output_dir) / f"{title}_{counter}.m4a"
        counter += 1
    out_path.write_bytes(content)
    log(f"✅ 音频已下载：{out_path.name} ({len(content) / 1024 / 1024:.1f} MB)")

    return {"title": title, "audio_path": str(out_path), "bv_id": bv}
