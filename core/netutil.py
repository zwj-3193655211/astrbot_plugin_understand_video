"""
网络小工具：构建可靠的 SSL 上下文

背景：部分 conda 环境的 OpenSSL 解析 Windows 证书库有 bug
（报 [ASN1: NOT_ENOUGH_DATA]），导致所有 HTTPS 请求失败。
对策：优先用 certifi 的 CA 包（不碰 Windows 证书存储），
没有 certifi 时回退默认上下文，最后回退到不校验（并打警告）。

纯标准库 + 可选 certifi（纯 Python 包，~300KB）。
"""
import ssl
import warnings


def make_ssl_context() -> ssl.SSLContext:
    """返回可用的 SSLContext（certifi → 系统默认 → 不校验兜底）"""
    # 1. certifi（最可靠，绕开 conda OpenSSL 的 Windows 证书库 bug）
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    # 2. 默认上下文（读系统证书）
    try:
        return ssl.create_default_context()
    except Exception:
        pass
    # 3. 兜底：不校验（仅用于无法构建证书环境的极端情况）
    warnings.warn(
        "无法加载可信 CA 证书，HTTPS 将不校验证书（不安全）。"
        "建议安装 certifi：pip install certifi",
        RuntimeWarning,
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
