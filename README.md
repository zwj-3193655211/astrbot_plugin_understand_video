# 视频理解助手（AstrBot 插件 v2.0）

<div align="center">

### 🎬 一键把视频变文字 + AI 总结

**B 站 · 抖音 · YouTube · 本地音视频 → 逐字转写 → LLM 总结**

[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.16-blue)](https://github.com/AstrBotDevs/AstrBot) [![Python](https://img.shields.io/badge/python-3.10--3.12-green)]() [![License](https://img.shields.io/badge/license-MIT-purple)]()

> **AstrBot 市场里唯一**支持多源（包含抖音 + 通用平台）+ 视频逐字 ASR + AI 总结的端到端插件

</div>

---

## 一句话总结

扔个 B 站/抖音/YouTube 链接（或本地视频路径）给机器人，它自动下载音频 → 逐字语音转写 → LLM 总结内容。无需字幕、无需手动操作，无需安装 ffmpeg、无需 1.1GB 模型（云端模式）。

## 一张表看懂

| 你给的输入 | 怎么转写 | 默认后端 |
|---|---|---|
| `BV1xx411c7mD` | B 站官方 DASH 音轨 | FunASR Paraformer-Large（GPU 5-10x） |
| `https://b23.tv/xxx` | 自动展开短链 | 同上 |
| `https://v.douyin.com/xxx/` | H5 分享页 → 浏览器 cookie → API | 同上 |
| `C:\Videos\lecture.mp4` | ffmpeg 抽音轨 | 同上 |
| `D:\audio\interview.wav` | 直读 | 同上 |

无 GPU / 不想下模型？配置 `asr_api_key` 走 SiliconFlow / 火山方舟豆包 / 任意 OpenAI 兼容端点，按用量计费，零下载。

## 真实使用

```
用户: /video https://www.bilibili.com/video/BV1xx411c7mD
Bot:  🎬 开始处理（B 站）
      输入：https://www.bilibili.com/video/BV1xx411c7mD
Bot:  ✅ 转写完成
      📌 标题：xxx
      📁 来源：B 站
      ⏱️ 时长：12分30秒
      📝 字符数：3254
      🕒 耗时：3分12秒
      💾 已保存：xxx
Bot:  🤖 正在生成 AI 总结...
Bot:  📺 视频理解结果
      📌 标题：xxx
      🤖 AI 总结：
      1. 视频主要内容概述
         ...
```

> 旧版本（v1.x）只支持 B 站 + SenseVoice 单模型 + 98MB ffmpeg.exe 捆绑。
> 新版本（v2.0）原生支持 B 站/抖音/通用平台/本地音视频 + 本地/云端两种 ASR 后端 + 零 ffmpeg 捆绑。

## 功能

- **多源输入** — B 站（BV 号 / bilibili.com / b23.tv 短链）/ 抖音（v.douyin.com / douyin.com）/ 通用平台（YouTube / AcFun / 微博，需 yt-dlp）/ 本地音视频（mp4 / mkv / mp3 / wav / m4a 等）
- **两种 ASR 后端** — 本地 FunASR Paraformer-Large（GPU 最佳）/ 云端 SiliconFlow / 火山方舟豆包 / 任意 OpenAI 兼容端点（按用量计费，零模型下载）
- **AI 总结** — 复用 AstrBot 现有的 LLM provider，提示词可配置
- **缓存管理** — 音频和转写文本自动缓存，重复处理秒回
- **数据规范** — 所有数据（缓存/模型/用户配置）放在 AstrBot 数据目录下，插件升级不丢数据

## 快速开始

1. 把插件放进 AstrBot 的 `plugins/` 目录（或通过 WebUI 上传）
2. 重启 AstrBot
3. 在 AstrBot WebUI 配置面板，填好 ASR 后端（推荐先用云端 API key 试通）：
   - `asr_backend` = `cloud`
   - `asr_api_key` = 你的 API key（[SiliconFlow](https://siliconflow.cn) 免费注册）
4. 群里发 `/video BV1xx411c7mD` 开始用

如要切换本地模式（无 API key），运行 `/video_init` 让插件自动下载 ~1.1GB 模型。

## 命令

| 命令 | 说明 |
|------|------|
| `/video <链接/路径>` | 处理视频（B 站/抖音/通用平台/本地音视频） |
| `/video_init` | 初始化环境（装依赖 + 下模型 + 写配置） |
| `/video_status` | 查看环境 / 后端 / 模型 / 缓存状态 |
| `/video_clear` | 清空缓存 |
| `/video_stop` | 取消任务（语义占位 - 阻塞任务需等当前完成） |
| `/video_open` | 打开缓存目录 |
| `/video_help` | 完整帮助 |

## 配置面板

| 字段 | 说明 | 默认 |
|------|------|------|
| `asr_backend` | `auto` / `local` / `cloud` | `auto` |
| `asr_provider` | `siliconflow` / `openai`（OpenAI 兼容） | `siliconflow` |
| `asr_api_key` | 云端 API key（也可用环境变量 `ASR_API_KEY`） | 空 |
| `asr_base_url` | `openai` 模式必填，例：火山方舟 `https://ark.cn-beijing.volces.com/api/v3` | 空 |
| `asr_model` | 供应商支持的转写模型 | `FunAudioLLM/SenseVoiceSmall` |
| `asr_verbose_json` | 请求 verbose_json 拿句级时间戳 | `false` |
| `use_punc` | 本地模式启用 CT-PUNC 标点恢复 | `true` |
| `device` | `auto` / `cuda` / `cpu` | `auto` |
| `cookie` | B 站/抖音 cookie（可选，登录态可拉大会员） | 空 |
| `model_root` | 已下完的模型目录（跳过 1.1GB 下载） | 空（用默认） |
| `max_duration_minutes` | 单视频上限（0 = 不限） | `70` |
| `summary_model` | AI 总结用的 LLM 模型（AstrBot provider） | 空（用默认） |
| `summary_prompt` | AI 总结提示词模板（`{content}` / `{title}` 占位） | 内置模板 |

## 三种运行路径

| 路径 | 适合谁 | 要什么 | 首次耗时 |
|------|--------|--------|----------|
| A. 本地 FunASR | 有 GPU / 想离线用 | 已有模型或下 ~1.1GB | 0 或 2–10 分钟 |
| B. 云端 SiliconFlow / 豆包 | 无 GPU / 不想下模型 | 免费 API key | ~1 分钟 |
| C. auto（默认） | 大多数用户 | 本地模型优先，没模型且有 key 走云端 | 自动 |

## 故障排查

**`/video_status` 必看** —— 一行能告诉你环境 / 后端 / 模型 / ffmpeg 状态。

| 现象 | 解决 |
|------|------|
| ffmpeg 缺失 | `winget install ffmpeg`（Windows） / `apt install ffmpeg`（Linux） / `brew install ffmpeg`（macOS） |
| Python 版本不兼容 | 重建 venv 用 3.10-3.12 |
| `asr_unavailable` | 两条路：跑 `/video_init` 下本地模型，或在配置面板填 `asr_api_key` 走云端 |
| B 站 403 / 大会员失败 | 在配置面板填 `cookie` 字段，或装 `yt-dlp` 自动兜底 |
| 抖音下载失败 | 装 `yt-dlp`，或在配置面板填 `cookie`（浏览器登录后复制） |
| 模型下载慢 | 设 `MODELSCOPE_CACHE` 环境变量，或 `model_root` 指向已下完的目录 |
| YouTube 等通用平台 | `pip install yt-dlp` 后 `/video <链接>` 即可 |

## 目录结构

```
astrbot_plugin_understand_video/
├── main.py                # AstrBot 插件入口（命令面 + 配置桥接）
├── metadata.yaml          # 插件元数据
├── _conf_schema.json      # 插件配置 schema
├── requirements.txt       # 依赖清单（默认空，按需安装）
├── README.md              # 本文件
├── core/                  # 嵌入的 video-transcriber skill
│   ├── __init__.py
│   ├── video_transcriber.py    # 主入口：run / setup / info / check_status
│   ├── asr_backend.py          # ASR 后端抽象：local / siliconflow / openai / auto
│   ├── bilibili.py             # B 站爬虫（纯标准库 + yt-dlp 兜底）
│   ├── douyin.py               # 抖音下载（多级 fallback）
│   ├── local_media.py          # 本地音视频 + ffmpeg
│   ├── recognizer.py           # FunASR Paraformer 识别
│   ├── download.py             # 模型下载
│   ├── cache.py                # 缓存管理
│   ├── config.py               # 配置
│   ├── netutil.py
│   └── manifest.json
└── LICENSE
```

## 数据存放位置

按 AstrBot 规范，所有运行时数据放 `data/plugin_data/astrbot_plugin_understand_video/`：

```
plugin_data/astrbot_plugin_understand_video/
├── config.json            # skill 的运行时配置（自动维护，不要手动编辑除非你知道在干嘛）
├── model/                 # ASR 模型（首次本地模式时下载 ~1.1GB）
│   ├── vad/...
│   ├── punc/...
│   └── paraformer/...
└── cache/                 # 处理缓存
    ├── audio/             # 下载的音频
    └── text/              # 转写文本
```

## 兼容性

- Python 3.10–3.12（3.13+ 与 funasr 不兼容；纯云端模式不受限）
- AstrBot >= 4.16
- Windows / Linux / macOS
- ffmpeg（必装；本地音视频/模型就绪性检查需要）

## License

MIT

## 致谢

- 底层 ASR：[FunASR](https://github.com/modelscope/FunASR) Paraformer-Large + FSMN-VAD + CT-PUNC
- 视频爬虫：[bilibili-API](https://github.com/SocialSisterYi/bilibili-API-collect) 参考实现
- 抖音多级 fallback：参考 [AI-VedioToText](https://github.com/...) 项目的 GetDouyinVideo.py
- 全部 skill 代码：[video-transcriber](https://github.com/zwj-3193655211/video-transcriber)
