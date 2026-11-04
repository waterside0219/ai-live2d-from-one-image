# 纯 AI 做 Live2D：从一张图到手机和桌宠

一份**真实项目复盘式**的教程：只用 AI agent，把一张平面角色立绘做成能动的 Live2D 模型（拆图、写 MOC3、物理、待机、口型），再装进安卓手机网页和 Windows 透明桌宠（字幕、出声、嘴型、按住侧键说话）。

- 施工：**GPT 6 Astra**（Codex 桌面版）按施工单干重活
- 写单、验收、上线：**Claude Opus 5.5**（沈砚）
- 原画、拍板、真机试用：阿年

不是理想化流程。返工、变脸、越改越坏、Windows 现场十一个坑，都写在里面，连同怎么爬出来、哪里要抠细、怎么验收。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/GUIDE-纯AI做Live2D-从一张图到桌宠.md](docs/GUIDE-纯AI做Live2D-从一张图到桌宠.md) | 主教程：总流程、分工、每一步的工具/做法/细节/踩坑/验收，施工单模板，Windows 桌宠 |
| [docs/LIVE2D-PLAYBOOK.md](docs/LIVE2D-PLAYBOOK.md) | 参考做法拆解、工具链限制、排查表、参数命名、验收清单 |

## 不包含

原画、拆好的图层、模型文件、录屏、私人记录都不在这里，只公开方法。Live2D Cubism Core 是专有运行库，请自行从 [Live2D 官网](https://www.live2d.com/en/sdk/download/web/) 按官方条款下载；公开发布或商用前请自己读官方许可。

## 致谢

流程参考了 [fifteen42/live2d-from-art](https://github.com/fifteen42/live2d-from-art) 和其作者的分享。
