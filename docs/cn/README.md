# GPT-SoVITS CPU 推理版

这是一个面向 Windows / Linux / macOS CPU 推理的 GPT-SoVITS 裁剪分支，只保留推理相关能力，目标是降低安装复杂度，并为后续 CPU 推理优化做准备。

[English](../../README.md)

## 这个仓库现在是什么

- 仅保留推理能力的 GPT-SoVITS 分支
- 设计目标是 CPU 推理，不再围绕 GPU 训练流程组织仓库
- 当前实际重点是 S2 `v2Pro` / `v2ProPlus` 路线，同时保留 `v1`、`v2`、`v3`、`v4`、`v2Pro`、`v2ProPlus` 的按版本推理权重下载

## 已删除内容

这个仓库已经移除了大部分上游训练与数据制作流程：

- 训练入口和训练专用工具
- 数据集切片、降噪、ASR、打标流程
- UVR5 和其他非推理 WebUI 工具

现在仓库的目标很单纯：更专注地跑 GPT-SoVITS CPU 推理。

## 当前保留的入口

- `webui.py`：最小推理启动器
- `GPT_SoVITS/inference_webui.py`：普通推理 WebUI
- `GPT_SoVITS/inference_webui_fast.py`：批量 / 快速推理 WebUI
- `api.py` 与 `api_v2.py`：推理 API

## 快速开始

### 1. 创建环境

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
```

### 2. 安装依赖并下载推理权重

CPU + ModelScope + `v2ProPlus` 示例：

```bash
bash install.sh --device CPU --source ModelScope --version v2ProPlus
```

Windows PowerShell：

```powershell
.\install.ps1 -Device CPU -Source ModelScope -Version v2ProPlus
```

可选版本：

- `v1`
- `v2`
- `v3`
- `v4`
- `v2Pro`
- `v2ProPlus`
- `all`

### 3. 启动

推荐方式：

```bash
python webui.py
```

直接启动普通推理 WebUI：

```bash
python GPT_SoVITS/inference_webui.py
```

直接启动快速推理 WebUI：

```bash
python GPT_SoVITS/inference_webui_fast.py
```

## 说明

- 这个分支的目标是推理，不再支持训练工作流。
- 中文推理前处理仍然比英 / 日 / 韩更重，因为需要额外做 `g2pw` 和 BERT 文本特征。
- `install.sh` 和 `install.ps1` 已改为按版本下载推理资源，不再整包下载全部预训练文件。
- `NLTK` 与 `OpenJTalk` 字典默认仍会下载。

## 上游项目与致谢

本项目基于并使用了以下上游项目代码：

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

下面保留上游及相关引用项目，避免丢失原始致谢信息。

## 引用项目

### 理论研究

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### 主模型 / 训练 / 声码器相关

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [SoVITS](https://github.com/voicepaw/so-vits-svc-fork)
- [GPT-SoVITS-beta](https://github.com/lj1995/GPT-SoVITS/tree/gsv-v2beta)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### 推理用文本前端

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### 继承自上游的工具引用

这些项目来自上游 GPT-SoVITS 的引用。虽然本裁剪分支已经删除了其中不少对应功能，但这里仍保留致谢。

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)
