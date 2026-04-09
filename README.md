# GPT-SoVITS CPU Inference Fork

Inference-only GPT-SoVITS fork focused on CPU deployment and CPU-side optimization on Windows, Linux, and macOS.

[中文简体](./docs/cn/README.md)

## What This Repo Is

- Inference-only fork of GPT-SoVITS.
- Designed around CPU usage rather than GPU-first features.
- Current practical focus is the S2 `v2Pro` / `v2ProPlus` path, while keeping versioned pretrained downloads for `v1`, `v2`, `v3`, `v4`, `v2Pro`, and `v2ProPlus`.

## What Was Removed

This repository no longer keeps most training and dataset-preparation features from upstream.

- Training entrypoints and training-only utilities
- Dataset slicing / denoise / ASR / labeling workflows
- UVR5 and other non-inference WebUI tools

The remaining goal is straightforward: run GPT-SoVITS inference on CPU with less installation friction and a smaller runtime surface.

## What Still Works

- `webui.py`: minimal inference launcher
- `GPT_SoVITS/inference_webui.py`: standard inference WebUI
- `GPT_SoVITS/inference_webui_fast.py`: batched / fast inference WebUI
- `api.py` and `api_v2.py`: inference APIs

## Quick Start

### 1. Create Environment

Use Miniconda or an existing Conda environment:

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
```

### 2. Install Dependencies and Download Inference Weights

CPU example with ModelScope and `v2ProPlus`:

```bash
bash install.sh --device CPU --source ModelScope --version v2ProPlus
```

Windows PowerShell:

```powershell
.\install.ps1 -Device CPU -Source ModelScope -Version v2ProPlus
```

Available versions:

- `v1`
- `v2`
- `v3`
- `v4`
- `v2Pro`
- `v2ProPlus`
- `all`

### 3. Launch

Recommended:

```bash
python webui.py
```

Direct inference WebUI:

```bash
python GPT_SoVITS/inference_webui.py
```

Fast inference WebUI:

```bash
python GPT_SoVITS/inference_webui_fast.py
```

## Notes

- This fork is aimed at CPU inference, not training.
- Chinese inference is still heavier than English / Japanese / Korean because text preprocessing needs extra frontend work such as `g2pw` and BERT features.
- `install.sh` and `install.ps1` now download inference assets by version instead of the full pretrained bundle.
- `NLTK` and `OpenJTalk` dictionary downloads remain enabled by default.

## Upstream and Credits

This project is based on and uses code from:

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

This fork keeps upstream credits and referenced projects below.

## Referenced Projects

### Theoretical Research

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### Main Model / Training / Vocoder Related

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [SoVITS](https://github.com/voicepaw/so-vits-svc-fork)
- [GPT-SoVITS-beta](https://github.com/lj1995/GPT-SoVITS/tree/gsv-v2beta)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### Text Frontend for Inference

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### Inherited Upstream Tool References

These projects were referenced by upstream GPT-SoVITS. Some related modules are removed in this inference-only fork, but credits are preserved here.

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)
