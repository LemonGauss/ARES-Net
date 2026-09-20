<div align="center">


# ARES-Net

### Commanding at the Speed of Sound
**Audio-Centric Referring Expression Segmentation for Aerial Scenes**

<p>
  <a href="https://www.modelscope.cn/datasets/LemonGauss/Aerial-AudioRES">Dataset</a> ·
  <a href="#quick-start">Quick start</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9" />
  <img src="https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.0.1" />
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 11.8" />
</p>

<p><a href="https://openreview.net/profile?id=~Fan_Liu7">Fan Liu</a><img src="assets/hhu_logo.png" alt="Hohai University" width="16"> · <a href="https://openreview.net/profile?id=~Yijun_Wang10">Yijun Wang</a><img src="assets/hhu_logo.png" alt="Hohai University" width="16"> · <a href="https://openreview.net/profile?id=~Chuanyi_Zhang1">Chuanyi Zhang</a><img src="assets/hhu_logo.png" alt="Hohai University" width="16"> · <a href="https://openreview.net/profile?id=~Liang_Yao4">Liang Yao</a><img src="assets/hhu_logo.png" alt="Hohai University" width="16"> · <a href="https://openreview.net/profile?id=~Yuexuan_An1">Yuexuan An</a><img src="assets/hhu_logo.png" alt="Hohai University" width="16"> · <a href="https://openreview.net/profile?id=~Xiang_Gu5">Xiang Gu</a><img src="assets/hhu_logo.png" alt="Hohai University" width="16"> · <a href="https://openreview.net/profile?id=~Pai_Peng2">Pai Peng</a><img src="assets/浙江大学-logo.png" alt="Zhejiang University" width="16"></p>

## Overview

ARES-Net is an end-to-end **audio-centric referring expression segmentation** framework for aerial scenes. It learns a direct soft alignment between continuous acoustic features and multi-scale visual representations, avoiding intermediate ASR transcription and reducing interaction latency to approximately one third of text-based pipelines.

<div align="center">

| Benchmark | Modalities | Task |
|:---:|:---:|:---:|
| [Aerial-AudioRES](https://www.modelscope.cn/datasets/LemonGauss/Aerial-AudioRES) | audio · text · image · mask | referring expression segmentation |

</div>

## Contents

- [Benchmark and data](#benchmark-and-data)
- [Quick start](#quick-start)
- [Pre-trained weights](#pre-trained-weights)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Benchmark and data

Experiments use **Aerial-AudioRES**, an audio-driven benchmark for aerial scenes. It contains audio–text–image–mask quadruples spanning synthetic voices, clean speech, environmental noise, and real-human speech.

Download the complete dataset from the [ModelScope repository](https://www.modelscope.cn/datasets/LemonGauss/Aerial-AudioRES), then keep this layout unchanged:

<details>
<summary><strong>Expected directory structure</strong></summary>

```text
Aerial-AudioRES/
├── images/
│   └── *.jpg
├── metadata/
│   ├── instances.json
│   ├── refs(unc)withvaltest.p
│   └── true_people-v2-jiasu_Filtered.p
├── synthetic_audio/
│   ├── voice_8people/
│   └── voice_8peoplev1_test_noise/
│       ├── Test_Domestic/
│       ├── Test_Human_Interference/
│       ├── Test_Mic_Hiss_Only/
│       ├── Test_Outdoor/
│       └── Test_Realistic_Mic_v2/
└── real_human_audio/
    ├── clean/
    └── noise_filtered/
        ├── Test_Domestic_Filtered/
        ├── Test_Human_Interference_Filtered/
        ├── Test_Outdoor_Filtered/
        └── Test_handware_Filtered/
```

</details>

`voice_8people` is used for synthetic clean training, validation, and testing. `voice_8peoplev1_test_noise` contains synthetic noisy test sets. `real_human_audio` contains clean and noisy real-human test audio.

## Quick start

### 1. Clone and install

The reference environment is Python 3.9, PyTorch 2.0.1, and CUDA 11.8.

```bash
git clone https://github.com/LemonGauss/ARES-Net.git
cd ARES-Net

conda env create -f environment.yml
conda activate aresnet
```

### 2. Download pre-trained weights

```bash
mkdir -p pretrained_weights
```

Download [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [HuBERT-base](https://huggingface.co/facebook/hubert-base-ls960), then place them here:

```text
pretrained_weights/
├── swin_base_patch4_window12_384_22k.pth
└── hubert-base-ls960/
```

## Training

ARES-Net uses PyTorch DistributedDataParallel. The reference recipe runs for 60 epochs at `480 × 480` resolution with AdamW, an initial learning rate of `3e-5`, and four-step gradient accumulation.

Set the dataset paths:

```bash
export ARES_DATA_ROOT=/path/to/Aerial-AudioRES
export ARES_REF_FILE="$ARES_DATA_ROOT/metadata/refs(unc)withvaltest.p"
export ARES_INSTANCES="$ARES_DATA_ROOT/metadata/instances.json"
export ARES_IMAGE_ROOT="$ARES_DATA_ROOT/images"
export ARES_AUDIO_ROOT="$ARES_DATA_ROOT/synthetic_audio/voice_8people"
```

Run on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 4345 \
  train.py \
  --epochs 60 \
  --img_size 480 \
  --lr 3e-5 \
  --ref_file_path "$ARES_REF_FILE" \
  --instances_path "$ARES_INSTANCES" \
  --imageFolder "$ARES_IMAGE_ROOT" \
  --audioFolder "$ARES_AUDIO_ROOT" \
  2>&1 | tee train_output.log
```

## Evaluation

Set the checkpoint and shared dataset paths:

```bash
export ARES_DATA_ROOT=/path/to/Aerial-AudioRES
export ARES_CHECKPOINT=/path/to/checkpoint.pth
export ARES_REF_FILE="$ARES_DATA_ROOT/metadata/refs(unc)withvaltest.p"
export ARES_INSTANCES="$ARES_DATA_ROOT/metadata/instances.json"
export ARES_IMAGE_ROOT="$ARES_DATA_ROOT/images"
export ARES_AUDIO_ROOT="$ARES_DATA_ROOT/synthetic_audio/voice_8people"
```

Run the standard evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash tasks/REF.sh
```

For noisy synthetic testing, point `ARES_AUDIO_ROOT` to one of:

```text
synthetic_audio/voice_8peoplev1_test_noise/{Test_Domestic,Test_Human_Interference,Test_Mic_Hiss_Only,Test_Outdoor,Test_Realistic_Mic_v2}
```

For real-human testing, use:

```text
real_human_audio/clean
real_human_audio/noise_filtered/{Test_Domestic_Filtered,Test_Human_Interference_Filtered,Test_Outdoor_Filtered,Test_handware_Filtered}
```

## Acknowledgements

- [RMSIN](https://github.com/Lsan2401/RMSIN), which this codebase builds upon.
- [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert), [VisDrone](https://github.com/VisDrone), and [SAM2](https://github.com/facebookresearch/segment-anything-2).

## Contact

Questions and feedback: **wangyijun@hhu.edu.cn**

## License

Released under the [MIT License](LICENSE).
