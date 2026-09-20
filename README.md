<div align="center">

# Commanding at the Speed of Sound: Audio-Centric Referring Expression Segmentation for Aerial Scenes

### [*Fan Liu*](https://openreview.net/profile?id=~Fan_Liu7)*, [*Yijun Wang*](https://openreview.net/profile?id=~Yijun_Wang10)*, [*Chuanyi Zhang*](https://openreview.net/profile?id=~Chuanyi_Zhang1)*, [*Liang Yao*](https://openreview.net/profile?id=~Liang_Yao4)*, [*Yuexuan An*](https://openreview.net/profile?id=~Yuexuan_An1)*, [*Xiang Gu*](https://openreview.net/profile?id=~Xiang_Gu5)*, [*Pai Peng*](https://openreview.net/profile?id=~Pai_Peng2)

Benchmark: [Aerial-AudioRES](https://www.modelscope.cn/datasets/LemonGauss/Aerial-AudioRES)

</div>

## Introduction

This is the official PyTorch implementation of **ARES-Net**, an end-to-end audio-centric referring expression segmentation framework for aerial scenes. Different from traditional text-centric paradigms, ARES-Net establishes direct soft alignment between continuous acoustic manifolds and multi-scale visual representations, eliminating intermediate ASR transcription errors and reducing total interaction latency to approximately 1/3 of text-based methods.

## Benchmark

We perform all experiments on our proposed **Aerial-AudioRES** benchmark. Aerial-AudioRES is an audio-driven referring expression segmentation benchmark for aerial scenes. It contains audio-text-image-mask quadruples with multiple synthetic voices, clean speech, environmental noise, and real-human speech.

The complete dataset can be downloaded from the [ModelScope dataset repository](https://www.modelscope.cn/datasets/LemonGauss/Aerial-AudioRES).

### Usage

After downloading the dataset, keep the following directory structure unchanged:

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

The `voice_8people` directory is used for synthetic clean training, validation, and clean testing. The `voice_8peoplev1_test_noise` directory contains synthetic noisy test sets. The `real_human_audio` directory contains clean and noisy real-human test audio.

## Setting Up

The reference environment uses Python 3.9, PyTorch 2.0.1, and CUDA 11.8, as specified in `environment.yml`.

### Package Dependencies

1. Clone this repository:

```shell
git clone https://github.com/LemonGauss/ARES-Net.git
cd ARES-Net
```

2. Create and activate the conda environment:

```shell
conda env create -f environment.yml
conda activate aresnet
```

### Pre-trained Weights

1. Create the directory for weight files:

```shell
mkdir ./pretrained_weights
```

2. Download the pre-trained weights of [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [HuBERT-base](https://huggingface.co/facebook/hubert-base-ls960), then place them at:

```text
./pretrained_weights/swin_base_patch4_window12_384_22k.pth
./pretrained_weights/hubert-base-ls960
```

## Training

We use PyTorch DistributedDataParallel for training. The main training settings follow the paper, including 60 epochs, 480×480 resolution, AdamW optimization, an initial learning rate of 3e-5, and 4-step gradient accumulation.

Set the dataset paths before training:

```shell
export ARES_DATA_ROOT=/path/to/Aerial-AudioRES
export ARES_REF_FILE="$ARES_DATA_ROOT/metadata/refs(unc)withvaltest.p"
export ARES_INSTANCES="$ARES_DATA_ROOT/metadata/instances.json"
export ARES_IMAGE_ROOT="$ARES_DATA_ROOT/images"
export ARES_AUDIO_ROOT="$ARES_DATA_ROOT/synthetic_audio/voice_8people"
```

Run training on two GPUs:

```shell
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
  2>&1 | tee ./train_output.log
```

## Evaluation

Set the checkpoint and dataset paths before evaluation:

```shell
export ARES_DATA_ROOT=/path/to/Aerial-AudioRES
export ARES_CHECKPOINT=/path/to/checkpoint.pth
export ARES_REF_FILE="$ARES_DATA_ROOT/metadata/refs(unc)withvaltest.p"
export ARES_INSTANCES="$ARES_DATA_ROOT/metadata/instances.json"
export ARES_IMAGE_ROOT="$ARES_DATA_ROOT/images"
export ARES_AUDIO_ROOT="$ARES_DATA_ROOT/synthetic_audio/voice_8people"
```

Run evaluation:

```shell
CUDA_VISIBLE_DEVICES=0 bash tasks/REF.sh
```

For noisy synthetic testing, use the following audio directories:

```text
synthetic_audio/voice_8peoplev1_test_noise/Test_Domestic
synthetic_audio/voice_8peoplev1_test_noise/Test_Human_Interference
synthetic_audio/voice_8peoplev1_test_noise/Test_Mic_Hiss_Only
synthetic_audio/voice_8peoplev1_test_noise/Test_Outdoor
synthetic_audio/voice_8peoplev1_test_noise/Test_Realistic_Mic_v2
```

For real-human testing, use:

```text
real_human_audio/clean
real_human_audio/noise_filtered/Test_Domestic_Filtered
real_human_audio/noise_filtered/Test_Human_Interference_Filtered
real_human_audio/noise_filtered/Test_Outdoor_Filtered
real_human_audio/noise_filtered/Test_handware_Filtered
```

## Acknowledgements

- Code in this repository is built on [RMSIN](https://github.com/Lsan2401/RMSIN). We sincerely thank the authors for their open-source work.
- We also acknowledge the contributions of [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert), [VisDrone](https://github.com/VisDrone), and [SAM2](https://github.com/facebookresearch/segment-anything-2).

## Contact

For questions, please contact:

wangyijun@hhu.edu.cn

## License

MIT License