<div align="center">

# Commanding at the Speed of Sound: Audio-Centric Referring Expression Segmentation for Aerial Scenes

[Fan Liu](https://multimodality.group/author/%E5%88%98%E5%87%A1/) 
<img src="assets/hhu_logo.png" alt="Hohai University" width="15">, &nbsp; &nbsp;
[Yijun Wang](https://multimodality.group/author/%E7%8E%8B%E7%BF%8C%E9%AA%8F/) ✉
<img src="assets/hhu_logo.png" alt="Hohai University" width="15">, &nbsp; &nbsp;
[Chuanyi Zhang](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm)
<img src="assets/hhu_logo.png" alt="Hohai University" width="15">, &nbsp; &nbsp;

[Liang Yao](https://multimodality.group/author/%E5%A7%9A%E4%BA%AE/)
<img src="assets/hhu_logo.png" alt="Hohai University" width="15">, &nbsp; &nbsp;
[Yuexuan An]()
<img src="assets/hhu_logo.png" alt="Hohai University" width="15">, &nbsp; &nbsp;
[Xiang Gu]()
<img src="assets/hhu_logo.png" alt="Hohai University" width="15">, &nbsp; &nbsp;

[Xiaohan Yu]()
<img src="assets/mq_logo.png" alt="Macquarie University" width="15">

\* ✉ Corresponding Author*

Benchmark : [Aerial-AudioRES](https://github.com/your-username/ARES-Net)

</div>


## Introduction

This is the official PyTorch implementation of **ARES-Net**, an end-to-end audio-centric referring expression segmentation framework for aerial scenes. Different from traditional text-centric paradigms, ARES-Net establishes direct soft alignment between continuous acoustic manifolds and multi-scale visual representations, eliminating intermediate ASR transcription errors and reducing total interaction latency to approximately 1/3 of text-based methods.


## Benchmark

We perform all experiments on our proposed **Aerial-AudioRES** benchmark. Aerial-AudioRES is the first audio-driven referring expression segmentation benchmark for aerial scenes, which contains 28,460 audio-text-image-mask quadruples with 8 distinct timbres and 4 types of realistic environmental noises. It can be downloaded from [GitHub](https://github.com/your-username/ARES-Net).

### Usage
1. Download our dataset.
2. Copy all the downloaded files to `./refer/data/`. The dataset folder should be like this:
```
$DATA_PATH
├── aerial_audiores
│   ├── refs(unc).p
│   ├── instances.json
├── images
│   └── aerial_audiores
│       ├── JPEGImages
│       └── ann_split
└── audio
    └── aerial_audiores
        ├── clean
        ├── noise_microphone
        ├── noise_outdoor
        ├── noise_human
        └── noise_indoor
```


## Setting Up

The code is verified to work with PyTorch 2.0+ and Python 3.9+.

### Package Dependencies
1. Clone this repository:
```shell
git clone https://github.com/your-username/ARES-Net.git
cd ARES-Net
```

2. Create and activate the conda environment from `environment.yml`:
```shell
conda env create -f environment.yml
conda activate aresnet
```

### Pre-trained Weights
1. Create the directory for weight files:
```shell
mkdir ./pretrained_weights
```

2. Download pre-trained weights of [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [HuBERT-base](https://huggingface.co/facebook/hubert-base-ls960), then place them into `./pretrained_weights`.


## Training

We use PyTorch DistributedDataParallel for training. All hyperparameters strictly follow the paper (60 epochs, 480×480 resolution, AdamW optimizer with initial learning rate 3e-5). More configurations can be modified in `args.py`.

Run on 2 GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 4345 \
  train.py \
  --epochs 60 \
  --img_size 480 \
  --lr 3e-5 \
  2>&1 | tee ./train_output.log
```


## Evaluation

- **Evaluation of Referring Expression Segmentation**
```shell
bash tasks/REF.sh
```


## Acknowledge
- Code in this repository is built on [RMSIN](https://github.com/Lsan2401/RMSIN). We sincerely thank the authors for their open-source work.
- We also acknowledge the contributions of [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert), [VisDrone](https://github.com/VisDrone), and [SAM2](https://github.com/facebookresearch/segment-anything-2).


## Contact
For any questions, please contact wangyijun@hhu.edu.cn.


## License
MIT License