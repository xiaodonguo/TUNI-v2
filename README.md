# TUNI-v2 (TCSVT version)
TUNI: Unifying Pre-training and Fine-tuning with Modality-Aware Mutual Learning and Rectification for RGB-T Semantic Segmentation
## Brief Introduction
This repository serves as an extension to the paper "TUNI: Real-time RGB-T Semantic Segmentation with Unified Multi-Modal Feature Extraction and Cross-Modal Feature Fusion", which has been accpeted by ICRA 2026. We include a summary of the following differences between the journal submission and the conference version:
1. We propose a novel multi-modal pre-training strategy, named **M**odal-**I**nverted **C**ontrastive **M**utual **L**earning (MI-CML).
2. We propose a novel **M**odality **R**ectification **L**earning (MRL) decoder to fully exploit thermal information during the fine-tuning phase.
3. A further trade-off between accuracy and model lightweightness is explored by introducing two encoder variants, TUNI-T and TUNI-B, which respectively target lightweight deployment and high-performance requirements.
4. We have expanded our experiments by including 6 SOTA models: SGFNet , MCNet-T, Sigma, CM-SSM, FGDNet-S, and TUNI. In addition, we conduct additional experiments on two public datasets, MSRS and SUS, to further verify the generalization capability of our model.

We have submitted this paper to IEEE TCSVT. To facilitate the review process, we release the evaluation code, experimental results, and pre-trained model weights in this repository.

## 1. 🌟  NEWS
- [2026/01/31] The conference version of TUNI has been accpected by ICRA 2026. You can find [paper](https://arxiv.org/abs/2509.10005) and [code](https://github.com/xiaodonguo/TUNI) here.
- [2026/02/07] The journal version of TUNI has submitted to IEEE TCSVT.
- [2026/02/08] We release the evaluation code, experimental results, and pre-trained model weights for journal version.
- [In Future] The training code will be released when our paper is accepted.

## 2. 🚀 Get Start
**0. Install**

```bash
conda create -n TUNI python=3.10-y
conda activate dformer

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
```
