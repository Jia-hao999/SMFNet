
# SMFNet [[PDF]](https://arxiv.org/pdf/2507.21857v1)|[[中文概要]](https://mp.weixin.qq.com/s/fUFah2ssS-6R0CgAbf6LLQ)


Pytorch implementation for SMFNet: Unleashing the Power of Motion and Depth: A Selective Fusion Strategy for RGB-D Video Salient Object Detection.

# Requirements
* Python 3.7.0 <br>
* Torch 1.7.1 <br>
* Torchvision 0.8.2 <br>
* Cuda 11.0 <br>

# Usage

## To Train 

### For training on RGB-D VSOD benchmarks
1. Download the datasets (RDVS and DVisal) from [Baidu Driver](https://pan.baidu.com/s/1vYEDy4uPbbB20Cvik-oriQ) (PSW: d4ew) and save it at './dataset/'. 
2. Download the pre_trained RGB, depth and flow stream models from [Baidu Driver](https://pan.baidu.com/s/1QIh1Fii5isWe0VOqcZxwCA) (PSW: lm6d) to './checkpoints/'.
3. Run `python train.py` in terminal.

### For training on VSOD benchmarks
1. Download VSOD datasets from [Baidu Driver](https://pan.baidu.com/s/1-slo_A3bjG9H61I1_wEwLQ) (PSW: hveg) and save the training datasets (DAVIS, DAVSOD, FBMS) at './vsod_dataset/train'.
2. Download the pre_trained RGB, depth and flow stream models from [Baidu Driver](https://pan.baidu.com/s/1-SgyEkNafLbXImqXswZWqQ) (PSW: 3c48) to './checkpoints/'.
3. Run `python train.py` in terminal.

### For pretraining single stream
Run `python pretrain.py` in terminal. When pretraining RGB stream, we additionally use DUTS-TR [Baidu Driver](https://pan.baidu.com/s/10mx3Oxy0PenTftHWInZYVw) (PSW: h5sn) and the pre_trained ResNet34 [Baidu Driver](https://pan.baidu.com/s/14PI0fHIawNlfBOubneMc0w) (PSW: mthj).

## To Test

### For testing on RGB-D VSOD benchmarks
1. Download the trained model from [Baidu Driver](https://pan.baidu.com/s/10XBOfTQ9V01_8rc-GsD2nw) (PSW: hgm3) to './checkpoints/'.
2. Run `python test.py` in the terminal.

### For testing on VSOD benchmarks
1. Download the trained model from [Baidu Driver](https://pan.baidu.com/s/1sp7brnKiVd2MUvVqCkMy8Q) (PSW: p2q0) to './checkpoints/'.
2. Run `python test.py` in the terminal.

## Saliency maps
1. The saliency maps of our SMFNet can be download from [Baidu Driver](https://pan.baidu.com/s/1A_jdsZErilXpgUatymNFLw) (PSW: u8rz, RGB-D VSOD benchmarks) and [Baidu Driver](https://pan.baidu.com/s/1MyR45E28WSb3YudHJaBjew) (PSW: 8mgu, VSOD benchmarks).
2. We have constructed the first RGB-D VSOD benchmark, which contains the results of 19 state-of-the-art (SOTA) methods evaluated on RDVS and DVisal.
   - We evaluate the originally trained models on the testing set of RDVS and DVisal. The saliency maps can be download from [Baidu Driver](https://pan.baidu.com/s/1vfiTxkrTjblQu5j13HS3_Q) (PSW: bjyk).
   - We first fine-tune the originally trained models on the training set of RDVS and DVisal, and then evaluate the fine-tuned models on the testing set of RDVS and DVisal. The saliency maps can be download from [Baidu Driver](https://pan.baidu.com/s/1rO7p6R7KJJjN3bDvxVEUcw) (PSW: hjwy).

