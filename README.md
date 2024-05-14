
# SMFNet

Pytorch implementation for SMFNet: Selective Modal Fusion for RGB-D Video Salient
Object detection.


# Requirements
* Python 3.7.0 <br>
* Torch 1.7.1 <br>
* Torchvision 0.8.2 <br>
* Cuda 11.0 <br>

# Usage

## To Train 

### For training on RGB-D VSOD benchmarks
1. Download the datasets (RDVS and DVisal) from [Baidu Driver](https://pan.baidu.com/s/1mVtAWJS0eC690nPXav2lwg) (PSW: 7yer) and save it at './dataset/'. 
2. Download the pre_trained RGB, depth and flow stream models from [Baidu Driver](https://pan.baidu.com/s/1HptTP81LXANJ9W0Lu3XCQA) (PSW: 8lux) to './checkpoints/'.
3. Run `python train.py` in terminal.

### For training on VSOD benchmarks
1. Download the training datasets (DAVIS, DAVSOD, FBMS) from [Baidu Driver](https://pan.baidu.com/s/1mVtAWJS0eC690nPXav2lwg) (PSW: 7yer) and save it at './vsod_dataset/train'. 
2. Download the pre_trained RGB, depth and flow stream models from [Baidu Driver](https://pan.baidu.com/s/1HptTP81LXANJ9W0Lu3XCQA) (PSW: 8lux) to './checkpoints/'.
3. Run `python train.py` in terminal.

### For pretraining single stream
Run `python pretrain.py` in terminal. When pretraining RGB stream, we additionally use DUTS-TR [Baidu Driver](https://pan.baidu.com/s/1mVtAWJS0eC690nPXav2lwg) (PSW: 7yer).

## To Test

### For testing on RGB-D VSOD benchmarks
1. Download the trained model from [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1) to './checkpoints/'.
2. Run `python test.py` in the terminal.

### For testing on VSOD benchmarks
1. Download the trained model from [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1) to './checkpoints/'.
2. Run `python test.py` in the terminal.

## Saliency maps
1. The saliency maps of our SMFNet can be download from [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1, RGB-D VSOD benchmarks) and [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1, VSOD benchmarks).
2. We have constructed the first RGB-D VSOD benchmark, which contains the results of 19 state-of-the-art (SOTA) methods evaluated on RDVS and DVisal.
   - We evaluate the originally trained models on the testing set of RDVS and DVisal. The saliency maps can be download from [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1).
   - We first fine-tune the originally trained models on the training set of RDVS and DVisal, and then evaluate the fine-tuned models on the testing set of RDVS and DVisal. The saliency maps can be download from [Baidu Driver](https://pan.baidu.com/s/1Z8Sut8bOGOwbUBf0Tmhm4w) (PSW: lze1).

