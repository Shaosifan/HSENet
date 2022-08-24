# HSENet for remote sensing image super-resolution
Official Pytorch implementation of the paper "[Hybrid-Scale Self-Similarity Exploitation for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/document/9400474)" accepted by IEEE TGRS.  

We propose a novel hybrid-scale self-similarity exploitation network (HSENet) for remote sensing image super-resolution.The HSENet effectively leverages the internal recurrence of information both in single- and cross-scale within the images. We introduce a single-scale self-similarity exploitation module (SSEM) to mine the feature correlation within the same scale image. Meanwhile, we design a cross-scale connection structure (CCS) to capture the recurrences across different scales. By combining SSEM and CCS, we further develop a hybrid-scale self-similarity exploitation module (HSEM) to construct the final HSENet. The ablation studies demonstrate the effectiveness of the main components of the HSENet. Our method obtains better super-resolved results on UCMerced data set than several state-of-the-art approaches in terms of both accuracy and visual performance. Moreover, experiments on real-world satellite data (GF-1 and GF-2) verify the robustness of HSENet, and the experiments on NWPU data set show that the details of ground targets recovered by our method can contribute to more accurate classification when given low-resolution inputs. 

## Requirements

- Python 3.6+
- Pytorch>=1.6
- torchvision>=0.7.0 
- matplotlib
- opencv-python
- scipy
- tqdm
- scikit-image

## Installation
Clone or download this code and install aforementioned requirements 
```
cd codes
```

## Train
Download the UCMerced dataset (the data has been splted into train/val/test set for x2, x3 and x4) [[Baidu Drive](https://pan.baidu.com/s/1bxHHqKpVSyj5CiW4S6ZzDQ),password:0oaj][[Google Drive](https://drive.google.com/file/d/1eKvoe6W7q5qD33MaqujzCAPtwNZqVsKB/view)], where the original images would be taken as the HR references and the corresponding LR images are generated by bicubic down-sample. 
```
# x4
python demo_train.py --model=HSENET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=HSENETx4_UCMerced
# x3
python demo_train.py --model=HSENET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=HSENETx3_UCMerced
# x2
python demo_train.py --model=HSENET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=HSENETx2_UCMerced
```

The train/val data pathes are set in [data/__init__.py](codes/data/__init__.py) 

## Test 
The test data path and the save path can be edited in [demo_deploy.py](codes/demo_deploy.py)

```
# x4
python demo_deploy.py --model=HSENET --scale=4
# x3
python demo_deploy.py --model=HSENET --scale=3
# x2
python demo_deploy.py --model=HSENET --scale=2
```

## Evaluation 
Compute the evaluated results in term of PSNR and SSIM, where the SR/HR paths can be edited in [calculate_PSNR_SSIM.py](codes/metric_scripts/calculate_PSNR_SSIM.py)

```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```

## Citation 
If you find this code useful for your research, please cite our paper:
``````
@article{lei2021hybrid,
  title={Hybrid-Scale Self-Similarity Exploitation for Remote Sensing Image Super-Resolution},
  author={Lei, Sen and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
``````

## Acknowledgements 
This code is built on [RCAN (Pytorch)](https://github.com/yulunzhang/RCAN) and [EDSR (Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch). We thank the authors for sharing the codes.  


