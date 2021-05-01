# HSENet
Official Pytorch implementation of the paper "[Hybrid-Scale Self-Similarity Exploitation for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/document/9400474)" accepted by IEEE TGRS.  

We propose a novel hybrid-scale self-similarity exploitation network (HSENet) for remote sensing image super-resolution.The HSENet effectively leverages the internal recurrence of information both in single- and cross-scale within the images. We introduce a single-scale self-similarity exploitation module (SSEM) to mine the feature correlation within the same scale image. Meanwhile, we design a cross-scale connection structure (CCS) to capture the recurrences across different scales. By combining SSEM and CCS, we further develop a hybrid-scale self-similarity exploitation module (HSEM) to construct the final HSENet. The ablation studies demonstrate the effectiveness of the main components of the HSENet. Our method obtains better super-resolved results on UCMerced data set than several state-of-the-art approaches in terms of both accuracy and visual performance. Moreover, experiments on real-world satellite data (GF-1 and GF-2) verify the robustness of HSENet, and the experiments on NWPU data set show that the details of ground targets recovered by our method can contribute to more accurate classification when given low-resolution inputs. 

## Requirements

- Python 3.6+
- Pytorch 1.0+ 
- ...


## Installation 


## Train


## Test 


## Results


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

