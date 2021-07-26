# face-forgery-detection

This is the author's personal implementaion of the core two-stream model from **Generalizing Face Forgery Detection with High-frequency Features (CVPR 2021)**. 

For more details, please refer to the original [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Generalizing_Face_Forgery_Detection_With_High-Frequency_Features_CVPR_2021_paper.html)].

## Overview

In this paper, we find that current CNN-based detectors tend to overfit to method-specific color textures and thus fail to generalize. Observing that image noises remove color textures and expose discrepancies between authentic and tampered regions, we propose to utilize the high-frequency noises for face forgery detection.

We carefully devise three functional modules to take full advantage of the high-frequency features. 

- The first is the multi-scale high-frequency feature extraction module that extracts high-frequency noises at multiple scales and composes a novel modality. 
- The second is the residual-guided spatial attention module that guides the low-level RGB feature extractor to concentrate more on forgery traces from a new perspective. 
- The last is the cross-modality attention module that leverages the correlation between the two complementary modalities to promote feature learning for each other. 

The two-stream model is shown as follows.

![image-20210428105010020](img/pipeline.png)

## Dependency

The model is implemented with PyTorch.

Pretrained Xception weights are downloaded from [this link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth).

## Contact

Please contact - LUO Yuchen - 592mcavoy@sjtu.edu.cn

## Citation

```
@InProceedings{Luo_2021_CVPR,
    author    = {Luo, Yuchen and Zhang, Yong and Yan, Junchi and Liu, Wei},
    title     = {Generalizing Face Forgery Detection With High-Frequency Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16317-16326}
}
```

## Notice
Thank you all for using this repo! I've received several emails regarding to the implementation and reproducing issues. Here I list some tips that might be useful :).
- Training and testing are conducted following the specifications in FaceForensics++ [paper](https://arxiv.org/abs/1901.08971).
- The training datasets can be downloaded and created following the official instructions of [FaceForensics++](https://github.com/ondyari/FaceForensics).
- The cross-dataset performance is largely influenced by the training data scale and training time. I found that the GPU version, the training batchsize, and the distributing setting also have impacts on the performance. Please refer to the detailed specifications in the paper.









