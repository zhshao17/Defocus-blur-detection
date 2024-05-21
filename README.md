# Defocus Blur Detection

This work was co-authored by [Zonghe Shao](https://github.com/zhshao17), [Qichao Wang](https://github.com/solomonWQC), [YuZhe Cao](https://github.com/yuzheCao423), Yijin Gong, Zhuodong Luo, advised by Prof. [Hao Lu](https://sites.google.com/site/poppinace/).

## **Topic**
[Defocus Blur Detection](https://github.com/zhshao17/Defocus-blur-detection) aims to separate in-focus and out-of-focus regions from a single image pixel-wisely. 

## **Method**

We proposed PRNet, based on Encoder-Decoder framework. In the Encoder, [ResNet18](https://arxiv.org/abs/1512.03385) is used for multi-scale image feature extraction and [Patch Attention Module(PAM)](https://arxiv.org/abs/2010.11929) is used to perform local to global attention analysis at different scales.The Decoder consist of embedded Residual Learning and Refinement Module(RLRM), which allows the top-down and bottom-up feature fusion and decodings.


<div  align="center">    
<img src="./img/model.png" alt="Method"  width="500" >
</div>



## **Datasets**

[DUT-DBD dataset](http://ice.dlut.edu.cn/ZhaoWenda/BTBCRLNet.html): Defocus Blur Detection via Multi-Stream Bottom-Top-Bottom Network

[CUHK dataset](https://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html): Discriminative Blur Detection Features

## Result

Comparison with existing work.

<img src="./img/result_datasets.png" alt="Method" weight=50% >

Challenges in some extreme scenarios.

<img src="./img/result_demo.png" alt="Method" weight=50% >
