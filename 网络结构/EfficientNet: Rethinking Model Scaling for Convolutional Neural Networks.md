>论文标题：EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks  
发表时间：2020  
研究组织：Google Research      
本文标签：网络结构、CVPR


# 速读概览：
## 1.针对什么问题？ 
    
## 2.采用什么方法？  
    
## 3.达到什么效果？  
    
## 4.存在什么不足？



# 论文精读
## 0.Abstract
* In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.
* We propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient.
* We use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets.

## 1.Introduction
* In previous work, it is common to scale only one of the three dimensions – depth, width, and image size. Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.
* Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio.
* Based on this observation, we propose a simple yet effective compound scaling method. Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

