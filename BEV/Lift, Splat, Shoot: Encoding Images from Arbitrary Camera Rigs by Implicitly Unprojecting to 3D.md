>论文标题：Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D  
发表时间：2020  
研究组织：NVIDIA      
本文标签：3D目标检测、时序模型、ECCV


# 速读概览：
## 1.针对什么问题？ 
    
## 2.采用什么方法？  
    
## 3.达到什么效果？  
    
## 4.存在什么不足？



# 论文精读
## 0.Abstract
* The goal of perception for autonomous vehicles is to extract semantic representations from multiple sensors and fuse these representations into a single “bird’s-eye-view” coordinate frame for consumption by motion planning.
* We directly extracts a bird’s-eye-view representation of a scene given image data from an arbitrary number of cameras.
* The core idea is to “lift” each image individually into a frustum of features for each camera, then “splat” all frustums into a rasterized bird’s-eye-view grid. 
* our model is able to learn not only how to represent images but how to fuse predictions from all cameras into a single cohesive representation of the scene while being robust to calibration error.

## 1.Introduction
* Computer vision algorithms generally take as input an image and output either a prediction that is coordinate-frame agnostic – such as in classification – or a prediction in the same coordinate frame as the input image – such as in object detection, semantic segmentation, or panoptic segmentation. But this paradigm does not match the setting for perception in self-driving out- of-the-box. In self-driving, multiple sensors are given as input, each with a different coordinate frame, and perception models are ultimately tasked with producing predictions in a new coordinate frame – the frame of the ego car – for consumption by the downstream planner
* For the problem of 3D object detection from n cameras, one can apply a single-image detector to all input images individually, then rotate and translate each detection into the ego frame according to the intrinsics and extrinsics of the camera in which the object was detected.
* 