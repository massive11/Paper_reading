# Paper_reading
# 目录
* [书籍笔记](#book)
  * 深度学习花书笔记
  * [动手学深度学习V2-李沐(Pytorch)](https://github.com/massive11/learning_dl_from_zero)
* [论文笔记](#paper)
  * [时间序记录](#time)


## 书籍笔记<span id = "book"></span>
* 深度学习花书笔记（2021.11.18 更新Ch10 序列建模：循环和递归网络）
* 动手学深度学习V2-李沐(Pytorch) （2021.12.16 更新16）

***

## 论文笔记<span id = "paper"></span>
论文共计22篇，其中包含
* [强化学习](#reinforcement) 1篇
* [语义分割](#segmentation) 6篇
* [SLAM综述](#SLAM) 2篇
* [深度估计](#estimation) 1篇
* [视觉定位](#localization) 1篇
* [自动驾驶](#driving) 1篇
* [车道线检测](#lane) 1篇
* [网络结构](#architecture) 2篇
* [网络模型](#model) 1篇
* [视频目标检测与分割](#video) 2篇
* [图像目标检测](#image) 4篇

***

### 强化学习<span id = "reinforcement"></span>
1.[Target-driving Visual Navigation in Indoor Scenes using Deep Reinforment Learning](https://github.com/massive11/Paper_reading/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/Target-driven%20Visual%20Navigation%20in%20Indoor%20Scenes.md)  
    使用深度强化学习的目标驱动的室内视觉导航  
「2017」「ICRA」 「Stanford University」  [[Article](https://arxiv.org/abs/1609.05143)]  

***

### 语义分割<span id = "segmentation"></span>
1.Road-map: A Light-Weight Semantic Map for Visual Localization towards Autonomous Driving.   
    Road-map：面向自动驾驶的视觉定位的轻量级语义地图  
「2021」「ICRA」 「HUAWEI 车BU」  [[Article](https://arxiv.org/abs/2106.02527)]

2.Long-term Visual Localization using Semantically Segmented Images  
    使用语义分割图像的长期视觉定位  
「2018」 「ICRA」 「Chalmers University of Technology」  [[Article]](http://www.liuxiao.org/wp-content/uploads/2018/08/Long-term-Visual-Localization-using-Semantically-Segmented-Images.pdf)

3.FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution.  
    FarSee-Net：使用高效多尺度上下文融合和特征空间超分辨率的实时语义分割  
「2020」 「ICRA」 「SenseTime Group Limited」  [[Article]](https://arxiv.org/abs/2003.03913)

4.SA-LOAM: Semantic-aided LiDAR SLAM with Loop Closure.  
    SA-LOAM：具有闭环的语义辅助的LiDar SLAM  
「2021」 「ICRA」 「Zhejiang University」  [[Article]](https://arxiv.org/abs/2106.11516)

5.Boosting Real-Time Driving Scene Parsing with Shared Semantics  
    使用共享语义促进实时驾驶场景解析  
「2020」「ICRA」「SJTU」「SAIC」   [[Article]](https://arxiv.org/pdf/1909.07038.pdf)

6.Fully Convolutional Networks for Semantic Segmentation  
    用于语义分割的全卷积网络  
「2015」「CVPR」「UC Berkeley」「Jonathan Long」 「Evan Shelhamer」「引用：26938」   [[Article]](https://arxiv.org/abs/1411.4038)

***

### SLAM综述<span id = "SLAM"></span>
1.基于单目视觉的同时定位与地图构建方法综述  
「2016」 「CCF-A」 「Zhejiang University」  [[Article]](http://www.cad.zju.edu.cn/home/gfzhang/projects/JCAD2016-SLAM-survey.pdf)

2.基于图优化的同时定位与地图创建综述  
「2013」 「中文核心」 「South China University of Technology」  [[Article]](http://robot.sia.cn/CN/10.3724/SP.J.1218.2013.00500)

***

### 深度估计<span id = "estimation"></span>
1.ClearGrasp:3D Shape Estimation of Transparent Objects for Manipulation.  
    ClearGrasp：用于操作的透明物体的3D形状估计  
「2020」 「ICRA」 「Google Research」  [[Article]](https://arxiv.org/abs/1910.02550)

***

### 视觉定位<span id = "localization"></span>
1.Differentiable Mapping Networks: Learning Structured Map Representations for Sparse Visual Localization.  
    可微建图网络：学习用于稀疏视觉定位的结构化地图表示  
「2020」「ICRA」「Google Research」  [[Article]](https://arxiv.org/abs/2005.09530)  

***

### 自动驾驶<span id = "driving"></span>
1.模型车自动驾驶平台及车道线识别算法设计与实现  
「2021」  「硕士学位论文」  「Zhejiang University」  

***

### 车道线检测<span id = "lane"></span>
1.Focus on Local: Detecting Lane Marker from Bottom Up via Key Point  
    专注于局部：通过关键点从下到上检测车道标记  
「2021」「CVPR」「Noah's Lab, Huawei」   [[Article]](https://arxiv.org/pdf/2105.13680.pdf)  

***

### 网络结构<span id = "architecture"></span>
1.Deep residual learning for image recognition  
    用于图像识别的深度残差学习（ResNet）  
「2015」「CVPR」「Microsoft」「Kaiming He」「引用：97068」   [[Article]](https://arxiv.org/abs/1512.03385)

2.Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  
    批标准化：通过减少内部协变量偏移来加速深度网络训练  
「2015」「ICML」「Google」「S Ioffe」「引用：31873」   [[Article]](https://arxiv.org/pdf/1502.03167.pdf)

***

### 网络模型<span id = "model"></span>
1.Mask R-CNN  
「2017」「CVPR」「FAIR」「Kaiming He」「引用：14842」   [[Article]](https://arxiv.org/pdf/1703.06870.pdf)  

***

### 视频目标检测<span id = "video"></span>
1.Flow-Guided Feature Aggregation for Video Object Detection  
    用于视频目标检测的流引导特征聚合  
「2017」「ICCV」「USTC」 「Microsoft」「Xizhou Zhu」「引用：385」  [[Article]](https://arxiv.org/abs/1703.10025)  

2.Reliable Propagation-Correction Modulation for Video Object Segmentation  
    视频对象分割的可靠传播校正调制  
「2022」「AAAI」「MSRA」「Xiaohao Xu」「引用：-」  [[Article]](https://arxiv.org/abs/2112.02853)  
***

### 图像目标检测<span id = "image"></span>
1.[Rich feature hierarchies for accurate object detection and semantic segmentation Tech report](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation%20Tech%20report.md)  
    用于准确目标检测和语义分割的丰富特征层次技术报告（R-CNN）  
「2014」「CVPR」「UC Berkeley」「R Girshick」「引用：20369」 [[Article]](https://arxiv.org/abs/1311.2524)  

2.[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/Spatial%20Pyramid%20Pooling%20in%20Deep%20Convolutional%20Networks%20for%20Visual%20Recognition.md)
    用于视觉识别的深度卷积网络的空间金字塔池化  
「2015」「IEEE」「Microsoft」「Kaiming He」「引用：7221」  [[Article]](https://arxiv.org/pdf/1406.4729.pdf)

3.[Fast R-CNN](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/Fast%20R-CNN.md)  
「2015」「ICCV」「Microsoft」「R Girshick」「引用：16908」  [[Article]](https://arxiv.org/pdf/1504.08083.pdf)  


4.Faster R-CNN  
「2015」「NIPS」「Microsoft」「Shaoqing Ren」「Kaiming He」「R Girshick」「引用：25019」  [[Article]](https://arxiv.org/pdf/1506.01497.pdf)  

***

## 时间序记录<span id = "time"></span>
* [2021年7月](#2107)
* [2021年8月](#2108)
* [2021年9月](#2109)
* [2021年10月](#2110)
* [2021年11月](#2111)
* [2021年12月](#2112)

## 2021年7月<span id = "2107"></span>
本月共3篇
  
### 强化学习
1.Target-driving Visual Navigation in Indoor Scenes using Deep Reinforment Learning.  
    使用深度强化学习的目标驱动的室内视觉导航  
「2017」「ICRA」 「Stanford University」  [[Article](https://arxiv.org/abs/1609.05143)]  

### 语义分割
1.Road-map: A Light-Weight Semantic Map for Visual Localization towards Autonomous Driving.   
    Road-map：面向自动驾驶的视觉定位的轻量级语义地图  
「2021」「ICRA」 「HUAWEI 车BU」  [[Article](https://arxiv.org/abs/2106.02527)]

2.Long-term Visual Localization using Semantically Segmented Images  
    使用语义分割图像的长期视觉定位  
「2018」 「ICRA」 「Chalmers University of Technology」  [[Article]](http://www.liuxiao.org/wp-content/uploads/2018/08/Long-term-Visual-Localization-using-Semantically-Segmented-Images.pdf)

***
  
## 2021年8月<span id = "2108"></span>
本月共4篇
  
### 语义分割
1.FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution.  
    FarSee-Net：使用高效多尺度上下文融合和特征空间超分辨率的实时语义分割  
「2020」 「ICRA」 「SenseTime Group Limited」  [[Article]](https://arxiv.org/abs/2003.03913)

2.SA-LOAM: Semantic-aided LiDAR SLAM with Loop Closure.  
    SA-LOAM：具有闭环的语义辅助的LiDar SLAM  
「2021」 「ICRA」 「Zhejiang University」  [[Article]](https://arxiv.org/abs/2106.11516)

### SLAM综述
1.基于单目视觉的同时定位与地图构建方法综述  
「2016」 「CCF-A」 「Zhejiang University」  [[Article]](http://www.cad.zju.edu.cn/home/gfzhang/projects/JCAD2016-SLAM-survey.pdf)

2.基于图优化的同时定位与地图创建综述  
「2013」 「中文核心」 「South China University of Technology」  [[Article]](http://robot.sia.cn/CN/10.3724/SP.J.1218.2013.00500)

***

## 2021年9月<span id = "2109"></span>
本月共3篇
  
### 深度估计
1.ClearGrasp:3D Shape Estimation of Transparent Objects for Manipulation.  
    ClearGrasp：用于操作的透明物体的3D形状估计  
「2020」 「ICRA」 「Google Research」  [[Article]](https://arxiv.org/abs/1910.02550)

### 视觉定位
1.Differentiable Mapping Networks: Learning Structured Map Representations for Sparse Visual Localization.  
    可微建图网络：学习用于稀疏视觉定位的结构化地图表示  
「2020」「ICRA」「Google Research」  [[Article]](https://arxiv.org/abs/2005.09530)  

### 自动驾驶
1.模型车自动驾驶平台及车道线识别算法设计与实现  
「2021」  「硕士学位论文」  「Zhejiang University」  

***

## 2021年10月<span id = "2110"></span>
本月共2篇
  
### 语义分割
1.Boosting Real-Time Driving Scene Parsing with Shared Semantics  
    使用共享语义促进实时驾驶场景解析  
「2020」「ICRA」「SJTU」「SAIC」   [[Article]](https://arxiv.org/pdf/1909.07038.pdf)

### 车道线检测
1.Focus on Local: Detecting Lane Marker from Bottom Up via Key Point  
    专注于局部：通过关键点从下到上检测车道标记  
「2021」「CVPR」「Noah's Lab, Huawei」   [[Article]](https://arxiv.org/pdf/2105.13680.pdf)  

***

## 2021年11月<span id = "2111"></span>
本月共8篇
  
### 网络结构
1.Deep residual learning for image recognition  
    用于图像识别的深度残差学习（ResNet）  
「2015」「CVPR」「Microsoft」「Kaiming He」「引用：97068」   [[Article]](https://arxiv.org/abs/1512.03385)

2.Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  
    批标准化：通过减少内部协变量偏移来加速深度网络训练  
「2015」「ICML」「Google」「S Ioffe」「引用：31873」   [[Article]](https://arxiv.org/pdf/1502.03167.pdf)

### 网络模型
1.Mask R-CNN  
「2017」「CVPR」「FAIR」「Kaiming He」「引用：14842」   [[Article]](https://arxiv.org/pdf/1703.06870.pdf)  

### 视频目标检测
1.Flow-Guided Feature Aggregation for Video Object Detection  
    用于视频目标检测的流引导特征聚合  
「2017」「ICCV」「USTC」 「Microsoft」「Xizhou Zhu」「引用：385」  [[Article]](https://arxiv.org/abs/1703.10025)  

### 图像目标检测
1.Rich feature hierarchies for accurate object detection and semantic segmentation Tech report  
    用于准确目标检测和语义分割的丰富特征层次技术报告（R-CNN）  
「2014」「CVPR」「UC Berkeley」「R Girshick」「引用：20369」  [[Article]](https://arxiv.org/abs/1311.2524.pdf)  

2.Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition  
    用于视觉识别的深度卷积网络的空间金字塔池化(SPPnet)  
「2015」「IEEE」「Microsoft」「Kaiming He」「引用：7221」  [[Article]](https://arxiv.org/pdf/1406.4729.pdf)

3.Fast R-CNN  
「2015」「ICCV」「Microsoft」「R Girshick」「引用：16908」  [[Article]](https://arxiv.org/pdf/1504.08083.pdf)  

4.Faster R-CNN  
「2015」「NIPS」「Microsoft」「Shaoqing Ren」「Kaiming He」「R Girshick」「引用：25019」  [[Article]](https://arxiv.org/pdf/1506.01497.pdf)  


## 2021年12月<span id = "2112"></span>
本月共2篇
  
### 语义分割
1.Fully Convolutional Networks for Semantic Segmentation  
    用于语义分割的全卷积网络  
「2015」「CVPR」「UC Berkeley」「Jonathan Long」 「Evan Shelhamer」「引用：26938」   [[Article]](https://arxiv.org/abs/1411.4038)


### 视频目标分割
2.Reliable Propagation-Correction Modulation for Video Object Segmentation  
    视频对象分割的可靠传播校正调制  
「2022」「AAAI」「MSRA」「Xiaohao Xu」「引用：-」  [[Article]](https://arxiv.org/abs/2112.02853)  