# Paper_reading
本仓库用于记录个人学习笔记，主要是自动驾驶的视觉感知相关技术涉及到的书籍和论文，目前着重关注视频目标检测领域。尚处研究初期，论文笔记的整理比较混乱，努力逐步提升中。

# 目录
* [书籍笔记](#book)
  * [深度学习花书笔记](https://github.com/massive11/Paper_reading/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%8A%B1%E4%B9%A6%E7%AC%94%E8%AE%B0)
  * [动手学深度学习V2-李沐(Pytorch)](https://github.com/massive11/learning_dl_from_zero)
* [论文笔记](#paper)
  * [时间序记录](#time)


## 书籍笔记<span id = "book"></span>
* 深度学习花书笔记（2021.11.18 更新Ch10 序列建模：循环和递归网络）
* 动手学深度学习V2-李沐(Pytorch) （2021.12.16 更新16）

***

## 论文笔记<span id = "paper"></span>

***

论文共计31篇，其中包含

| 主题 | 数量 |
| ------ | :------: |
| [视频目标识别、检测与分割](#video) | 5篇 |
| [图像目标识别、检测与分割](#image) | 7篇 |
| [语义分割](#segmentation) | 7篇 |
| [网络结构](#architecture) | 3篇 |
| [强化学习](#reinforcement) | 1篇 |
| [SLAM综述](#SLAM) | 2篇 |
| [深度估计](#estimation) | 1篇 |
| [视觉定位](#localization) | 1篇 |
| [自动驾驶](#driving) | 1篇 |
| [车道线检测](#lane) | 1篇 |
| [语义定位](#semantic_localization) | 1篇 |
| [非监督学习](#unsupervised) | 1篇 |

***


### 视频目标识别、检测与分割<span id = "video"></span>
| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [Deep Feature Flow for Video Recognition](https://github.com/massive11/Paper_reading/blob/master/%E8%A7%86%E9%A2%91%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Deep%20Feature%20Flow%20for%20Video%20Recognition.md) | 2017 | CVPR | USTC MSRA | Xizhou Zhu | 396 | [Article](https://arxiv.org/pdf/1611.07715) |
| [Flow-Guided Feature Aggregation for Video Object Detection](https://github.com/massive11/Paper_reading/blob/master/%E8%A7%86%E9%A2%91%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Flow-Guided%20Feature%20Aggression%20for%20Video%20Object%20Detection.md) | 2017 | ICCV | USTC MSRA | Xizhou Zhu | 385 | [Article](https://arxiv.org/abs/1703.10025) |
| [Towards High Performance Video Object Detection](https://github.com/massive11/Paper_reading/blob/master/%E8%A7%86%E9%A2%91%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Towards%20High%20Performance%20Video%20Object%20Detection.md) | 2018 | CVPR | MSRA | Xizhou Zhu | 157 | [Article](https://arxiv.org/abs/1711.11577) |
| [ViViT: A Video Vision Transformer](https://github.com/massive11/Paper_reading/blob/master/%E8%A7%86%E9%A2%91%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/ViViT:%20A%20Video%20Vision%20Transformer.md) | 2021 | ICCV | Google Research | Anurag Arnab | 110 | [Article](https://arxiv.org/abs/2103.15691) |
| [Reliable Propagation-Correction Modulation for Video Object Segmentation](https://github.com/massive11/Paper_reading/blob/master/%E8%A7%86%E9%A2%91%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Reliable%20Propagation-Correction%20Modulation%20for%20Video%20Object%20Segmentation.md) | 2022 | AAAI | MSRA | Xiaohao Xu | - | [Article](https://arxiv.org/abs/2112.02853) |

***

### 图像目标识别、检测与分割<span id = "image"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [Rich feature hierarchies for accurate object detection and semantic segmentation Tech report](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation%20Tech%20report.md) | 2014 | CVPR | UC Berkeley | R Girshick | 20369 | [Article](https://arxiv.org/abs/1311.2524) |  
| [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Spatial%20Pyramid%20Pooling%20in%20Deep%20Convolutional%20Networks%20for%20Visual%20Recognition.md) | 2015 | SCI | Microsoft | Kaiming He | 7221 | [Article](https://arxiv.org/pdf/1406.4729.pdf) |
| [Fast R-CNN](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Fast%20R-CNN.md) | 2015 | ICCV | Microsoft | R Girshick  | 16908 | [Article](https://arxiv.org/pdf/1504.08083.pdf) | 
| [Faster R-CNN](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Faster%20R-CNN.md) | 2015 | NIPS | Microsoft | Shaoqing Ren | 25019 | [Article](https://arxiv.org/pdf/1506.01497.pdf) |  
| [An Image is Worth 16X16 Words: Transformer for Image Recognition at Scale](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/AN%20IMAGE%20IS%20WORTH%2016X16%20WORDS:%20TRANSFORMERS%20FOR%20IMAGE%20RECOGNITION%20AT%20SCALE.md) | 2021 | ICLR | Google Research | Alexey Dosovitskiy | 2116 | [Article](https://arxiv.org/abs/2010.11929) |
| [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.md) | 2021 | ICCV | MSRA | Ze Liu | 618 | [Article](https://arxiv.org/abs/2103.14030) |
| [Masked Autoencoders Are Scalable Vision Learners](https://github.com/massive11/Paper_reading/blob/master/%E5%9B%BE%E5%83%8F%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E3%80%81%E6%A3%80%E6%B5%8B%E4%B8%8E%E5%88%86%E5%89%B2/Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.md) | 2021 | - |FAIR | Kaiming He | 17 | [Article](https://arxiv.org/abs/2111.06377) |

***

### 语义分割<span id = "segmentation"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Fully Convolutional Networks for Semantic Segmentation | 2015 |CVPR |UC Berkeley | Jonathan Long | 26938 | [Article](https://arxiv.org/abs/1411.4038) |
| Mask R-CNN | 2017 | CVPR | FAIR | Kaiming He | 14842 | [Article](https://arxiv.org/pdf/1703.06870.pdf) |
| Long-term Visual Localization using Semantically Segmented Images | 2018 | ICRA | Chalmers University of Technology | Erik Stenborg | 77 | [Article](http://www.liuxiao.org/wp-content/uploads/2018/08/Long-term-Visual-Localization-using-Semantically-Segmented-Images.pdf) |
| FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution | 2020 | ICRA | SenseTime Group Limited | Zhanpeng Zhang | 9 | [Article](https://arxiv.org/abs/2003.03913) |
| Boosting Real-Time Driving Scene Parsing with Shared Semantics | 2020 |ICRA | SJTU | Zhenzhen Xiang | 3 | [Article](https://arxiv.org/pdf/1909.07038.pdf) |
| SA-LOAM: Semantic-aided LiDAR SLAM with Loop Closure | 2021 | ICRA | Zhejiang University | Lin Li | 1 | [Article](https://arxiv.org/abs/2106.11516) |
| [Road-map: A Light-Weight Semantic Map for Visual Localization towards Autonomous Driving](https://github.com/massive11/Paper_reading/blob/master/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2/Road-map:A%20light-weight%20semantic%20map%20for%20visual%20localization.md) | 2021 | ICRA | HUAWEI | Tong Qin | 3 | [Article](https://arxiv.org/abs/2106.02527) | 

***

### 网络结构<span id = "architecture"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Deep residual learning for image recognition | 2015 | CVPR | Microsoft |Kaiming He | 97068 | [Article](https://arxiv.org/abs/1512.03385) |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | 2015 | ICML | Google | S Ioffe | 31873 | [Article](https://arxiv.org/pdf/1502.03167.pdf) |
| Attention is all you need | 2017 | NIPS | Google | Ashish Vaswani | 33638 | [Article](https://arxiv.org/abs/1706.03762) |

***

### 强化学习<span id = "reinforcement"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [Target-driving Visual Navigation in Indoor Scenes using Deep Reinforment Learning](https://github.com/massive11/Paper_reading/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/Target-driven%20Visual%20Navigation%20in%20Indoor%20Scenes.md) | 2017 | ICRA | Stanford University | Yuke Zhu | 1149 | [Article](https://arxiv.org/abs/1609.05143) |

***

### SLAM综述<span id = "SLAM"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 基于图优化的同时定位与地图创建综述 | 2013 | 机器人 | South China University of Technology  |  梁明杰 | 315 | [Article](http://robot.sia.cn/CN/10.3724/SP.J.1218.2013.00500) |
| 基于单目视觉的同时定位与地图构建方法综述 | 2016 | 计算机辅助设计与图形学学报 | Zhejiang University | 刘浩敏 | 385 | [Article](http://www.cad.zju.edu.cn/home/gfzhang/projects/JCAD2016-SLAM-survey.pdf) |

***

### 深度估计<span id = "estimation"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ClearGrasp:3D Shape Estimation of Transparent Objects for Manipulation | 2020 | ICRA | Google Research | Shreeyak S. Sajjan | 551 | [Article](https://arxiv.org/abs/1910.02550) |

***

### 视觉定位<span id = "localization"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Differentiable Mapping Networks: Learning Structured Map Representations for Sparse Visual Localization | 2020 | ICRA | Google Research | Peter Karkus | 5 |  [Article](https://arxiv.org/abs/2005.09530)  |

***

### 自动驾驶<span id = "driving"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 模型车自动驾驶平台及车道线识别算法设计与实现 | 2021 |  硕士学位论文 |  Zhejiang University | 谢荀 | - | CNKI |

***

### 车道线检测<span id = "lane"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Focus on Local: Detecting Lane Marker from Bottom Up via Key Point | 2021 | CVPR | Noah's Lab, Huawei | Zhan Qu | 4 |  [Article](https://arxiv.org/pdf/2105.13680.pdf) | 

***

### 语义定位<span id = "semantic_localization"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Visual Semantic Localization based on HD Map for Autonomous Vehicles in Urban Scenarios | 2021 |ICRA |Noah's Lab, Huawei |Huayou Wang | - |  [Article](https://ieeexplore.ieee.org/document/9561459) | 

***

## 非监督学习<span id = "unsupervised"></span>

| 论文 | 时间 | 会议 | 研究机构 | 作者 | 引用 | 原文 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Momentum Contrast for Unsupervised Visual Representation Learning | 2020 |CVPR | Facebook AI Research | Kaiming He | 2400 |  [Article](https://arxiv.org/abs/1911.05722) |

***

## 时间序记录<span id = "time"></span>
* [2021年7月](#2107)
* [2021年8月](#2108)
* [2021年9月](#2109)
* [2021年10月](#2110)
* [2021年11月](#2111)
* [2021年12月](#2112)
* [2022年1月](#2201)

## 2021年7月<span id = "2107"></span>
本月共3篇

| 主题 | 论文 | 来源 | 原文 |
| -----| ---- | :----: | :----: |
| 强化学习 | Target-driving Visual Navigation in Indoor Scenes using Deep Reinforment Learning | ICRA 2017 | [Article](https://arxiv.org/abs/1609.05143) |
| 语义分割 | Long-term Visual Localization using Semantically Segmented Images | ICRA 2018 | [Article](http://www.liuxiao.org/wp-content/uploads/2018/08/Long-term-Visual-Localization-using-Semantically-Segmented-Images.pdf) |
| 语义分割 | Road-map: A Light-Weight Semantic Map for Visual Localization towards Autonomous Driving | ICRA 2021 | [Article](https://arxiv.org/abs/2106.02527) |

***
  
## 2021年8月<span id = "2108"></span>
本月共4篇

| 主题 | 论文 | 来源 | 原文 |
| -----| ---- | :----: | :----: |
| 语义分割 | FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution | ICRA 2020 | [Article](https://arxiv.org/abs/2003.03913) |
| 语义分割 | SA-LOAM | ICRA 2021 | [Article](https://arxiv.org/abs/2106.11516) |
| SLAM综述 | 基于单目视觉的同时定位与地图构建方法综述 | CCF-A 2016 | [Article](http://www.cad.zju.edu.cn/home/gfzhang/projects/JCAD2016-SLAM-survey.pdf) |
| SLAM综述 | 基于图优化的同时定位与地图创建综述 | 中文核心 2013 | [Article](http://robot.sia.cn/CN/10.3724/SP.J.1218.2013.00500) |

***

## 2021年9月<span id = "2109"></span>
本月共3篇

| 主题 | 论文 | 来源 | 原文 |
| -----| ---- | :----: | :----: |
| 深度估计 | ClearGrasp:3D Shape Estimation of Transparent Objects for Manipulation | ICRA 2020 | [Article](https://arxiv.org/abs/1910.02550) |
| 视觉定位 | Differentiable Mapping Networks: Learning Structured Map Representations for Sparse Visual Localization | ICRA 2020 | [Article](https://arxiv.org/abs/2005.09530) |
| 自动驾驶 | 模型车自动驾驶平台及车道线识别算法设计与实现 | Master Thesis 2021 | CNKI |

***

## 2021年10月<span id = "2110"></span>
本月共2篇

| 主题 | 论文 | 来源 | 原文 |
| -----| ---- | :----: | :----: |
| 语义分割 | Boosting Real-Time Driving Scene Parsing with Shared Semantics | ICRA 2020 | [Article](https://arxiv.org/pdf/1909.07038.pdf) |
| 车道线检测 | Focus on Local: Detecting Lane Marker from Bottom Up via Key Point | CVPR 2021 | [Article](https://arxiv.org/pdf/2105.13680.pdf) | 

***

## 2021年11月<span id = "2111"></span>
本月共8篇

| 主题 | 论文 | 来源 | 原文 |
| -----| ---- | :----: | :----: |
| 网络结构 | ResNet | CVPR 2015 | [Article](https://arxiv.org/abs/1512.03385) |
| 网络结构 | Batch Normalization | ICML 2015 | [Article](https://arxiv.org/pdf/1502.03167.pdf) |
| 网络模型 | Mask R-CNN | CVPR 2017 | [Article](https://arxiv.org/pdf/1703.06870.pdf) | 
| 视频目标识别、检测与分割 | FGFA | ICCV 2017 | [Article](https://arxiv.org/abs/1703.10025) |
| 图像目标识别、检测与分割 | R-CNN | CVPR 2014 | [Article](https://arxiv.org/abs/1311.2524.pdf) |
| 图像目标识别、检测与分割 | SPPNet | IEEE 2015 | [Article](https://arxiv.org/pdf/1406.4729.pdf) |
| 图像目标识别、检测与分割 | Fast R-CNN | ICCV 2015 | [Article](https://arxiv.org/pdf/1504.08083.pdf) |
| 图像目标识别、检测与分割 | Faster R-CNN | NIPS 2015 | [Article](https://arxiv.org/pdf/1506.01497.pdf) |

***

## 2021年12月<span id = "2112"></span>
本月共3篇

| 主题 | 论文 | 来源 | 原文 |
| ----- | ---- | :----: | :----: |
| 语义分割 | FCN | CVPR 2015 | [Article](https://arxiv.org/abs/1411.4038) |
| 视频目标识别、检测与分割 | Reliable Propagation-Correction Modulation for Video Object Segmentation | AAAI 2022 | [Article](https://arxiv.org/abs/2112.02853) |
| 语义定位 | Visual Semantic Localization based on HD Map for Autonomous Vehicles in Urban Scenarios | ICRA 2021 | [Article](https://ieeexplore.ieee.org/document/9561459) | 

***

## 2022年1月<span id = "2201"></span>
本月共7篇

| 日期 | 主题 | 论文 | 来源 | 原文 |
| :----: | ----- | ---- | :----: | :----: |
| - | 网络结构 | Transformer | NIPS 2017 | [Article](https://arxiv.org/abs/1706.03762)
| - | 视频目标识别、检测与分割 | DFF | CVPR 2017 | [Article](https://arxiv.org/pdf/1611.07715)
| - | 视频目标识别、检测与分割 | Towards High Performance Video Object Detection | CVPR 2018 | [Article](https://arxiv.org/abs/1711.11577)
| - | 图像目标识别、检测与分割 | ViT | ICLR 2021 | [Article](https://arxiv.org/abs/2010.11929)
| - | 图像目标识别、检测与分割 | MAE | arxiv 2021 | [Article](https://arxiv.org/abs/2111.06377)
| 1.18 | 非监督学习 | MoCo | CVPR 2020 | [Article](https://arxiv.org/abs/1911.05722) |
| 1.20 | 图像目标识别、检测与分割 | Swin Transformer | ICCV 2021 | [Article](https://arxiv.org/abs/2103.14030) |

***

## 2022年2月<span id = "2201"></span>
本月共1篇

| 日期 | 主题 | 论文 | 来源 | 原文 |
| :----: | ----- | ---- | :----: | :----: |
| 2.14 | 视频目标识别、检测与分割 | ViViT | ICCV 2021 | [Article](https://arxiv.org/abs/2103.15691)