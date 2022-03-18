>论文标题：Projecting Your View Attentively: Monocular Road Scene Layout Estimation via Cross-view Transformation  
发表时间：2021  
研究组织：福州大学、 南方科技大学、香港大学等 
本文标签：自动驾驶、语义地图、高精地图、跨视图学习、CVPR


# 速读概览：
## 1.针对什么问题？ 
    

## 2.采用什么方法？  
    生成式网络

## 3.达到什么效果？  
    

## 4.存在什么不足？
    



# 论文精读
## 0.Abstract
* HD map重建对自动驾驶而言是非常关键的。基于LiDAR的方法由于部署了昂贵的传感器和耗时的计算受到限制。 基于相机的方法通常需要分别进行道路分割和视图变换，这往往会导致失真和内容缺失。为了突破技术的极限，本文提出了一个新的框架，该框架能够在仅给定前视单目图像的情况下，在BEV中重建由道路布局和车辆占用形成的局部地图。特别是，我们提出了一个跨视图转换模块，它考虑到视图之间的循环一致性约束，并充分利用它们的相关性来加强视图转换和场景理解。 考虑到车辆和道路之间的关系，我们还设计了一个上下文感知鉴别器来进一步细化结果。
* 在public benchmarks上的实验表明，我们的方法在道路布局估计和车辆占用估计任务中实现了最先进的性能。 特别是对于后一项任务，我们的模型大大优于所有竞争对手。 此外，我们的模型在单个 GPU 上以 35 FPS 的速度运行，这是高效的，适用于实时全景高清地图重建。

## 1.Introduction
* 本文关注的问题：局部高精地图的建立（道路布局以及 3D 世界中附近车辆的占用情况）
  * With the rapid progress of autonomous driving technologies, many recent efforts have been spent on the related research topics, e.g., scene layout estimation, 3D object detection, vehicle behavior prediction, and lane detection, etc. **Among these tasks, high-definition map (HD map) reconstruction is fundamental and critical for perception, prediction, and planning of autonomous driving.** Its major issues are concerned with the estimation of a local map including the road layout as well as the occupancies of nearby vehicles in the 3D world.
* 现有方法的问题：
  * LiDAR-based方法依赖贵的传感器且需要耗时的计算
    * Existing techniques rely on expensive sensors like LiDAR and require time-consuming computation for cloud point data. 
  * camera-based方法需要分别进行道路分割和视图变换，会导致失真和内容缺失。
    * Besides, the camera-based techniques usually need to separately perform road segmentation and view transformation, which thus causes distortion and the absence of content.
* 本文旨在给定单个单目front-view图像的情况下，以top view 或BEV估计道路布局和车辆占用率
  * To push the limits of the technology, our work aims to address this realistic yet challenging problem of estimating the road layout and vehicle occupancy in top view or bird’s-eye view (BEV), given a single monocular front-view image (see Fig. 1).
![avatar](img/Projecting%20Your%20View%20Attentively-f1.png)
* 从top-view到front-view转换中由于大的视角gap和严重的视野变形存在困难，转换需要充分利用正视图图像的信息和先天推理看不见的区域的能力
  * However, due to the large view gap and severe view deformation, understanding and estimating the top-view scene layout from the front-view image is an extremely difficult problem even for a human observer. Particularly, the same scene has significantly different appearances in the images of bird’s-eye view and frontal view. Thus, **parsing and projecting the road scenes of frontal view to top view require the ability of fully exploiting the information of the frontal view image and innate reasoning the unseen regions.**
* 传统方法侧重于通过估计相机参数和执行图像坐标变换来研究透视变换，但由于几何翘曲引起的 BEV 特征图中的间隙导致结果不佳。
  * Traditional methods focus on investigating the perspective transformation by estimating the camera parameters and performing image coordinate transformation, but gaps in the resulting BEV feature maps caused by geometric warping lead to poor results.
* Deep learning based方法依靠深度CNN的幻觉能力来推断视图之间看不见的区域。这些方法不是对视图之间的相关性进行建模，而是直接利用 CNN 以有监督的方式学习视图投影模型。 这些模型需要深度网络结构来通过多层传播和转换正面视图的特征，以在空间上与顶视图布局对齐。
  * Recent deep learning based approaches mainly rely on the hallucination capability of deep Convolutional Neural Networks to infer the unseen regions between views. In general, instead of modeling the correlation between views, these methods directly leverage CNNs to learn the view projection models in a supervised manner. These models require deep network structures to propagate and transform the features of frontal view through multiple layers to spatially align with the top-view layout. 
* 由于CNN的感受野的局限性，拟合视图投影模型和识别小尺度车辆存在困难。现有的道路场景解析方法通常会忽略车辆和道路之间的空间关系。
  * However, due to the locally confined receptive fields of convolutional layers, it causes the difficulty of fitting a view projection model and identifying the vehicles of small scales. Moreover, road layout provides the crucial context information to infer the position and orientation of vehicles, e.g., vehicles parked alongside the road. **Yet, the prior road scene parsing methods usually ignore the spatial relationship between vehicles and roads.**
* 本文提出了一种GAN-based框架，从单目front-view图像中估计top-view的道路布局和车辆占用率。为了处理视图之间的巨大差异，在生成器网络中提出了一个cross-view转换模块，该模块由两个子模块组成：桥接各自域中的视图特征的Cycled View Projection和关联视图的Cross-View Transformer (CVT) 
  * To address the aforementioned concerns, we derive a novel GAN-based framework to estimate the road layout and vehicle occupancies from top view, given a single monocular front-view image. To handle the large discrepancy between views, we present a cross-view transformation module in the generator network, which is composed of two sub-modules: Cycled View Projection (CVP) module bridges the view features in their respective domains and Cross-View Transformer (CVT) correlates the views, as shown in Fig. 1.
* CVP利用MLP投影视图，超越了通过卷积层的标准信息流，并涉及循环一致性的约束以保留与视图投影相关的特征。将正面视图转换为顶视图需要对视觉特征进行全局空间转换。 然而，标准 CNN 层只允许对特征图进行局部计算，因此需要几个层才能获得足够大的感受野。
  * the CVP utilizes a multi-layer perceptron (MLP) to project views, which overtakes the standard information flow passing through convolutional layers, and involves the constraint of cycle consistency to retain the features relevant for view projection.
  * transforming frontal views to top views requires a global spatial transformation over the visual features. Yet, standard CNN layers only allow local computation over feature maps, which thus takes several layers to obtain a sufficiently large receptive field.
* 全连接层可以更好地促进cross-view转换。 然后，CVT 显式关联从 CVP 获得的投影前后视图的特征，这可以显着增强视图投影后的特征。 特别是，我们在 CVT 中涉及一个特征选择方案，该方案利用两个视图的关联来提取最相关的信息。 此外，为了利用车辆和道路之间的空间关系，我们提出了一个上下文感知判别器，它不仅评估估计的车辆掩码，还评估它们的相关性。
  * Fully connected layers can better facilitate the cross-view transformation. Then, CVT explicitly correlates the features of the views before and after projection obtained from CVP, which can significantly enhance the features after view projection. In particular, we involve a feature selection scheme in CVT which leverages the associations of both views to extract the most relevant information. Furthermore, to exploit the spatial relationship between vehicles and roads, we present a context-aware discriminator that evaluates not only the estimated masks of vehicles but also their correlation.
* 实验结果
  * We show that our cross-view transformation module and the context-aware discriminator can elevate the performance of road layout and vehicle occupancy estimation. For both tasks, we compare our model against the state-of-the-art methods on public benchmarks and demonstrate that our model is superior to all the other methods.
  * For the estimation of vehicle occupancies, our model achieves a significant advantage over the other comparison methods by at least 28.5% in the KITTI 3D Object dataset and by at least 48.8% in the Argoverse dataset. We also show that our framework is able to process 1024 × 1024 images in 35 FPS using a single GPU, and it is applicable for real-time reconstruction of panorama HD map.
* 本文贡献总结：
  * We propose a novel framework that reconstructs a local map formed by top-view road scene layout and vehicle occupancy using a single monocular front-view image only. In particular, we propose a cross-view transformation module which leverages the cycle consistency between views and their correlation to strengthen the view transformation.
  * We also propose a context-aware discriminator that considers the spatial relationship between vehicles and roads in the task of estimating vehicle occupancies.
  * On public benchmarks, it is demonstrated that our model achieves the state-of-the-art performance for the tasks of road layout and vehicle occupancy estimation.


## 2.Related Work
* The related literature on road layout estimation, vehicle detection, and street view synthesis on top view representation and the recent progress of transformers on vision tasks.
### BEV-based Road layout estimation and vehicle detection.
* Most road scene parsing works focus on semantic segmentation, while there are a few attempts that derive top view representation for road layout. 
* Schulter et al. propose to estimate an occlusion-reasoned road layout on top view from a single color image **by depth estimation and semantic segmentation.**
* 文献25（Monocular semantic occupancy grid mapping with convolutional variational encoder–decoder networks） proposes a variational autoencoder (VAE) model to predict road layout from a given image, **yet without attempting to reason about the unseen layout from observation.**
* Pan et al. present a **cross-view semantic segmentation** by transforming and fusing the observation from multiple cameras. 文献34,36（Lift, splat, shoot、Predicting semantic map representations from images using pyramid occupancy networks） directly transform features from images to 3D space and finally to BEV grids.
* 

### View transformation and synthesis
* 

### Transformer for vision tasks

## 3.Our Proposed Method

## 4.Experimental Results

## 5.Conclusion
* We present a novel framework to estimate road layout and vehicle occupancy in top views given a front-view monocular image. In particular, we propose a cross-view transformation module that is composed of cycled view projection structure and cross-view transformer, in which the features of the views before and after projection are explicitly correlated and the most relevant features for view projection are fully exploited in order to enhance the transformed features. Besides, we propose a context-aware discriminator that takes into account the spatial relationship of vehicles and roads. We demonstrate that our proposed model can achieve the state-of-the-art performance and run at 35 FPS on a single GPU, which is efficient and applicable for real-time paranoma HD map reconstruction.