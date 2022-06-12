>论文标题：Learning to Look around Objects for Top-View Representations of Outdoor Scenes  
发表时间：2018  
研究组织：NEC-Labs   
本文标签：自动驾驶、目标检测、高精地图、cross-view、ECCV  


# 速读概览：
## 1.针对什么问题？ 
    

## 2.采用什么方法？  
    

## 3.达到什么效果？  
    

## 4.存在什么不足？
    


# 论文精读
## 0.Abstract
* 给定透视图中复杂户外道路场景的单个 RGB 图像，我们提出了在top-view中估计一个基于遮挡的语义场景布局的新问题。 这个具有挑战性的问题不仅需要准确理解 3D 几何和可见场景的语义，还需要准确理解遮挡区域。
* 我们提出了一个卷积神经网络，它通过观察汽车或行人等前景物体来学习预测场景布局的遮挡部分。 但是，相对于产生幻觉 RGB 值，我们展示了直接预测被遮挡区域的语义和深度，可以更好地转换到top-view。我们进一步表明，通过从模拟数据或如果可用的地图数据中学习有关典型道路布局的先验和规则，可以显着增强这种初始top-view表示。 至关重要的是，训练我们的模型不需要对遮挡区域或top-view进行昂贵或主观的人工注释，而是使用现成的注释进行标准语义分割。 我们广泛评估和分析我们在 KITTI 和 Cityscapes 数据集上的方法。

## 1.Introduction
* 视觉补全是智能体在三维世界中导航和交互的关键能力。一些任务，例如在城市场景中驾驶，或机器人在杂乱的桌子上抓取物体，需要对看不见的区域进行天生的推理。在这种情况下，如果已经解决了场景的top-view或 BEV 中的遮挡关系的表示会很有用。 它是对具有语义和几何一致关系的agents和场景元素的紧凑描述，对于人类可视化来说是直观的，对于自主决策来说是精确的。
* 本文核心目的
  * In this work, we derive such top-view representations through a novel framework that simultaneously reasons about geometry and semantics from just a single RGB image, which we illustrate in the particularly challenging scenario of outdoor road scenes.
* 工作重点
  * The focus of this work lies in the estimation of the scene layout, although foreground objects can be placed on top using existing 3D localization methods. Our learning-based approach estimates a geometrically and semantically consistent spatial layout even in regions hidden behind foreground objects, like cars or pedestrians, without requiring human annotation for occluded pixels or the top-view itself. 请注意，对这种基于遮挡的top-view地图的人工监督可能是主观的，当然，采购成本很高。 相反，我们通过模拟和 OpenStreetMap 数据从透视图中语义分割的现成注释、深度传感器或立体（用于可见区域）和典型道路场景的知识库中获取监督信号。 图 1 提供了一个说明。
![avatar](img/Learning%20to%20Look%20around-f1.png)
* 文章结构
  * In Section 3.1, we propose a novel CNN that takes as input an image with occluded regions (corresponding to foreground objects) masked out, and estimates the segmentation labels and depth values over the entire image, essentially hallucinating distances and semantics in the occluded regions. In contrast to standard image in-painting approaches, we operate in the semantic and depth spaces rather than the RGB image space. Section 3.1 shows how to train this CNN without additional human annotations for occluded regions. The hallucinated depth map is then used to map the hallucinated semantic segmentation of each pixel into the BEV, see Section 3.2.

## 2.Related work

## 3.Generating bird's eye view representations

## 4.Experiments

## 5.Conclusion
* Our work addresses a complex problem in 3D scene understanding, namely, occlusion-reasoned semantic representation of outdoor scenes in the top-view, using just a single RGB image in the perspective view.
* 这需要解决在被前景物体遮挡的区域中产生幻觉语义和几何形状的典型挑战，为此我们提出了一个仅使用透视图像中的标准注释进行训练的 CNN。 此外，我们表明，对抗性和基于扭曲的细化允许利用模拟和地图数据作为有价值的监督信号来学习先验知识。
* Quantitative and qualitative evaluations on the KITTI and Cityscapes datasets show attractive results compared to several baselines. While we have shown the feasibility of solving this problem using a single image, incorporating temporal information might be a promising extension for further gains.
*  We finally note that with the use of indoor data sets like [36,37], along with simulators and floor plans, a similar framework may be derived for indoor scenes, which will be the subject of our future work.