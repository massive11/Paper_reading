>论文标题：Differentiable Mapping Networks: Learning Structured Map Representations for Sparse Visual Localization  
发表时间：2020  
研究组织：Google、National University of Singapore  
本文标签：视觉定位、CNN


# 速读概览：
## 1.针对什么问题？ 
    
## 2.采用什么方法？  
    
## 3.达到什么效果？  
    
## 4.存在什么不足？
    


# 论文精读
## 0.摘要
* 从少量的观察中进行建图和定位对机器人来说是一项基本任务。我们通过在一个新颖的神经网络架构：可区分建图网络（DMN）中结合稀疏特征（可微建图）和端到端学习解决了这些任务。DMN构造了一个空间特征的视图嵌入图，并将其用于后续的带有粒子滤波器的视觉定位。由于 DMN 架构是端到端可微的，我们可以使用梯度下降联合学习地图表示和定位。我们将DMN用于稀疏视觉定位，机器人需要定位在一个从已知视图中获取到相对少量图像的新的环境中。我们使用模拟环境和一个具有挑战性的真实世界街景场景的数据集评估了DMN。我们发现DMN学习了用于视觉定位的高效地图表示。空间结构的好处随着更大的环境、更多的制图视点以及训练数据稀缺而增加。项目网页：https:// sites.google.com/view/differentiable-mapping


## 1.介绍
* 