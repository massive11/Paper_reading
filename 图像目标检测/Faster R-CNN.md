>论文标题：Faster R-CNN  
发表时间：2015  
研究组织：Microsoft  
本文标签：Faster RCNN、深度学习、NIPS


# 速读概览：
## 1.针对什么问题？ 
    
## 2.采用什么方法？  
    
## 3.达到什么效果？  
    
## 4.存在什么不足？
    


# 论文精读
## 0.摘要
* 最先进的目标检测网络依赖于region proposal算法来假设目标位置。SPPnet和Fast rcnn提出的改进已经降低了这些检测网络的运行时间，但也暴露了region proposal的计算是一个瓶颈。在本文中，我们介绍了一种Region Proposal Network（RPN），它与检测网络共享全图的卷积特征，使得region proposal几乎是免费的。RPN是一个全卷积网络，能够同时预测每个位置的目标边界和客观分数。RPN是端到端训练的生成高质量的region proposal，然后由Fast R-CNN进行检测。通过共享卷积特征，我们进一步将RPN和Fast R-CNN合并到一个单独的网络中，使用最近很流行的具有注意极致的神经网络术语，RPN组件告诉这个统一的网络要关注哪里。对于非常深的VGG16网络，我们的检测系统在GPU上的速率是5fps，同时在Pascal VOC 2007、2012和MS Coco数据集上实现了最先进的目标检测精度，每幅图像仅有300个proposal。在ILSVRC和COCO2015比赛中，faster R-CNN和RPN是在几个赛道上获得第一名的基础。

## 1.Introduction
* 目标检测领域的最新进展是由region proposal方法和基于region的卷积神经网络的成功所引导的。尽管原本提出的基于region的方法计算代价较高，由于在proposal间共享卷积的方法的出现这个开销已经大幅降低。最新的化身Fast R-CNN在忽略花费在region proposal上的时间的情况下，可以使用非常深的网络实现几乎实时的速度。如今，**proposal在最先进的检测系统中成为测试阶段的计算瓶颈**。
* region proposal方法通常依赖于廉价的特征和经济的推理方案。Selective search作为最流行的方法之一，贪婪地基于工程的低级特征合并超像素。然而，与最高效的检测网络相比，selective search的速度要慢一个数量级，CPU实现的条件下每幅图像需要2秒。EdgeBoxes提供了当下在proposal质量和速度上最好的权衡，速度是每张图像0.2秒。然而，region proposal步骤消耗和检测网络一样多的运行时间。
* 有的观点可能会说速度比较快的基于region的CNN利用了GPU的优势，而研究中使用的region proposal方法是在CPU上实现的，这样对比运行时间是不对等的。比较直观的加速proposal计算的方式是在GPU上重新实现一遍。这可能是个有效的工程上的解决方案，但是重实现忽略了下游的检测网络，因此错过了共享计算的重要机会。
* 在本文中，我们展示了算法上的改变——使用深度卷积网络计算proposal——得到了一种优雅而有效的解决方案，在给定检测网络的计算量的情况下，方案计算几乎是免费的。为此，我们引入了与最先进的目标检测网络、共享卷积层的新型region proposal网络(RPN)。通过在测试阶段共享卷积，计算proposal的边缘成本很小(例如每幅图像10ms)
* 我们观察到基于region的检测器使用的卷积特征图也能用于生成region proposal。在这些卷积特征之上，我们通过添加一些额外的卷积层构建了一个RPN，这些卷积层同时回归规则网格上每个位置的区域边界和客观性分数。因此，RPN是一种完全卷积网络(FCN)，并且可以针对生成检测proposal的任务进行端到端的专门训练
* 