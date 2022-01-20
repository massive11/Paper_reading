>论文标题：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows  
发表时间：2021  
研究组织：MSRA    
本文标签：图像目标识别、ICCV
论文讲解：https://www.bilibili.com/video/BV13L4y1475U/?spm_id_from=333.788

# 速读概览：
## 1.针对什么问题？ 
    
## 2.采用什么方法？  
    
## 3.达到什么效果？  
    
## 4.存在什么不足？



# 论文精读
## 0.摘要
* 本文提出了一个叫做Swin Transformer的新的vision Transformer，它可以用作CV领域的通用目的的backbone。将 Transformer 从语言适应到视觉的挑战来自两个领域之间的差异，例如视觉实体的规模变化很大，以及与文本中的单词相比，图像中像素的高分辨率。为了解决这些差异，我们提出了一个分层 Transformer，其表示是用 Shifted windows 计算的。Shifted windows方案通过将 self-attention 计算限制在不重叠的local windows上，同时还允许跨窗口连接，从而带来更高的效率。这种分层架构具有在各种尺度上建模的灵活性，并且具有相对于图像大小的线性计算复杂度。Swin Transformer 的这些品质使其与广泛的视觉任务兼容，包括图像分类（ImageNet-1K 上 87.3 top-1 准确度）和密集预测任务，例如目标检测（COCO 上 58.7 box AP 和 51.1 mask AP test-dev）和语义分割（ADE20K val 上的 53.5 mIoU）。 它的性能在 COCO 上超过了 +2.7 box AP 和 +2.6 mask AP，在 ADE20K 上超过了 +3.2 mIoU，大大超过了之前的 state-of-the-art，展示了基于 Transformer 的模型作为视觉骨干的潜力。 分层设计和移位窗口方法也证明对全 MLP 架构有益。 代码和模型可在 https://github.com/microsoft/Swin-Transformer 上公开获得。。