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
* 本文提出了一个叫做Swin Transformer的新的vision Transformer，它可以用作CV领域的通用目的的backbone。将 Transformer 从语言适应到视觉的挑战来自两个领域之间的差异，例如视觉实体的尺度变化很大，以及与文本中的单词相比，图像中像素的高分辨率。为了解决这些差异，我们提出了一个层级式的 Transformer，其表示是用 Shifted windows 计算的。Shifted windows方案通过将 self-attention 计算限制在不重叠的local windows上，同时还允许跨窗口连接，从而带来更高的效率。这种分层架构具有在各种尺度上建模的灵活性，并且计算复杂度与图像大小成线性关系。Swin Transformer 的这些品质使其与广泛的视觉任务兼容，包括图像分类（ImageNet-1K 上 87.3 top-1 准确度）和密集预测任务，例如目标检测（COCO 上 58.7 box AP 和 51.1 mask AP test-dev）和语义分割（ADE20K val 上的 53.5 mIoU）。 它的性能在 COCO 上超过了 +2.7 box AP 和 +2.6 mask AP，在 ADE20K 上超过了 +3.2 mIoU，大大超过了之前的 state-of-the-art，展示了基于 Transformer 的模型作为视觉骨干的潜力。 分层设计和上公开获得。方法也证明对全 MLP 架构有益。 代码和模型可在 https://github.com/microsoft/Swin-Transformer 上公开获得。

## 1.Introduction
* CV领域的模型已经在很长时间内由CNN主导。从AlexNet及其在ImageNet图像分类挑战上的革命性表现开始，通过更大的规模、更广泛的连接和更复杂的卷积形式，CNN 架构已经发展得越来越强大。 随着 CNN 作为各种视觉任务的backbone，这些架构上的进步带来了性能改进，从而广泛提升了整个领域。
* 另一方面，NLP领域的网络架构的发展也发生了很大的变化，如今流行的架构是Transformer。为序列建模和转导任务设计的Transformer由于其使用注意力机制来对数据中的长范围依赖建模而闻名。它在语言领域的巨大成功引领了许多研究者探索它在CV领域的应用，它最近在某些任务上展示了有希望的结果，特别是图像分类和联合视觉-语言建模。
* 在本文中，我们希望可以扩展Transformer的能力以使其可以作为CV领域通用的backbone，如同它在NLP领域和CNN在视觉领域的地位。我们观察到将其在语言领域的高性能迁移到视觉领域的挑战可以用两种模式之间的差异来解释。尺度是这些差异中的一个。与作为语言Transformer中处理的基本元素的word tokens不同，视觉元素在规模上可以有很大的不同，这是在目标检测等任务中受到关注的问题。在现有的基于Transformer的方法中，tokens都是固定的尺度，不适合这些视觉应用的属性。另一个差异在于图像中像素的分辨率要比文本中文章的词高多了。许多视觉任务如语义分割需要在像素级别密集预测，这对于高分辨率图像上的 Transformer 来说是难以处理的，因为其自注意力的计算复杂性与图像大小成二次方。为了克服这个问题，我们提出了一个通用的Transformer架构，称为Swin Transformer，它构建层级式特征图且计算复杂度与小的patches成线性关系。Swin Transformer 通过从小尺寸的patches（灰色轮廓）开始并逐渐合并更深的 Transformer 层中的相邻patches来构建分层表示。借助这些分层特征图，Swin Transformer 模型可以方便地利用高级技术进行密集预测，例如特征金字塔网络 (FPN) 或 U-Net 。

## 5.Conclusion
* 本文提出了Swin Transformer，这是一种引入分层特征表示的vision transformer，相对于输入图像size而言就有线性计算复杂度。Swin Transformer实现了COCO目标检测和ADE20K语义分割上最好的性能。我们希望Swin Transformer在各种视觉任务上的强大表现可以促进视觉和语言信号的通用模型。
* 基于自注意力的Shifted window作为Swin Transformer中的关键元素展示了它在视觉任务上的高效性，我们期待其在NLP领域的应用研究出现。