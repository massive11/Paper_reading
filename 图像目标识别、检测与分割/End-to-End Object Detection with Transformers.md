>论文标题：End-to-End Object Detection with Transformers  
发表时间：2020 
研究组织：Facebook AI    
本文标签：图像目标识别、ICCV

# 速读概览：
## 1.针对什么问题？ 
    

## 2.采用什么方法？
    
    
## 3.达到什么效果？  
    
  
## 4.存在什么不足？
    



# 论文精读
## 0.Abstract
* 本文提出了一种将目标检测视为直接的set prediction问题的新方法。我们的方法简化了检测流程，有效地消除了对许多手动设计的组件的需求，例如显式编码我们关于任务的先验知识的非极大值抑制过程或anchor generation。新框架的主要成分，称为 DEtection TRansformer 或 DETR，是一个set-based的全局损失，通过二分匹配强制进行独特的预测，以及一个transformer encoder-decoder architecture。 Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors.DETR 在具有挑战性的 COCO 目标检测数据集上展示了与完善且高度优化的 Faster R-CNN 基线相当的准确性和运行时性能。 此外，DETR 可以很容易地推广，以统一的方式产生全景分割。 我们表明它显着优于竞争基线。 训练代码和预训练模型可在 https://github.com/facebookresearch/detr 获得。

## 1.Introduction
* The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest.现代检测器通过在大量proposals、anchors或window centers上定义代理回归和分类问题，以间接方式解决这一set prediction任务。它们的性能受到折叠近乎重复预测的后处理步骤、anchor sets的设计以及将目标框分配给anchor的启发式方法的显着影响。
* To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks.这种端到端的理念在复杂的结构化预测任务（例如机器翻译或语音识别）方面取得了重大进展，但在目标检测方面还没有：以前的尝试要么添加其他形式的先验知识，或者在具有挑战性的基准上没有被证明具有强大的基线竞争力。This paper aims to bridge this gap.
* We streamline the training pipeline by viewing object detection as a direct set prediction problem.We adopt an encoder-decoder architecture based on transformers, a popular architecture for sequence prediction.Transformer的自注意力机制明确地对序列中元素之间的所有成对交互进行建模，使这些架构特别适用于set prediction的特定约束，例如删除重复预测。
![avatar](img/DETR-f1.png)
* 我们的DETR一次预测了所有目标,并使用一组损失函数进行端到端训练，该函数在预测目标和真实目标之间执行二分匹配。DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression.DETR 不需要任何自定义层，因此可以在任何包含标准 CNN 和Transformer类的框架中轻松复制。
* Compared to most previous work on direct set prediction，DETR 的主要特点是二分匹配损失和Transformer与（非自回归）并行解码的结合。作为对比，之前的工作集中在RNNs的自回归decoding。我们的匹配损失函数将预测唯一地分配给真值目标，并且对预测目标的排列保持不变，因此我们可以并行发出它们。
* We evaluate DETR on one of the most popular object detection datasets, COCO, against a very competitive Faster R-CNN baseline.Faster R-CNN 经历了多次设计迭代，其性能自最初发表以来得到了极大的提升。我们的结果表明我们的模型实现了相媲美的结果。
* 更详细的说，DETR收到transformer中non-local computations的影响，在大目标上的表现非常出色。但是在小目标上的表现则稍显逊色。We expect that future work will improve this aspect in the same way the development of FPN did for Faster R-CNN.
* DETR的training settings在许多方面与标准的目标detectors不同。新模型需要超长的训练计划并受益于Transformer中的辅助解码损失。 我们彻底探索了哪些组件对展示的性能至关重要。
* DETR的设计很容易拓展到更多的复杂任务上。在我们的实验中，我们在预先训练的 DETR 之上训练的简单分割头在全景分割上优于竞争基线，这是一项具有挑战性的像素级识别任务，最近受到欢迎。

## 2.Related work
* Our work build on prior work in several domains: bipartite matching losses for set prediction, encoder-decoder architectures based on the transformer, parallel decoding, and object detection methods.

### 2.1 Set Prediction
* 没有规范的深度学习模型可以直接predict sets。基础的set prediction 任务是一个多标签分类问题，其基线方法 one-vs-rest 不适用于诸如检测存在元素之间的底层结构（即，几乎相同的框）。

## 3.The DETR model

## 4.Experiments

## 5.Conclusion
* 本文提出了DETR，一种基于Transformer和二分匹配损失的目标检测系统的新设计，用于直接的set prediction。DETR在具有挑战性的COCO数据集上取得了与优化过的Faster R-CNN基线相媲美的结果。DETR 易于实施，具有灵活的架构，可轻松扩展到全景分割，并具有竞争性结果。 此外，它在大型对象上的性能明显优于 Faster R-CNN，这可能要归功于 self-attention 对全局信息的处理。
* 这种新的detector设计也带来了新的挑战，特别是在小物体的训练、优化和性能方面。 当前的detector需要几年的改进才能解决类似的问题，我们希望未来的工作能够成功解决 DETR 的问题。