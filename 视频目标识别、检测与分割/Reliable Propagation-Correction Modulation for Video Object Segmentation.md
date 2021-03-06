>论文标题：Reliable Propagation-Correction Modulation for Video Object Segmentation  
发表时间：2022  
研究组织：MSRA    
本文标签：视频目标分割、AAAI


# 速读概览：
## 1.针对什么问题？ 

## 2.采用什么方法？  

## 3.达到什么效果？  

## 4.存在什么不足？



# 论文精读
## 0.摘要
* 误差传播在在线半监督视频目标分割领域是一个常见但至关重要的问题。我们力图通过一个高可靠性的校正机制来抑制误差传播。关键的观点是用可靠的线索将校正与传统的掩模传播过程分开。我们引入了两个调制器，传播调制器和校正调制器，分别根据局部时间相关性和可靠参考对目标帧嵌入分别执行通道级别的重新校准。特别地，我们用级联传播校正方案组装调制器。这避免了传播调制器覆盖可靠校正调制器的影响。尽管具有真值标签的参考帧提供了可靠的线索，它可能会与目标帧非常不同，并引入不确定性或不完整的相关性。我们通过向维护池补充可靠的特征补丁来增加参考线索，从而为调制器提供更全面和更具表现力的对象表示。另外，设计了一个可靠的过滤器检索可靠的补丁并将它们传递到后续帧中。我们的模型实现了YouTube-VOS18/19 和 DAVIS17-Val/Test基准上最先进的表现。大量实验表明，通过充分利用可靠的指导，校正机制提供了可观的性能提升。代码：https://github.com/JerryX1110/RPCMVOS.

## 1.Introduction
* 半监督视频目标分割（VOS），也称为掩码跟踪，旨在在给定第一（或参考）帧的真值掩码的情况下分割视频序列中的目标对象。
* 近来，序列到序列的方法实现了惊人的结果，但是具有开销大的缺点。在线方法（online learning）仅将当前帧与图像参考作为输入，对于快速和流式应用非常实用。 我们在本文中只关注在线方法。
* 半监督 VOS 问题通常被表述为最大后验 (MAP) 问题，以目标帧、前导帧和参考帧以及标签为条件。考虑到在线 VOS 的概率模型，当前标签可以从逐帧传播路径或来自可靠参考标签的直接翻译路径（？）中进行预测。为了利用局部时间连续性，很多方法根据传播路径实现掩膜传播，但是由于在每次迭代中不可避免的预测的不确定性，误差将随着时间累积。
* 带有真值标签的参考帧提供了可靠的目标线索，因此具有降低误差传播的潜力。最近的一些方法证明即使通过特征连接和匹配来naively（？）操作参考也可以提高 VOS 性能。（这句话是什么意思不太理解）。这鼓励我们在传播的时候充分利用可靠的参考线索来纠正误差。然而，随着时间的推移目标帧可能会与参考帧非常不同，失去了与参考帧之间明确的对应关系。例如仅包含对象的一部分的参考并不能全面代表整个对象，如参考帧中的脚。在这种情况下估计参考和目标之间的相关性是不确定和不完整的，这可能会对 VOS 任务产生负面影响。
* 使用附加条件重新校准特征嵌入的网络调制在 VOS 中取得了巨大成功。调制是轻量级的，可以按帧执行，满足流媒体要求。 调制的关键是构建富有表现力的条件权重并提取高度相关的嵌入。对于VOS任务，调制权重对参考物体来说要具有代表性，嵌入（embeddings）也需要编码参考与目标之间可靠的对应关系。本文提出了一个针对VOS问题，使用可靠的传播校正调制的新的端到端的框架，它可以为调制提供代表性的目标代理权重，并在传播和校正调制器的级联组件中整合目标嵌入。
* 为了实现高可靠性的校正，我们增加了从参考到更全面的修正路径的翻译路径（？）。由于参考帧中的目标线索可能是不完整的，我们在每次迭代中逐渐使用可靠的信息补充他们。我们还维护一个可靠的补丁记忆池来存储历史可靠的特征补丁，在后续帧中进一步使用。 可靠的补丁池有两种用途，即用综合信息增强目标代理以获得富有表现力的调制权重，以及用更可靠的相关性巩固帧嵌入。
* 在网络设计方面，我们引入了两种类型的调制器，传播调制器和校正调制器，分别在传播路径中中根据局部时间相关线索和校正路径中根据可靠的参考线索增加嵌入。为了避免覆盖校正效应，校正调制器被嵌入到传播调制器之后。我们还提出了一个可靠性过滤器来评估预测质量。 从图 1（b）中的示例可靠性图中，与参考相比具有较大外观变化的区域被预测为不确定，而其他可靠区域可以传递到以下帧进行校正。 实验表明，传播和校正调制器的组装对 VOS 性能有相当大的影响。 
* 本文贡献有三方面：
  * 我们提出了一个针对VOS问题的新的可靠的传播校正调制模型，能够显著抑制误差传播。我们的模型在YouTube-VOS 和 DAVIS17基线上都具有最先进的表现。
  * 我们使用单独的记忆化调制器从传统的错误掩模传播过程中分离出可靠的校正。
  * 我们用全面可靠的线索来增强目标代理和目标嵌入，以加强校正调制。


## 2.Related work
* Propagation-based VOS。基于传播的方法从之前的帧中的语义或空间线索来预测当前帧的掩膜。早起的方法使用在线学习方法来消除飘逸问题，但是耗时严重。光流法和目标跟踪法也被证明是掩膜传播的有效指导。尽管基于传播的模型能够保证好的时间一致性，但这些基于传播的方法容易出现错误累积，这可能会在很大程度上降低 VOS 性能，尤其是对于长视频剪辑。
* Matching-based VOS。基于匹配的方法学习目标物体的嵌入空间。Chen等直接构建当前帧与第一帧之间的对应关系。Lin等进一步利用第一帧和前一帧。一些最新的方法转而使用几个最新的帧来进一步改进局部时间引导。此外，基于STM的网络通过记忆来自过去帧的信息以供进一步重用的记忆网络提高了性能，这在一定程度上缓解了错误传播。 然而，如何减少不确定性传播仍然是一个难题，并没有得到完美解决。 我们的方法通过可靠性来解开引导，并通过精心设计的方案进一步抑制不确定性。
* Conditional modulation。最近，Yang 等介绍了条件调制方法来处理与视频相关的任务，例如基于实例的视频目标分割或空间引导。 然而，视频中掩码传播的调制权重或嵌入不可避免地会出现错误。 为了协同抑制错误传播并充分利用条件调制的力量，我们提出了一种新的条件调制方法，称为可靠条件调制来解决 VOS。

## 3.Preminaries
* 首先回顾一下帧到帧的VOS问题中的概率模型，并从两方面对其进行分析，分别是帧到帧的传播路径和可靠推理中的校正路径。我们还介绍了一种广泛使用的预测可靠性测量方法。

### Probablistic model of frame-to-frame VOS
* 给定直到第t帧所有可用的观察${x_{1:t}}$，直到第t帧的${y_{1:t}}$的标签由最大后验（MAP）估计预测：
  $${p(y_{1:t}|X_{1:t}) = \frac{p(X_{1:t}|y_{1:t})p(y_t)}{p(X_{1:t})} \propto p(X_{1:t}|y_{1:t})p(y_t)}$$

* 此处的${p(X_{1:t}|y_{1:t})}$是观察模型，通常由似然${p(X_{1:t}|D)}$，其中${D}$代表训练数据。${p(y_{1:t-1}|X_{1:t-1})}$是前一帧的后验。${p(y_{1:t})}$是先验模型，可以用一阶马尔科夫假设展开。
