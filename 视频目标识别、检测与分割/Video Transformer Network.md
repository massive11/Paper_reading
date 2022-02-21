>论文标题：Video Transformer Network  
发表时间：2021  
研究组织：Theator    
本文标签：视频目标识别、IEEE/CVF

# 速读概览：
## 1.针对什么问题？ 
    作者认为Video recognition is a perfect candidate for Transformers，因为视频序列和语言建模类似。虽然Transformer在图像领域已经取得了一定的进展，但是Transformer在处理长视频的时候，受自注意力机制的限制，仍然存在计算量大的瓶颈。

## 2.采用什么方法？
    本文提出了基于Transformer的模块化网络VTN，其时间处理组件是基于Longformer的。Longformer using sliding window attention that enables a linear computation complexity，能够处理包含上千个tokens的长序列。Longformer提出的注意力机制使得超越短片处理并保持全局注意力成为可能，它关注输入序列中的所有标记。
    
## 3.达到什么效果？  
    In terms of wall runtime, it trains 16.1× faster and runs 5.1× faster during inference while maintaining competitive accuracy compared to other state-of-the-art methods.
  
## 4.存在什么不足？
    对具有实时检测要求的任务并不适用。



# 论文精读
## 0.Abstract
* 本文提出了VTN，这是针对视频识别的基于Transformer的框架。受视觉Transformer近期发展的启发，我们放弃了依赖 3D ConvNets 的视频动作识别的标准方法，并引入了一种通过关注整个视频序列信息对动作进行分类的方法。我们的方法是通用的，并且建立在任何给定的 2D 空间网络之上。在wall runtime方面，与其他最先进的方法相比，它在推理期间的训练速度提高了 16.1 倍，运行速度提高了 5.1 倍，同时保持了具有竞争力的准确性。它通过单个端到端通道实现整个视频分析，同时需要的 GFLOP 减少 1.5 倍。 我们报告了 Kinetics-400 和 Moments in Time 基准的竞争结果，并介绍了 VTN 属性的消融研究以及准确性和推理速度之间的权衡。我们希望我们的方法将作为一个新的基线，并在视频识别领域开始一条新的研究路线。代码和模型可在https://github.com/bomri/SlowFast/blob/master/projects/vtn/README.md查看。

## 1.Introduction
* Attention matters。近十年来，ConvNets 一直统治着计算机视觉领域。在视觉识别任务中使用ConvNets产生了很多SOTA的结果，如图像分类、目标检测、语义分割、目标实例分割、面部识别和视频动作识别等方面。但是，最近，随着基于Transformer的模型在许多此类任务中显示出可喜的结果，这种统治开始破裂。
* 视频识别任务严重依赖于ConvNets。为了处理时间维度，fundamental方法是使用3D ConvNets。与直接从input clip level添加时间维度的其他研究相比，我们的目标是远离 3D 网络。我们使用SOTA的2D架构来学习空间特征表示，并通过在结果特征之上使用注意力机制添加时间信息在数据流后面。我们的方式输入只有RGB视频帧，没有任何bells and whistles（如optical flow，streams lateral connections，multi-scale inference，multi-view inference， longer clips fine-tuning等），实现了与其他SOTA方法相媲美的结果。
* 视频识别是Transformer的perfect candidate。与语言建模类似，其输入的单词或字符表示为一系列tokens，视频表示为一系列图像（帧）。然而这种相似在处理长序列时也是一种限制。像长文档一样，长视频也很难处理。 即使是 10 秒的视频，例如 Kinetics-400 基准中的视频，在最近的研究中也被处理为短的约 2 秒的剪辑。
* 但是这种基于剪辑的推理如何在更长的视频（即电影、体育赛事或外科手术）上发挥作用？仅使用几秒钟的片段就可以掌握几小时甚至几分钟的视频中的信息似乎违反直觉。 然而，当前的网络并非旨在在整个视频中共享长期信息。
* VTN的时间处理组件是基于Longformer的。这类基于Transformer的模型能够处理包含上千个tokens的长序列。Longformer提出的注意力机制使得超越短片处理并保持全局注意力成为可能，它关注输入序列中的所有标记。
* 在长序列处理之外，我们也探索了机器学习中的一个重要的trade-off——速度和精度。我们的框架在训练和推理时都展示了这种权衡的优越平衡。在训练过程中，尽管每个epoch的wall runtime与其他网络要么相同要么更好，但是我们的方法需要更少的训练数据集就能达到其最大性能；我们的方法是端到端的，与其他SOTA网络相比，这导致了训练过程中16倍以上的加速。在推理时候，我们的方法能够在保持相似的精度的同时解决多视角和完整视频分析的问题。相比之下，其他网络的性能在一次分析完整视频时显着下降。In terms of GFLOPS x Views, their inference cost is considerably higher than those of VTN, which concludes to a 1.5× fewer GFLOPS and a 5.1× faster validation wall runtime.
![avatar](./img/VTN-f1.png)
* 如图1所示，我们的架构结构元件是模块化的。首先，2D spatial components可以被任何给定的网络期待。attention-based的模块可以堆叠更多层、更多head，或者可以设置为可以处理长序列的不同 Transformers 模型。 最后，可以修改分类head以促进不同的基于视频的任务，例如时间动作定位。

## 2.Related work
### Spatial-temporal networks
* 最近在视频识别领域的研究提出了基于3D ConvNets的架构。在文献（Spatiotemporal Residual Networks for Video Action Recognition）中。
  * 使用了双流的架构，一个stream用于RGB输入，另一个用于Optical Flow输入。
  * 残差连接被插入到双流架构中，以允许 RGB 和 OF 层之间的直接链接。
* 将 2D ConvNets 膨胀到它们的 3D 对应物 (I3D) 的想法在 文献（Quo vadis, action recognition? a new model and the kinetics dataset） 中引入。
  * I3D采用 2D ConvNets 并将其层扩展为 3D。因此它允许使用时空域预训练的SOTA的图像识别架构，并将其应用于基于视频的任务。
* Non-local Neural Networks（NLN）引入了non-local操作，也是自注意力的一种，它根据输入信号中不同位置之间的关系计算响应。
  * NLN 证明了 the core attention mechanism in Transformers can produce good results on video tasks, however it is confined to processing only short clips.
* 为了extract long temporal context，文献（Long-term feature banks for detailed video understanding）引入了 a long-term feature bank
  * acts as the entire video memory and a Feature Bank Operator (FBO) that computes interactions between short-term and long- term features.
  * 缺点：需要precomputed features，在特征提取backbone中不够高效以支持端到端训练
* SlowFast探索了一种以两种路径和不同帧速率运行的网络架构。 
  * 横向连接融合了关注空间信息的慢速路径和关注时间信息的快速路径之间的信息。
* The X3D study builds on top of SlowFast
  * 它认为，与通过严格演化开发的图像分类架构相比，视频架构尚未被详细探索，并且在历史上是基于扩展的image-based的网络以适应时间域。
  * X3D 引入了一组网络，这些网络在不同的轴上逐渐扩展，例如时间、帧速率、空间、宽度、瓶颈宽度和深度。 与 SlowFast 相比，它提供了具有相似性能的轻量级网络（在 GFLOPS 和参数方面）。

### Transformers in Computer vision
* Transformer架构在许多NLP任务中实现了SOTA的结果，最近也开始入侵依赖于deep ConvNets的CV领域。
* Studies like
  * ViT and DeiT for image classification
  * DETR for object detection and panoptic segmentation
  * VisTR for video instance segmentation
  are some examples showing promising results when using Transformers in the computer vision field.
* Binding these results with the sequential nature of video makes it a perfect match for Transformers.

### Applying Transformers on long sequences
* BERT and its optimized version RoBERTa are transformer-based language representation models. 
  * They are pre-trained on large unlabeled text and later fine-tuned on a given target task.
  * With minimal modification, they achieve state-of-the- art results on a variety of NLP tasks.
* One significant limitation of these models, and Transformers in general, is their ability to **process long sequences**.
  * Due to the self-attention operation, which has a complexity of O(n2) per layer (n is sequence length) 
* Longformer addresses this problem and enables lengthy document processing by introducing an attention mechanism with a complexity of O(n).
  * This attention mechanism combines a local-context self-attention, performed by a sliding window, and task-specific global attention.
  * 与 ConvNets 类似，堆叠多个windowed attention layers会产生更大的感受野。 Longformer 的这一特性使其能够整合整个序列的信息。 全局注意力部分专注于预先选择的标记（如 [CLS] 标记），并且可以关注输入序列中的所有其他标记。

## 3.Video Transformer Network
* VTN is a generic framework for video recognition. 它使用从帧级别到目标任务头的单个数据流进行操作。 在本研究的范围内，我们通过将输入视频分类到正确的动作类别来展示我们使用动作识别任务的方法。
  ![avatar](img/VTN-f1.png)
* The architecture of VTN is modular and composed of three consecutive parts.图1展示了架构布局。
  * A 2D spatial feature extraction model (spatial backbone), a temporal attention-based encoder, and a classification MLP head.
* VTN 在推理期间的视频长度方面是可扩展的，并且能够处理非常长的序列。 由于内存限制，我们建议几种类型的推理方法。 
  * (1) 以端到端的方式处理整个视频。 
  * (2) 分块处理视频帧，首先提取特征，然后将它们应用于基于时间注意力的编码器。
  * (3) 提前提取所有帧的特征，然后将它们提供给时间编码器。
  
### 3.1 Spatial backbone
* The spatial backbone operates as a learned feature extraction module. It can be any network that works on 2D images, either deep or shallow, pre-trained or not, convolutional- or transformers-based. And its weights can be fixed (pre-trained) or trained during the learning process.

### 3.2 Temporal attention-based encoder
* Inspired by Transformer, we use a Transformer model architecture that applies attention mechanisms to make global dependencies in a sequence data.但是Transformer被他们可以同时处理的tokens数量所限制。这限制了他们处理长输入（例如视频）以及整合远程信息之间的联系的能力。
* 我们提出在推理过程中一次处理整个视频. Use an efficient variant of self-attention, that is not all-pairwise, called Longformer. 
  * Longformer operates using sliding window attention that enables a linear computation complexity. 
  * The sequence of feature vectors of dimension $d_{backbone$ is fed to the Longformer encoder. These vectors act as the 1D tokens embedding in the standard Transformer setup.
* 类似BERT，we add a special classification token ([CLS]) in front of the features sequence.在通过 Longformer 层传播序列后，我们使用与该分类标记相关的特征的最终状态作为视频的最终表示，并将其应用于给定的分类任务头。 Longformer 还保持着对该特殊 [CLS] token的全局关注。

### 3.3 Classification MLP head
* 与文献（ViT）类似，the classification token is processed with an MLP head to provide a final predicted category.The MLP head contains two linear layers with a GELU non-linearity and Dropout between them. The input token representation is first processed with a Layer normalization.

### 3.4 Looking beyond a short clip context
* The common approach in recent studies for video action recognition uses 3D-based networks. 在推理过程中，由于增加了一个时间维度，这些网络受限于内存和运行时间，只能处理小空间尺度和少量帧的剪辑。
* 在文献（Quo vadis, action recognition? a new model and the kinetics dataset）中，作者在推理阶段使用了整个视频，在时间上平均预测。最近取得最先进结果的研究在推理过程中处理了大量但相对较短的剪辑。
* 在文献（Non-local neural networks）中，推理是通过从全长视频中均匀采样 10 个剪辑并平均 softmax 分数来实现最终预测的。 
* SlowFast 遵循相同的做法并引入了术语“view”——具有空间裁剪的时间剪辑。 SlowFast 在推理时使用十个时间剪辑和三个空间裁剪； 因此，最终预测平均有 30 个不同的视图。 
* X3D 遵循相同的做法，但此外，它使用更大的空间尺度在 30 个不同的视图上实现最佳结果。
 ![avatar](img/VTN-f1.png)
* 这种多视图推理的常见做法有些违反直觉，尤其是在处理长视频时。一种更直观的方法是在决定动作之前“查看”整个视频上下文，而不是只查看其中的一小部分。 图 2 显示了从下降类别的视频中均匀提取的 16 帧。 在视频的几个部分中，实际动作模糊不清或不可见； 这可能会导致许多视图中的错误动作预测。 专注于视频中最相关的片段的潜力是一种强大的能力。 然而，完整的视频推理在使用短片训练的方法中会产生较差的性能。 此外，由于硬件、内存和运行时方面的原因，它在实践中也受到限制。

## 4.Video Action Recognition with VTN
* In order to evaluate our approach and the impact of context attention on video action recognition, we use several spatial backbones pre-trained on 2D images.

#### ViT-B-VTN
* Combining the state-of-the-art image classification model, ViT-Base, as the backbone in VTN.
* We use a ViT-Base network that was pre-trained on ImageNet-21K. Using ViT as the backbone for VTN produces an end-to-end transformers-based network that uses attention both for the spatial and temporal domains.

#### R50/101-VTN
* As a comparison, we also use a standard 2D ResNet-50 and ResNet-101 networks, pre-trained on ImageNet.

#### DeiT-B/BD/Ti-VTN
*  Since ViT-Base was trained on ImageNet-21K we also want to compare VTN by using sim- ilar networks trained on ImageNet. We use the recent work of 文献（Training data-efficient image transformers & distillation through at- tention.） and apply DeiT-Tiny, DeiT-Base, and DeiT-Base- Distilled as the backbone for VTN.

### 4.1 Implementation Details
#### Training

#### Inference

## 5.Experiments
### 5.1 Ablation Experiments on Kinetics-400
#### Kinetics-400 dataset

#### Spatial backbone variations

#### Longformer depth

#### Longformer positional embedding

#### Temporal footprint and number of frames in a clip

#### Finetune the 2D spatial backbone

#### Does attention Matter?
* 我们方法的一个关键组成部分是注意力功能对 VTN 感知完整视频序列的方式的影响。为了传达这种影响，我们训练了两个 VTN 网络，使用了 Longformer 中的三层，但每一层都有一个头。In one network the head is trained as usual, while in the second network instead of computing attention based on query/key dot products and softmax, we replace the attention matrix with a hard-coded uniform distribution that is not updated during back-propagation.
* 尽管训练有类似的趋势，但学习到的注意力表现更好。 相比之下，统一注意力的验证在几个时期后就崩溃了，这表明该网络的泛化能力很差。 此外，我们通过使用单头训练网络处理图 2 中的相同视频来可视化 [CLS] 标记注意权重，并在图 3 中描绘了与视频帧对齐的第一个注意层的所有权重。 有趣的是，与下降类别相关的部分的权重要高得多。附录 A 中展示了更多示例。

#### Training and validation runtime

#### Data augmentation
* Recent studies showed that data augmentation significantly improves the performance of transformers-based models.
* We apply extensive data augmentation as suggested in DeiT and RandAugment. Our method reaches 79.8% top-1 accuracy, a 1.2% improvement vs. the same model trained without such augmentations. Training with augmentations requires 10 more epochs but didn’t impact the training wall runtime.

#### Final inference computational complexity

### 5.2 Experiments on Moments in Time

## 6.Conclusion
* 我们为视频识别任务提出了一个基于模块化Transformer的框架。 我们的方法引入了一种有效的方法来大规模评估视频，包括计算资源和wall runtime。 它允许在测试期间进行完整的视频处理，使其更适合处理长视频。 尽管当前的视频分类基准对于测试长期视频处理能力并不理想，但希望在未来，当此类数据集可用时，与 3D ConvNets 相比，VTN 等模型将显示出更大的改进。