>论文标题：Faster R-CNN  
发表时间：2015  
研究组织：Microsoft  
本文标签：Faster RCNN、anchor、NIPS


# 速读概览：
## 1.针对什么问题？ 
* 当下检测网络的主流依然是基于region proposal的方法，但是基于region proposal的方法在计算region的过程中存在耗时长的问题，是目前的瓶颈。
## 2.采用什么方法？  
* 提出与检测网络共享卷积特征的RPN网络，因此两个网络实际上可以组成一个统一的网络，进行端到端的训练，共享卷积特征也能极大的缩短检测时间，同时也进一步提高了精度。
## 3.达到什么效果？  
* 本文提出的检测系统在GPU上的速率是5fps，在Pascal VOC 2007、2012和MS Coco数据集上实现了最先进的目标检测精度，每幅图像仅有300个proposal。
## 4.存在什么不足？
* 使用NMS的方法很容易滤掉被遮挡的目标
* 在RPN和RCNN都会通过超参数来限制正负样本的个数来保证样本的均衡。筛选Proposal得到RoI时候，会选择不超过64个正样本，负样本比例基本满足是正样本的3倍。在RCNN时也最多存在64个正样本进行回归计算，负样本不参与回归。事实上，对于不同任务和数据，这种直接限定正、负样本数量的均衡方法是否都是最有效的也值得研究。（https://blog.csdn.net/qq_41214679/article/details/114595001）


# 论文精读
## 0.摘要
* 最先进的目标检测网络依赖于region proposal算法来假设目标位置。SPPnet和Fast rcnn提出的改进已经降低了这些检测网络的运行时间，但也暴露了region proposal的计算是一个瓶颈。在本文中，我们介绍了一种Region Proposal Network（RPN），它与检测网络共享全图的卷积特征，使得region proposal几乎是免费的。RPN是一个全卷积网络，能够同时预测每个位置的目标边界和客观分数。RPN是端到端训练的生成高质量的region proposal，然后由Fast R-CNN进行检测。通过共享卷积特征，我们进一步将RPN和Fast R-CNN合并到一个单独的网络中，使用最近很流行的具有注意力机制的神经网络术语，RPN组件告诉这个统一的网络要关注哪里。对于非常深的VGG16网络，我们的检测系统在GPU上的速率是5fps，同时在Pascal VOC 2007、2012和MS Coco数据集上实现了最先进的目标检测精度，每幅图像仅有300个proposal。在ILSVRC和COCO2015比赛中，faster R-CNN和RPN是在几个赛道上获得第一名的基础。

## 1.Introduction
* 目标检测领域的最新进展是由region proposal方法和基于region的卷积神经网络的成功所引导的。尽管原本提出的基于region的方法计算代价较高，由于在proposal间共享卷积的方法的出现这个开销已经大幅降低。最新的化身Fast R-CNN在忽略花费在region proposal上的时间的情况下，可以使用非常深的网络实现几乎实时的速度。如今，**proposal在最先进的检测系统中成为测试阶段的计算瓶颈**。
* region proposal方法通常依赖于廉价的特征和经济的推理方案。Selective search作为最流行的方法之一，贪婪地基于工程的低级特征合并超像素。然而，与最高效的检测网络相比，selective search的速度要慢一个数量级，CPU实现的条件下每幅图像需要2秒。EdgeBoxes提供了当下在proposal质量和速度上最好的权衡，速度是每张图像0.2秒。然而，region proposal步骤消耗和检测网络一样多的运行时间。
* 有的观点可能会说速度比较快的基于region的CNN利用了GPU的优势，而研究中使用的region proposal方法是在CPU上实现的，这样对比运行时间是不对等的。比较直观的加速proposal计算的方式是在GPU上重新实现一遍。这可能是个有效的工程上的解决方案，但是重实现忽略了下游的检测网络，因此错过了共享计算的重要机会。
* 在本文中，我们展示了算法上的改变——使用深度卷积网络计算proposal——得到了一种优雅而有效的解决方案，在给定检测网络的计算量的情况下，方案计算几乎是免费的。为此，我们引入了与最先进的目标检测网络、共享卷积层的新型region proposal网络(RPN)。通过在测试阶段共享卷积，计算proposal的边缘成本很小(例如每幅图像10ms)
* 我们观察到基于region的检测器使用的卷积特征图也能用于生成region proposal。在这些卷积特征之上，我们通过添加一些额外的卷积层构建了一个RPN，这些卷积层同时回归规则网格上每个位置的区域边界和客观性分数。因此，RPN是一种完全卷积网络(FCN)，并且可以针对生成检测proposal的任务进行端到端的专门训练
* RPN是被设计使用一定范围内的尺度和纵横比例来高效的预测region proposal。与使用图像金字塔或filter金字塔方法的相关方法相比，我们提出了一种新的anchor box机制，它可以作为多尺度和纵横比例的参考。我们的方案可以被看作是a pyramid of regression references，它避免了枚举多个尺度或纵横比的图像或过滤器。该模型在使用单尺度图像进行训练和测试时表现良好，从而提高了运行速度。
* 为了将RPN和Fast R-CNN统一起来，我们提出了一种训练方案，即在保持proposal固定的同时，交替进行region proposal任务的微调和目标检测任务的微调。这个方案能够快速收敛，并产生一个在两个任务重共享卷积特征的统一的网络。

## 2.Related Work
### Object Proposals
* 广泛使用的目标proposal方法包括基于分组超像素的selective search、CPMC、MCG方法和那些基于滑动窗口的EdgeBoxes等方法。目标proposal方法被用作独立于检测器的外部模块（如selective search、R-CNN、Fast R-CNN）

### 用于目标检测的深度网络
* R-CNN方法端到端的训练CNN对proposal region进行分类。R-CNN主要还是用作分类器，并不对目标边界进行预测，其精度依赖于region proposal模块的表现。
* OverFeat方法训练了全连接层对定位任务中的边框坐标进行预测。全连接层随后又转化为卷积层用于检测特定类别的物体。
* MultiBox方法通过网络生成region proposal，该网络的最后一个全连接层同时预测了多个与类别无关的框，集成了OverFeat的single-vox风格。这些与类别无关的框用于作为R-CNN的proposal。与我们的全卷积方案相比，MultiBox 网络应用于单个图像裁剪或多个大图像裁剪（例如，224×224）。 MultiBox 在proposal和检测网络之间不共享特征。 
* 出于对高效、精准、可视的识别的需求，共享计算的卷积吸引了越来越多的视线。OverFeat从用于分类、定位和检测的图像金字塔中计算卷积特征。 共享卷积特征图上的自适应大小池化 (SPP)被开发用于高效的基于region的对目标检测和语义分割。 Fast R-CNN 支持对共享卷积特征进行端到端检测器训练，并显示出令人信服的准确性和速度。

## 3.Faster R-CNN
* Faster R-CNN包括两个模块。第一个模块是一个深度全卷积网络用于找出region，第二个模块是Fast R-CNN检测器。整个系统是用于目标检测的独立、联合网络。使用最近流行的注意力机制，RPN模块能够告诉Fast R-CNN模块关注哪里。（RPN serves as the attention of the unified network）

### 3.1 Region Proposal Networks
* RPN的输入是任何大小的图像，输出是一系列的矩形目标proposal。每一个都有相应的目标得分。
* 由于我们最终的目标是与Fast R-CNN共享计算，我们假设两个网络共享相同的卷积层设计。
* 为了生成region proposal，我们在最后一个共享卷积层输出的卷积特征图上滑动一个小网络。这个小网络将输入卷积特征图的n × n 空间窗口作为输入。每个滑动窗口都映射到一个低维特征。这个特征随后被送入两个兄弟全连接层——一个边界框回归层和一个边界框分类层。本文中使用n=3，输入图像的有效感受野很大。由于迷你网络以滑动窗口的方式运行，因此所有空间位置共享全连接层。这种架构自然是通过一个 n×n 卷积层和两个兄弟 1×1 卷积层（分别用于 reg 和 cls）来实现的。

#### 3.1.1 Anchors
* 在每个滑动窗口的位置，我们同时预测多个region proposal，每个位置最多产生k个proposal。因此，reg层有4k个输出编码k个框的坐标，cls层要针对每个proposal进行是目标还是不是目标的概率估计，共产生2k个得分。k个proposal是相对于k个参考框进行参数化的，我们称之为anchors。锚点位于相关滑动窗口的中心，并与其尺度和纵横比例相关联。默认设置3种尺度和3个相对比例，在每个滑动位置产生9个anchor。

##### Translation-Invariant Anchors
* 我们的方法的一个重要特性就是它是平移不变的，对anchors和计算相对于anchor的proposal的函数来说都是如此。如果要在一张对象中平移一个物体，那么相同的函数对处于任意位置的proposal都应该能计算出。与之相对，MultiBox方法使用k-means生成了800个anchor，这种方法就不具有平移不变性。故其不能保证对于平移后的物体能生成相同的proposal。
* 这种特性帮助减少了模型的尺寸和参数的量。

##### Multi-Scale Anchors as Regression Reference
* 我们设计的anchor提出了一种新的解决多尺度和纵横比例问题的方案。
* 过去有两种流行的方法进行多尺度的预测。第一种是基于image/feature的金字塔。图像被resized到多种尺度，对每个尺度计算一次特征图。这种方法很有用但也很耗时。第二种方法是在特征图上使用多尺度或多种纵横比例的滑动窗口。第二种方法通常与第一种方法联合使用。
* 而我们采用的是anchor 金字塔方法，使用多尺度和多纵横比例的anchor box作为参考进行分类和边界框回归。它只依赖单一尺度的图像和特征图，并使用单一尺寸的过滤器。
* 多尺度anchor的设计是共享特征的重要组成部分，并且无需为解决多尺度付出额外的代价。

#### 3.1.2 Loss Function
* 为了训练RPN，我们为每个anchor分配了一个二元类标签。对于两种类型的anchor将打上正标签：1.与真值框具有最高IoU的anchor；2.与任何真值框的IoU超过0.7的anchor。单个真值框可能会为多个anchor分配正标签。如果非积极的anchor与所有真值框的 IoU 比率低于 0.3，为其分配负标签。 既不是正面也不是负面的anchor对训练目标没有贡献。
* 损失函数可以定义为两部分之和，即分类损失和回归损失之和。分类损失是两个类（对象与非对象）的对数损失，回归损失使用的是smooth L1.其中回归损失只对有正标签的anchor有效。
* 两项都进行了归一化，并由平衡参数${\lambda}$加权。默认情况下设置 λ = 10，因此 cls 和 reg 项的权重大致相等。
* 过去的一些文献中，对从任意大小的 RoI 汇集的特征执行边界框回归，并且所有区域尺度共享回归权重。 在我们的公式中，用于回归的特征在特征图上具有相同的空间大小 (3 × 3)。 为了考虑不同的大小，学习了一组 k 个边界框回归器。 每个回归器负责一个尺度和一个纵横比，k 个回归器不共享权重。 因此，由于anchor的设计，即使特征具有固定大小/比例，仍然可以预测各种大小的框。

#### 3.1.3 Training RPNs
* RPN能够通过反向传播和随机梯度下降算法进行端到端的训练
* 我们遵循“以图像为中心”的采样策略来训练这个网络。每个 mini-batch 都来自一个包含许多正面和负面示例anchor的图像。 可以针对所有anchor的损失函数进行优化，但这将偏向于负样本，因为它们占主导地位。 相反，我们在图像中随机采样 256 个锚点来计算 mini-batch 的损失函数，其中采样的正负锚点的比例高达 1:1。 如果图像中的正样本少于 128 个，我们用负样本填充小批量。
* 我们通过从标准差为 0.01 的零均值高斯分布中绘制权重来随机初始化所有新层。 所有其他层（即共享卷积层）通过预训练 ImageNet 分类模型进行初始化，这是标准做法。

### 3.2 RPN和Fast R-CNN共享特征
* 独立训练的 RPN 和 Fast R-CNN 都会以不同的方式修改它们的卷积层。 因此，我们需要开发一种技术，允许在两个网络之间共享卷积层，而不是学习两个单独的网络。 我们讨论了三种具有共享特征的训练网络的方法：
  * Alternating training。这个方法是首先训练RPN，然后使用得到的proposal训练Fast R-CNN。然后使用由 Fast R-CNN 调整的网络来初始化 RPN，并迭代此过程。 这是本文所有实验中使用的解决方案。<font color=red>不理解，为什么用Fast R-CNN 调整的网络来初始化 RPN能提高性能？</font>
  * Approximate joint training。在此解决方案中，RPN 和 Fast R-CNN 网络在训练期间合并为一个网络。在每次SGD迭代中，前向传递会生成在训练Fast R-CNN检测器的过程中，被视为固定的、预计算的region proposal。反向传播像往常一样设计，对于共享层，来自 RPN 损失和 Fast R-CNN 损失的反向传播信号被组合在一起。该解决方案易于实施。 但是这个解决方案忽略了导数 w.r.t. proposal框的坐标也是网络响应，所以是近似的。 在我们的实验中，我们凭经验发现该求解器产生了接近的结果，但与交替训练相比，训练时间减少了约 25-50%。<font color=red>没懂，一些长句子不太理解定语状语究竟修饰的哪个部分</font>
  * Non-approximate joint training。RPN预测的边界框也是输入的函数。Fast R-CNN中的RoI池化层将卷积特征和预测的边界框作为输入，所以理论上来说一个有效的反向传播求解器应当包括box坐标的梯度。上面的近似联合训练中忽略了这些梯度。在非近似联合训练解决方案中，我们需要一个相对于box坐标可微分的RoI池化层。
* 4-step Alternating Training。在本文中，我们采用实用的 4 步训练算法通过交替优化来学习共享特征。
  * 第一步我们按照3.1.3中描述的方法训练RPN。这个网络是使用ImageNet预训练的模型和微调过的针对端到端的region proposal任务进行初始化的。
  * 第二步使用第一步中的RPN生成的proposal训练一个单独的基于Fast R-CNN的检测网络。这个检测网络也是由ImageNet初始化的。这时候两个网络还没有共享卷积层。
  * 第三步使用检测网络初始化RPN训练，但是固定了共享卷积层，只微调RPN独有的层。现在这两个网络就共享了卷积层。最后，保持共享卷积层固定，仅微调Fast R-CNN中独有的层。因此，两个网络共享相同的卷积层并形成一个统一的网络。 类似的交替训练可以运行更多次迭代，但我们观察到的改进可以忽略不计。

### 3.3 Implementation Details
* 多尺度特征提取（使用图像金字塔）可以提高精度，但没有表现出良好的速度-精度平衡。
* 对于锚点，我们使用 3 种比例，框区域为 128×128、256×256 和 512×512 像素，以及 1:1、1:2 和 2:1 的 3 种纵横比。 这些超参数不是为特定数据集精心选择的，我们将在下一节中提供有关其效果的消融实验。 正如所讨论的，我们的解决方案不需要图像金字塔或滤波器金字塔来预测多个尺度的区域，从而节省了大量的运行时间。
* 跨越图像边界的锚框需要小心处理。 在训练期间，我们忽略了所有跨边界锚点，因此它们不会导致损失。 对于典型的 1000 × 600 图像，总共将有大约 20000（≈ 60 × 40 × 9）个锚点。 在忽略跨边界锚点的情况下，每张图像大约有 6000 个锚点用于训练。 如果在训练中不忽略跨越边界的异常值，它们会在目标中引入大的、难以纠正的误差项，并且训练不会收敛。 然而，在测试期间，我们仍然将完全卷积的 RPN 应用于整个图像。 这可能会生成跨边界建议框，我们将其裁剪到图像边界。
* 一些 RPN porposal彼此高度重叠。 为了减少冗余，我们根据region proposal的 cls 分数对建议区域采用非最大抑制（NMS）。 我们将 NMS 的 IoU 阈值固定为 0.7，这为每张图像留下了大约 2000 个region proposal。 正如我们将要展示的，NMS 不会损害最终的检测精度，但会大大减少region的数量。 在 NMS 之后，我们使用排名前 N 的region proposal进行检测。 在下面，我们使用 2000 个 RPN 建议训练 Fast R-CNN，但在测试时评估不同数量的建议。

## 4.Experiments

## 5.Conclusion
* 我们提出使用RPN高效准确的生成region proposal。通过与下游的检测网络共享卷积特征，region proposal步骤几乎是cost-free的。我们的方法使统一的、基于深度学习的对象检测系统能够以近乎实时的帧速率运行。 学习到的 RPN 还提高了region proposal的质量，从而提高了整体目标检测的准确性。