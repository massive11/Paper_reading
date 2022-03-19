>论文标题：Projecting Your View Attentively: Monocular Road Scene Layout Estimation via Cross-view Transformation  
发表时间：2021  
研究组织：福州大学、 南方科技大学、香港大学等  
本文标签：自动驾驶、语义地图、高精地图、跨视图学习、CVPR


# 速读概览：
## 1.针对什么问题？ 
    HD map重建对自动驾驶而言是非常关键的,基于LiDAR的方法由于部署了昂贵的传感器和耗时的计算受到限制,基于相机的方法通常需要分别进行道路分割和视图变换，这往往会导致失真和内容缺失。

## 2.采用什么方法？  
    本文的目标是在给定单目正视图图像的情况下，以语义masks的形式估计鸟瞰图上的道路场景布局和车辆占用率。在具体的实现方面，在基于 GAN 的框架的生成器中引入了一个cross-view转换模块，该模块增强了提取的视觉特征，用于将frontal view投影到top view。交叉视图变换模块由cycled view projection和cross-view transformer组成。（采用生成式网络实现）

## 3.达到什么效果？  
    We demonstrate that our proposed model can achieve the state-of-the-art performance and run at 35 FPS on a single GPU, which is efficient and applicable for real-time paranoma HD map reconstruction.

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
* 大多数道路场景解析工作都集中在语义分割上，同时也有一些尝试为道路布局推导顶视图表示。 
  * Schulter et al. propose to estimate an occlusion-reasoned road layout on top view from a single color image **by depth estimation and semantic segmentation.**
  * 文献25（Monocular semantic occupancy grid mapping with convolutional variational encoder–decoder networks） proposes a variational autoencoder (VAE) model to predict road layout from a given image, **yet without attempting to reason about the unseen layout from observation.**
  * Pan et al. present a **cross-view semantic segmentation** by transforming and fusing the observation from multiple cameras. 
  * 文献34,36（Lift, splat, shoot、Predicting semantic map representations from images using pyramid occupancy networks）directly transform features from images to 3D space and finally to BEV grids.
* 许多基于单目图像的 3D 车辆检测技术也被提出
  * Several methods handle this problem by **mapping the monocular image to the top view**. 文献37（Orthographic feature transform for monocular 3d object detection） proposes to map a monocular image to the top view representation, and **treats 3D object detection as a task of 2D segmentation**. 
  * BirdGAN also leverages **adversarial learning** for mapping images to BEV.
  * 文献47（Monocular plan view networks for autonomous driving） does not focus on explicit scene layout estimation, focusing instead more on the motion planning side. 
  * Most related to our work, 文献27（Monolayout: Amodal scene layout from a single image） presents a unified model to tackle the task of road layout (static scene) and traffic participant (dynamic scene) estimations from a single image. Differently, we propose an approach to explicitly **model the large view projection and exploits the context information for producing high-quality results.**

### View transformation and synthesis
* 处理交通场景中透视变换的传统方法（文献17、文献23、文献45）
* 基于深度学习的方法
  * 文献56（Generative adversarial frontal view to bird view synthesis）提出了一项开创性的工作，即根据驾驶员的视角生成鸟瞰图。 他们将跨视图合成视为图像翻译任务，并采用基于 GAN 的框架来完成它。 由于难以为真实数据收集注释，他们的模型是从视频游戏数据中训练出来的。
  * 文献1（A geometric approach to obtain a bird’s eye view from an image）专注于将相机图像变形为 BEV 图像，而不执行任何下游任务，例如目标检测。 
  * 最近关于视图合成的尝试旨在将航拍图像转换为街景图像，反之亦然。 与这些工作相比，我们的目的截然不同，不仅需要从正面视图到顶视图的隐式视图投影，还需要在统一框架下估计道路布局和车辆占用率。

### Transformer for vision tasks
* With recent success of the transformer, its **ability of explicitly modeling pair-wise interactions for elements in a sequence** has been leveraged in many vision tasks. 
* Unlike these transformer-based models, our proposed cross-view transformer attempts to establish the correlation between the features of views. In addition, we incorporate a feature selection scheme along with the non-local cross-view correlation scheme, which significantly enhances the representativeness of the features.

## 3.Our Proposed Method
### 3.1 Network Overview
* 本文工作的目标是在给定单目正视图图像的情况下，以语义masks的形式估计鸟瞰图上的道路场景布局和车辆占用率。
* Our network architecture is based on a GAN-based framework, as shown in Fig. 2.
![avatar](img/Projecting%20Your%20View%20Attentively-f2.png)
* **The generator is an encoder-decoder architecture, in which the input frontal view image I is first passed through the encoder that adopts ResNet as the backbone network to extract visual features, then our proposed cross-view transformation module that enhances the features for view projection, and finally the decoder to produce the top-view masks $\hat{M}$.**
![avatar](img/Projecting%20Your%20View%20Attentively-f5.png)
* **On the other hand, we propose a context-aware discriminator (see Fig. 5) that discriminates against the masks of vehicles by taking the road context into account.**

### 3.2 Corss-view Transformation Module
* 传统方法得到的结果无法正确投影不同视角转换中缺失的部分
  * Due to the large gap between frontal views and top views, there exists a large amount of missing image content during view projection, so the traditional view projection techniques lead to defective results. 
* 对于这个问题，有方法利用CNN的能力进行解决，但两种视图上patch-level的相关性在深度网络中建模并非易事。
  * To this end, the hallucination ability of CNN-based methods has been exploited to address the problem, but the patch-level correlation of both views is not trivial to model within deep networks.
* 为了在利用深度网络的能力的同时加强视图相关性，本文在基于 GAN 的框架的生成器中引入了一个交叉视图转换模块，该模块增强了提取的视觉特征，用于将frontal view投影到top view。 我们提出的交叉视图变换模块的结构如图2所示，它由cycled view projection和cross-view transformer组成。
  * In order to strengthen the view correlation while exploiting the capability of deep networks, we introduce a cross-view transformation module into the generator of GAN-based framework, which enhances the extracted visual features for projecting frontal view to top view. The structure of our proposed cross-view transformation module is shown in Fig. 2, which is composed of two parts: cycled view projection and cross-view transformer.
#### Cycled View Projection(CVP)
* 视角差异导致转换中无法对齐，因此使用包含两个全连接层的MLP来投影特征，它可以超越堆叠卷积层的标准信息流
  * Since the features of frontal views are not spatially aligned with the ones of top views due to their large gap, we deploy the MLP structure consisting of two fully-connected layers to project the features of frontal view to top view, which can overtake the standard information flow of stacking convolution layers. As shown in Fig. 2, $X$ and $X'$ represent the feature maps before and after view projection, respectively. Hence, the holistic view projection can be achieved by: $X' = F_{MLP} (X)$, where X refers to the features extracted from the ResNet backbone.
* 引入循环自监督机制巩固视图投影，它将top-view特征投影回frontal views的域。
  * However, such a simple view projection structure cannot guarantee the information of frontal views to be effectively delivered. Here, we introduce a cycled self-supervision scheme to consolidate the view projection, which projects the top-view features back to the domain of frontal views. As illustrated in Fig. 2, $X'$ is computed by cycling $X'$ back to the frontal view domain via the same MLP structure, i.e., $X'' = F'_{MLP} (X')$. To guarantee the domain consistency between $X'$ and $X''$, we incorporate a cycle loss, i.e. Lcycle, as expressed below.
$${L_{cycle} = ||X-X''||_1 \tag{1}}$$
* 循环结构一方面可以先天地提高特征的代表性，因为将顶视图特征循环回正面视图域将加强两个视图之间的连接。其次，当$X$和$X''$的差异不能进一步缩小时，$X''$实际上保留了与视图投影最相关的信息，因为$X''$是从$X'$倒数投影的。
  * The benefits of the cycle structure are two-fold. First, similar to the cycle consistency based approaches, the cycle loss can innately improve the representativeness of features, since cycling back the top-view features to the frontal view domain will strengthen the connection between both views. 
  * Second, when the discrepancy of $X$ and $X''$ cannot be further narrowed down, $X''$ actually retains the most relevant information for view projection, since $X''$ is reciprocally projected from $X'$. 
  * Hence, $X'$ and $X'$ refer to the features before and after view projection. $X''$ contains the most relevant features of the frontal view for view projection.
![avatar](img/Projecting%20Your%20View%20Attentively-f3.png)
* 图3展示了front view 和 top view中特征的可视化
  * In Fig. 3, we show two examples by visualizing the features of the front view and top view. Specifically, the way we visualize them is to select the typical channels of the feature maps (i.e., the 7th and 92nd for two examples of Fig. 3) and align them with the input images. As observed, $X$ and $X''$ are similar, but quite different from $X'$, due to the domain difference. We can also observe that, via cycling, $X''$ concentrates more on the road and the vehicles. $X$, $X'$ and $X''$ will be fed into the cross-view transformer.

#### Cross-View Transformer(CVT)
* CVT 的主要目的是关联视图投影前的特征（即 X ）和视图投影后的特征（即 $X'$ ），以加强后者。 由于$X''$ 包含用于视图投影的正面视图的大量信息，因此它也可以用于进一步增强特征。
  * The main purpose of CVT is to correlate the features before view projection (i.e. X ) and the features after view projection (i.e. $X'$ ) in order to strengthen the latter ones. Since $X''$ contains the substantial information of the frontal view for view projection, it can be involved to further enhance the features as well.
![avatar](img/Projecting%20Your%20View%20Attentively-f4.png)
* CVT 大致可以分为两种方案：明确关联视图特征以实现注意力图 W 来增强 $X'$ 的cross-view关联方案以及从$X''$提取最多的相关信息的特征选择方案。
  * As illustrated in Fig. 4, CVT can roughly be divided into two schemes: the cross-view correlation scheme that explicitly correlates the features of views to achieve an attention map W to strengthen $X'$ as well as the feature selection scheme that extracts the most relevant information from $X''$.
* 具体设计
  * $X$, $X'$, and $X''$ serve as the key K (K ≡ X),the query Q(Q≡$X'$),and the value V (V ≡X′′) of CVT. In our model, the dimensions of $X$, $X'$, and $X''$ are set as the same. $X'$ and X are both flattened into patches, and each patch is denoted as $x'_i \in X'(i ∈ [1,...,hw])$ and $x_j \in X(j ∈ [1,...,hw])$, where hw refers to the width of X times its height. Thus, the relevance matrix R between any pairwise patches of X and $X'$ can be estimated, i.e., for each patch $X'_i$ in $X'$ and $x_j$ in X , their relevance $r_{ij} (∀r_{ij} \in R)$ is measured by the normalized inner-product:
$${r_{ij} = <\frac{x'_i}{||x'_i||}, \frac{x'_j}{||x'_j||}> \tag{2}}$$
* 使用相关矩阵 R，我们创建两个向量 W ($W = {w_i}, \forall i \in [1,...,hw]$) 和 H ($H = {h_i}, \forall i \in [1, ..., hw]$) 分别基于 R 的每一行的最大值和对应的索引：
$$w_i = max_j r_{ij}, \forall r_{ij} \in R, \tag{3}$$
$$h_i = arg max_j r_{ij}, \forall r_{ij} \in R, \tag{4}$$
* W 的每个元素都暗示了$X'$的每个patch与$X$的所有patch之间的相关程度，可以作为注意力图。 H的每个元素表示X中最相关的patch相对于$X'$的每个patch的索引。
* $X$和$X''$都是frontal view的特征，只是 X 包含其完整信息，而 $X''$保留了视图投影的相关信息。 假设 X 和$X'$之间的相关性与$X''$和$X'$之间的相关性相似，那么利用 X 和$X'$的相关性（即 R）从$X''$中提取最重要的信息是合理的。为此，我们引入了一个特征选择机制$F_{fs}$。使用H和W，$F_{fs}$能够通过从$X''$ 中检索最相关的特征来引入新的特征图$T(T={t_i}, \forall i \in [1, ..., hw])$
$$t_i = F_{fs}(X'', h_i), \forall h_i \in H \tag{5}$$
其中，$F_{fs}$从$X''$的$h_i -th$位置检索特征向量$t_i$。
* Hence, T stores the most relevant information of $X''$ for each patch of $X'$. It can be reshaped as the same dimension as $X'$ and concatenated with $X'$ . Then, the concatenated features will be weighted by the attention map W and finally aggregated with $X'$ via a residual structure. To sum up, the process can be formally expressed as below:
$$X_{out} = X' + F_{conv(Concat(X', T))\odot W} \tag{6}$$
其中，$\odot$代表逐元素乘法，$F_{conv}$表示具有 3 × 3 内核大小的卷积层。 $X_out$ 是 CVT 的最终输出，然后将传递给解码器网络以生成顶视图的分割掩码。

### 3.3 Context-aware Discriminator
* 利用车辆与其上下文（即道路）之间的空间关系，可以进一步细化车辆的合成mask。为此，我们提出了一个上下文感知鉴别器，它不仅试图区分输出的车辆掩码和真实的掩码，而且还明确地利用车辆和道路之间的相关性来加强区分。
  * In the discriminator of GAN-based framework, to further refine the synthetic masks of vehicles, the spatial relationship between the vehicles and their context (i.e. road) can be exploited. To accomplish this, we propose a context-aware discriminator that not only attempts to distinguish the output vehicle masks and the ground-truth ones, but also explicitly utilizes the correlation between the vehicles and the roads to strengthen the discrimination.
* 使用同一场景中车辆$\hat{M_v}$的估计mask和道路$M_r$的真实掩码，我们部署共享的 CNN $F_D$ 来分别提取 $\hat{M_v}$ 的特征以及 $\hat{M_v}$ 和 $M_r$ 的连接，然后计算它们的特征的内积来评估它们的相关性，即，
  * Particularly, with the estimated masks of vehicles $\hat{M_v}$ and the ground-truth mask of the road $M_r$ in the same scene, we deploy a shared CNN $F_D$ to separately extract the features of $\hat{M_v}$ and the concatenation of $\hat{M_v}$ and $M_r$, and then calculate the inner-product of their features to evaluate their correlation, i.e.,
$$\hat{C_{v, r}} = <F_D(\hat{M_v}), F_D({\hat{M_v}, M_r})> \tag{7}$$
* 同样，车辆$M_v$的真实掩码以及 $M_v$ 和 $M_r$ 的连接通过具有共享参数的相同网络馈送，然后以相同方式评估真实车辆和道路的相关性。
* To this end, \hat{M_v} and $M_v$ are fed into a classifier $F_D$ for a foreground object discrimination, while the correlations $\hat{C_{v,r}}$ and $C_{v,r}$ are sent into the other classifier $F'_D$ for discrimination. In practice, for both classifiers, we adopt multiple convolutional layers and insert spectral normalization after each layer along with hinge losses for stabilizing training. Thus, the losses of the discriminator are:
$${L_1^D = E[max(0, 1+F_D(\hat{M_v}))] + E[max(0, 1-F_D(M_v))] \tag{8}}$$
$${L_2^D = E[max(0, 1+F'_D(\hat{C_{v,r}}))] + E[max(0, 1-F'_D(C_{v, r}))] \tag{9}}$$
* 因此，我们的上下文感知鉴别器允许我们区分估计的车辆和真实车辆，同时区分车辆和道路之间的各自相关性，这强调了车辆和道路之间的空间关系。

### 3.4 Loss Function
* Overall, the loss function of our framework is defined as:
$${L = L_{BCE}+\lambda L_{cycle} + \beta (L_1^D + L_2^D) \tag{10}}$$
其中，$L_{BCE}$是一种二元交叉熵损失，它作为生成网络的主要目标，以缩小合成语义掩码和地面实况掩码之间的差距。 $\lambda$ 和 $\beta$ 分别是循环损失和对抗损失的平衡权重。 在实践中，$\lambda$ 和 $\beta$ 分别设置为 0.001 和 1。

## 4.Experimental Results
### 4.1 Implementation Details
### 4.2 Datasets and Comparison Methods
### 4.3 Performance Evaluation
### 4.4 Ablation Study
### 4.5 Panorama HD Map Generation

## 5.Conclusion
* We present a novel framework to estimate road layout and vehicle occupancy in top views given a front-view monocular image. 
  * In particular, we propose a cross-view transformation module that is composed of cycled view projection structure and cross-view transformer, in which the features of the views before and after projection are explicitly correlated and the most relevant features for view projection are fully exploited in order to enhance the transformed features. 
  * Besides, we propose a context-aware discriminator that takes into account the spatial relationship of vehicles and roads. 
* We demonstrate that our proposed model can achieve the state-of-the-art performance and run at 35 FPS on a single GPU, which is efficient and applicable for real-time paranoma HD map reconstruction.