>主题：Attention    
本文标签：Attention


## Attention（2014-Google Mind-Recurrent Models of Visual Attention）
### Attention含义
* 核心逻辑：从关注全部到关注重点
* 注意力机制就是希望网络能够自动学出来图片或者文字序列中的需要注意的地方，其实就是一系列注意力分配系数，也就是一系列权重参数。
* 实现：注意力机制通过神经网络的操作生成一个掩码mask, mask上的值一个打分，评价当前需要关注的点的评分。

## Attention分类
* 注意力机制可以分为：
  * 通道注意力机制：对通道生成掩码mask，进行打分，代表是senet, Channel Attention Module
  * 空间注意力机制：对空间进行掩码的生成，进行打分，代表是Spatial Attention Module
  * 混合域注意力机制：同时对通道注意力和空间注意力进行评价打分，代表的有BAM, CBA
* 目前演化出了两种注意力，一种是软注意力，另一种是抢注意力
  * 软注意力：更关注区域或者通道，是确定性的注意力，学习完成后直接可以通过网络生成，最关键的地方是软注意力是可微的。可以微分的注意力就可以通过神经网络算出梯度并且前向传播和后向反馈来学习得到注意力的权重
  * 强注意力与软注意力不同点在于，首先强注意力是更加关注点，也就是图像中的每个点都有可能延伸出注意力，同时强注意力是一个随机的预测过程，更强调动态变化。最关键是强注意力是一个不可微的注意力，训练过程往往是通过增强学习(reinforcement learning)来完成的。

## Attention在CV中的应用
* CV中早期的Attention，通常是在通道或者空间计算注意力分布，例如：SENet，CBAM。


## Self-Attention（2017-Google-Attention is all you need）
### Attention与Self-Attention的区别
* 一般说的Attention，他的输入Source和输出Target内容是不一样的，比如在翻译的场景中，Source是一种语言，Target是另一种语言，Attention机制发生在Target元素Query和Source中所有元素之间。而Self Attention指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。

### Attention的优点
* 引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征。因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。
* Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，Self Attention对于增加计算的并行性也有直接帮助作用。

### 具体实现
* Self Attention就是Q、K、V均为同一个输入向量映射而来的Encoder-Decoder Attention，它可以无视词之间的距离直接计算依赖关系，能够学习一个句子的内部结构，实现也较为简单并且可以并行计算。
* Self-attention（NLP中往往称为Scaled-Dot Attention）的结构有三个分支：query、key和value。计算时通常分为三步：
  * 第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，cos相似度，拼接，感知机等；
  * 第二步一般是使用一个softmax函数对这些权重进行归一化；
  * 第三步将权重和相应的键值value进行加权求和得到最后的attention。


## Multi-Head Attention
* Multi-Head Attention同时计算多个Attention，并最终得到合并结果，通过计算多次来捕获不同子空间上的相关信息。
