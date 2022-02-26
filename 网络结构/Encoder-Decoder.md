>主题：Encoder-Decoder    
时间：2022年2月24日  
本文标签：Encoder-Decoder


# Encoder-Decoder
* Encoder-Decoder框架是一种文本处理领域的研究模式，并不特指某种具体的算法，是一类算法统称。
* Encoder和Decoder部分可以是任意的文字，语音，图像，视频数据，模型可以采用CNN，RNN，BiRNN、LSTM、GRU等等。基于Encoder-Decoder，我们可以设计出各种各样的应用算法。
* 由此，Encoder-Decoder 有 2 点需要注意：
  * 不管输入序列和输出序列长度是什么，中间的向量c 长度都是固定的，这也是它的一个缺陷。
  * 不同的任务可以选择不同的编码器和解码器 (RNN，CNN，LSTM，GRU)。