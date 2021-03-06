>书名：深度学习（花书）    
时间：2021年11月9日      
本文标签：深度学习、正则化

# 7 深度学习中的正则化
* 机器学习中的核心问题是设计不仅在训练数据上表现好，而且能在新输入上泛化好的算法，许多策略被显式地设计来减少测试误差，这些策略统称为正则化。

## 7.1 参数范数惩罚
* 许多正则化方法通过对目标函数 J添加一个参数范数惩罚${\Omega(\theta)}$，限制模型的学习能力。将正则化后的目标函数记为${\widetilde J}$
  $${\widetilde{J}(\theta;X,y) = J(\theta;X,y) + \alpha \Omega(\theta)}$$
  其中，${\alpha \in [0,\infty]}$是权衡范数惩罚项和标准目标函数相对贡献的超参数，其值越大对应正则化惩罚越大
* 通常只对权重做惩罚而不对偏置做正则惩罚。正则化偏执参数可能会导致明显的欠拟合。
* 为了减少搜索空间，会在所有层使用相同的权重衰减。

### 7.1.1 ${L^2}$参数正则化
* ${L^2}$参数范数惩罚通常被称为参数衰减，在其他学术圈也被称为岭回归。
* 这个正则化策略通过向目标函数添加一个正则项${\Omega(\theta) = \frac{1}{2}\| \omega \|^2_2}$，使权重更接近原点
* ${L^2}$正则化能让学习算法感知到具有较高方差的输入x，因此与输出目标的协方差较小（相对增加方差）的特征的权重将会收缩。

### 7.1.2 ${L^1}$参数正则化
* 对模型参数${\omega}$的${L^1}$正则化被定义为
  $${\Omega(\theta) = \| \omega \|_1 = \sum_i |\omega_i|}$$
  即各个参数的绝对值之和
* 相比${L^2}$正则化，${L^1}$正则化会产生更稀疏的解。由${L^1}$正则化导出的稀疏性质已经被广泛地用于特征选择机制。

## 7.2 作为约束的范数惩罚
* 经过参数范数正则化的代价函数可以表示为
  $${\widetilde{J}(\theta;X,y) = J(\theta;X,y) + \alpha \Omega(\theta)}$$
  可以构造一个广义Lagrange函数来最小化带约束的函数，即在原始目标函数上添加一系列惩罚项。每个惩罚是一个被称为Karush-Kuhn-Tucker乘子的系数以及一个表示约束是否满足的函数之间的乘积。若想约束${\Omega(\theta)}$小于某个常数k。可以构建广义Lagrange函数
  $${L(\theta,\alpha;X,y) = J(\theta;X,y) + \alpha(\Omega(\theta) - k)}$$
  该约束问题的解由下式给出
  $${\theta^* = argmin_{\theta}max_{\alpha,\alpha \ge 0}L(\theta,\alpha)}$$
* 使用显式的限制而不是惩罚能够知道什么样的k值是合适的，而不必花时间寻找对应于此k处的${\alpha}$。并且惩罚可能会导致目标函数非凸而使算法陷入局部极小（对应于小的${\theta}$）。通过重投影实现的显式约束不鼓励权重接近原点，只在权重变大并试图离开限制区域时产生作用。此外，还对优化过程增加了一定的稳定性。
* 在实践中，列范数的限制总是通过重投影的显式约束来实现

## 7.3 正则化和欠约束问题
* 大多数形式的正则化能够保证应用于欠定问题的迭代方法收敛