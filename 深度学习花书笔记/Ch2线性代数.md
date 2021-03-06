>书名：深度学习（花书）    
时间：2021年11月1日      
本文标签：深度学习、线性代数  

# 2 线性代数
## 2.1 标量、向量、矩阵和张量
* 张量：一个数组中的元素分布在若干维坐标的规则网格中称为张量。

* 深度学习允许矩阵和向量相加，产生另一个矩阵：**C** = **A** + b，其中
  $$C_{i,j} = A_{i,j} + b_j$$
  即向量b与矩阵A中的每一行相加。这种隐式地复制向量b到很多未知的方式称为**广播**。

## 2.2 矩阵和向量相乘
* 矩阵乘积服从分配律、结合律，但不满足交换律
* 矩阵乘积的转置
  $$(AB)^T = B^TA^T$$

## 2.3 单位矩阵和逆矩阵
* 单位矩阵：所有沿主对角线的元素都是1，而其他位置都是0
* 矩阵的逆称作${A^{-1}}$
  $${A^{-1}A = I_n}$$
  $${AA^{-1} = I_n}$$

## 2.4 线性相关和生成子空间
* 一组向量的 **线性组合** 是指每个向量乘以对应标量系数之后的和，即
  $${\sum_i c_i v^{(i)}}$$
* 一组向量的 **生成子空间（span）** 是原始向量线性组合后所能抵达的点的集合
* 确定${Ax = b}$是否有解，相当于确定向量b是否在A列向量的生成子空间中。这个特殊的生成子空间被称为A的列空间或者A的值域
  <font color=red>（为什么是A列向量不是A行向量？是因为b是列向量吗）</font>
* 线性无关：如果一组向量中的任意一个向量都不能表示成其他向量的线性组合，那么这组向量称为线性无关。
* 否则，若存在冗余，则称为线性相关。
* 奇异的方阵：列向量线性相关
* 如果矩阵A不是一个方针或是一个奇异的方针，该方阵仍可能有解，但不能用矩阵逆去求解

## 2.5 范数
* 在机器学习中，使用范数来衡量向量大小，${L^p}$范数定义如下
  $${\|x\|_p = (\sum_i \vert x_i \vert^p)^{\frac 1 p}}$$

* 范数是将向量映射到非负值的函数。向量x的范数衡量从原点到x的距离。
* 范数是满足下列性质的任意函数：
  *  $${f(x) = 0 \Rightarrow x = 0}$$
  *  $${f(x+y) \le f(x) + f(y)}$$
  *  $${\forall \alpha \in \mathbb{R}, f(\alpha x) = \vert \alpha \vert f(x)}$$
* 当$p=2$时，称为欧几里得范数，表示从原点出发到向量x确定的点的欧几里得距离，经常简化表示为${\|x\|}$
* 平方${L^2}$范数也常用来衡量向量的大小，可以简单的通过点积${x^Tx}$计算。  
  平方${L^2}$范数比${L^2}$范数本身要更方便，对每个元素的倒数只取决于对应的元素，但是在原点附近增长的十分缓慢，此时可以转用${L^1}$范数
* ${L^1}$范数可以简化如下
  $${\|x\|_1 = \sum_i \vert x_i \vert}$$
  通常当机器学习问题中0和非0元素之间的差异非常重要时使用。
* ${L^{\infty}}$范数也称最大范数，表示向量中具有最大幅值的元素的绝对值
  $${\|x\|_\infty = max_i \vert x_i \vert}$$
* 衡量矩阵的大小使用Frobenius范数（类似向量的${L^2}$范数）
  $$\|A\|_F = \sqrt {\sum_{ij} A^2_{i,j}}$$
* 两个向量的点积可以用范数来表示
  $${x^Ty = \|x\|_2 \|y\|_2 cos \theta}$$

## 2.6 特殊类型的矩阵和向量
* 对角矩阵：只要主对角线上含有非零元素，其他位置都是零。用$diag(v)$表示对角元素由向量v中元素给定的一个对角方阵。
* 对角方阵的逆矩阵存在，当且仅当对角元素都是非零值
* 对称矩阵是转置和自己相等的矩阵，即
  $${A = A^T}$$
* 单位向量是具有单位范数的向量，即
  $${\|x\|_2 = 1}$$
* 如果${x^Ty = 0}$，那么向量x和向量y互相**正交**。如果两个向量都有非零范数，那么这两个向量之间的夹角是90度。如果这些向量不但互相正交，且范数都为1，则称他们是**标准正交**。
* 正交矩阵指行向量和列向量是分别标准正交的方阵，即
  $${A^TA = AA^T = I}$$
  这意味着
  $${A^{-1} = A^T}$$

## 2.7 特征分解
* 特征分解是使用最广的矩阵分解之一，即我们将矩阵分解成一组特征向量和特征值。
* 方阵A的特征向量是指与A相乘后相当于对该向量进行缩放的非零向量v：
  $${Av = \lambda v}$$
  标量${\lambda}$称为这个特征向量对应的特征值（事实上是右特征向量）
* 假设矩阵A有n个线性无关的特征向量${\{v^{(1)},...,v^{(n)}\}}$，对应着特征值${\{\lambda_1,...,\lambda_n\}}$。将特征向量连接成一个矩阵，使得每一列是一个特征向量：${V = \{v^{(1)},...,v^{(n)}\}}$。类似地，可以将特征值连成一个向量${\lambda = [\lambda_1,...,\lambda_n]^T}$。因此A的特征分解可以记作
  $${A = V diag(\lambda)V^{-1}}$$
* 并非每一个矩阵都可以分解成特征值和特征向量。本书只讨论实对称矩阵，每个实对称矩阵都可以分解成实对称向量和实特征值：
  $${A = Q\Lambda Q^{-1}}$$
  其中，Q是A的特征向量组成的正交矩阵，${\Lambda}$是对角矩阵。特征值${\Lambda_{i,i}}$对应的特征向量是矩阵Q的第i列，记作${Q_{:,i}}$
* 矩阵是奇异的，当且仅当含有零特征值
* 所有特征值都是正数的矩阵称为正定；
* 所有特征值都是非负数的矩阵称为半正定；
* 所有特征值都是负数的矩阵称为负定；
* 所有特征值都是非正数的矩阵称为半负定；

## 2.8 奇异值分解（SVD）
* 奇异值分解是将矩阵分解为奇异向量和奇异值。
* 每个实数矩阵都有一个奇异值分解，但不一定都有特征分解（非方阵的矩阵没有特征分解）
* 奇异值分解将A分解成三个矩阵的乘积
  $${A = UDV^T}$$
  假设A是一个${m\times n}$的矩阵，那么U是一个${m\times m}$的正交矩阵，D是一个${m\times n}$的对角矩阵（不一定是方阵），V是一个${n\times n}$的正交矩阵。
* 对角矩阵D对角线上的元素称为矩阵A的奇异值。矩阵U的列向量称为左奇异向量，矩阵V的列向量称为右奇异向量。
* A的左奇异向量是${AA^T}$的特征向量，A的右奇异向量是${A^TA}$的特征向量。
* A的非零奇异值是${A^TA}$特征值的平方根，同时也是${AA^T}$特征值的平方根。
* <font color=red>SVD最有用的性质是拓展矩阵求逆到非方矩阵上</font>

## 2.9 Moore-Penrose 伪逆
* 非方矩阵没有逆矩阵的定义
* 定义矩阵A的伪逆为
  $${A^{+} = lim_{\alpha \rightarrow 0}(A^TA + \alpha I)^{-1}A^T}$$
* 实际算法使用公式如下
  $${A^+ = VD^+U^T}$$
  其中，矩阵U、D和V是矩阵A奇异值分解后得到的矩阵。对角矩阵D的伪逆${D^+}$是其非零元素取倒数之后再转置得到的。

## 2.10 迹运算
* 迹运算返回的是矩阵对角元素的和
  $${Tr(A) = \sum_i A_{i,i}}$$
* 迹运算提供了另一种描述矩阵Frobenius范数的方式
  $${\|A\|_F = \sqrt{Tr(AA^T)}}$$

## 2.11 行列式
* 行列式，记作${det(A)}$，是一个将方阵A映射到实数的函数。行列式等于矩阵特征值的乘积。
* 行列式的绝对值可以用来衡量矩阵参与矩阵乘法后空间扩大或缩小了多少。
  （若行列式是0，则失去了所有体积；若为1，则保持空间体积不变）