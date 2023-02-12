[TOC]



### 1. VoxelMorph: A Learning Framework for Deformable Medical Image Registration (2019 CVPR VoxeMorph)

1. 方法

   ![image-20230127094531717](E:\data\biomedical\registeration\images\markdown\1_1.png)

   - 使用U-net得到flow field,然后使用spatial transformer得到配准图像；
   - ncc loss和梯度平滑loss；
   - 有分割图像作为辅助信息，计算分割图像配准前后的dice作为总loss的一部分。

2. 代码

   - 生成flow field的网络的最后一层使用0均值，1e-5方差的正态分布初始化参数。

     ```python
     self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
     self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
     ```

   - flow field和标准网格相加然后再归一化到（-0.1,0.1）。

     ```python
     def forward(self, src, flow):
         # new locations
         new_locs = self.grid + flow
         shape = flow.shape[2:]
     
         # need to normalize grid values to [-1, 1] for resampler
         for i in range(len(shape)):
             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
     
         # move channels dim to last position
         # also not sure why, but the channels need to be reversed
         if len(shape) == 2:
             new_locs = new_locs.permute(0, 2, 3, 1)
             new_locs = new_locs[..., [1, 0]]
         elif len(shape) == 3:
             new_locs = new_locs.permute(0, 2, 3, 4, 1)
             new_locs = new_locs[..., [2, 1, 0]]
     
         return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
     ```

   - 医学图像处理包 nibabel

3. 问题：

   ~~代码VoxelMorph-torch/Model/model.py/forward()中new_locs = new_locs[..., [2, 1, 0]],为什么要交换x,y,z的位置；~~

### 2. A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with dual consistency constraint (2020 TMI)

1. 动机

   - 先叙述了多对比图像和多对比图像分析的意义，然后指出了传统的多对比分析方法的缺点，最后直接说提出了这个框架。

2. 方法

   <img src="E:\data\biomedical\registeration\images\markdown\2_1.png" style="zoom:150%;" />

   - 先使用一个网络输出仿射变换参数进行仿射变换，然后使用Unet结构输出flow field，然后进行配准操作。

   - 创新点：逆变换场，和两个损失函数。

     - 提出JL loss，即除了similarity（这里是MI）term之外对于图像background的配准结果计算MSE，抑制warped image中处于fixed image background区域中出现object （作者基于磁共振背景的灰度信号接近于0提出了背景抑制损失函数，该函数将f中灰度值小于某个值γ（由数据集据统计得到）的部分特别额外做一个MSE运算（其实也就是相当于让配准后图像对应位置也应该是0）。

       ![2.2](E:\data\biomedical\registeration\images\markdown\2.2.png)

     - 双重一致性损失：变形配准后的图像经过逆变换后和配准前的图像进行MSE或NCC的损失。

   - 总loss等于上述两项loss加上梯度平滑loss（下图）：

     ![1_2](E:\data\biomedical\registeration\images\markdown\1_2.png)

3. 代码

   有，没看完。对于形变场中的一个点（位移向量），如果它的雅克比行列式小于0，则表示该点发生了折叠（folding），代码如下：

   ```python
   def jacobian_determinant(disp):
       """
       jacobian determinant of a displacement field.
       NB: to compute the spatial gradients, we use np.gradient.
       Parameters:
           disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
                 where vol_shape is of len nb_dims
       Returns:
           jacobian determinant
       """
   
       # check inputs
       volshape = disp.shape[:-1]
       nb_dims = len(volshape)
       assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
   
       # compute grid
       grid_lst = nd.volsize2ndgrid(volshape)
       grid = np.stack(grid_lst, len(volshape))
   
       # compute gradients
       J = np.gradient(disp + grid)
   
       # 3D glow
       if nb_dims == 3:
           dx = J[0]
           dy = J[1]
           dz = J[2]
   
           # compute jacobian components
           Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
           Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
           Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
   
           return Jdet0 - Jdet1 + Jdet2
   
       else:  # must be 2
           dfdx = J[0]
           dfdy = J[1]
           return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
   ```

   

4. 问题

   逆形变场的理解。

### 3. A deep learning framework for unsupervised affine and deformable image registration （2019 Medical Image Analysis DLIR）

1. 动机

   作者列举出了一些基于深度学习的方法，指出了这些方法虽然显示了准确的配准性能，但这些方法都是有监督的，即它们依赖于示例配准进行训练或需要手动分割。然后，又叙述了无监督DL方法已经用于光流估计领域。最后作者直接说明了提出了一种深度学习图像配准（DLIR）框架：一种无监督技术，用于训练CNN进行医学图像配准任务。

2. 方法

   <img src="E:\data\biomedical\registeration\images\markdown\3.1.png" style="zoom:150%;" />

   ![image-20230127112039947](E:\data\biomedical\registeration\images\markdown\3.2.png)

   - 仿射变换网络输出12个仿射变换参数：三个平移，三个旋转，三个缩放和三个剪切参数；

   - 按顺序训练，每个阶段都为其特定的注册任务进行训练，同时保持前几个阶段的权重固定；

   - ncc loss和bending energy penalty loss:

     ![3.3](E:\data\biomedical\registeration\images\markdown\3.3.png)

3. 代码

   无

4. 问题

   不能理解bending energy penalty loss。

### 4. Attention for Image Registration (AiR): an unsupervised Transformer approach  (2021 arxiv AiR)

1. 动机

   - Transformer在各个领域展示出了强大的能力，本文试图将Transformer模型引入图像配准领域，提出的框架是第一个基于Transformer的图像配准方法。

2. 方法

   <img src="E:\data\biomedical\registeration\images\markdown\4_1.png" alt="4_1" style="zoom:130%;" />

   

   <img src="E:\data\biomedical\registeration\images\markdown\4_2.png" alt="4_2" style="zoom:130%;" />

   - 将配准视为翻译任务，输入移动图像和固定图像，经过一个一个Transformer输出形变场。具体来说，AiR将固定图像分成一些patch序列，输入到Encoder中；将移动图像分成一些patch序列输入到Decoder中，Transformer整体结构不变，最后输出形变场，经过一个STN网络得到配准图像。
   - 提出了一种多尺度注意力并行Transformer(MAPT)，它可以从不同的感知尺度学习特征。MAPT由N个Transformer（N个解码器和N个编码器）组成。对于每个变压器，他们采用不同大小的patch作为输入，生成N个不同的注意力特征图$F_N$ 。然后将N个特征图采样成统一的大小，并按归一化加权比例相加，得到最终的可变形特征图$F$。感觉这个部分论文中没有讲清楚。
   - 实验有点少，说服力也不够强。

3. 代码

   给出了链接，但无法正常访问。

4. 问题

   无

### 5. BIRNet: Brain image registration using dual-supervised fully convolutional networks (2018 MedIA BIRNet)

1. 动机

   - 与基于深度学习的配准方法相比，文章旨在解决缺少理想groundtrue形变场（有形变场，但不理想）的问题，进而进一步提高配准精度。用其他方法获得的形变场辅助配准的方法。groundtrue形变场用于快速粗配准，图像相似性损失用于细配准。

2. 方法

   ![](E:\data\biomedical\registeration\images\markdown\5_1.png)

   - motivation: 

   - point:

     1. Hierarchical dual-supervision双重监督策略：预测变形场与现有groudtrue变形场之间的差异  *$loss_\phi$* ，从groudtrue形变场中抽取出$24^3,14^3,9^3$三种分辨率的patch  (方法如下图), 与 U-net对应各层输出的形变场计算loss，将它们相加得到 *$loss_\phi$；配准图像与固定图像的差异*$loss_M$* 。

        ![5_21](E:\data\biomedical\registeration\images\markdown\5_2.jpg)

     2. Gap filling：为了提高预测精度，在u型末端之间进一步插入额外的卷积层来连接低级特征和高级特征，即图中绿色部分。

     3. Multi-channel inputs: 差分图和梯度图也被用作网络的输入，与原图像进行拼接。

   - 从图像中抽取出64\*64\*64的patch作为输入，输出24\*24\*24 patch大小的形变场，对应输入patch的中心区域。在对整个图像进行训练或应用网络时，提取重叠的patch，步长为24，即输出patch大小。这样，所有不重叠的输出patch就可以形成整个形变场。

3. 代码

   无

4. 问题

   ~~差分图和梯度图的理解和计算。~~

### 6. End-to-End Unsupervised Deformable Image Registration with a Convolutional Neural Network (2017 DLMIA&ML-CDS&MICCAI DIRNet)


1. 动机

   以前的方法中要么是传统的方法，要么是有监督的深度学习配准方法，作者提出了第一个无监督端到端的可形变的深度学习配准方法。

2. 方法

   ![6_1](E:\data\biomedical\registeration\images\markdown\6_1.png)

   - ConvNet regressor是一个由Conv,Pool,BatchNorm组成的普通的神经网络，输出每个像素点在x和y方向的位移大小，然后经过一个STN网络得到配准图像。

3. 代码

   有，理解

4. 问题

   无

### 7. Inverse-Consistent Deep Networks for Unsupervised Deformable Image Registration (2018 arxiv ICNet)

1. 动机

   ![8_1](E:\data\biomedical\registeration\images\markdown\7_1.png)

   - 现有的大多数算法仅利用空间平滑惩罚来约束变换，这不能完全避免配准映射中的折叠（通常指示错误）。如果使用较大的权值作为平滑约束，过度鼓励待估计流动的局部平滑，如上图（a）所示，获得的配准结果会有全局错误。如果如果使用较小的权值作为平滑约束，学习到的流中会出现大量的折叠，如图（b）所示，从而由于局部缺陷而产生错误配准。如何适当地调整平滑度约束的贡献，同时避免估计流量中的折叠，并保持较高的配准精度是很有挑战性。
   - 以往的研究通常独立地估计从图像A到图像B或从图像B到图像A的变换，因此不能保证这些变换是彼此的逆映射，即忽略了一对图像之间转换的固有逆一致特性。

2. 方法

   ![8_2](E:\data\biomedical\registeration\images\markdown\7_2.png)

   - 为了解决这两个问题，本文提出了一种逆一致深度神经网络（ICNet）用于无监督变形图像配准。上图中两个Fully Convolutional Network实际上是同一个Network。（a）中绿色部分的是$L_{sim}$， 灰色部分是$L_{smo}$，橙色部分是$L_{ant}$，蓝色部分是$L_{inv}$。 $F_{AB}$ 是将图像A配准到图像B的形变场。 $\widetilde{F}_{BA}$是通过Inverse Network得到的形变场。  在Inverse Network中，相当于把$-F_{AB}$当做图像，用$F_{AB}$对$-F_{AB}$进行采样。

   - 提出了一种反向一致约束$L_{inv}$，以鼓励一对图像在多次传递中相互对称变形，直到双向变形的图像被匹配以实现正确的配准。

     ![8_2](E:\data\biomedical\registeration\images\markdown\7_3.png)

   - 提出了一个反折叠约束$L_{ant}$，以避免形变场发生折叠。

     ![](E:\data\biomedical\registeration\images\markdown\7_4.png)

     解释：以下图为例，$i$表示其中一个坐标轴方向（x，y或z），$m+1$是$m$在$i$方向的相邻点，$F_{AB}^i(m)$是作用在点$m$的$i$方向上的位移，$m+F_{AB}^i(m)$表示点$m$位移后的位置，其它符合类似。为了避免发生折叠，点$m$和点$m+1$位移后形成的新的两个点应满足：
     $$
     m+F_{AB}^i(m)<m+1+F_{AB}^i(m+1)\quad\quad \Rightarrow\quad\quad F_{AB}^i(m+1)-F_{AB}^i(m)+1>0
     $$
     点$m$在$i$方向上的梯度定义为：
     $$
     \begin{aligned}\nabla{F_{AB}^i(m)}  & = \frac{F_{AB}^i(m+1)-F_{AB}^i(m)}{(m+1)-m}\quad \\&=F_{AB}^i(m+1)-F_{AB}^i(m) \end{aligned} \tag{2}
     $$
     结合公式（1）和（2）可以得到：
     $$
     \nabla{F_{AB}^i(m)} +1>0 \tag{3}
     $$
     如果公式（3）则表示没有发生折叠，反之，在$m$点发生折叠。

     ![](E:\data\biomedical\registeration\images\markdown\7_5.png)

   - 总的损失函数$L=L_{smi}+\alpha L_{smo}+\beta L_{inv}+\gamma L_{ant}$ ，其中$L_{smo}=\sum_{p \in \Omega}(\parallel \nabla{F_{AB}(p)} \parallel_2^2+\parallel \nabla{F_{BA}(p)} \parallel_2^2)$，$L_{smi}=(\parallel B-\widetilde A \parallel_F^2+\parallel A-\widetilde B \parallel_F^2$，$\widetilde A$和$\widetilde B$表示分别用用$F_{AB}$和$F_{BA}$配准后的图像。

3. 代码：

   有。只看了损失函数部分，基本理解。

4. 问题

   无

### 8. Learning a Deformable Registration Pyramid (2021 MICCAI)

1. 动机

   提出了一种三维变形图像配准方法，灵感来自PWC-Net，一种流行于计算机视觉的二维光流估计的方法。

2. 方法

   ![9_1](E:\data\biomedical\registeration\images\markdown\8_1.png)

   - 图（b）中紫色部分表示有可训练参数，白色部分表示没有可训练参数。

   - 四个层级输出四种不同大小的特征图，level 4的特征图最小。$w_f^{(l)}$ 是输入固定图像得到的特征图，$w_m^{(l)}$ 是输入移动图像得到的特征图。从最上面的层开始每一层都要执行图（b）。$U(\phi^{(l+1)})$是上一层得到的形变场，初始值$U(\phi^{(5)})$为全0的张量。**W**表示warp操作，用得到的形变场$U(\phi^{(l+1)})$ warp特征图$w_m^{(l)}$。**A**表示放射变换网络，输入$w_f^{(l)}$和**W**中得到的特征图，输出12个放射变换参数，上图可能少画了$w_f^{(l)}$指向**W**的箭头。**CV**计算运动图像中的扭曲特征图与固定图像中的特征图之间的相关性，输出一个特征图。**D**是一个3D DenseNet, 输入仿射变换后的特征图、CV得到的特征图和$w_f^{(l)}$，输出形变场$ \phi^{(l)}$ 。重复这个过程。

   - 损失函数：

     ![9_2](E:\data\biomedical\registeration\images\markdown\8_2.png)

3. 代码：

   有，浏览了一部分，不是很理解。

4. 问题：

   **CV**模块的理解和计算没搞清楚。

### 9. Unsupervised 3D End-to-End Medical Image Registration with Volume Tweening Network (2019 JBHI VTN)

1. 动机
   - 受FlowNet 2.0 （一种用在光流估计中的网络）和STN的启发，作者提出了Volume Tweening Network (VTN)，它能够对端到端的CNN进行无监督训练，执行体素级3D医学图像配准。
   - VTN包含了了3个技术组件：（1）级联了注册子网络，这提高了注册大量移位图像的性能，并且没有太大的减速；（2）将仿射配准集成到我们的网络中，这被证明是有效的，比使用单独的工具更快；（3）在训练过程中加入了额外的可逆性损失，从而提高了配准性能。
2. 方法

![10_1](E:\data\biomedical\registeration\images\markdown\9_1.png)

- 首先用一个FCN回归出12个仿射配准参数进行仿射变换，后面接n个级联的U-Net。黄色部分只向第一个子网络传递梯度，蓝色部分向前两个子网络传播梯度。除了级联多个网络，网络本身没有什么大的创新，提出了Invertibility Loss。

- Orthogonality Loss：对于特定任务（医学图像配准），通常情况下，输入图像只需要小的缩放和旋转就可以仿射对齐。我们想对产生过度非刚性变换的网络进行惩罚。为此，我们引入了$I+A$的非正交性的损失，其中$I$表示单位矩阵，$A$表示仿射配准网络产生的变换矩阵（不包含平移项）。
  $$
  L_{ortho}=-6+\sum_{i=1}^3(\lambda_i^2+\lambda_i^{-2}) \tag{1}
  $$
  其中，$\lambda_{1,2,3}$是$I+A$的奇异值。如果$A$具有很小的缩放和旋转，那么$I+A$将接近与$I$，$I$是正交的。当且仅当一个矩阵的所有奇异值都是1时，它是正交的。因此，$I+A$与正交矩阵的偏差越大(即奇异值偏离1越多)，其正交性损失越大。如果$I+A$是正交的，其值将为0。

- Determinant Loss：假设图像具有相同的手性，因此，不允许包含反射的仿射变换。这就要求 $det(I+A)>0$。结合正交性要求，设行列式损失为：
  $$
  L_{det}=(-1+det(A+I))^2
  $$
  因为正交矩阵行列式为正负1，当行列式为-1时代表A存在反射变换？这时$L_{det}$比较大。当$I+A$不存在反射变换且满足正交性时$L_{det}$接近0，否则会变大。

- Invertibility Loss：不理解

  ![10_2](E:\data\biomedical\registeration\images\markdown\9_2.png)

  其中，$f_{12}\star f_{21}=f_{21}+warp(f_{12},f_{21})$。

3. 代码
    无

4. 问题

   Determinant Loss和Invertibility Loss不理解。没有明白网络的训练过程。

### 10. Recursive Cascaded Networks for Unsupervised Medical Image Registration (2019 ICCV)

1. 动机

   - 一些研究也试图叠加多个网络。它们以非递归的方式为每个级联分配不同的任务和输入，并逐个训练它们，但它们的性能在只有少数（不超过3个）级联时接近极限。另一方面，级联在处理不连续和闭塞时可能没有多大帮助。因此，根据直觉，作者认为具有递归架构的级联网络适合可变形配准的设置。
   - 然而，大多数提出的网络被强制进行简单的预测，这被证明是处理复杂变形时的负担，特别是大位移。DLIR和VTN也堆叠它们的网络，尽管它们都局限于少量的级联。DLIR一个接一个地训练每个级联，即在固定前级联的权重之后。VTN联合训练级联，而所有连续扭曲的图像都通过与固定图像的相似度来衡量。这两种训练方法都不允许中间级联逐步注册一对图像。这些非合作级联不考虑其他级联的存在而学习自己的目标，因此即使进行更多的级联，也很难实现进一步的改进。
   - 级联方法已经涉及到计算机视觉的各个领域，例如级联分类器逐步改进了从监督训练数据中学习到的姿态估计，加快了目标检测的过程。
   - 因此，作者提出递归级联体系结构，它鼓励对可以在现有基础网络上构建的无限数量的级联进行无监督训练，以提高技术水平。我们的体系结构与现有级联方法的不同之处在于，我们的每个级联通常将当前扭曲图像和固定图像作为输入，并且仅在最终扭曲图像上测量相似性（与DLIR，VTN相反），使所有级联能够协同学习渐进对齐。

2. 方法

   ![10_1](E:\data\biomedical\registeration\images\markdown\10_1.png)

   - 最终的预测可以被认为是递归预测的流场的组合，而每个级联只需要学习一个简单的小位移对齐，可以通过更深的递归来细化，如下图所示。

     ![10_2](E:\data\biomedical\registeration\images\markdown\10_2.png)

   - 每一个子网络都是类U-net网络。

   - 在递归过程中可以重复应用一个级联，也就是说，多个级联可以使用相同的参数共享，这被称为共享权重级联。在每个级联之后立即插入一个或多个共享权重级联，即通过将每个$f_k$替换为$n$倍的$f_k$来构造总共$r*n$级联。这种方法在实验中被证明是有效的。当输出流场的质量可以通过进一步细化来提高时，测试过程中的共享权重级联是一种选择。然而，z作者注意到这种技术并不总是得到积极的增益，并可能导致过度变形。递归级联只能确保扭曲的运动图像与固定图像之间的相似性不断增加，但如果图像过于完美匹配，聚合流场就会变得不那么自然。在训练中不使用共享权重级联的原因是，在使用的平台（Tensorflow）的梯度反向传播过程中，共享权重级联消耗的GPU内存与非共享权重级联一样大。要训练的级联的数量受到GPU内存的限制，但当数据集足够大以避免过拟合时，如果允许学习不同的参数，它们会表现得更好。

   - 理论上，递归级联网络保持图像拓扑，只要每个子形变场都保持。然而，目前提出的方法中，折叠区域是常见的，并且在递归过程中可能会被放大，这给权值共享技术的使用带来了挑战。通过仔细研究正则化项，或设计一个保证可逆性的基本网络，可以减少这个问题。

3. 代码

   有，没看

4. 问题

   无

### 11. Affine Medical Image Registration with Coarse-to-Fine Vision Transformer (2022 CVPR C2FViT)

1. 动机：
   - 现有的基于CNN的仿射配准方法要么关注输入的局部错位，要么关注输入的全局方向和位置来预测仿射变换矩阵，对空间初始化敏感，脱离训练数据集的泛化能力有限。
   - 在综合图像配准框架中，目标图像对通常在使用可变形(非刚性)配准之前基于刚性或仿射变换进行预对齐，消除了目标图像对之间可能的线性和大空间错位。
   - 

