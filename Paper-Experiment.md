





| 名称                                                     | 数据集                                                       | 指标                                                         | 对比方法                                                     | 内容                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Voxelmorpher（TMI/CVPR 2018）                            | OASIS, ABIDE, ADHD200, MCIC, PPMI, HABS, Harvard GSP,Buckner40。用FreeSurfer对每张图像进行标准的预处理步骤，包括仿射空间归一化和大脑提取，并将得到的图像裁剪为160 × 192 × 224。所有MRI都使用FreeSurfer进行解剖分割。train: val: test=3231:250:250 ，Buckner40数据集仅用于测试。 | DSC，Jacobian determinant                                    | SyN, ANTs                                                    | 1)  将每个MRI配准到（来自其他数据集的）atlas中。展示了配准图像和形变场；表格展示了DSC、测试时间、Jacobian determinant 数量和百分比。2）图展示了loss权重对DSC的影响 。3）评估了训练集大小对准确性的影响（图像展示）。4）在有手动分割标签的数据集Buckner40上比较DSC的结果（其他数据集是用软件自动分割的），展示了CC和MSE两种损失函数的结果。5）subject to subject 配准，即随机选择MR图像对进行配准，展示了DSC结果。6）评估了不同ROI数量对配准结果的影响（one, half, all）。7）评估了粗标签下的结果。8）评估了评估了不同ROI数量对Jacobian determinant ≤ 0的结果。 |
| TMI_multi-contrast-registration （2021 TMI）             | 私有数据集。train: test=426:40所有数据大小调整为224 × 224，强度归一化[0,1]。 | Dice，Recall，Precision，Jacobian determinant                | VoxelMorph， VoxelMorph-diff (VM-diff) ，LT-Net，SyN         | 1）展示了不同方法不同指标的表格结果是实例图像。2）展示了这些方法的测试时间。3）展示了形变场的可视化结果。4）消融实验：学习率和损失权重。5）在没有扫描信息的情况下对数据进行实验（和他们数据有关）。 |
| C2FViT （2022 CVPR）                                     | OASIS和LPBA。对于OASIS，调整到256 × 256 × 256，然后使用FreeSurfer对每个MRI进行标准预处理步骤，包括运动校正、颅骨剥离和皮层下结构分割。将OASIS数据集分为255、10和149个卷，分别用于训练集、验证集和测试集。对于LPBA数据集，将所有40个MR作为测试集。1）OASIS所有图像仿射配准到MNI152脑模板（脑模板配准）。2）分别从OASIS和LPBA数据集的测试集中随机选择3个MRI作为地图集（基于地图集的本地空间配准）。 | DSC，HD95，                                                  | ANTs， Elastix，ConvNet-Affine， VTN-Affine                  | 1）展示了不同方法的DSC、HD95、参数量和测试时间的结果。2）消融实验。 |
| CycleMorph（2021 MIA）                                   | 1）RaFD（2D人脸表情）。OASIS-3（1249 MRI，用FreeSurfer得到分割标签），仿射空间归一化和大脑提取，裁剪到160 × 192 × 224，并除以255。train: val: test=1027:93:129。2）Multiphase liver CT，train: test=555:50 | RaFD: NMSE和SSIM。OASIS-3: Dice，Jacobian determinant  。liver CT：TRE (landmark的距离) | Elastix ，ANTs， Voxel-Morph。VoxelMorph-diff, ICNet, MS-DIRNet | 1）RaFD：展示了人脸配准后的图像和NMSE和SSIM指标。2）OASIS-3和liver CT：展示了配准图像结果、差分图像、形变场的可视化结果、Dice、TRE、Jacobian determinant指标和测试时间。3）消融实验：损失函数的分析、收敛性分析、局部形变场组合。 |
| DLIR (2021 MIA)                                          | Sunnybrook Cardiac Data（45*20 MRI ，256 × 256 train: val: test=1:1:1）。Chest CT from  NLST (2060 images)。DIR-Lab 4D chest CT 。 | Dice，Jacobian determinant (百分比和标准差），ASD , HD。     | SimpleElastix                                                | 1）展示了配准结果与形变场叠加的图像，Jacobian determinant的图像。2）展示了四个评价指标的箱线图和表格。3）消融实验。 |
| BIRNet（2019 MIA）                                       | LPBA40（第1张图像作为模板图像，1 - 30张图像作为训练样本，31 - 40张图像作为验证数据。使用FLIRT将所有MR线性配准到模板空间，提取大小为64 × 64 × 64的patch，总共有54,0 00个训练patch）。将LPBA40应用于四个不同的测试数据集：IBSR18、CUMC12、MGH10和IXI30。 | Dice                                                         | Diffeomorphic Demons，LCC-Demons，FNIRT，SyN，U-Net          | 1）展示了训练loss曲线，Dice结果表格，配准结果图像。2）展示了雅克比行列式图像，和损失系数对它的影响。3）比较了测试时间。 |
| Dual-PRNet++ (2022 MIA)                                  | LPBA40 (train:test=30:10，生成30个×29个图像对，并将56个标签合并为7个区域，裁剪到160 × 192 × 160)。Mindboggle101（数据被分为42名受试者(1722对)进行训练，20名受试者(380对)进行测试，裁剪为160 × 192 × 160） | Dice，Jacobian determinant，ASD , HD。                       | Affine, SyN, Voxelmorpher, FAIM, PMRNet，LapIRN，Contrastive Registration ，CycleMorp | 1）展示了这些方法的Dice箱线图和表格。2）消融实验：不同模块的作用。3）展示了联合分割和配准的Dice结果。4）展示了在大位移下的配准图像。5）评估了模型对不同大小切片的鲁棒性。6）展示了分割标签的配准图像。 |
| Recursive Cascaded Networks（2019 ICCV）                 | MSD, BFH，LiTS，LSPIG （这四个是CT图像，在MSD和BFH上进行无监督训练，共$1025^2$对图像，LiTS和LSPIG用于测试）。ADNI，ABIDE，ADHD，LPBA40（这四个是MR图像，ADNI, ABIDE, ADHD用于无监督训练，LPBA用于测试。）。所有图像裁剪到128 × 128 × 128 | Dice，Landkmark Distance，Similarity，测试时间               | VTN，Voxelmorpher，SyN, Elastix                              | 1）展示了对比实验的几个指标的结果，不同级联数量的结果，不同级联方式的结果。 |
| VTN (2019 JBHI)                                          | LITS，BFH，MICCAI’07 (将LITF和BFH的所有图像混合，两两组成图像对作为训练集，MICCAI’07用于测试)。ADNI，ABIDE-1，ABIDE-2，ADHD。 | Dice，Landkmark Distance，Jacobian determinant（百分比和标准差），测试时间 | ANTs，Elastix，Voxelmorpher                                  | 先展示CT图像的实验结果后展示MR图像的结果：1）展示了对比实验的几个指标的结果。2）展示了不同级联数量的指标结果和配准图像。3）展示了训练集大小对模型的影响。 |
| TransMorph (2022 MIA)                                    | brain MRI私有数据集 (train:val:test=182:26:52, Inter-patient brain MRI registration)。IXI (train: val:test=403:58:115，裁剪到160 × 192 × 224，Atlas-to-patient brain MRI registration)。OASIS (来自2021年Learn2Reg挑战赛，train:val:test=394:19:38，裁剪到160 × 192 × 224)。私有CT数据集（XCAT-to-CT registration） | DSC，SSIM，Jacobian determinant, HdDist95,SDlogJ             | SyN，NiftyReg，deedsBCV，LDDMM，VoxelMorpher，VoxelMorpher-diff，CycleMorph，MIDIR，ViT-V-Net，PVT，nnFormer | 1）展示了对比实验的几个指标的结果、形变场的可视化结果；2）评估了模型参数、计算复杂度；3）消融实验：位置编码、跳连接、模型设置等。4）不确定性估计、有效接受野的比较、位移幅度比较、损失landscape比较、收敛速度的比较。 |
| Learning a Deformable Registration Pyramid (2021 MICCAI) | 2020年Learn2Reg挑战 (Task2 （CT），Task3（CT）， Task4（MRI）)。下采样到64 × 64 × 64并归一化到[0,1]。比赛第五名的方法。 | TRE，DSC，HD，SDlogJ，测试时间                               | 初始化的结果                                                 | 1）展示了一张这些指标下测得的结果和一副配准图像。            |
| XMorpher（2022 MICCAI）                                  | MM-WHS 2017挑战（有20张标记图像和40张未标记图像）和ASOCA（有60张未标记图像）的CT数据集。使用所有的未标记图像（100张）和5张标记图像组成500张标记-未标记图像对和9900张未标记-未标记图像对作为无监督和半监督实验的训练集。其余15张标记图像组成210对图像作为测试集。所有图像都经过仿射变换处理。 | DSC，Jacobian determinant                                    | BSpline， Voxelmorph， PC-Reg， Transmorph                   | 无监督和半监督：1）展示了不同方法的指标结果；2）配准图像和形变场的可视化结果；3）窗口大小对模型的影响；4）展示了注意力的可视化图像。 |
| DeepAtlas （2019 MICCAI）                                | 3D knee MRIs from the OAI（train:val:test=200:53:542，为了测试配准性能，我们使用测试集中的10,000个随机图像对，裁剪到160×160×160，强度归一化到[0,1]）。MindBoogle101（train:val:test=65:15:5，裁剪为168 × 200 × 169大小，使用翻转来增强训练数据） | Dice                                                         | 单个分割/配准网络。                                          | 1）展示了不同标签数量下的Dice结果；2）展示了配准后图像和形变场的可视化结果 |
|                                                          |                                                              |                                                              |                                                              |                                                              |
|                                                          |                                                              |                                                              |                                                              |                                                              |

DSC：Dice similarity coefficient就是Dice

Jacobian determinant ($|J_\phi|$ ≤ 0) ：计算形变场的每个像素点雅克比行列式，统计行列式为负数的比例，衡量形变场是否光滑（是否微分同胚）。参考De Vos B D, Berendsen F F, Viergever M A, et al. A deep learning framework for unsupervised affine and deformable image registration[J]. Medical image analysis, 2019, 52: 128-143. 的第四节内容。

SSIM： structural similarity index 量化变形后的图像与固定图像之间的结构差异。

TRE：target registration error，两组对应的landmark点之间欧式距离的平均值。

ASD：average symmetric surface distance，两个点之间的距离。

HD：Hausdorff distance，两个点之间的距离。

Similarity：配准后的图像和固定图像之间的相似性。





|                                            | similarity | segmentation  | regularization |
| ------------------------------------------ | ---------- | ------------- | -------------- |
| VoxelMorpher                               | 1          | 0.01  default | 0.01  default  |
| TMI_multi-contrast-registration            | 1          |               | 100            |
| C2FViT                                     | 1          | 0.5           |                |
| learning-a-deformable-registration-pyramid | 5          | 5             | 1              |
| Recursive-Cascaded-Networks                | 1          | None          | 1              |
| TransMorph                                 | 1          | 1             | 1              |
| XMorpher                                   | 1          | 1             | 1              |
| Dual-Stream-PRNet-Plus                     | 1          | 1             | 1              |
|                                            |            |               |                |
