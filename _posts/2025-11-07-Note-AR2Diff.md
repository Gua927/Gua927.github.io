---
layout: distill
title: Training-Free Method for Parallel Decoding of Autoregressive Models
date: 2025-11-07 01:13:00
description: This blog post investigates the possibility of parallel decoding for autoregressive models. The author notes that autoregressive and diffusion models both fundamentally model data probability distributions, and that each has advantages—autoregressive models in training and diffusion models in sampling. The goal is to achieve a training-free way to perform parallel decoding with a pretrained autoregressive model, enabling low-cost accelerated generation.
tags: Gen-Model autoregressive diffusion-model score-matching
categories: Notes
bibliography: papers.bib
authors:
  - name: Runze Tian
    url: "https://gua927.github.io"
    affiliations:
      name: GenSI Lab, THU-AIR
      url: "https://www.gensi-thuair.com/#/portal_home"
giscus_comments: true
related_posts: true
toc:
  - name: Introduction
  - name: Current GenModels
    subsections:
      - name: Autoregressive Model
      - name: Diffusion model
      - name: Masked Diffusion Model
  - name: Score Function and Concrete Score
    subsections:
      - name: Score function and Langevin Dynamics
      - name: Concrete score and masked diffusion
  - name: AR2Diff in consistency space
  - name: AR2Diff in discrete space
  - name: Conclusion
---

## Introduction

In recent years, the rapid rise of artificial intelligence has been driven not only by advances in discriminative models but, more fundamentally, by the evolution of generative models — models that learn to represent and simulate the underlying probability distribution of data. From text and images to audio and 3D scenes, the essence of generation lies in one universal goal: to capture the complexity of real-world data distributions and to reproduce samples that are both coherent and diverse.

However, modeling such high-dimensional and structured distributions directly is intractable. Instead of storing an explicit probability function, modern generative models encode this distribution implicitly within their network parameters and decode it through learned stochastic processes. The diversity of generative paradigms — from autoregressive (AR) models to diffusion and energy-based models — stems primarily from the different ways they design and interpret this encoding–decoding process.

{% include figure.liquid
  loading="eager"
  path="assets/img/posts/2025-11-07-Note-AR2Diff/figure1.png"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

Autoregressive models, by factorizing joint probabilities into sequential conditionals, offer a simple and efficient training pipeline. Their major limitation, however, lies in the sequential nature of generation, which prevents parallel sampling. Diffusion models, in contrast, model the joint distribution directly through iterative denoising steps, enabling parallel decoding at the cost of heavier and slower training. Recent studies, such as <d-cite key="kim2025trainworstplanbest"></d-cite>, have quantitatively confirmed this asymmetry between training cost and sampling efficiency.

This blog explores a conceptual bridge between these two paradigms — a path to train with autoregression and decode with diffusion. Since both families ultimately learn to approximate the same probability distribution, it may be possible to transfer or reinterpret an autoregressive model into a diffusion-like sampling mechanism without extensive retraining — a direction we tentatively refer to as AR2Diff. Early efforts, such as Parallel and Flexible Sampling from Autoregressive Models via Langevin Dynamics (ICML 2021)<d-cite key="jayaram2021parallelflexiblesamplingautoregressive"></d-cite>, have hinted at this possibility, though the idea remained underexplored. With the resurgence of masked diffusion models (MDM) and consistency-based sampling, this line of thought now regains both practical relevance and theoretical elegance.

In the following sections, we will review the current landscape of generative modeling, connect autoregression and diffusion through the lens of score functions, and discuss how AR2Diff might be realized in both continuous and discrete spaces.

<!-- Removed manual Distill bibliography to avoid duplicate rendering; using Jekyll Scholar citations. -->

## Current GenModel

### Autogressive Models

The autoregressive model (AR) is fundamentally grounded in the chain rule decomposition of the joint distribution
$p(\mathbf{x})$. Let
$\mathbf{x} = (x_1, x_2, \dots, x_D) \in \mathcal{X}^D$, where
$\mathcal{X}$ may denote either a discrete vocabulary or a continuous real-valued space. The model factorizes the distribution as:

$$
\begin{equation*}
p(\mathbf{x}) = \prod_{i=1}^{D}p(x_i \mid x_{<i}),\quad \text{where } x_{<i}:= (x_1, \dots, x_{i-1}).
\end{equation*}
$$

In parametric modeling, a neural network with parameters
$\theta$ is employed to approximate each conditional distribution. For discrete data, this often takes the form of a softmax output:

$$
p_\theta(x_i = k \mid x_{<i}) = \frac{\exp(f_\theta(x_{<i})_k)}{\sum_{k' \in \mathcal{V}} \exp(f_\theta(x_{<i})_{k'})},
$$

while for continuous variables, a Gaussian parameterization is common:

$$
p_\theta(x_i \mid x_{<i}) = \mathcal{N}\big(x_i; \mu_\theta(x_{<i}), \sigma_\theta^2(x_{<i})\big).
$$

During training, maximum likelihood estimation (MLE) is used to minimize the negative log-likelihood:

$$
\mathcal{L}_{\text{AR}}(\theta) = -\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \sum_{i=1}^{D} \log p_\theta(x_i \mid x_{<i}) \right].
$$

This objective is fully parallelizable under teacher forcing, as all conditioning contexts $x_{<i}$ are taken from the ground-truth data. However, **sampling remains inherently sequential**: the generation process must proceed step-by-step,

$$
x_i^{(s)} \sim p_\theta\big(x_i \mid x_1^{(s)}, \dots, x_{i-1}^{(s)}\big), \quad i = 1, \dots, D,
$$

resulting in inference latency that scales linearly with the data dimensionality $D$. More fundamentally, this factorization imposes a **fixed ordering** $\pi$ (typically the natural index order) on the variables, despite the fact that the true data distribution $p_{\text{data}}(\mathbf{x})$ possesses no intrinsic sequential structure. While this inductive bias facilitates tractable modeling, it may constrain the model’s ability to capture non-local, symmetric, or graph-structured dependencies that do not conform to a unidirectional causal chain.

### Diffusion Model

Diffusion models construct a forward Markov chain that progressively corrupts data with noise, then learn the reverse process to enable generation. Let $\mathbf{x}_0 \sim p_{\text{data}}$. The forward process is defined as:

$$
q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1}), \quad q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}, \beta_t \mathbf{I}),
$$

where $\beta_t \in (0,1)$ is a pre-specified noise schedule. This process admits a closed-form expression:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}),
$$

with $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

The reverse process aims to learn a sequence of conditionals $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ such that the resulting generative chain $p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ approximates the true data distribution. In practice, the reverse transitions are often parameterized as Gaussians, and training is performed by minimizing a variational lower bound (ELBO) or, more commonly, by direct noise regression:

$$
\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_t, t) \|^2 \right],
$$

where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}$.

From the perspective of score matching, diffusion models equivalently learn the time-dependent score function:

$$
s_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t),
$$

and sampling can be carried out by solving the reverse stochastic differential equation (SDE):

$$
d\mathbf{x} = \left[ -\frac{1}{2} \beta(t) \left( \mathbf{x} + s_\theta(\mathbf{x}, t) \right) \right] dt + \sqrt{\beta(t)} \, d\mathbf{w},
$$

where $\mathbf{w}$ denotes a standard Wiener process. This formulation reveals the essential nature of diffusion models: rather than explicitly modeling the density, they learn a vector field that steers samples toward high-probability regions of the data manifold.

This implicit, geometry-aware approach endows diffusion models with remarkable expressiveness and robustness to long-range dependencies—making them especially well-suited for structured data like natural images or 3D scenes. However, this flexibility comes at a cost: training requires optimization across multiple noise levels, resulting in significantly higher computational overhead compared to autoregressive models. Moreover, the continuous-time foundation of diffusion poses fundamental challenges when applied to discrete domains (e.g., text), where gradients are undefined and semantic continuity breaks down.

### Masked Diffuison Model

Masked diffusion models (MDMs)<d-cite key="shi2025simplifiedgeneralizedmaskeddiffusion"></d-cite> extend the diffusion paradigm to discrete data by replacing additive Gaussian noise with a **masking mechanism**. Let $\mathbf{x}_0 \in \mathcal{V}^D$ be a discrete sequence (e.g., image tokens or natural language tokens). A masking rate $\alpha_t \in [0,1]$ is defined to increase with time step $t$, for instance as $\alpha_t = 1 - (1 - \gamma)^t$. The forward process randomly masks $\lceil \alpha_t D \rceil$ positions:

$$
\mathbf{x}_t = \mathcal{M}_t \odot \mathbf{x}_0 + (1 - \mathcal{M}_t) \odot \mathbf{1}_{[\text{MASK}]},
$$

where $\mathcal{M}_t \in \{0,1\}^D$ is a binary mask vector satisfying $\mathbb{E}[\|\mathcal{M}_t\|_1] = (1 - \alpha_t) D$.

The reverse process trains a unified model $p_\theta(x_i \mid \mathbf{x}_t, t)$ to predict the original token at any masked position. The training objective is typically formulated as a masked cross-entropy loss:

$$
\mathcal{L}_{\text{MDM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0} \left[ \sum_{i: \mathcal{M}_{t,i} = 0} \log p_\theta(x_{0,i} \mid \mathbf{x}_t, t) \right].
$$

Notably, at early timesteps (e.g., $t=1$) with high masking rates, $\mathbf{x}_1$ is nearly all `[MASK]`, forcing the model to perform “cold-start” prediction with minimal context. In contrast, as $t \to T$, $\mathbf{x}_T \approx \mathbf{x}_0$, and the task reduces to self-supervised reconstruction. This progression naturally induces a **multi-scale modeling hierarchy**, progressing from global structure to fine-grained detail.

MDMs exhibit a deep formal connection to autoregressive models. In the limiting case where the masking strategy is fixed to “mask only the last position,” the MDM exactly recovers a standard AR model. Conversely, under fully random masking, the model must learn the conditional distribution over **any subset** of variables given the rest—that is, for any $\mathcal{S} \subset \{1,\dots,D\}$, it implicitly learns $p(\mathbf{x}_{\mathcal{S}^c} \mid \mathbf{x}_{\mathcal{S}})$. This capability far exceeds the unidirectional conditioning of AR models, yet the training objective retains the same semantic essence: _predicting missing information from partial observations_.

It is precisely this **semantic equivalence** coupled with **structural disparity** that provides a theoretical foundation for AR2Diff: if an autoregressive model has already internalized rich contextual dependencies through sequential training, can we reinterpret it as a denoiser in a mask-based iterative refinement loop, thereby enabling parallel sampling without retraining? The answer may lie in the dual relationship between conditional probabilities and score functions—an insight we will explore in the next section.

## Score Function and Concrete Score

<!-- 并行生成的通用语言要将 AR 模型的密度估计能力转化为并行生成能力，必须引入分数函数。 -->

### Score Function and Langevin Dynamics

<!-- $s(x)$ 定义为对数概率密度函数的梯度：$$s(x) = \nabla_x \log p(x)$$这个梯度向量场指向数据流形上概率密度更高的区域 。SGM 的优势在于只需要建模和估计这个梯度，从而避开了计算概率密度函数 $p(x)$ 中可能无法解析的归一化常数（partition function） 12。实现并行性的关键是利用分数函数指导朗之万动力学（Langevin Dynamics）：$$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta} \epsilon_t$$与顺序的 AR 采样不同，这个更新步骤可以在 $x$ 的所有维度上同时操作，从而实现了大规模的并行化 5。AR 模型通过 MLE 训练确定了完整的联合似然 $p(x)$ 2。由于 $\log p(x)$ 理论上是可导出的，因此其空间梯度 $\nabla_x \log p(x)$ 也理论上可得 15。 -->

### Concrete Score

<!-- 离散数据的梯度泛化当 $x$ 是离散的（如文本令牌），传统的梯度 $\nabla_x \log p(x)$ 在数学上是未定义的 15。这构成了直接将 AR 似然转换为并行采样器的核心障碍 19。为了将分数基方法的优势扩展到离散领域，研究人员提出了 Concrete Score $c_p(x)$ 15。定义： Concrete Score 是连续领域 Stein Score 的推广 20。它不依赖于微分，而是基于概率随输入局部方向性变化的速率来定义 20。机制： Concrete Score 通过考虑在预定义的邻域结构下（例如在离散空间中使用曼哈顿距离代替欧几里得距离 3）两个相邻样本之间的相似性，在离散空间中构造出替代梯度信息 15。这种推广使得分数基生成模型得以应用于文本、图形和基因序列等结构化离散数据 3。 -->

## AR2Diff in consistency space

<!-- 连续或潜在空间的转换在 AR 模型输出为连续值（如原始音频信号）或通过变分自编码器（VAE）等映射到连续潜在码的情况下，梯度是可计算或可近似的。 -->

<!-- (PnF)针对连续域，**“并行且灵活采样”（Parallel and Flexible Sampling, PnF）技术实现了真正的“训练免费”**转换 。平滑操作： PnF 针对 AR 模型输出的离散化（例如 8 位音频样本）问题，采用高斯卷积 $\phi_{\sigma}$ 对离散分布 $p(x)$ 进行平滑处理 $p_{\sigma}(x) = (\phi_{\sigma} * p)(x)$ 5。这有效地将离散的概率质量函数转换为连续、可微的概率密度函数 5。解析推导： 平滑后的分布 $p_{\sigma}(x)$ 的梯度 $\nabla_x \log p_{\sigma}(x)$ 可以通过解析形式计算，这个过程完全利用原始预训练的 AR 模型，无需任何额外的训练或梯度估计 9。并行性： PnF 利用朗之万动力学在序列维度上并行生成，将确定性的顺序计算替换为了随机的并行 MCMC 过程 。PnF 采样相对于祖先采样实现了显著的加速，墙钟时间对序列长度 $L$ 表现出对数线性依赖 5。 -->

## AR2Diff in discrete space

<!-- 分类令牌的融合之路对于文本令牌等纯粹的、不可微的分类离散数据，PnF 所依赖的连续性假设不再成立 5。将 AR 似然转化为并行解码器需要引入训练步骤。 -->

<!-- ### A.Concrete Score Matching (CSM)

离散域，通常采用 Concrete Score Matching (CSM) 方法来训练一个专用的分数模型 $\tilde{s}_\theta(\tilde{x})$ 20。这标志着从“训练免费”到**“训练简单”或“训练微调”**的路径转换 22。去噪 CSM (D-CSM)： CSM 的一个常用变体是 D-CSM 23，它训练分数模型 $\tilde{s}_\theta(\tilde{x})$ 通过去噪目标来匹配 Concrete Score 22。这个训练目标与**离散扩散模型（DDM）**的学习目标本质上是高度一致的：即学习逆转加性噪声（如随机掩码）的过程 24。AR 知识的迁移： Concrete Score 充当了将 AR 模型的精确似然知识转化为 DDM 并行去噪能力的学习目标 19。

### B. 与大规模语言模型的集成与迭代细化通过对齐推理过程

可以将预训练的 AR-trained Transformer 主干网络与并行生成所需的去噪/扩散步骤相结合 26。例如，Diffusion-NAT 方法将离散扩散的去噪过程与预训练的非自回归解码器（如 BART）统一起来 26，使得模型能够利用预训练的知识进行迭代的掩码令牌（masked token）细化 26。这种方法在保持生成速度（比 AR Transformer 快 20 倍）的同时，利用了 AR 模型的预训练知识 26。迭代细化范式（例如 Mask-Predict 7）本身也高度依赖于隐式的“信心分数”（Confidence Score）来指导修正，这与基于 Concrete Score 的去噪过程在操作上具有趋同性——即通过识别低似然或噪声区域并同时改进它们来实现并行生成 7。 -->

## Conclusion

<!-- 融合 AR 精度与 SGM 并行性的新范式将 AR 模型与分数基采样方法相结合，为生成建模提供了一个强大的框架，旨在同时利用 MLE 训练的高保真度和 SGM 的并行采样效率 3。A. 范式融合的实现与挑战范式核心推理原则适用领域推理并行性训练要求自回归 (AR)顺序条件似然离散/连续无 (顺序)MLE (高保真度) 1AR-to-Score (PnF)基于平滑似然的梯度引导 MCMC连续/潜在编码 5高 (MCMC 步骤 $O(T)$)训练免费 (后验处理) 9掩码语言模型 (Mask-Predict)基于信心的迭代细化离散 (文本) 7中 (块并行)训练简单 (非 AR 目标) 7离散分数/扩散 (CSM)通过显式分数匹配进行去噪离散 (文本, 图形) 20高 (去噪步骤 $O(T)$)训练较重 (需要分数匹配) 3关键权衡：速度与质量： AR-to-Score 和 DDM 的总延迟取决于迭代 MCMC/去噪步骤的数量 ($T_{steps}$) 5。虽然它们在序列长度上是并行的，但样本质量高度依赖于 $T_{steps}$ 5。泛化挑战： 对于纯粹的分类离散数据，找到一种数学上严谨且真正的“训练免费”方法来推导其 Concrete Score 仍然是一个主要的开放挑战 15。目前的解决方案大多需要通过 CSM 引入新的训练目标 22。B. 未来研究展望未来的研究应专注于优化迭代效率，即大幅减少朗之万或去噪的迭代步骤数 $T_{steps}$，同时保持样本的高质量 5。此外，开发更高效、更通用的离散分数估计器，使其能够从现有的离散 AR 似然中低成本地提取，将是扩大“训练简单”方法适用范围的关键 20。通过架构集成，在同一模型中同时训练 AR 似然和 Concrete Score 匹配目标，从而在训练阶段就内建并行采样的能力，将进一步优化总体训练和推理预算 24。 -->
