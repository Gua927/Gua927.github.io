---
layout: distill
title: 融合序列精确性与并行效率：连接自回归模型与分数基采样
date: 2025-11-07 12:00:00
description: 统一自回归模型的精确似然建模与分数基/扩散模型的并行采样能力，讨论连续与离散空间下的训练免费与训练简单路径。
tags: Gen-Model autoregressive diffusion-model score-matching parallel-sampling
categories: Notes
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
	- name: Current GenModel
	- name: Score Function and Concrete Score
	- name: AR2Diff in consistency space
	- name: AR2Diff in discrete space
	- name: Conclusion
---

# 融合序列精确性与并行效率：连接自回归模型与分数基采样

## Introduction

序列化精度与并行化效率的矛盾统一在生成模型领域，自回归模型（Autoregressive, AR）和分数基生成模型（Score-Based Generative Models, SGM，如扩散模型）各自占据优势。AR 模型依赖于最大似然估计（MLE）进行精确的密度建模 1，在数据效率和训练初期的稳定收敛上表现出色 3，是当前 AI 算力持续增长而高质量训练数据增长开始放缓时代下的理想选择 4。然而，AR 模型固有的**祖先采样（Ancestral Sampling）**机制，要求逐令牌（token-by-token）生成，导致推理速度受到序列长度 $L$ 的严格限制 5。相比之下，扩散模型等分数基方法虽然训练难度和资源消耗较大，但在推理阶段可以利用朗之万动力学（Langevin Dynamics）等技术进行大规模并行采样，实现显著加速 8。本文的核心探究目标在于实现一种范式合成：保留 AR 模型在 MLE 训练中所建立的高保真度密度估计器，同时采用 SGM 固有的、通过梯度引导的并行采样能力。 重点在于探索如何通过分数函数（Score Function），尤其是其针对离散数据的推广形式——Concrete Score，实现从 AR 似然到并行解码的“训练免费”（Training-Free）或“训练简单”（Training-Easy）转换 9。

## Current GenModel

自回归与扩散模型的关键权衡

A. 自回归模型的精准似然与训练优势AR 模型的核心优势在于其对联合概率分布 $p(x)$ 的解析建模，即：$$p(x) = \prod_{i=1}^L p(x_i | x_{<i})$$这种基于链式法则的分解允许对数据分布进行精确的对数似然计算和 MLE 优化

在训练初期（如单次迭代周期），AR 模型通常能取得显著优于扩散模型的性能（如损失值10.65 vs. 7.07） 3。这种训练的严谨性和稳定性使其成为在追求高保真度时最大限度利用现有数据的有效选择 4。B. 顺序推理的固有成本与并行化探索AR 模型在生成长度为 $L$ 的序列时，需要 $L$ 次顺序计算步骤。这种固有的顺序依赖性限制了模型在推理时的墙钟并行性（wall-clock parallelism）

6。为了突破这一瓶颈，研究人员探索了各种并行解码方法，例如基于迭代细化的非自回归（Non-AR）方法，如 Mask-Predict 7、变分自回归模型（VAR） 10 或离散扩散模型 ，这些方法旨在通过并行块生成来实现显著的解码加速 11。这些并行路径最终都指向了一个共同的数学工具：分数函数。

## Score Function and Concrete Score

并行生成的通用语言要将 AR 模型的密度估计能力转化为并行生成能力，必须引入分数函数。

### A. 连续分数函数与朗之万动力学分数函数

$s(x)$ 定义为对数概率密度函数的梯度：$$s(x) = \nabla_x \log p(x)$$这个梯度向量场指向数据流形上概率密度更高的区域 。SGM 的优势在于只需要建模和估计这个梯度，从而避开了计算概率密度函数 $p(x)$ 中可能无法解析的归一化常数（partition function） 12。实现并行性的关键是利用分数函数指导朗之万动力学（Langevin Dynamics）：$$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta} \epsilon_t$$与顺序的 AR 采样不同，这个更新步骤可以在 $x$ 的所有维度上同时操作，从而实现了大规模的并行化 5。AR 模型通过 MLE 训练确定了完整的联合似然 $p(x)$ 2。由于 $\log p(x)$ 理论上是可导出的，因此其空间梯度 $\nabla_x \log p(x)$ 也理论上可得 15。

### Concrete Score

离散数据的梯度泛化当 $x$ 是离散的（如文本令牌），传统的梯度 $\nabla_x \log p(x)$ 在数学上是未定义的 15。这构成了直接将 AR 似然转换为并行采样器的核心障碍 19。为了将分数基方法的优势扩展到离散领域，研究人员提出了 Concrete Score $c_p(x)$ 15。定义： Concrete Score 是连续领域 Stein Score 的推广 20。它不依赖于微分，而是基于概率随输入局部方向性变化的速率来定义 20。机制： Concrete Score 通过考虑在预定义的邻域结构下（例如在离散空间中使用曼哈顿距离代替欧几里得距离 3）两个相邻样本之间的相似性，在离散空间中构造出替代梯度信息 15。这种推广使得分数基生成模型得以应用于文本、图形和基因序列等结构化离散数据 3。

## AR2Diff in consistency space

连续或潜在空间的转换在 AR 模型输出为连续值（如原始音频信号）或通过变分自编码器（VAE）等映射到连续潜在码的情况下，梯度是可计算或可近似的。

### A.训练免费的并行且灵活采样

(PnF)针对连续域，**“并行且灵活采样”（Parallel and Flexible Sampling, PnF）技术实现了真正的“训练免费”**转换 。平滑操作： PnF 针对 AR 模型输出的离散化（例如 8 位音频样本）问题，采用高斯卷积 $\phi_{\sigma}$ 对离散分布 $p(x)$ 进行平滑处理 $p_{\sigma}(x) = (\phi_{\sigma} * p)(x)$ 5。这有效地将离散的概率质量函数转换为连续、可微的概率密度函数 5。解析推导： 平滑后的分布 $p_{\sigma}(x)$ 的梯度 $\nabla_x \log p_{\sigma}(x)$ 可以通过解析形式计算，这个过程完全利用原始预训练的 AR 模型，无需任何额外的训练或梯度估计 9。并行性： PnF 利用朗之万动力学在序列维度上并行生成，将确定性的顺序计算替换为了随机的并行 MCMC 过程 。PnF 采样相对于祖先采样实现了显著的加速，墙钟时间对序列长度 $L$ 表现出对数线性依赖 5。

### B. 变分潜在空间（VSSM）

桥接的另一种方法是利用变分状态空间模型（VSSM） 或类似的 VAE 结构。VSSM 采用 VAE 结构，并通过 Gumbel 重参数化技巧 (Gumbel-Softmax) 在离散潜在空间 $z$ 中进行采样 。Gumbel-Softmax 提供了对离散采样过程进行可微分近似的能力 ，使得潜在序列 $z$ 的并行解码对梯度优化变得可行 。这展示了 AR 模型可以将其序列化和离散化转移到可并行优化的潜在空间，从而重用强大的 AR 训练模型 。

## AR2Diff in discrete space

分类令牌的融合之路对于文本令牌等纯粹的、不可微的分类离散数据，PnF 所依赖的连续性假设不再成立 5。将 AR 似然转化为并行解码器需要引入训练步骤。

### A.Concrete Score Matching (CSM)

离散域，通常采用 Concrete Score Matching (CSM) 方法来训练一个专用的分数模型 $\tilde{s}_\theta(\tilde{x})$ 20。这标志着从“训练免费”到**“训练简单”或“训练微调”**的路径转换 22。去噪 CSM (D-CSM)： CSM 的一个常用变体是 D-CSM 23，它训练分数模型 $\tilde{s}_\theta(\tilde{x})$ 通过去噪目标来匹配 Concrete Score 22。这个训练目标与**离散扩散模型（DDM）**的学习目标本质上是高度一致的：即学习逆转加性噪声（如随机掩码）的过程 24。AR 知识的迁移： Concrete Score 充当了将 AR 模型的精确似然知识转化为 DDM 并行去噪能力的学习目标 19。

### B. 与大规模语言模型的集成与迭代细化通过对齐推理过程

，可以将预训练的 AR-trained Transformer 主干网络与并行生成所需的去噪/扩散步骤相结合 26。例如，Diffusion-NAT 方法将离散扩散的去噪过程与预训练的非自回归解码器（如 BART）统一起来 26，使得模型能够利用预训练的知识进行迭代的掩码令牌（masked token）细化 26。这种方法在保持生成速度（比 AR Transformer 快 20 倍）的同时，利用了 AR 模型的预训练知识 26。迭代细化范式（例如 Mask-Predict 7）本身也高度依赖于隐式的“信心分数”（Confidence Score）来指导修正，这与基于 Concrete Score 的去噪过程在操作上具有趋同性——即通过识别低似然或噪声区域并同时改进它们来实现并行生成 7。

## Conclusion

融合 AR 精度与 SGM 并行性的新范式将 AR 模型与分数基采样方法相结合，为生成建模提供了一个强大的框架，旨在同时利用 MLE 训练的高保真度和 SGM 的并行采样效率 3。A. 范式融合的实现与挑战范式核心推理原则适用领域推理并行性训练要求自回归 (AR)顺序条件似然离散/连续无 (顺序)MLE (高保真度) 1AR-to-Score (PnF)基于平滑似然的梯度引导 MCMC连续/潜在编码 5高 (MCMC 步骤 $O(T)$)训练免费 (后验处理) 9掩码语言模型 (Mask-Predict)基于信心的迭代细化离散 (文本) 7中 (块并行)训练简单 (非 AR 目标) 7离散分数/扩散 (CSM)通过显式分数匹配进行去噪离散 (文本, 图形) 20高 (去噪步骤 $O(T)$)训练较重 (需要分数匹配) 3关键权衡：速度与质量： AR-to-Score 和 DDM 的总延迟取决于迭代 MCMC/去噪步骤的数量 ($T_{steps}$) 5。虽然它们在序列长度上是并行的，但样本质量高度依赖于 $T_{steps}$ 5。泛化挑战： 对于纯粹的分类离散数据，找到一种数学上严谨且真正的“训练免费”方法来推导其 Concrete Score 仍然是一个主要的开放挑战 15。目前的解决方案大多需要通过 CSM 引入新的训练目标 22。B. 未来研究展望未来的研究应专注于优化迭代效率，即大幅减少朗之万或去噪的迭代步骤数 $T_{steps}$，同时保持样本的高质量 5。此外，开发更高效、更通用的离散分数估计器，使其能够从现有的离散 AR 似然中低成本地提取，将是扩大“训练简单”方法适用范围的关键 20。通过架构集成，在同一模型中同时训练 AR 似然和 Concrete Score 匹配目标，从而在训练阶段就内建并行采样的能力，将进一步优化总体训练和推理预算 24。
