---
title: "Research Plan: Encoder"
publish: false
notes: "Planning document only. This file is intentionally outside Astro content collections and is not rendered on the website."
---

# Research Plan: Encoder

## 序章

00｜Representation First：为什么 Encoder 是智能系统的状态入口？

## 第一部分：表示学习的思想地基

01｜什么是 Representation：从原始数据到可计算状态

02｜信息论视角：压缩、保真与信息瓶颈

03｜几何视角：表示空间、流形假设与相似性

04｜概率视角：Latent Variable Model 与隐变量表示

05｜归纳偏置：结构如何决定模型能学到什么表示

06｜如何评价一个表示：Probe、迁移、检索与 Collapse

## 第二部分：语言 Encoder

07｜离散符号的连续化：从 one-hot 到 word embedding

08｜序列如何被编码：CNN、RNN、LSTM 与 Seq2Seq Encoder

09｜Attention 与 Transformer Encoder：现代语言表示的骨架

10｜上下文化表示：从 ELMo 到 BERT

11｜预训练目标如何塑造语言 Encoder：MLM、Denoising、RTD

12｜LLM 时代的语言表示：Decoder-only 模型还需要 Encoder 吗？

## 第三部分：视觉 Encoder

13｜CNN 时代：局部性、层级结构与视觉表示

14｜图像 Tokenization：Patch、Region、Mask 与视觉 Token

15｜Vision Transformer：把图像变成 Token 序列

16｜视觉自监督学习：从对比学习到自蒸馏表示

17｜Masked Image Modeling：MAE、BEiT 与恢复缺失世界

18｜通用视觉特征：DINOv2、SAM 与 Dense Representation

## 第四部分：多模态 Encoder 与表示对齐

19｜CLIP：图像和文本如何进入同一个语义空间

20｜Dual Encoder、Fusion Encoder 与 Query Encoder

21｜Multimodal LLM：视觉表示如何接入语言模型

22｜音频、视频与时间表示：Encoder 如何处理动态世界

23｜统一表示空间的梦想与问题：Alignment、Grounding 与 Collapse

## 第五部分：领域中的 Encoder

24｜AI4S 中的 Encoder：分子、蛋白质、材料与科学表示

25｜Graph 与 Geometric Encoder：结构、关系与等变表示

26｜World Model 中的 Encoder：从感知状态到可预测世界

27｜具身智能与 VLA：从视觉语言表示到可行动表示

28｜领域 Encoder 的共同规律：结构、约束、尺度与任务闭环

## 第六部分：表示学习如何帮助生成

29｜Encoder-Decoder：表示学习与生成模型的基本接口

30｜Autoencoder、VAE 与 VQ-VAE：生成模型中的 Latent Space

31｜Latent Diffusion：为什么生成模型要在表示空间里工作

32｜Representation Autoencoder：从语义 Latent 到高质量生成

33｜Encoder 的未来：从表示空间到世界状态空间
