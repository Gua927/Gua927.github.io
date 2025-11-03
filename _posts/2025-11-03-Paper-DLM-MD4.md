---
layout: post
title: Simplified and Generalized Masked Diffusion for Discrete Data
date: 2025-11-03 14:17:00
description: The x_0-prediction framework for masked Diffusion Model published on NeurIPS 2024.
tags: discrete-diffusion diffusion-models NIPS2024
categories: paper-reading
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

- NIPS 2024
- Paper Link: http://arxiv.org/abs/2406.04329
- Code Link: https://github.com/google-deepmind/md4
- Video Link: https://www.youtube.com/watch?v=0--pr5c2U4E
- Author:

{% include figure.liquid path="assets/img/MD4-pic1.png" class="img-fluid rounded z-depth-1" %}

## Core Contributions

- 建立了更好的加噪去噪过程
- 推导了一个简单的ELBO，并证明它对应于SE Loss对时间的加权积分
- 统一理解先前的Discrete Diffusion，表明训练中的大方差和正反向过程的一致性存在冲突
- 提出了Generlized Masked Diffusion，允许 State-dependent mask schedules，进一步提升了似然测试。

---

## Masked Diffusion

我们希望将一个离散序列（句子中的词序列）逐步替换为特殊的mask token，并根据mask的过程，学习去mask过程，最终实现由纯mask生成文本。

设总词汇数也即离散状态空间大小为m，表示m种token，对应的one-hot向量分别为e_0,e_1,...,e_m-1，再增加一个mask状态，索引为m，记为e_m，最终得到具有m+1个状态的状态空间。

### Discrete-time forward process

先考虑单一token的情况，forward process是一个离散时间马尔科夫链：

$$
x_t\in \{0,1,\cdots,m \},t\in[0,1]
$$

其中 x_t 表示token在时间 t 的状态（或者对应的one-hot向量）。
将区间 [0,1] 离散化为 T 个时间步，定义：

$$
s(i)=\frac{i-1}{T},\quad t(i)=\frac{i}{T}
$$

那么在时间步 s(i)-->t(i) 的状态转移可以由转移矩阵 $Q_i\in\mathbb R^{(m+1)\times(m+1)}$ 来表示，定义为：

$$
Q_i = (1-\beta_i) I + \beta_i \, \mathbf{1} e_m^\top
\newline
[Q_i]_{jk} = (1-\beta_i)\,\delta_{jk} + \beta_i\,\delta_{k m}
$$

这表明每一步以概率 $1-beta_i$ 保持原状态，以概率 beta_i 转移为mask状态m。且mask为吸收态，因为 $[Q_i]*{m,m}=1$。
由以上设定，我们可以计算给定初始状态 x_0 ，在时刻 t(i) 的分布为：

$$
q(x_{t(i)}|x_0) = \mathrm{Cat}(x_{t(i)};\ \bar Q_i^\top x_0)=x_0^\top \bar Q_ix_t(i)
$$

Cat(x;p) 表示一个类别分布，其中 p 是概率向量。$\bar Q_i=\prod_{j=1}^i Q_j=\alpha_i I+(1-\alpha)\bold 1e_m^\top$ 表示从第一步到第i步的累积转移矩阵，$\alpha_i=\prod_{j=1}^i (1-\beta_j)$。

### Continuous-time Limit

通过令 $T-->∞$，并将构造连续 beta(t) 函数使得 beta\*i=beta(t(i))/T ，我们可以得到连续情形下的公式：

$$
\bar Q(t)\triangleq \lim*{T\to\infty} \bar Q_i=\alpha_tI+(1-\alpha_t)]\bold 1e_m^\top,\quad \alpha_t\triangleq \exp \left( -\int_0^t\beta(s)ds \right)
$$

于是 $q(x_t|x_0)=\text{Cat}(x_t;\bar Q(t)^\top x_0)$，同时对于任意两个时间s，t满足 $0<=s<t<=1$，条件概率可以写为：

$$
q(x_t|x_s)=\text{Cat}(x_t;\bar Q(x,t)^\top x_s)=x_s^\top\bar Q(s,t)x_t,
\newline
\text{where }
\bar Q(s,t)\triangleq \bar Q(s)^{-1}\bar Q(t)=\frac{\alpha_t}{\alpha_s}I+(1-\frac{\alpha_t}{\alpha_s})\bold 1e_m^\top
$$

公式推导：

1. $\bar Q(t)$的推导：
   令$\beta_i=\frac{\beta(t(i))}{T}，且~T\to\infty：$
   $$
   \begin{aligned}
   \prod_{j=1}^i(1-\beta_j)&=\exp\left(\sum_{j=1}^{i}\log (1-\beta_j) \right)\\
   &=\exp\left(\sum_{j=1}^{i}-\frac{\beta(t(j))}{T}+o(1/T) \right)\\
   &\to \exp(-\int_0^{t(i)}\beta(s)ds)
   \end{aligned}
   $$
   考虑瞬时状态转移矩阵：
   $$
   Q(t)=\lim_{T\to\infty}\frac{Q_i-I}{1/T}= \beta(t)(\mathbf{1}e_m^T - I)
   $$
   令$Q=\bold 1e_m^\top-I$，则
   $$\bar{Q}(t) = \exp\left(\int_0^t Q(s)ds\right)=\exp(\bar{\beta}(t)Q)$$
   其中 $$\bar\beta(t)\triangleq \int_0^t\beta(s)ds $$，注意到 $$Q^2=-Q
   $$
   ，可得
   $$\begin{aligned}
   \bar{Q}(t) &= \exp(\bar{\beta}(t) Q) \\
   &= I + \bar{\beta}(t) Q + \frac{1}{2} \bar{\beta}(t)^2 Q^2 + \frac{1}{3} \bar{\beta}(t)^3 Q^3 + \ldots \\
   &= I + Q - (1 - \bar{\beta}(t) + \frac{1}{2} \bar{\beta}(t)^2 - \frac{1}{3} \bar{\beta}(t)^3 + \ldots) Q \\
   &= I + Q - \exp(-\bar{\beta}(t)) Q \\
   &= \alpha_t I + (1 - \alpha_t) \mathbf{1} e_m^\top.
   \end{aligned}
   $$
2. $$\bar Q(s,t)$$的推导：
   由连续时间马尔可夫过程微分方程：
   $$\frac{d}{dt}\bar{Q}(s, t) = \bar{Q}(s, t)Q(t),\text{where } Q(t)=\beta(t)Q$$
   在初始条件 $$\bar Q(s,s)=I$$下，该微分方程的解为：
   $$
   \begin{aligned}
   \bar{Q}(s, t) &= \exp\left(\int_s^t Q(\tau)d\tau\right) = \exp\left(Q\int_s^t \beta(\tau)d\tau\right)\\
   &=\exp((\bar{\beta}(t) - \bar{\beta}(s))Q)\\
   &=\bar{Q}(s)^{-1}\bar{Q}(t)\\
   &= \frac{\alpha_t}{\alpha_s} I + \left(1 - \frac{\alpha_t}{\alpha_s}\right)\mathbf{1}e_m^T
   \end{aligned}
   $$

### Masking Schedule

有效的噪声调度应该满足在起始时间 t=0 时，alpha_t=1，而在结束时间 t=1 时，alpha_1 必须等于或非常接近于0，这保证了前向过程的终点是一个几乎完全被掩码的句子。
目前已存在的策略及对应的效果如下：

Time reversal of the forward process given $$\bold x_0$$
通过学习 $$p(x_s|x_t),s<t$$，我们希望最终能够学习到 $$p(x_0|x_1)$$，即从一个完全掩码的句子 x_1 恢复出原始句子 x_0。
论文给出了这个反向过程的精确解：
$$q(x_s | x_t, x_0) = Cat(x_s; \bar{R}^{x_0}(t, s)^T x_t)$$
其中反向转移矩阵为 $$\bar{R}^{x_0}(t, s) = I + \frac{\alpha_s - \alpha_t}{1 - \alpha_t}e_m(x_0 - e_m)^T$$。
公式推导：

1. 反向转移矩阵
   根据贝叶斯公式，我们可以求得：
   $$
   \begin{aligned}
   q(x_s|x_t,x_0)&=\frac{q(x_t|x_s)q(x_s|x_0)}{q(x_t|x_0)}\\
   &=\begin{cases}
       \frac{\alpha_s - \alpha_t}{1 - \alpha_t} x_s^T x_0 & \text{if } x_s \neq m, x_t = m \\
       \frac{1 - \alpha_s}{1 - \alpha_t} & \text{if } x_s = m, x_t = m \\
       x_s^T x_t & \text{if } x_t \neq m
   \end{cases}
   \end{aligned}
   $$
   上述分情况的讨论可以被一个更优雅的矩阵形式所统一。论文指出，这个概率分布可以被写作：
   $$q(x_s|x_t, x_0) = \text{Cat}(x_s; \bar{R}^{x_0}(t, s)^T x_t)$$
   其中反向转移矩阵为：
   $$\bar{R}^{x_0}(t, s) = I + \frac{\alpha_s - \alpha_t}{1 - \alpha_t}e_m(x_0 - e_m)^T$$
2. 反向转移速率矩阵
   考察一个无穷小的时间间隔，即令 s=t-delta t，且 delta t-->0，此时
   $$\alpha_{t-\Delta t}-\alpha_t=-\alpha_t'\Delta t+o(\Delta t)$$
   带入反向转移矩阵即得
   $$\bar{R}^{x_0}(t, t-\Delta t) = I - \frac{\alpha'_t}{1 - \alpha_t}e_m(x_0 - e_m)^T\Delta t+ + o(\Delta t)$$
   根据速率矩阵定义即可得
   $$R^{x_0}(t) = -\frac{\alpha'_t}{1 - \alpha_t}e_m(x_0 - e_m)^T$$

---

Model and Objective
Reverse Process Prediction
我们通过 $$q(x_s | x_t, x_0)$$来从mask解码出token，但由于 x*0 是未知的，因此我们需要使用生成模型 $$p*{\theta}(x*s|x_t)$$ 来近似：直接借用 q 的数学形式，但把其中未知的 x_0 替换为神经网络的预测结果。
$$p*\theta(x*s | x_t) \triangleq q(x_s | x_t, \mu*\theta(x*t, t))$$
$$\mu*\theta(x_t, t)$$ 是整个模型的核心——一个由神经网络驱动的函数。它的输入是当前被部分掩码的句子 x_t 和当前时间步 t，它的输出是对原始句子 x_0 的预测。这个输出是一个概率向量，其实现为：

$$
\mu_\theta(x_t, t) =
\begin{cases}
    \text{softmax}(f_\theta(x_t, t)) & \text{if } x_t = m  \\
    x_t & \text{if } x_t \neq m
\end{cases}
$$

在时间 t=1 时，句子已经完全变为mask了，因此将先验分布 p(x*1) 定义为mask，也即 $$p(x_1)=\delta*{x*1,m}$$，而在去噪的最后一个时间步 t(1)=1/T，我们需要生成干净的句子 x_0，可以直接设定 $$p(x_0|x*{t(1)})\propto q(x*{t(1)}|x_0)$$.
[图片]
Loss function
为优化似然，我们只需要优化其证据下界：
$$\log p(x_0) \ge \mathbb E*{q(x*{t(1)}|x_0)}[\log p(x_0|x*{t(1)})] - \text{KL}(q(x_1|x_0)||p(x_1)) - \mathcal L_T$$
该损失函数分为三项：

- 重建项： $$\mathbb E[\log p(x_0|x_{t(1)})]$$
- 先验匹配项： $$\text{KL}(q(x_1|x_0)||p(x_1))$$，在论文的 x_1 设定下，这一项为0
- 去噪匹配项： $$\mathcal L_T=\sum \text{KL}(q(x_s|x_t, x_0) || p_θ(x_s|x_t))$$
  其中去噪匹配项中每一个和都可以被简化为交叉熵损失：
  $$\mathrm{KL}(q(x_s|x_t,x_0)\|p_\theta(x_s|x_t)) = -\frac{\alpha_s - \alpha_t}{1 - \alpha_t} \delta_{x_t,m} \cdot x_0^\top \log \mu_\theta(x_t,t),$$
  当 T-->∞ 时，去噪匹配项的求和可以写为积分形式：
  $$L_∞ = ∫_{t(1)}^{1} -\frac{\alpha'_t}{1 - \alpha_t} \mathbb E_{q(x_t|x_0)}[\delta_{x_t, m} \cdot x_0^T \log \mu_\theta(x_t, t)] dt$$
  通过引入信噪比，我们可以重新参数化该损失函数，令 $$\lambda_t=\log\frac{\alpha_t}{1-\alpha_t}$$，则上述损失变为：
  $$\mathcal{L}_\infty = \int_{\lambda_{t(1)}}^{\lambda_1} \sigma(\lambda) \mathbb{E}_{\tilde{q}(x_\lambda|x_0)} \left[ \delta_{x_\lambda,m} \cdot x_0^\top \log \tilde{\mu}_\theta(x_\lambda, \lambda) \right] \mathrm{d}\lambda.$$
  其中： $$\tilde{\mu}_\theta(x, \lambda) := \mu_\theta(x, t),\tilde{q}(x_\lambda|x_0) := q(x_t|x_0),t = \log\text{-SNR}^{-1}(\lambda)$$
  公式推导：
  将具体的 p\_{theta} 带入即得
  $$
  \begin{aligned}
  \mathrm{KL}(q(x_s|x_t,x_0)\|p_\theta(x_s|x_t)) &= \mathrm{KL}(q(x_s|x_t,x_0)\|q(x_s|x_t,\mu_\theta(x_t,t))) \\
  &=
  \begin{cases}
  \sum_{x_s=0}^{m} q(x_s|x_t,x_0) \log \frac{q(x_s|x_t,x_0)}{q(x_s|x_t,\mu_\theta(x_t,t))} & x_t = m \\
  0 & x_t \neq m
  \end{cases} \\
  &= \delta_{x_t=m} \sum_{k \neq m} \frac{\alpha_s - \alpha_t}{1 - \alpha_t} x_0^\top e_k \log \frac{x_0^\top e_k}{\mu_\theta(x_t,t)^\top e_k} \\
  &= -\delta_{x_t=m} \frac{\alpha_s - \alpha_t}{1 - \alpha_t} x_0^\top \log \mu_\theta(x_t,t).
  \end{aligned}
  $$
  求和得：
  $$
  \begin{aligned}
  \mathcal{L}_\infty &\triangleq \lim_{T \to \infty} \mathcal{L}_T \\
  &= \lim_{T \to \infty} \sum_{i=2}^{T} \frac{-\alpha_{s(i)} - \alpha_{t(i)}}{s(i) - t(i)} \cdot \frac{s(i) - t(i)}{1 - \alpha_{t(i)}} x_0^\top \mathbb{E}_{q(x_{t(i)}|x_0)} \left[ \delta_{x_{t(i)},m} \log \mu_\theta(x_{t(i)}, t(i)) \right] \\
  &= \int_{t(1)}^{1} \frac{\alpha_t'}{1 - \alpha_t} x_0^\top \mathbb{E}_{q(x_t|x_0)} \left[ \delta_{x_t,m} \log \mu_\theta(x_t, t) \right] \mathrm{d}t.
  \end{aligned}
  $$
  Multidimensional data
  到目前为止，我们所有的数学推导都是围绕一个孤立的词元 x*t 进行的。但我们的最终目标是生成连贯的句子，也就是一个由 N 个词元组成的序列 $$x_t=(x_t^{(1)},x_t^{(2)},...,x_t^{(N)})$$。本节就是要解决如何将之前的理论框架应用到这个序列上的问题。
  论文做出了一个非常重要且简洁的假设：对句子中每个词元进行掩码（masking）的前向过程是相互独立的。这意味着，在任何时间点，第 n 个词是否被掩码，只取决于它自己的初始状态 $$x_t^{(n)}$$，而与它旁边的词是什么或者是否被掩码完全无关。这表明，整个序列的转移概率可以写为各个词元转移概率的乘积：
  $$q(x_t|x_s) = \prod*{n=1}^{N} q(x*t^{(n)}|x_s^{(n)})$$
同样的，反向过程以及模型预测也可以写成联合乘积的形式：
$$p*\theta(x*s|x_t) \triangleq \prod*{n=1}^{N} q(x*s^{(n)}|x_t^{(n)}, \mu*\theta^{(n)}(\color{red}{x*t}, t\color{black}))$$
注意，虽然前向过程各词元独立加噪，但反向过程中，如果让各词元独立进行去噪，这会导致无法关注到各词元之间的依赖，造成语义和语序的混乱。因此预测第 n 个词元的神经网络输出 $$\mu*{\theta}^{(n)}(x*t,t)$$的输入是整个序列 x_t 与时间 t。
既然前向和后向过程都是（在形式上）可分解的，那么总的对数似然也可以分解为每个位置的对数似然之和。因此，整个序列的损失函数 $$\mathcal L*{\infty}^{(n)}$$就是我们之前推导的单个词元损失函数 $$\mathcal L_{\infty}$$ 在所有 N 个位置上的总和。
  $$\color{blue}\mathcal{L}_\infty^{(N)} \triangleq \int_0^1 -\frac{\alpha'_t}{1 - \alpha_t} \mathbb E_{q(x_t|x_0)} \left[ \sum_{n:x_t^{(n)}=m} (x_0^{(n)})^T \log \mu_\theta^{(n)}(x_t, t) \right] dt$$
  在这一部分我们完成了Discrete Diffusion的全部训练准备，包括正向过程、反向过程、拟合函数、损失函数。
- 正向过程中，根据预定义的noise schedule，对每一个token独立加噪，以1-alpha的概率变为mask，并在mask之后不再进行改变，到达时间1后，该token一定变为mask；
- 反向过程中，模型根据当前整个序列的状态以及时间 t 预测每一个token的状态概率分布；在时刻t，如果token为mask状态，则计算损失，将模型的预测输出与真实one-hot拉近，；如果token已经被解码，则不进行损失计算，仅仅依靠重建项来进行优化。
- 根据以上讨论，可以认为在mask状态时，模型不断积累，提升根据当前序列预测真实token的embedding的能力，但并不更新序列；直到当时刻 t_0 到来，mask变为token，真实序列也随之更新，并基于新的序列来继续优化其他mask处的预测结果。
- 但这样做同样存在弊端：由于mask操作是独立的，我们在训练时，总是随机得到一个 x_t，并根据此预测每个token的概率分布，
  - 时间t时，每个token的预测质量不仅与模型的参数有关，同时也与 x_t 有关：如果我们能够获得更好的 x_t，实际上模型可以更快更好的获得准确的预测，
  - 同时不同unmask时间也会影响该mask位置token概率分布优化的次数，如果被umask需要的时间长，则此处的token概率分布会得到更多个loss损失；但实际上该token的预测可能很早就已经被优化很好了，这会造成一定时间步上的浪费。
  - 在某个 x*t 下已经优化了很好的token概率分布，当面对新的 x*{t-1}时可能会存在冲突，这样会造成大量冲突优化，增大了学习难度，造成训练效率低下；但实际上，这一做法正是在使模型不断学习语言模式，反复根据上下文的变化来动态调整自己的预测，从而使模型习得更强的表达模式。
    Training Algorihm

---

Sampling
Sampling Algorihm

在生成时，向模型输入序列长度N以及初始被mask的位置，随后，在每个时间步t，对所有mask的token进行采样更新，但保持已被unmask的token不变。
这一采样算法未免有些草率：当一个token被unmask后，则不再进行任何改变，这显然不合理，因为模型完全失去了自我纠错的能力。
Impact of schedules and discretization
作者使用图像上的FID来对模型的生成效果进行评测，一开始使用“线性策略”来训练模型。这意味着 alpha_t 的值是线性下降的。这个策略在训练时表现很好，能够让模型的最终 ELBO（证据下界，与损失函数相关）达到最优。尽管训练得很好，但在生成（采样）时，效果却不理想。他们使用的是标准的“均匀离散化网格”，即 t(i) = i/T。这意味着在生成过程的每一步，时间 t 的变化量是恒定的。这导致了unmask 的“速度”也是相对恒定的。
作者推断，问题出在生成的早期阶段。在生成初期（t 接近1），序列中几乎全是 [MASK]，上下文信息极度稀疏。此时，如果模型同时解封多个词元，这些新生成的词元之间很可能会相互冲突或不连贯。
个人认为，token conflict并不应该是采用linear schedule的问题，而是采样时，模型不具备自我纠错能力所导致的
他们在训练时换用了“余弦策略”。余弦函数的导数在起点附近接近于0，然后逐渐变大。这个策略的特性是，在反向生成过程的开始阶段，解掩码的速度非常慢。而在后期，当大部分词元已经被揭示，上下文信息非常丰富时，解掩码的速度会加快。
作者提出，即使模型是用线性策略训练的，我们在采样时也可以模拟出“慢启动”的效果。方法就是不使用均匀的时间步长，而是使用一个“余弦离散化网格”：
$$t(i) = \cos(\frac{\pi}{2}(1 - \frac{i}{T}))$$
这就在采样阶段人为地创造了一个“慢启动、后加速”的节奏，即使模型本身是在一个恒定节奏（线性策略）下训练的。结果表明，这种方法同样能获得高质量的样本。

---

Relation to Existing Work
Continuous-Time Markov Chains (CTMC)
暂时无法在飞书文档外展示此内容
Score parameterization
Lou et al. 和 Benton et al. 提出。他们不让模型预测 x_0，而是预测score：
$$s(x_t,t)_j\triangleq \frac{q_t(j)}{q_t(x_t)}$$
并由此推导出了一系列基于score的discrete diffusion。
作者在本节中证明了该工作的等价性：

$$
s(m, t)_j = \frac{\alpha_t}{1 - \alpha_t} E[x_0|x_t = m]^T e_j,\text{ where } \sum_{j\neq m}s(m,t)_j=\frac{\alpha_t}{1-\alpha_t}
$$

这个公式告诉我们：当当前状态是 [MASK] 时（x*t = m），指向某个具体词 j 的真实分数，正比于“在看到 [MASK] 的情况下，原始词是 j 的条件期望”。
$$s*\theta(m, t)_j = \frac{\alpha_t}{1 - \alpha_t} \mu_\theta(m, t)*j$$
既然真实分数 s 和真实期望 E[x_0|...] 是关联的，那么一个合理的、由神经网络参数化的分数模型 s*θ，就应该和一个合理的、由神经网络参数化的均值模型 μ_θ 通过同样的方式关联起来。
注意上式中规定了所有指向非 [MASK] 状态的分数值的总和必须等于一个特定值，在本文参数化下，这一条件被天然满足，但在score参数化下，这一条件可能被违反，从而可能导致学到的反向模型与理论上的前向过程不兼容，造成效果下降。
Any-order autoregressive models
我们的掩码扩散模型在生成时，其行为可以被完美地重新诠释为一个以均匀随机顺序生成词元的任意顺序自回归模型。它们的损失函数（ELBO）在数学上也被证明是等价的。
但扩散模型引入了掩码策略 α_t 作为一个全新的、可设计的参数。通过调控 α_t 的形状，我们可以控制生成过程中“解封事件”的节奏（例如实现“慢启动”），从而优化 AO-ARM 的生成性能，获得比传统方法更好的结果。

---

Generalization to State-dependent Masking Schedules
在这一节中，作者关注到了不同token实际具有不同的重要性，希望能够结合token重要性来定制mask schedule。
为了实现这种“区别对待”，核心的数学改动是将原来描述解掩码概率的标量 α*t，推广为一个与词典大小相同的向量 α_t：一个 m+1 维的向量。它的第 i 个元素 α*{t,i} 表示在时间 t，类型为 i 的那个特定词（比如 "cat"）保持不被掩码的概率。
现在，α*{t, "artist"} 的函数曲线可以被设计成比 α*{t, "a"} 的函数曲线下降得更快。这意味着在反向生成过程中，模型会被“鼓励”更早地解封 "artist"。
对应的，相应的前后过程公式也会发生转变：
$$\bar{Q}(s, t) = \mathrm{diag}\left(\frac{\alpha_t}{\alpha_s}\right) + \left(I - \mathrm{diag}\left(\frac{\alpha_t}{\alpha_s}\right)\right) \mathbf{1} e_m^\top$$
并且边缘分布将变为：
$$q(x_s|x_0) = \mathrm{Cat}(x_s; \bar{Q}(s)^\top x_0) \quad \text{for} \quad \bar{Q}(s) = \mathrm{diag}(\alpha_s) + (I - \mathrm{diag}(\alpha_s)) \mathbf{1} e_m^\top.$$
有了前向过程，但还需要解决以下的问题：

- 如何构建一个可训练的生成模型 来近似这个更复杂的状态依赖反向过程？

1. 如何让模型自动地、从数据中学习出最佳的、针对不同词类型的掩码策略？
   为此，作者引入了更强大的模型GenMD4：
   GenMD4
   理想的反向过程为：
   $$q(x_s|x_t = m, x_0) = q(x_s|x_t = m, \color{blue}x_0, \color{red}x_0 x_0^\top\color{black}) = \left(\frac{1 - \alpha_s}{1 - \alpha_t}\right)^\top \color{blue}x_0 \color{black}e_m^\top x_s + \left(\frac{\alpha_s - \alpha_t}{1 - \alpha_t}\right)^\top \color{red}x_0 x_0^\top \color{black}x_s.$$
   我们通过用神经网络的预测 μ*θ 来替换公式中未知的真实期望 E[x_0|x_t]，并用 diag(μ*θ) 来近似 $$\mathbb E[x_0x_o^\top|x_t]=\text{diag}(\mathbb E[x_0|x_t])$$，最终的损失函数为：
   $$\mathcal{L}_\infty = \int_0^1 \left( \frac{\alpha_t'}{1 - \alpha_t} \right)^\top \mathbb{E}_{q(x_t|x_0)} \left[ \delta_{x_t,m} \cdot (x_0 - \mu_\theta(x_t,t)) + x_0 x_0^\top \log \mu_\theta(x_t,t) \right] \mathrm{d}t.$$
   公式推导
   Alpha-Learning
   将 α*t 向量的第 i 个元素（对应词 i 的策略）定义为一个简单的函数:$$α*{t,i} = 1 - t^{w*i}$$。这里的 w_i 是一个可学习的参数。我们的优化目标不仅包括神经网络的参数 θ，还包括这 m+1 个掩码速度参数 w = {w_0, w_1, ..., w_m}。
   但直接优化 w 会遇到一个巨大的技术障碍：
   我们的损失函数 L*∞ 中包含一项期望 E\_{q(x_t|x_0)}[...]。而该期望是在分布 q(x_t|x_0) 上计算的。在实践中，我们通过从这个分布中采样一个 x_t 来近似这个期望。分布 q(x_t|x_0) 本身是依赖于参数 w 的（因为 w 决定了 α_t，而 α_t 决定了 q）。但采样是一个离散的、不可微分的操作。梯度不能从损失函数“穿透”这个采样步骤，返回到参数 w。
   为解决该问题，作者提出使用REINFORCE算法，用当前的 w 采样一个 x_t，然后计算一个“奖励”（这里是负的损失）。如果这个“奖励”比我们预期的平均奖励要高，我们就调整 w，使得未来更有可能采样到类似 x_t 的样本。反之，如果奖励低，我们就调整 w 来降低采样到类似 x_t 的可能性。通过这种方式，REINFORCE 算法可以为参数 w 提供一个无偏的梯度估计，即使它隔着一个离散的采样操作。这使得我们能够有效地、端到端地同时优化神经网络参数 θ 和掩码策略参数 w。
   该噪声调度仍存在一定的问题，首先是优化的困难，由于无法梯度下降，噪声调度参数难以优化；其次，该噪声调度关注了每个词元的特性，但并没有关注到每个句子的语义，实际上噪声调度不仅仅与token的特征有关，也和token所处的序列语义有关。

---

Experiments
Text
文本是天然的离散数据，本文在三大文本数据集上进行了实验：

- text8: character-level text modeling benchmark
- OpenWebText: an open clone of the unreleased WebText dataset used to train GPT-2
- FineWeb-Edu: a high-quality dataset of fine educational text
  作者在text8与OpenWebText上进行了似然函数的评估，并在FineWeb-Edu上进行了下游任务表现测试。
  指标：
- Perplexity: $$\text{PPL} = 2^{ - \frac{1}{N} \sum_{i=1}^N \log_2 p(x_i) }$$ 困惑度表示模型在每个预测时“平均困惑多少种选择”。
- BPC：Bits Per Character， $$\text{BPC} = - \frac{1}{N} \sum_{i=1}^{N} \log_2 p(x_i)$$，BPC = “压缩一个字符需要多少比特”。
  OpenWebText

Text8

FineWeb-Edu

相比于AR模型，达到相同效果，Diffuison需要更大的训练步数。
Pixel-level image modeling
作者在CIFAR-10与ImageNet 64×64上进行了评测，采用FID这一指标
