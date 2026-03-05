---
layout: distill
title: Masking Schedulers of Mask Diffusion Model
date: 2026-02-01 05:09:27
blog_hidden: false
description: In Mask Diffusion Models (MDM), the Noise Scheduler is pivotal for learning capacity and sampling quality. This paper presents a unified analysis addressing three core challenges —— Exposure Bias induced by Absorb mechanisms, efficiency bottlenecks from Intrinsic Order, and joint probability deviations from Independence Assumptions. We systematically review mainstream strategies, comparing their efficacy in semantic capture, remasking, and efficiency to elucidate how refined scheduling reshapes token dependencies. Finally, we outline future directions for overcoming these underlying logical defects.
tags: Masked-Diffusion-Model scheduler Intrinsic-Order
categories: Notes
authors:
  - name: Runze Tian
    url: "https://gua927.github.io"
    affiliations:
      name: GenSI Lab, THU-AIR
      url: "https://www.gensi-thuair.com/#/portal_home"
giscus_comments: true
related_posts: true
bibliography: papers.bib
toc:
  - name: Introduction
  - name: Background
    subsections:
      - name: Masked diffusion model
      - name: Exposure Bias
      - name: Intrinsic Order
      - name: Token Independence
      - name: Remark
  - name: Current Schedulers
    subsections:
      - name: Random Schedulers
      - name: Confidence-based Schedulers
      - name: Planer-based Schedulers
      - name: RL-Policy Planer
      - name: Manual Selected Schedulers
  - name: Final Remark
---

[Click! Read Chinese Version](/blog/2026/masking-schedulers-cn/)

## Introduction

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure0.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="During inference, adaptive MDM can avoid difficult problem instances, improving performance" %}

In the research of Diffusion Models, the selection of the Noise Scheduler not only determines the sampling trajectory but also directly influences the upper bound of the model's learning capability for complex distributions. Compared to Gaussian diffusion in the continuous domain, Mask Diffusion Models (MDM) have demonstrated unique potential in discrete data modeling. However, the Absorb Transition Kernel universally employed in MDM, while endowing the model with logical simplicity, also introduces three key theoretical and engineering challenges.

Firstly, Exposure Bias is particularly severe in MDM. Due to the Absorb mechanism, generated tokens possess an "immutable" characteristic. Any slight deviation during the sampling process accumulates rapidly over timesteps due to the lack of a correction mechanism, leading to the collapse of the generated sequence.

Secondly, the complexity of Intrinsic Order limits training efficiency. Real-world data (such as natural language or code) often exhibits diverse dependency orders, while MDM attempts to capture all possible generation paths within a limited parameter space. Since it is impossible to enumerate and achieve optimal learning for every generation order in polynomial time, the model often faces insufficient training efficiency and struggles to achieve convergence on complex structured data.

The most fundamental challenge lies in the probability deviation caused by the Independence Assumption. During the modeling phase, MDM often assumes conditional independence between tokens, which contradicts the essence of strong couplings in discrete data. This leads to a severe deviation of the estimated product of marginal distributions from the true joint probability distribution.

Therefore, **designing a sophisticated Noise Scheduler is no longer merely a simple scaling of mask ratios, but a pivotal mechanism for compensating for the underlying logical defects of MDM.** An optimized scheduler not only needs to alleviate exposure bias through fine-grained mask rate balancing but, more critically, <u>aims to reshape the dependency dynamics between tokens through the evolution of the time trajectory.</u>

In addition to seeking a "zero-full-correlation path" to statistically approximate an independent distribution, a more forward-looking correction scheme lies in breaking the single-token processing paradigm. For instance, by introducing Block-wise Masking, the Scheduler can force the model to capture locally strongly coupled semantic clusters, alleviating the joint probability deviation caused by the independence assumption at a mechanistic level. Meanwhile, introducing Iterative Remasking endows the model with the ability to "regret" and correct on the inference trajectory, breaking the unidirectional irreversibility of the Absorb Kernel. Coordinated dynamically by the Scheduler, these strategies not only reduce information entropy loss in a statistical sense but essentially act as a calibrator between the independence assumption and structured dependencies, thereby achieving accurate restoration of the deep joint probability distribution of discrete data while ensuring computational efficiency.

Based on the above observations, this paper reviews and summarizes current improvement strategies for Noise Schedulers in the field of Mask Diffusion Models, focusing on exploring how to break through the aforementioned bottlenecks through the refined design of the scheduler.

## Background

### Masked Diffusion Model

#### Forward Process

Suppose there are $C$ types of tokens. Including the mask, the state space of each token is $\{0, 1, \dots, C\}$. Let the token sequence be denoted as $x = x^1 x^2 \dots x^L$, where $x_0 \sim p_{\text{data}}$. The conditional probability is given by $q(x_t \mid x_0) = \prod_{i=1}^L q(x_t^i \mid x_0^i)$, where

$$q(x_t^i \mid x_0^i) = (1 - \alpha_t) \delta_{\text{m}}(x_t^i) + \alpha_t \delta_{x_0^i}(x_t^i)$$

indicates that at time $t$, each token independently transitions to a mask with probability $1 - \alpha_t$ or remains unchanged with probability $\alpha_t$.

#### Reverse Process

Let $0 \leq s \le t \leq 1$. In the reverse process, to calculate $p(x_s^i \mid x_t^i)$, we first consider $q(x_t^i \mid x_s^i)$:

$$
q(x_t^i \mid x_s^i) =
\begin{cases}
\frac{\alpha_s - \alpha_t}{\alpha_s} \delta_{\text{m}}(x_t^i) + \frac{\alpha_t}{\alpha_s} \delta_{x_s^i}(x_t^i) & \text{if } x_s^i \in \mathcal{X}, \\
\delta_{x_s^i}(x_t^i) & \text{if } x_s^i = \text{m}.
\end{cases}
$$

By Applying Bayes' theorem, we obtain:

$$
q(x_s^i \mid x_t^i, x_0^i) =
\begin{cases}
\delta_{x_t^i}(x_s^i) & \text{if } x_t^i \in \mathcal{X}, \\
\frac{1 - \alpha_s}{1 - \alpha_t} \delta_{\text{m}}(x_s^i) + \frac{\alpha_s - \alpha_t}{1 - \alpha_t} \delta_{x_0^i}(x_s^i) & \text{if } x_t^i = \text{m}.
\end{cases}
$$

This indicates that when a token $x_t^i$ is a mask, it remains a mask with probability $\frac{1 - \alpha_s}{1 - \alpha_t}$ and transitions to $x_0^i$ with probability $\frac{\alpha_s - \alpha_t}{1 - \alpha_t}$. Once a token is unmasked, it is no longer updated.

#### Model and Loss

Since $p(x_s \mid x_t) = \mathbb{E}\_{p(x_0 \mid x_t)}[q(x_s \mid x_t, x_0)]$, the reverse process can be implemented by first sampling $x_0 \sim p(\cdot \mid x_t)$ and then sampling $x_s \sim q(\cdot \mid x_t, x_0)$. As $x_0$ is unknown, we model it via $p_{\theta}(x_0 \mid x_t) = \prod_{i=1}^L p_{\theta}(x_0^i \mid x_t)$ and optimize it using the following loss function:

$$
\mathcal{L}_{\text{vb}}(\mathbf{x}_0; \theta) = \int_0^1 \frac{\alpha_t'}{1 - \alpha_t} \mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)} \left[ \sum_{i=1}^{L} \log p_\theta(x_0^i \mid x_t^i) \right] dt,
$$

#### Inference

In the inference phase, given a fully masked sequence $\mathbf{x}\_L = [\text{m}, \text{m}, \dots, \text{m}]$, we need to obtain $\mathbf{x}\_0$ through $T$ denoising steps. Based on previous discussions, the state transition at each time step $t$ depends not only on the model's predictive capability $p_\theta(x_0^i \mid \mathbf{x}\_t)$ but also crucially on the choice of the Sampling Path.

We model the inference from $t+1 \rightarrow t$ as a two-step process: first, a selection strategy $\pi(i \mid x_{t+1}), i=1, 2, \dots, L$ is used to determine the positions to be unmasked, denoted as $M_{t+1}$. Then, for the positions in $M_{t+1}$, we sample according to the logits predicted by the model, i.e., $x_t^i \sim p_{\theta}(x_t^i \mid x_{t+1})$. Thus, a single inference step is given by:

$$x_t^i = \begin{cases}  \sim p_{\theta}(x^i \mid \mathbf{x}_{t+1}), & \text{if } i \in M_{t+1} \quad \text{(Selected to be unmasked in the current step)} \\ x_{t+1}^i, & \text{if } i \notin M_{t+1} \quad \text{(State from the previous step is maintained)} \end{cases}$$

Namely:

$$[\mathbf{T}_{t+1 \rightarrow t}^M](\mathbf{x}_{t+1}, \mathbf{x}_t) =  \begin{cases}   \underbrace{\pi(M \mid \mathbf{x}_{t+1})}_{\text{Selection}} \cdot \underbrace{\prod_{i \in M} p_\theta(x_t^i \mid \mathbf{x}_{t+1})}_{\text{Generation}} & \text{if } \mathbf{x}_t^{-M} = \mathbf{x}_{t+1}^{-M} \\  0 & \text{otherwise}  \end{cases}$$

### Exposure Bias

First, we define two core probability measures existing in the same state space $\mathcal{X}^L$:

- Training State Distribution ($P_{t}$): The marginal distribution of real data after being projected by the forward process $q$.

$$P_{t}(\mathbf{x}_t) = \int p_{\text{data}}(\mathbf{x}_0) q(\mathbf{x}_t \mid \mathbf{x}_0) d\mathbf{x}_0$$

- Inference State Distribution ($\hat{P}_{t}$): The distribution generated iteratively from the fully masked state $\mathbf{x}_L$ using the inference operator $\mathbf{T}^M$.

$$\hat{P}_{t}(\hat{\mathbf{x}}_t) = \sum_{\hat{\mathbf{x}}_{t+1}} \hat{P}_{t+1}(\hat{\mathbf{x}}_{t+1}) \mathbf{T}_{t+1 \to t}^M(\hat{\mathbf{x}}_t \mid \hat{\mathbf{x}}_{t+1})$$

Our objective is the total distributional shift at time $t$: $\Delta_t = D_{KL}(P_t \Vert \hat{P}_t)$.

By applying the chain rule of KL divergence, we can establish the evolutionary relationship of $\Delta_t$.

Let $\mathcal{Q}\_{t+1 \to t}$ be the optimal reverse transition operator induced by the ground-truth data distribution. Clearly, $P_t = \mathcal{Q}\_{t+1 \to t} P_{t+1}$. Then:

$$\Delta_t = D_{KL}\left( \mathcal{Q}\_{t+1 \to t} P_{t+1} \big\| \mathbf{T}\_{t+1 \to t}^M \hat{P}\_{t+1} \right)$$

By introducing an intermediate term $\mathbf{T}\_{t+1 \to t}^M P_{t+1}$ and utilizing the triangle inequality of divergence, the following upper bound can be derived:

$$\Delta_t \le \underbrace{D_{KL}(\mathcal{Q}_{t+1 \to t} P_{t+1} \| \mathbf{T}_{t+1 \to t}^M P_{t+1})}_{\color{red}{\text{Single-step local error } \mathcal{E}_{\text{loc}}}} + \underbrace{D_{KL}(\mathbf{T}_{t+1 \to t}^M P_{t+1} \| \mathbf{T}_{t+1 \to t}^M \hat{P}_{t+1})}_{\color{red}{\text{History error propagation}}} \leq \mathcal{E}_{\text{loc}} + \Delta_{t+1}$$

By optimizing $\mathbf{T}_{t+1 \to t}^M$, we can simultaneously minimize individual local errors, thereby reducing $\Delta_t$.

### Intrinsic Order

We first formalize the concept of "intrinsic order" of data. Let the complete sequence be

$$
\mathbf{x}_0 = (x_0^1, x_0^2, \dots, x_0^L) \sim p_{\text{data}}
$$

For any subset $S \subset \{1,\dots,L\}$, we define its conditional generation difficulty as:

$$
\mathcal{H}(S)
\;\triangleq\;
\mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}}
\left[
H\big( \mathbf{x}_0^S \mid \mathbf{x}_0^{\bar S} \big)
\right]
$$

This quantity characterizes the uncertainty of the tokens in the set $S$ given that the remaining tokens are known.

Thus, the data distribution $p_{\text{data}}$ induces a family of equivalent optimal generation orders:

$$
\mathcal{O}^*
=
\arg\min_{\pi \in \mathfrak{S}_L}
\sum_{k=1}^L
\mathcal{H}\big(\{\pi_k\} \mid \{\pi_1,\dots,\pi_{k-1}\}\big)
$$

where $\pi$ is a permutation (generation order) of the tokens.

However, defining a single or fixed optimal generation order at the distribution level is unreasonable in practice, as different samples $\boldsymbol{x}\_0$ often correspond to different $\pi^*(\boldsymbol{x}\_0)$. This phenomenon reflects the diversity of generation orders ubiquitously existing in real-world discrete data. For a specific sample $x_0\sim p_{data}$, its optimal generation order generally depends on the embodied local structure and dependency relationships, which can be formalized as:

$$\pi^*(\boldsymbol{x}_0)=\operatorname*{arg\,min}_{\pi\in \mathfrak{S}_L }\sum_{k=1}^LH\bigg(x_0^{\pi_k}\mid \boldsymbol{x}_0^{\{\pi_1,\cdots,\pi_{k-1}\}}\bigg)$$

#### Impact on Training Complexity

In standard MDM training, the noise scheduler commonly employs a uniform sampling strategy, assuming that all tokens have an equal probability of being masked. This implies that the model is forced to learn the average of all possible generation permutations $\pi \in \mathfrak{S}_L$ during training.

However, for data with strong structural dependencies, there is a tremendous difference in the learning difficulty of different generation paths. We can decompose the expected loss during training into a permutation-based weighted sum:

$$\mathcal{L}_{\text{total}} \approx \mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{\pi \sim U(\mathfrak{S}_L)} \left[ \sum_{k=1}^L -\log p_\theta(x_0^{\pi_k} \mid \mathbf{x}_0^{\{\pi_1, \dots, \pi_{k-1}\}}) \right]$$

Due to the limited parameter space, the model is unable to fit the marginal distributions of all $L!$ paths in polynomial time. <span style="color: #8e44ad; font-weight: bold;">The uniform scheduler forces the model to predict high-entropy, highly-dependent tokens at early time steps, which leads to gradient conflicts in the optimization objective.</span>

This conflict not only slows down the convergence speed but also theoretically lowers the upper bound of the model's achievable performance. We can define the Optimization Gap, denoted as $\Delta \mathcal{L}_{\text{order}}$, to quantify the additional complexity introduced by neglecting the intrinsic order of the data:

$$
\begin{aligned}
\Delta \mathcal{L}_{\text{order}} = \mathbb{E}_{\mathbf{x}_0} \left[ \sum_{k=1}^L \left( \underbrace{\mathbb{E}_{\pi \sim U} [H(x_0^{\pi_k} \mid \mathbf{x}_0^{\pi_{\lt k}})]}_{\color{purple}{\text{Avg. Complexity (Uniform)}}} - \underbrace{H(x_0^{\pi^*_k} \mid \mathbf{x}_0^{\pi^*_{\lt k}})}_{\color{purple}{\text{Min. Complexity (Optimal)}}} \right) \right]
\end{aligned}
$$

where $\pi^*$ represents the optimal generation order that minimizes the sum of sequence conditional entropies. The presence of $\Delta \mathcal{L}_{\text{order}}$ reveals a core contradiction: the model not only has to learn "easy-to-learn paths" that comply with causality ($P(\text{Result}\mid\text{Cause})$), but is also forced to consume a massive amount of parameters to fit counter-intuitive reverse or disordered paths $P(\text{Cause}\mid\text{Result})$.

This phenomenon triggers the problem of capacity dilution. According to Rate-Distortion Theory, if we consider model parameters $\theta$ as an information channel $C_\theta$ with limited capacity, the uniform sampling strategy artificially elevates the required information encoding rate of the task from $R_{\text{opt}}$ to $R_{\text{uni}}$. When $R_{\text{uni}} > C_\theta$, the model cannot perfectly fit this high-entropy mixture distribution, inevitably yielding irreducible distortion.

In other words, **$\Delta \mathcal{L}\_{\text{order}}$ essentially measures how much model capacity is wasted on fitting invalid noisy paths.** <u>This passive dispersion of resources leads to a reduction in the available parameters when the model deals with truly important forward dependencies,</u> thus coercing the joint distribution $P_\theta(\mathbf{x}_0)$ learned by the model to exhibit a smoothed characteristic, making it difficult to capture the originally sharp multimodal structures in the data distribution. Therefore, designing a Scheduler that can approximate $\pi^*$ is essentially conducting a form of curriculum learning, aiming to eliminate the entropy surge brought by $\Delta \mathcal{L}\_{\text{order}}$, thereby approaching a higher performance upper bound with a limited parameter scale.

#### Impact on Inference Accuracy

During the inference phase, when the denoising path selected by the noise scheduler $\pi$ violates the intrinsic order $\pi^*$ of the data, it triggers a cascading accuracy collapse.

When the scheduler forces the model to prioritize forecasting downstream tokens in the absence of context, the model confronts a high-entropy distribution.

Suppose $x^A \to x^B$ is the true causal dependency. If $\pi$ reveals $x^B$ first, the model has to sample from the marginal distribution:

$$
x^B \sim p_\theta(x^B \mid \emptyset) = \sum_{x^A} p(x^B \mid x^A)p(x^A)
$$

Compared to the conditional distribution $p(x^B\mid x^A)$, this distribution is extremely flat, making the sampling highly susceptible to hitting low-probability or incorrect tokens.

### Token Independence

Let $\mathcal{M}\_t$ denote the set of masked tokens at time step $t$, and $c = x\_{t+1}$ be the current context. In the step $t+1 \to t$, we need to unmask a subset $\mathcal{U} \subset \mathcal{M}\_{t+1}$. The model approximates the true distribution $P(x_\mathcal{U} \mid c)$ using an independent distribution $Q_\theta(x_\mathcal{U} \mid c) = \prod_{i \in \mathcal{U}} Q_\theta(x_i \mid c)$. Our objective to minimize is $D_{\text{KL}}(P \Vert Q_\theta)$. By introducing the product of true marginal distributions $\prod\_{i \in \mathcal{U}} P(x_i \mid c)$ as an intermediate term, we can perfectly decompose this KL divergence:

$$\begin{aligned} D_{\text{KL}}(P \| Q_\theta) &= \mathbb{E}_{x \sim P} \left[ \log \frac{P(x_\mathcal{U}\mid c)}{\prod_{i \in \mathcal{U}} Q_\theta(x_i\mid c)} \right] \\ &= \mathbb{E}_{x \sim P} \left[ \log \left( \frac{P(x_\mathcal{U}\mid c)}{\color{blue}{\prod_{i \in \mathcal{U}} P(x_i\mid c)}} \cdot \frac{\color{blue}{\prod_{i \in \mathcal{U}} P(x_i\mid c)}}{\prod_{i \in \mathcal{U}} Q_\theta(x_i\mid c)} \right) \right] \\ &= \underbrace{\mathbb{E}_{x \sim P} \left[ \log \frac{\prod_{i \in \mathcal{U}} P(x_i\mid c)}{\prod_{i \in \mathcal{U}} Q_\theta(x_i\mid c)} \right]}_{\color{green}{\text{Term 1: Sum of Marginal Errors}}} + \underbrace{\mathbb{E}_{x \sim P} \left[ \log \frac{P(x_\mathcal{U}\mid c)}{\prod_{i \in \mathcal{U}} P(x_i\mid c)} \right]}_{\color{green}{\text{Term 2: Total Correlation (Dependency)}}} \end{aligned}$$

- Term 1: Sum of Marginal Errors (corresponding to standard CE Loss)

  $$\text{Term 1} = \sum_{i \in \mathcal{U}} D_{\text{KL}}(P(x_i\mid c) \| Q_\theta(x_i\mid c))$$

  Since $D_{\text{KL}}(P_i \Vert Q_i) = \text{CrossEntropy}(P_i, Q_i) - H(P_i)$, and $H(P_i)$ is constant with respect to the model parameters $\theta$, minimizing this term is entirely equivalent to minimizing the standard Cross Entropy Loss.

- Term 2: Total Correlation Error

  $$\text{Term 2} = D_{\text{KL}}\left( P(x_\mathcal{U}\mid c) \bigg\| \prod_{i \in \mathcal{U}} P(x_i\mid c) \right) = \left( \sum_{i \in \mathcal{U}} H(x_i\mid c) \right) - H(x_\mathcal{U}\mid c)$$

  This measures the degree to which the variables deviate from independence. If the tokens in $x_\mathcal{U}$ are mutually conditionally independent given $c$, this term is 0. The stronger the dependency, the larger this term becomes.

Based on the mathematical decomposition of Total Correlation (TC) above, we can gain a profound understanding of the core mission of the Noise Scheduler during the inference phase. Because the model parameters $\theta$ are fixed during inference, the upper bound of Term 1 (Marginal Errors) has already been determined at the end of training. Therefore, the improvement of inference quality depends entirely on whether Term 2 (Total Correlation) can be minimized through the strategic selections of the Scheduler.

In other words, during the inference process, <span style="color: #e74c3c; font-weight: bold;">the Scheduler is no longer merely a simple progress bar controller (deciding how many tokens to unmask), but instead plays the role of a "dependency decoupler."</span>

### Remark

Recalling the previous derivation of Exposure Bias, the accumulation of the total error $\Delta_t$ originates from the local error $\mathcal{E}\_{\text{loc}}$ at each step. In the $t+1 \to t$ stage of inference, given the context $c = \mathbf{x}\_{t+1}$, the Scheduler $\pi$ first decides the subset to unmask $\mathcal{U} \sim \pi(\cdot\mid c)$, which is subsequently filled by the model $p_\theta$.
We re-formalize the single-step local error $\mathcal{E}\_{\text{loc}}$ as an expectation over the policy $\pi$:

$$\mathcal{E}_{\text{loc}}(\pi, \theta) = \mathbb{E}_{\mathcal{U} \sim \pi(\cdot \mid c)} \left[ D_{KL}\Big(~P(x_\mathcal{U} \mid c) ~\Big\| \underbrace{\prod_{i \in \mathcal{U}} p_\theta(x_i \mid c)}_{\text{Independence Assumption}} \Big) \right]$$

We can further decompose this error into two parts, thereby clearly separating the influences of Intrinsic Order and Independence:

$$\mathcal{E}_{\text{loc}}(\pi, \theta) = \mathbb{E}_{\mathcal{U} \sim \pi} \left[ \underbrace{\sum_{i \in \mathcal{U}} D_{KL}(P(x_i\mid c) \| p_\theta(x_i\mid c))}_{\color{orange}{\text{(I) Marginal Precision Error}}} + \underbrace{\mathbb{I}(\lvert\mathcal{U}\rvert > 1) \cdot \text{TC}(x_\mathcal{U} \mid c)}_{\color{orange}{\text{(II) Structural Dependency Error}}} \right]$$

where $\mathbb{I}(\cdot)$ is the indicator function, emphasizing that term (II) only exists when multiple tokens are unmasked in a single step.

<span style="color: #27ae60; font-weight: bold;">In summary, the intrinsic order and independence of data simultaneously affect training and inference. When the number of decoded tokens in a single step $\mathcal U$ is large, independence will significantly restrict model accuracy, but at the same time it reduces the violation of the data's intrinsic order; conversely, when $\mathcal U$ is very small, the impact of independence is greatly weakened, but the restriction of the data's intrinsic order is enhanced.</span> By designing a rational mask/unmask policy $\pi$ such that model training and inference can conform to the inherent order of the data, while mitigating the impact of Independence as much as possible, we can significantly improve the model's training efficiency, model capacity, and alleviate exposure bias.

## Current Schedulers

### Random Schedulers

The random strategy serves as the baseline implementation of MDMs (such as MDLM, MD4, SEDD)<d-cite key="sahoo2024simpleeffectivemaskeddiffusion,shi2025simplifiedgeneralizedmaskeddiffusion,lou2023discretediffusionestimating"></d-cite>. It assumes that all tokens possess an equal status, thereby ignoring the intrinsic order of the data.

- Training
  During training, the time step $t \in [0, 1]$ is sampled uniformly, with a corresponding mask ratio of $1 - \alpha_t$. For any position $i$ in the sequence, its probability of being selected is independent:

  $$\pi(M \mid \mathbf{x}_0) = \binom{L}{k}^{-1}, \quad \text{where } k \approx (1-\alpha_t)L$$

  At this point, the loss function $\mathcal{L}_{\text{vb}}$ degrades into a uniform expectation over all possible permutations $\mathfrak{S}_L$.

- Inference
  At each inference step $t+1 \to t$, the policy $\pi$ randomly selects a fixed proportion (determined by the scheduler) of positions from the current mask set to unmask:

  $$M_{t+1} \subset \{i \mid x_{t+1}^i = \text{m}\}, \quad \text{s.t. } \lvert M_{t+1} \rvert = \lfloor (\alpha_t - \alpha_{s})L \rfloor$$

  The selection process is independent of the context $c$, i.e., $\pi(M \mid \mathbf{x}_{t+1}) = \pi(M)$. The model independently predicts the probability distribution of all masked positions at each step:

  $$x_t^i \sim p_\theta(x^i \mid \mathbf{x}_{t+1}) \quad \forall i \in M_{t+1}$$

### Confidence-based Schedulers

Unlike random strategies, confidence-based strategies (Train for the worst, Plan for the best, MGDM)<d-cite key="kim2025trainworstplanbest,ye2024beyondautoregressiondiscretediffusion"></d-cite> attempt to dynamically discover "easy-to-learn paths" through the model's prediction results. The core logic is that the more confident the model is about a token, the more inclined it is to be unmasked first, thereby conforming to the intrinsic order of the data $\pi^*$. Under this framework, the policy $\pi$ is no longer a uniform distribution but depends on the probability distribution $p_\theta(x^i \mid \mathbf{x}_{t+1})$ output by the model.

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure1.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="The figure illustrates the comparison between generation perplexity (left axis) and predicted entropy (right axis) under different sampling steps: confidence-driven adaptive inference can maintain a lower perplexity with fewer steps, and approaches high-step performance with a smoother entropy trajectory." %}

- Selection score ($s_i$):
  For each masked position $i \in \{i \mid x_{t+1}^i = \text{m}\}$, its score is calculated:
  - Logit Top-1: $s_i = \max_x \log p_\theta(x^i = x \mid \mathbf{x}\_{t+1})$
  - Logit Margin: $s_i = \text{logit}\_{max1} - \text{logit}\_{max2}$ (measuring class distinguishability)
  - Negative Entropy: $s_i = -H(p_\theta(x^i \mid \mathbf{x}\_{t+1}))$ (measuring prediction certainty)
- Inference ($\pi_{\text{inf}}$):

  At step $t+1 \to t$, the $s_i$ of all masked positions is calculated, and the top $k$ positions with the highest scores are selected:

  $$M_{t+1} = \text{arg-topk}_{i} \{s_i\}, \quad \text{where } k = \lfloor (\alpha_t - \alpha_s)L \rfloor$$

  The corresponding generation process is:

  $$\pi(M \mid \mathbf{x}\_{t+1}) = \begin{cases} 1 & \text{if } M = M_{t+1} \\ 0 & \text{otherwise} \end{cases}$$

  To preserve a certain degree of randomness, a topk ratio can also be designed to control the proportion of topk selections, meaning a portion of topk tokens and a portion of randomly selected tokens are placed into $M_{t+1}$.

- Training

  In current works, there is no explicit implementation of a confidence-based policy for masking during training. However, some works implement preference training for specific tokens or specific generation orders by reweighting the loss function. For instance, in MGDM<d-cite key="ye2024beyondautoregressiondiscretediffusion"></d-cite>, the model's preference for different generation paths is implicitly adjusted through token-level reweighting of the loss function. Its training objective function is defined as:

  $$L_{\text{MGDM}} = \sum_{n=1}^N \sum_{t=1}^T w(t)v(\mathbf{x}_{t,n})u(\mathbf{x}_0, \mathbf{x}_t, n; \theta)$$

  where $v(\mathbf{x}\_{t,n}) = \alpha(1 - \exp(-u(\cdot)))^\beta$ is an adaptive token-level reweighting term. When $\beta > 0$, this term reduces the weight of easy-to-learn tokens (low loss) and amplifies the weight of hard-to-learn tokens (high loss). From an optimization objective perspective, this training strategy aims to improve global training efficiency by "forcing" the model to focus on those hard samples that are currently inaccurately predicted.

- Advantages:

  - The logits output by the denoiser after adequately learning the data can be viewed as a good estimation of token difficulty. The generation path guided by logits can adapt to different inputs to obtain an appropriate generation order.

- Shortages:

  - The estimation of logits may be inadequate if the training data is biased or the training is insufficient.
  - When the vocabulary is very large (30k+), the logits of many tokens are close, leading to unreasonable sorting among tokens.
  - This method has no effect on mitigating Token Independence under small-step sampling.

### Planer-based Schedulers

In this section, we introduce several schedulers learned by the model to reduce inductive bias and obtain better generalization.

#### Noise Predict Planer

**_Paper: THINK WHILE YOU GENERATE: DISCRETE DIFFUSION WITH PLANNED DENOISING (ICLR 2025)<d-cite key="liu2024thinkwhilegeneratediscretediffusion"></d-cite>_**

DDPD consists of two independent neural networks:

- Planner ($p_\theta(z_t^d \mid x_t)$): Executes a binary classification task to predict whether each dimension $d$ in the sequence is corrupted (i.e., in a noise state $z_t^d = N$). It only outputs one logit for each dimension.

- Denoiser ($p_{1\mid t}^\theta(x_1^d \mid x_t, z_t^d = N)$): Predicts the true data distribution conditioned on the corresponding position being classified as noise.
  DDPD no longer relies on a preset time step but inversely deduces the actual time through the global noise level estimated by the planner: $\tilde{t} = \alpha_t^{-1}(1 - \sum_{d'} p_\theta(z_t^{d'} = N \mid x_t) / D)$, and uses the actual time as the time embedding for the denoiser.

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure2.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="Schematic diagram of the DDPD (Discrete Diffusion with Planned Denoising) framework: The model consists of a Planner and a Denoiser. The Planner first outputs the noise state probability $p_\theta(z_t^d\mid x_t)$ for each position, identifying the token that needs repair most urgently; subsequently, the Denoiser performs conditional prediction $p_{1\mid t}^\theta(x_1^d\mid x_t,z_t^d=N)$ on the selected noise positions. Unlike fixed-step schedulers, DDPD reversely deduces the actual time embedding $\tilde t=\alpha_t^{-1}\!\left(1-\frac{1}{D}\sum_{d'}p_\theta(z_t^{d'}=N\mid x_t)\right)$ by estimating the global noise ratio, thereby iterating the generation in a 'plan first, denoise later' sequence, improving sampling efficiency and final quality." %}

- Advantages:

  - An additional planner is trained, avoiding model capacity constraints.
  - Enables remasking.
  - Due to single-step sampling of one token, there is no impact from Token Independence.

- Shortages:

  - The rank of logits output by the planner is not necessarily a good estimate of the data's intrinsic order.
  - Iteration of single-step sampling of one token + remasking makes the generation speed excessively slow.

### RL-Policy Planer

RL-Policy Planer aims to model the denoising process of discrete diffusion models as a Markov Decision Process, using reinforcement learning to learn the optimal denoising trajectory or unmask policy. Unlike heuristic-based schedulers (such as random or confidence sampling), RL-Policy can adaptively adjust the generation path based on the final reward of downstream tasks (e.g., accuracy of mathematical problems, logical correctness of reasoning), thereby exhibiting superiority in more complex generation scenarios.

#### Freeze backbone with trainable Policy head

**_Paper: IMPROVING DISCRETE DIFFUSION UNMASKING POLICIES BEYOND EXPLICIT REFERENCE POLICIES (ICLR 2026)_**<d-cite key="hong2026improvingdiscreteunmasking"></d-cite>

The core idea of this architecture is a clear division of labor: the underlying backbone already possesses powerful conditional probability modeling capabilities, while the Policy Head acts as a scheduler, specifically responsible for evaluating the unmasking priority of each position based on the current context.

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure3.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="Schematic of the RL-Policy Planner architecture with a frozen backbone and trainable Policy Head: At each step $t+1\to t$, the policy network outputs an unmasking priority $\pi_\phi(i\mid x_{t+1})$ for each position based on the current sequence state, selects the tokens to be preferentially denoised, and then the frozen foundation MDM executes conditional generation. This design decouples 'positional decision' and 'content generation', effectively reducing training overhead while optimizing the sampling sequence through reward signals to elevate inference quality." %}

- Implementation:

  - Architecture Decoupling: Freezes the pretrained discrete diffusion model parameters $\theta$, and additionally designs a lightweight Policy Head $\phi$ (usually a Multi-Layer Perceptron or small Transformer layers).
  - Policy Parameterization: Given the current sequence containing masks $x_{t+1}$, the Policy Head outputs a score vector $s = h_\phi(x_{t+1})$, which is converted into a probability distribution $\pi_\phi(i \mid x_{t+1})$ for selecting position $i$ via Softmax.
  - Reward Guidance: If the finally generated sequence $x_0$ is correct, all decisions along that path are rewarded.

- Advantages:

  - Lightweight, and the learned policy head may possess generalization potential and support transfer.

- Shortages:
  - Training only a lightweight head results in limited capability enhancement.

#### Trainable backbone with fixed path

**_Paper: MDPO: OVERCOMING THE TRAINING-INFERENCE DIVIDE OF MASKED DIFFUSION LANGUAGE MODELS_**<d-cite key="he2025mdpoovercomingtraininginference"></d-cite>

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure4.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="MDPO generates a group of answers given a query for RL rollouts. Then all completions at intermediate and final steps are verified with a reward model. Based on verified rewards, MDPO estimates the advantage of step t by considering rewards of the other steps in the current rollout and step t from other rollouts in the group. These estimated advantages are used for policy optimization." %}

Unlike the aforementioned schemes, MDPO does not alter the logic of position selection (still following the confidence-based heuristic) but directly optimizes the backbone itself through RL training, enabling its output logits to more perfectly adapt to the scheduling trajectory during inference.

- Implementation

  - End-to-end Optimization: The backbone parameters $\theta$ are trainable. In the RL stage, the model performs sampling generation according to the confidence-based policy used during inference.

  - RL simulates the "progressive unmasking" path during the inference process. By comparing the cumulative rewards of different trajectories (such as using Group Relative Policy Optimization, GRPO), the model is forced to produce more accurate predictions under the current inference distribution.

    The RL objective increases the logit values of tokens on the correct path and suppresses the values of erroneous/interfering paths. This ensures the model's confidence scores during inference are no longer merely a byproduct of the training loss, but a heuristic function equipped with authentic navigational capabilities.

- Advantages:

  - Small-step sampling can be directly rewarded, achieving better acceleration effects.
  - Updating the policy directly on a confidence-based foundation aligns training and inference, and avoids model capacity dilution.
  - The upper bound is higher after training the backbone.

- Shortages:
  - By compressing the paths into a single one, the model's generalization capability may be significantly reduced.

### Manual Selected Schedulers

In this section, we introduce several manually selected schedulers.

#### Subsequence-based Schedulers (SPMDM)

**_Paper: SPMDM: Enhancing Masked Diffusion Models through Simplifying Sampling Path_**<d-cite key="zhu2025spmdmenhancingmaskeddiffusion"></d-cite>

Unlike traditional global random scheduling, SPMDM asserts that an efficient sampling path should possess two characteristics: local priority (prioritizing the unmasking of masks near already solved tokens) and logical sequence priority (prioritizing the unmasking of early subsequences in the reasoning chain).

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure5.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="SPMDM partitions the input sequence x into K subsequences and introduces noise with varying magnitudes to each of them. Compared to MDLM, our method not only models token-level dependencies but also explicitly encourages the learning of inter-subsequence relationships. In contrast to BDLM, SP" %}

- Insight

  - Same noise within a block → learning relationships among adjacent tokens becomes more stable;
  - Different noises across blocks → inherently biases learning towards "earlier recovery" of certain subsequences during training.

- Training Method: Subsequence Partitioning and Non-uniform Noise

  - Data Preparation: Partitions the input sequence $x$ of length $N$ into $K$ non-overlapping subsequences $\{x^1, x^2, \dots, x^K\}$ of length $L$.
  - Mask Schedule: During training, instead of applying a uniform noise level $t$ to the entire sequence, independent noise scales $t_k$ are assigned to each subsequence $x^k$.
  - Time Embedding: Each subsequence has its own $t_k$, thus $K$ groups of time embeddings are added to the Transformer input.
  - Target: Optimized through subsequence-level NELBO, forcing the model to predict a specific subsequence $x^k$ given the context $x_t^{-k}$:

    $$\mathcal{L}_{NELBO}=\sum_{k=1}^{K}\mathbb{E}_{t_{k}\sim[0,1],q_{t_{k}/0}}\left[\frac{\alpha_{t_{k}}^{\prime}}{1-\alpha_{t_{k}}}\log p_{\theta}(x^{k}\mid x_{t_{k}}^{k},{x_{t}}^{-k})\right]$$

- Sampling Method:
  Sampling starts from $x^1 = [m, m, \dots, m]$. At each step:
  1. Calculate the number of recovered tokens $n_k$ for each subsequence;
  2. Update the time for that subsequence: $t_k = 1 - \frac{n_k}{L};$
  3. Use the model for prediction: $\hat{x}^0_k = p_\theta(x_k \mid x^{t_k}_k, x^{-k}_t, t_k);$
  4. Convert mask to token according to a certain policy (probabilistic or adaptive oracle can be used).
- Advantages:
  - Consistency between training and inference, which reduces total correlation errors.
  - The setting of subsequences aligns with intuition.
  - Lightweight without extra burden.
- Shortages:
  - The $t_k$ for different subsequences are obtained randomly, which could be further optimized.

#### Block Diffusion Language Model

**_Paper: BLOCK DIFFUSION: INTERPOLATING BETWEEN AUTOREGRESSIVE AND DIFFUSION LANGUAGE MODELS_<d-cite key="arriola2025blockdiffusion"></d-cite>**

BD3-LMs propose a new paradigm that interpolates between autoregressive (AR) modeling and discrete diffusion modeling. Its core motivation lies in overcoming the limitations of standard diffusion models in generating sequences of arbitrary length, inference efficiency (KV Caching), and likelihood modeling quality (Perplexity) by introducing block-wise structures.

{% include figure.liquid loading="eager" path="assets/img/posts/2026-03-04-Note-Schedulers-of-Mask-Diffusion-Model/figure6.png" figure_class="my-1" class="img-fluid rounded z-depth-1 float-right float-end ml-3 ms-3 mb-2" zoomable=true caption="Block diffusion sequentially generates blocks of tokens by performing diffusion within each
block and conditioning on previous blocks. By combining strength from autoregressive and diffusion
models, block diffusion overcomes the limitations of both approaches by supporting variable-length,
higher-quality generation and improving inference efficiency with KV caching and parallel sampling." %}

BD3-LM partitions a sequence of length $L$ into $B$ blocks of length $L'$. The model follows an autoregressive decomposition across blocks, while executing a discrete diffusion process within each block:

$$\log p_{\theta}(x) = \sum_{b=1}^{B} \log p_{\theta}(x^{b} \mid x^{\lt b})$$

- Advantages:
  - Conforms to human sequential priors regarding text.
  - Consistency between training and inference, which reduces total correlation errors.
  - Facilitates the utilization of KV-caching.

## Final Remark

<style>
  .remark-table-center {
    overflow-x: auto;
    width: 100%;
    margin-left: 0%;
  }

  .remark-table-center table {
    width: 100%;
    min-width: 1300px;
    margin: 0;
    table-layout: fixed;
  }

  .remark-table-center th,
  .remark-table-center td {
    text-align: center;
    vertical-align: middle;
    white-space: normal;
    word-break: break-word;
    line-height: 1.55;
  }

  .remark-table-center th:nth-child(1),
  .remark-table-center td:nth-child(1) {
    width: 15%;
  }

  .remark-table-center th:nth-child(2),
  .remark-table-center td:nth-child(2) {
    width: 25%;
  }

  .remark-table-center th:nth-child(3),
  .remark-table-center td:nth-child(3) {
    width: 20%;
  }

  .remark-table-center th:nth-child(4),
  .remark-table-center td:nth-child(4) {
    width: 20%;
  }

  .remark-table-center th:nth-child(5),
  .remark-table-center td:nth-child(5) {
    width: 20%;
  }

  @media (max-width: 900px) {
    .remark-table-center {
      width: 100%;
      margin-left: 0;
    }
  }
</style>

<div class="remark-table-center" markdown="1">

|         Schedulers          |                                                                                                              Policy $\pi(M \mid c)$                                                                                                               |                                                                       Sampler $p_{\theta}$                                                                       |                                                              Advantages                                                              |                                                            Shortages                                                            |
| :-------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
|      Random Schedulers      |                                                                               $\pi(\mathcal{U}) = \displaystyle\frac{1}{\binom{ \vert \mathcal{M}_{t+1}\vert}{k}}$                                                                                |                                             Average distribution of all possible permutation paths $\mathfrak{S}_L$                                              | Closed-form solution for the forward process;<br>No need for additional logit sorting or planner forward computation during sampling |                 Squanders training resources;<br>Ignores intrinsic order, leading to severe error accumulation                  |
| Confidence-based Schedulers | $\mathcal{U} = \text{arg-topk}\_{i \in \mathcal{M}\_{t+1}} \{ s_i \mid s_i = \max_{x \in \mathcal{X}} \log p_\theta(x^i = x \mid \mathbf{x}\_{t+1}) \}$<br> $\pi(\mathcal{U} \mid \mathbf{x}\_{t+1}) = \mathbb{1}(\mathcal{U} = \text{arg-topk})$ |        Usually introduces token-level reweighting during training to make the model delineate boundaries for low-confidence (hard) samples more clearly.         |                               Adaptively adjusts generation order according to different input samples                               | Insufficient model training or overly large vocabulary leads to noisy logit sorting;<br>Cannot resolve Total Correlation errors |
|   Planer-based Schedulers   |                                                                                     $\mathcal{U} = \{ i \mid p_\phi(z^i = N \mid \mathbf{x}_{t+1}) < \tau \}$                                                                                     | $p_\theta$ is only responsible for dimensions selected by $p_\phi$, and its input usually contains the real-time time step $\tilde{t}$ estimated by the planner. |                                           Supports Remasking;<br>Avoids capacity dilution;                                           |                              Generates token by token with remasking, resulting in low efficiency;                              |
|    RL-Policy Schedulers     |                                                                                  $\pi_\phi(i \mid \mathbf{x}\_{t+1}) = \text{Softmax}(h_\phi(\mathbf{x}_{t+1}))$                                                                                  |                                     Training objective shifts from simple maximum likelihood to preferring high-reward paths                                     |                                Consistency in training and inference;<br>High performance upper bound                                |                                                        Complex training                                                         |
|   Manual/Block Schedulers   |                                                    $\pi(\mathcal{U}) \implies \mathcal{U} \subset \text{Block}_k,\ \text{where } k \text{ is determined by preset order or block noise } t_k$                                                     |                                                            Learns locally strongly coupled semantics                                                             |                                       Significantly reduces $\text{TC}(x_\mathcal{U} \mid c)$                                        |                  Fixed block partition may fail to perfectly match the nonlinear logical dependencies of data                   |

</div>

\
\
From a unified perspective, <u><span style="color: #2980b9; font-weight: bold;">the Noise Scheduler in a Mask Diffusion Model is fundamentally a "Structural Regulator" rather than a time controller in the traditional sense.</span></u> By determining when, what to unmask, and how many tokens to unmask at once, it **directly shapes the conditional distributions confronted by the model during training and inference**, thereby dictating the tradeoff among three core errors:

1. Time propagation error from Exposure Bias (accumulated from single-step local deviations);
2. Optimization complexity and capacity dilution introduced by violating Intrinsic Order;
3. Inevitable Total Correlation deviations under the Token Independence assumption.

Analysis reveals that these three issues are not independent variables; rather, they are jointly regulated along the same error tradeoff curve through the strategic choices of the Scheduler:

- Large-step decoding mitigates sequential violation but amplifies structural dependency errors;
- Small-step decoding diminishes independence bias but imposes higher demands on intrinsic order alignment and inference stability;
- Random scheduling is theoretically unbiased but introduces significant optimization redundancy under limited capacity;
- Schedulers based on confidence or policy learning attempt to approximate the sample-level optimal generation order, yet they are constrained by the model's inherent imperfections and estimation noise.

In conclusion, RL-Policy Schedulers comprehensively address the constraints of intrinsic order and total correlation errors through rewards. Furthermore, because their sampling trajectories adhere seamlessly to the learned policy, they manifest enhanced training-inference consistency, ultimately tending toward optimal generation performance.
