---
title: "SAT-Mask: Self-Aligned Trajectory Masking for Diffusion Language Models"
date: 2026-06-14
excerpt: SAT-Mask aligns masked diffusion language model training with inference trajectories by replacing random mask states with self-aligned, confidence-guided rollout states.
category: PUB-NOTE
tags: ["MDLM", "Order"]
author: Runze Tian
affiliation: GenSI Lab, THU-AIR
bib: public/assets/blog/sat-mask-diffusion-language-model-training/references.bib
---

## Introduction

Autoregressive language models scale well, but their left-to-right factorization fixes generation to a single causal order. That order is useful for next-token prediction, yet it also limits bidirectional dependency modeling, non-monotonic planning, and intrinsic error correction. Masked diffusion models (MDMs) are interesting precisely because they replace this monotone generation path with a global denoising process: tokens can be filled in any order, partial contexts are bidirectional, and multiple positions can be updated in parallel [@D3PM; @MDLM; @SEDD; @LLaDA].

The same flexibility creates the central dilemma of the paper. During training, the usual MDM objective corrupts a clean sequence by independently masking positions. This gives convenient order-agnostic supervision, but it also forces the denoiser to fit a combinatorially large family of arbitrary mask patterns. During inference, however, generation is not arbitrary. A sampler follows a concrete unmasking trajectory, often selecting positions using confidence, margin, entropy, or a planner [@TrainingForTheWorst; @PAPL; @SPMDM]. The model is therefore trained on one state distribution and evaluated along another.

Existing fixes mostly intervene at inference time: use a better selection rule, choose safer tokens first, or allow remasking. These samplers can reduce local uncertainty, but they do not change the training distribution that shaped the denoiser in the first place. Training-side schedules are closer in spirit, but if they still rely on heuristic or uniformly random states, they leave the capacity dilution problem unresolved.

SAT-Mask addresses the issue at the source. It constructs training states by first over-noising a clean example and then partially denoising it with the same kind of confidence-guided transition used at inference. The resulting state is no longer an arbitrary mask subset; it is a local sample from the model's own easy-to-hard trajectory.

The paper's logic is:

1. Random masking pays an explicit information tax by asking the model to distinguish arbitrary mask states.
2. Train-inference mismatch accumulates as exposure bias because local transition errors compound along the reverse chain.
3. SAT-Mask reduces both failures by using a shared training-inference transition kernel.
4. This improves quality and efficiency across problem solving, text generation, and math reasoning.

## Background

Let $\mathbf{x}_0\in\mathcal{X}^L$ be a clean sequence and let $\mathbf{x}_t\in\bar{\mathcal{X}}^L$ be a corrupted state, where $\bar{\mathcal{X}}=\mathcal{X}\cup\{\mathrm{m}\}$ and $\mathrm{m}$ is the mask token. In the standard absorbing corruption process, each coordinate is independently kept with probability $\alpha_t$ and masked otherwise:

$$
q_{t|0}(\mathbf{x}_t\mid\mathbf{x}_0)
= \prod_{i=1}^{L}
\left[
\alpha_t\delta_{x_0^i}(x_t^i)
+ (1-\alpha_t)\delta_{\mathrm{m}}(x_t^i)
\right].
$$

The denoiser learns a clean-token posterior

$$
p_\theta^i(a\mid \mathbf{x}_t,t)
\equiv p_\theta(x_0^i=a\mid \mathbf{x}_t,t),
$$

and the continuous-time objective reduces to a weighted masked-token reconstruction loss:

$$
\mathcal{L}_{\mathrm{MDM}}(\theta)
=
\mathbb{E}_{\mathbf{x}_0}
\int_0^1
\mathbb{E}_{\mathbf{x}_t\sim q_{t|0}}
\left[
\frac{-\dot{\alpha}_t}{1-\alpha_t}
\sum_{i\in\mathcal{M}(\mathbf{x}_t)}
-\log p_\theta^i(x_0^i\mid \mathbf{x}_t,t)
\right]dt.
$$

This objective trains a denoiser, but sampling needs one more ingredient: a rule for which masked coordinates to reveal at each reverse step. On a discrete grid $1=t_T>\cdots>t_0=0$, a sampler transition from $\mathbf{x}_{t_{k+1}}$ to $\mathbf{x}_{t_k}$ can be written as a transition kernel. A policy $\pi_k(M\mid\mathbf{x})$ first selects a subset of still-masked positions, and the denoiser then fills those positions:

$$
\mathbf{T}^{\pi}_{k}(\mathbf{x},\mathbf{y})
=
\sum_{M\subseteq\mathcal{M}(\mathbf{x})}
\pi_k(M\mid\mathbf{x})
\left[
\prod_{i\in M}p_\theta^i(y^i\mid\mathbf{x},t_{k+1})
\right]
\mathbf{1}\{\mathbf{y}^{-M}=\mathbf{x}^{-M}\}.
$$

This formulation makes the mismatch precise. Training samples $\mathbf{x}_t$ from independent random masking, while inference moves through states induced by $\mathbf{T}^{\pi}_k$. SAT-Mask is built around making these two paths share a local transition structure.

## Motivation

The motivation section of the paper has two parts. The first explains why random masking wastes capacity. The second explains why that waste is not only a training inefficiency, but also becomes exposure bias during generation.

### Random masking causes capacity dilution

Consider a generic masking policy $Q(\mathbf{x}_t\mid\mathbf{x}_0)$. At a fixed time, the masked-token objective can be decomposed into the true conditional entropy plus the denoiser approximation error:

$$
\mathcal{L}_Q(\theta)
=
H_Q(\mathbf{x}_0\mid\mathbf{x}_t)
+
\mathbb{E}_{\mathbf{x}_t\sim p_Q}
\left[
D_{\mathrm{KL}}\!\left(
p_Q(\mathbf{x}_0\mid\mathbf{x}_t)
\parallel
p_\theta(\mathbf{x}_0\mid\mathbf{x}_t)
\right)
\right].
$$

The important point is that $Q$ determines the state space the model must represent. The paper formalizes this with an additive capacity law:

> **Theorem (Additive capacity law).** For any masking policy $Q$, the idealized capacity requirement satisfies
>
> $$
> \mathcal{C}_{\mathrm{needed}}(Q)
> \ge
> H_Q(\mathbf{x}_0\mid\mathbf{x}_t)
> +
> H_Q(\mathbf{x}_t)
> =
> H(\mathbf{x}_0)
> +
> H_Q(\mathbf{x}_t\mid\mathbf{x}_0).
> $$

The term $H(\mathbf{x}_0)$ is intrinsic to the data. The extra term $H_Q(\mathbf{x}_t\mid\mathbf{x}_0)$ is created by the masking policy. For independent random masking with mask probability $m_t$, every coordinate contributes one Bernoulli mask decision:

$$
H_{Q_{\mathrm{rand}}}(\mathbf{x}_t\mid\mathbf{x}_0)
=
L\,\mathcal{H}_b(m_t).
$$

Under a linear mask schedule $m_t=t$ and uniform time sampling, this gives the closed-form overhead:

$$
\mathbb{E}_{t\sim\mathcal{U}(0,1)}
H_{Q_{\mathrm{rand}}}(\mathbf{x}_t\mid\mathbf{x}_0)
=
L\int_0^1 \mathcal{H}_b(t)dt
=
\frac{L}{2\ln 2}
\approx
{\color{red}0.721L\;\text{bits}}.
$$

<figure class="blog-wrap-figure right">
  <img src="/assets/blog/sat-mask-diffusion-language-model-training/fig/any-order-masking.png" alt="Random masking versus order-aware masking">
  <figcaption>Figure 1. Random masking visits arbitrary mask states that dilute capacity; SAT-Mask follows an order-aware trajectory and releases capacity for meaningful context.</figcaption>
</figure>

For length $1024$, this is about $738$ bits spent only on arbitrary mask subsets. A deterministic fixed order would remove this entropy, but it would also give up the bidirectional, any-order advantage of MDMs. The target is therefore not "make the order fixed"; it is "make the order structured while preserving bidirectional context."

### Misaligned intrinsic order induces exposure bias

The capacity argument explains why random masking is inefficient. Exposure bias explains why it hurts generation.

Let $P_t$ denote the distribution of states seen during training, and let $\hat{P}_t$ denote the distribution of states produced by the sampler. Their mismatch is

$$
\Delta_t = D_{\mathrm{KL}}(P_t\parallel \hat{P}_t).
$$

For one reverse step $t_{k+1}\to t_k$, the paper derives a recursion:

$$
\Delta_{t_k}
\le
\Delta_{t_{k+1}}
+
{\color{red}
\mathbb{E}_{\mathbf{x}\sim P_{t_{k+1}}}
\left[
D_{\mathrm{KL}}\!\left(
\mathcal{Q}^{\mathrm{train}}_k(\cdot\mid\mathbf{x})
\parallel
\mathbf{T}^{\pi}_k(\cdot\mid\mathbf{x})
\right)
\right]
}.
$$

This is the key exposure-bias statement. The mismatch at the next cleaner state is bounded by the previous mismatch plus a local transition mismatch. If the training transition and sampler transition disagree at every step, the error is not a one-shot defect; it accumulates along the whole denoising path.

The next question is: which path should they agree on? The paper argues that good inference seeks an empirical intrinsic order $\pi_\theta^*$: for a given sample and model state, reveal positions that have lower local surprisal first, so easy tokens become anchors for harder tokens. This is the easy-to-hard order induced by the current denoiser, not a fixed left-to-right order.

Using this intrinsic order as a reference, the local mismatch can be decomposed into three terms:

> **Proposition (Local error decomposition).**
>
> $$
> \begin{aligned}
> D_{\mathrm{KL}}\!\left(
> \mathcal{Q}^{\mathrm{train}}_k
> \parallel
> \mathbf{T}^{\pi}_k
> \right)
> &=
> \underbrace{
> D_{\mathrm{KL}}\!\left(
> \mathcal{Q}^{\mathrm{train}}_k
> \parallel
> \mathcal{Q}^{\pi_\theta^*}_k
> \right)
> }_{\mathcal{E}_{\mathrm{capacity}}}
> +
> \underbrace{
> D_{\mathrm{KL}}\!\left(
> \mathcal{Q}^{\pi_\theta^*}_k
> \parallel
> \mathbf{T}^{\pi}_k
> \right)
> }_{\mathcal{E}_{\mathrm{align}}}
> +
> \mathcal{R}_{\mathrm{shift}}.
> \end{aligned}
> $$

This decomposition clarifies why inference-only fixes are insufficient.

- $\mathcal{E}_{\mathrm{capacity}}$ measures how far the training transition is from the model's intrinsic easy-to-hard order. Uniform random masking makes this large because it trains on states unrelated to the denoising frontier.
- $\mathcal{E}_{\mathrm{align}}$ measures how far the sampler is from that intrinsic order. Confidence, margin, and entropy samplers mainly attack this term.
- $\mathcal{R}_{\mathrm{shift}}$ captures the residual distribution shift between the states used in training and the states induced by the intrinsic trajectory.

So a better sampler can reduce $\mathcal{E}_{\mathrm{align}}$, but if the training distribution remains random, $\mathcal{E}_{\mathrm{capacity}}$ and $\mathcal{R}_{\mathrm{shift}}$ remain. This is the precise reason SAT-Mask moves the planner-like transition into training. The training state itself must be produced by a trajectory that approximates the model's empirical intrinsic order.

## The SAT-Mask Framework

SAT-Mask constructs training states by matching the inference-time two-stage update. It first rolls back to a higher-noise state, then executes an uncertainty-aware denoising step back to the target mask budget. The architecture and reconstruction loss stay the same; the supervised state distribution changes.

![SAT-Mask method overview](/assets/blog/sat-mask-diffusion-language-model-training/fig/SAT-Mask.png "Figure 2. Overview of SAT-Mask. Left: starting from a more corrupted state, SAT-Mask uses denoiser confidence to fill high-confidence tokens along a zigzag self-aligned trajectory, while computing loss only on the retained masks. Right: the shared training-inference path reduces capacity dilution and exposure bias.")

### SAT-Masking schedule

Given a clean sequence $\mathbf{x}_0$ and time $t$, SAT-Mask first samples a more corrupted state at

$$
t^+ = \min(t+\Delta t,1).
$$

The masks at $t$ and $t^+$ are coupled by shared uniforms $u_i\sim\mathrm{Unif}(0,1)$:

$$
\mathcal{M}_t=\{i:u_i<m_t\},
\qquad
\mathcal{M}_{t^+}=\{i:u_i<m_{t^+}\}.
$$

Therefore $\mathcal{M}_t\subseteq\mathcal{M}_{t^+}$. The target budget at time $t$ is

$$
N_t=|\mathcal{M}_t|.
$$

SAT-Mask then calls the current denoiser once on $\mathbf{x}_{t^+}$ and asks a sampler operator $S$ to unmask exactly

$$
B_t=|\mathcal{M}_{t^+}|-N_t
$$

positions:

$$
\mathbf{x}'_t
=
S\!\left(
\mathbf{x}_{t^+},
f_\theta(\mathbf{x}_{t^+},t^+),
B_t
\right),
\qquad
|\mathcal{M}(\mathbf{x}'_t)|=N_t.
$$

This is the zigzag move: go to a slightly noisier point, then take one sampler-like step back to the original mask budget. The visible tokens in $\mathbf{x}'_t$ are no longer a uniformly random subset. They are the tokens the current model would prefer to reveal along its trajectory.

### Sampler-compatible transition

The operator $S$ is an interface. It can be random, confidence-based, entropy-based, margin-based, remasking-aware, or replaced by future sampler designs. The core requirement is that the same local rule used at inference can also be used to construct training states.

In the experiments, the default is `downk-margin`. Let $\ell_\theta^i$ be the logits at a masked position $i$, and let

$$
r_i = \ell_{(1)}^i-\ell_{(2)}^i
$$

be the margin between the largest and second-largest log-probabilities. SAT-Mask selects a set $\mathcal{U}_t$ by mixing deterministic high-margin unmasking with random coverage:

$$
\mathcal{U}_t
=
\operatorname{TopK}_{i\in\mathcal{M}_{t^+}}
(r_i,\lfloor\gamma B_t\rfloor)
\cup
\operatorname{Uniform}(\mathrm{rest},B_t-\lfloor\gamma B_t\rfloor).
$$

For selected positions, tokens are sampled from a temperature-controlled distribution:

$$
x_t^{\prime i}
\sim
\mathrm{Cat}\!\left(
\operatorname{softmax}
\left(
\ell_\theta^i(\cdot\mid\mathbf{x}_{t^+},t^+)/\tau
\right)
\right),
\qquad
i\in\mathcal{U}_t.
$$

The margin score gives an easy-to-hard signal. The random part avoids collapsing the schedule to a single deterministic path. The temperature $\tau$ preserves stochasticity in the rollout.

### Training objective

SAT-Mask keeps the standard reconstruction target but evaluates it on the sampler-induced state distribution:

$$
\mathcal{L}_{\mathrm{SAT}}(\theta)
=
\mathbb{E}_{\mathbf{x}_0}
\int_0^1
\mathbb{E}_{
{\color{red}\mathbf{x}'_t\sim
\operatorname{sg}[
\mathcal{Q}^{S}_{\theta,t}(\cdot\mid\mathbf{x}_0)
]}}
\left[
\frac{-\dot{\alpha}_t}{1-\alpha_t}
\sum_{i\in\mathcal{M}({\color{red}\mathbf{x}'_t})}
-\log p_\theta^i(x_0^i\mid {\color{red}\mathbf{x}'_t},t)
\right]dt.
$$

The stop-gradient notation indicates that the rollout constructs the state but does not receive gradients directly. Gradients are applied through the final reconstruction loss. Thus SAT-Mask changes neither the denoiser architecture nor the target mask schedule. It changes which contexts the model treats as normal during optimization.

```algorithm
caption: SAT-Mask Training
input: Dataset $\mathcal{D}$, denoiser $f_{\theta}$, rollout step $\Delta t$, sampler $S$, temperature $\tau$
while not converged do
  Sample clean data $\mathbf{x}_0 \sim \mathcal{D}$ and diffusion time $t \sim \mathcal{U}(0,1)$
  Sample state $\mathbf{x}_{t^+}$ at $t^+=\min(t+\Delta t,1)$ using masks coupled with the target budget $N_t$
  Compute logits $\ell_\theta=f_\theta(\mathbf{x}_{t^+},t^+)$
  $B_t \leftarrow |\mathcal{M}_{t^+}|-N_t$ // Number of tokens to unmask from $\mathbf{x}_{t^+}$
  $\mathbf{x}'_t \leftarrow S(\mathbf{x}_{t^+},\ell_\theta,B_t)$ // One sampler step; default uses downk-margin
  Update $\theta$ by descending $\nabla_\theta\left(\lambda(t)\sum_{i\in\mathcal{M}(\mathbf{x}'_t)}-\log p_\theta^i(x_0^i\mid \mathbf{x}'_t,t)\right)$ // Reconstruction loss
end while
```

### Effectiveness analysis

For capacity dilution, SAT-Mask replaces arbitrary random mask subsets with states produced by a model-dependent map. In the deterministic core, many over-noised states can collapse to the same supervised state when they induce the same high-margin reveals:

> **Proposition (State-space collapse of SAT-Mask).**
>
> $$
> H_{\mathrm{SAT}}(\tilde{\mathbf{x}}_t\mid\mathbf{x}_0)
> =
> H_q(\mathbf{x}_{t^+}\mid\mathbf{x}_0)
> -
> H(\mathbf{x}_{t^+}\mid\tilde{\mathbf{x}}_t,\mathbf{x}_0,\theta).
> $$

The second term is the collapse entropy released by SAT-Mask. Intuitively, states that differ only by irrelevant arbitrary mask choices no longer need to be separately modeled if they lead to the same high-margin reveal pattern.

For exposure bias, the same margin-guided local kernel appears in training and inference. Because margin is a target-free proxy for low local surprisal, the training states become closer to the empirical intrinsic order $\pi_\theta^*$. This directly reduces the capacity and shift terms that inference-only samplers cannot remove.

## Experiments

The experiments evaluate SAT-Mask on problem solving, text generation, and math reasoning. In each setting, the core comparison is the same: replace vanilla random masking with SAT-Mask while keeping the denoiser and decoding policy controlled.

### Sudoku and Countdown

For Sudoku, the paper uses the one-million solved-game corpus and trains on the first 100k puzzles. Each $9\times 9$ grid is serialized as a digit sequence, with `0` marking an empty cell. For Countdown-4, it generates 500k arithmetic problems following Stream of Search, with 10% of targets held out for out-of-distribution evaluation [@park20161million; @cd]. Both tasks use the same 6M DiT denoiser.

SAT-Mask consistently improves both benchmarks. On Sudoku, accuracy increases from 39.1% with vanilla random masking to 63.5% at $T=2.5$, a 62.4% relative gain. On Countdown-4, the best SAT-Mask setting reaches 35.6% at $T=1.5$, improving over the 30.7% vanilla baseline by 16.0%.

<figure class="blog-wrap-figure right wide">
  <img src="/assets/blog/sat-mask-diffusion-language-model-training/fig/problem-solving.png" alt="Problem-solving results">
  <figcaption>Figure 3. Problem-solving results. Left: Sudoku accuracy across training steps. Right: Countdown-4 accuracy and relative improvement over vanilla across temperatures.</figcaption>
</figure>

These tasks make the trajectory problem visible. A wrong early reveal can constrain later decisions, while a reliable early reveal can become an anchor. SAT-Mask helps because the model is trained on contexts that already reflect that easy-to-hard dependency.

### OpenWebText generation

For text generation, the paper evaluates unconditional OpenWebText generation with a 169M DiT-based MDM initialized from the MDLM checkpoint, using the GPT-2 tokenizer and length $L=1024$ [@gokaslan2019openwebtext; @MDLM; @gpt-2]. The evaluation reports MAUVE, GenPPL, and entropy over 5000 samples [@mauve; @prism; @ReMDM].

<div class="paper-table-wrap">
  <table class="paper-results-table">
    <caption>Table 1. OpenWebText unconditional generation. MAUVE is higher-is-better; GenPPL is lower-is-better; entropy is reported as a sanity metric. Best scores within each method family and sampling budget are bolded.</caption>
    <colgroup>
      <col class="paper-col-method">
      <col class="paper-col-metric" span="9">
    </colgroup>
    <thead>
      <tr>
        <th scope="col"></th>
        <th scope="colgroup" colspan="3">T=128</th>
        <th scope="colgroup" colspan="3">T=256</th>
        <th scope="colgroup" colspan="3">T=512</th>
      </tr>
      <tr>
        <th scope="col">Method</th>
        <th scope="col">MAUVE</th>
        <th scope="col">GenPPL</th>
        <th scope="col">Ent.</th>
        <th scope="col">MAUVE</th>
        <th scope="col">GenPPL</th>
        <th scope="col">Ent.</th>
        <th scope="col">MAUVE</th>
        <th scope="col">GenPPL</th>
        <th scope="col">Ent.</th>
      </tr>
    </thead>
    <tbody>
      <tr class="paper-table-group"><td colspan="10">MDM without remask</td></tr>
      <tr><th scope="row">MDLM</th><td>0.016</td><td>79.37</td><td><strong>5.57</strong></td><td>0.027</td><td>73.02</td><td><strong>5.55</strong></td><td>0.034</td><td>70.21</td><td><strong>5.54</strong></td></tr>
      <tr class="sat-row"><th scope="row">MDLM+SAT-Mask</th><td><strong>0.034</strong></td><td><strong>78.16</strong></td><td>5.55</td><td><strong>0.038</strong></td><td><strong>72.58</strong></td><td>5.53</td><td><strong>0.039</strong></td><td><strong>67.96</strong></td><td>5.51</td></tr>
      <tr class="paper-table-group"><td colspan="10">MDM with remask</td></tr>
      <tr><th scope="row">ReMDM-conf</th><td>0.02</td><td>74.50</td><td><strong>5.57</strong></td><td>0.03</td><td>66.50</td><td><strong>5.54</strong></td><td>0.04</td><td>52.50</td><td><strong>5.49</strong></td></tr>
      <tr><th scope="row">ReMDM</th><td>0.06</td><td>42.50</td><td>5.43</td><td>0.22</td><td>30.50</td><td>5.35</td><td>0.35</td><td>21.00</td><td>5.22</td></tr>
      <tr><th scope="row">PRISM</th><td>0.18</td><td><strong>18.10</strong></td><td>5.11</td><td>0.30</td><td>18.00</td><td>5.15</td><td>0.42</td><td>17.12</td><td>5.12</td></tr>
      <tr class="sat-row"><th scope="row">PRISM+SAT-Mask</th><td><strong>0.31</strong></td><td>23.30</td><td>5.20</td><td><strong>0.42</strong></td><td><strong>15.60</strong></td><td>5.05</td><td><strong>0.43</strong></td><td><strong>11.08</strong></td><td>4.91</td></tr>
    </tbody>
  </table>
</div>

SAT-Mask improves the non-remasking MDLM baseline at every sampling budget and also improves PRISM in the remasking setting. This supports the paper's claim that training-state alignment is complementary to sampler design.

### Math reasoning

For GSM8K, the paper follows the SMDM supervised fine-tuning setup and evaluates different model scales under 32, 64, 128, and 256 sampling steps [@gsm8k; @smdm]. The comparison includes LLAMA, Plaid, MDLM, SPMDM, SEDD, SMDM, and PAPL numbers where applicable [@llama; @plaid; @SEDD; @SPMDM; @PAPL].

<div class="paper-table-wrap compact">
  <table class="paper-results-table">
    <caption>Table 2. GSM8K-CoT math reasoning accuracy under different sampling steps. Higher is better. Missing entries indicate results not reported in the corresponding baseline.</caption>
    <colgroup>
      <col class="paper-col-method">
      <col class="paper-col-param">
      <col class="paper-col-metric" span="4">
    </colgroup>
    <thead>
      <tr>
        <th scope="col">Method</th>
        <th scope="col">Param.</th>
        <th scope="col">32 steps</th>
        <th scope="col">64 steps</th>
        <th scope="col">128 steps</th>
        <th scope="col">256 steps</th>
      </tr>
    </thead>
    <tbody>
      <tr><th scope="row">LLAMA</th><td>7B</td><td>58.60</td><td>58.60</td><td>58.60</td><td>58.60</td></tr>
      <tr><th scope="row">Plaid</th><td>1.3B</td><td>--</td><td>--</td><td>--</td><td>32.60</td></tr>
      <tr class="paper-table-rule"><th scope="row">SMDM</th><td>1.1B</td><td>53.82</td><td>55.11</td><td>54.96</td><td>56.10</td></tr>
      <tr class="sat-row"><th scope="row">SAT-Mask (ours)</th><td>1.1B</td><td><strong>54.58</strong></td><td><strong>55.11</strong></td><td><strong>57.01</strong></td><td><strong>58.75</strong></td></tr>
      <tr class="paper-table-rule"><th scope="row">SMDM</th><td>336M</td><td>52.08</td><td>53.52</td><td>54.20</td><td>54.96</td></tr>
      <tr><th scope="row">PAPL</th><td>336M</td><td>51.40</td><td>53.52</td><td>54.89</td><td><strong>55.64</strong></td></tr>
      <tr class="sat-row"><th scope="row">SAT-Mask (ours)</th><td>336M</td><td><strong>53.52</strong></td><td><strong>54.35</strong></td><td><strong>55.64</strong></td><td>55.49</td></tr>
      <tr class="paper-table-rule"><th scope="row">SEDD</th><td>170M</td><td>--</td><td>--</td><td>--</td><td>45.30</td></tr>
      <tr><th scope="row">SMDM</th><td>170M</td><td>49.65</td><td>50.01</td><td>50.79</td><td>51.25</td></tr>
      <tr><th scope="row">PAPL</th><td>170M</td><td>49.65</td><td>52.01</td><td>53.37</td><td>53.60</td></tr>
      <tr class="sat-row"><th scope="row">SAT-Mask (ours)</th><td>170M</td><td><strong>51.63</strong></td><td><strong>53.52</strong></td><td><strong>53.67</strong></td><td><strong>54.28</strong></td></tr>
      <tr class="paper-table-rule"><th scope="row">MDLM</th><td>127M</td><td>--</td><td>--</td><td>--</td><td>46.10</td></tr>
      <tr><th scope="row">SPMDM</th><td>127M</td><td>--</td><td>--</td><td>--</td><td>51.30</td></tr>
      <tr><th scope="row">SMDM</th><td>113M</td><td>47.76</td><td>49.05</td><td>50.34</td><td>50.56</td></tr>
      <tr><th scope="row">PAPL</th><td>113M</td><td>45.11</td><td>48.67</td><td>49.50</td><td>50.64</td></tr>
      <tr class="sat-row"><th scope="row">SAT-Mask (ours)</th><td>113M</td><td><strong>48.74</strong></td><td><strong>51.25</strong></td><td><strong>51.70</strong></td><td><strong>52.76</strong></td></tr>
    </tbody>
  </table>
</div>

The gains are stronger in smaller models. At 170M, SAT-Mask reaches 54.28%, close to the 336M SMDM baseline at 54.96%, while requiring fewer training steps in the paper's efficiency comparison. This matches the theory: capacity-limited models pay more for arbitrary mask-state entropy, so they benefit more from aligned state construction.

### Efficiency and ablation

The paper measures efficiency by baseline-equivalent training steps required to reach comparable performance. SAT-Mask reduces the required steps by 68.2% on Sudoku, 61.1% on Countdown, and 16.7% on OpenWebText. On GSM8K, the reductions are 9.1%, 28.3%, 3.5%, and 16.7% for 113M, 170M, 336M, and 1028M models.

![Training efficiency of SAT-Mask](/assets/blog/sat-mask-diffusion-language-model-training/fig/efficiency.png "Figure 4. Training efficiency of SAT-Mask. Bars report the baseline-equivalent training steps required by vanilla random masking and SAT-Mask; percentages denote relative step reduction.")

The ablations isolate the schedule design. Temperature controls the diversity of filled tokens in the zigzag rollout: too little exploration or too much noise hurts, while $\tau=2.5$ performs best on Sudoku. The selection rule is also crucial: random selection behaves like vanilla masking, top-$k$ alone hurts, and `downk-margin` performs best by preserving the easy-to-hard order. Finally, the over-noise offset $\Delta t$ should be moderate; on GSM8K-113M, performance improves up to $\Delta t=1/16$ and then drops when the over-noised state is too far from the target budget.

![Ablation studies of SAT-Mask](/assets/blog/sat-mask-diffusion-language-model-training/fig/ablation-acc.png "Figure 5. Ablation studies. Temperature controls token-filling stochasticity, the selection function controls which positions are unmasked, and the offset controls the distance between the over-noised state and the target mask budget.")

## Conclusion

SAT-Mask is best understood as a training-side alignment method for masked diffusion language models. Random masking asks the denoiser to spend capacity on arbitrary mask states and creates a state-distribution gap with the sampler. SAT-Mask replaces those arbitrary states with local rollout states produced by a shared confidence-guided transition.

This closes the loop between training and inference without changing the architecture or the reconstruction objective. The result is a masking schedule that follows the model's empirical easy-to-hard order, reduces exposure bias by construction, and improves quality and training efficiency across structured reasoning, open-ended generation, and math reasoning.
