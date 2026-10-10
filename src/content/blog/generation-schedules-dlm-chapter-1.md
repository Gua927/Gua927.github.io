---
title: "Generation Schedules for Diffusion Language Models | What Kind of DLM Do We Actually Want?"
date: 2026-10-10
excerpt: Diffusion language models inherit a recipe from images, but language has no natural generation space in which noise, distance, frequency, and meaning line up. A language-native DLM needs both a structured generation space and a semantic generation schedule.
category: RESEARCH-NOTE
tags: ["MDLM", "Order"]
---

> **Takeaway**
>
> Diffusion language models (DLMs) are usually understood as carrying the success of image diffusion over to language modeling. That framing may hide the real problem.
>
> The difficulty of diffusing language is not only that language is discrete, nor that tokens are harder to corrupt than pixels. The deeper issue is that language has no natural generation space in which noise, distance, frequency, and semantic abstraction align on their own.
>
> In images, low and high frequencies have fairly clear physical meanings. In language, "frequency" might refer to semantic level, dependency distance, information density, degree of abstraction, discourse structure, or surface form. The spectrum of language is not given by the data domain; it is a structure that has to be modeled, learned, or even redefined.
>
> So the central question for DLMs should not only be *how to denoise tokens*, but ***where* language generation should happen, and *along what trajectory* it should move.** A truly language-native diffusion model should not merely learn to recover tokens from noise; it should learn how meaning, structure, and surface text are connected step by step.

## 0. Introduction: are we really diffusing language?

Diffusion models succeed at image generation not only because they offer an effective noising–denoising framework, but because images have a structure that suits a diffusion process.

Intuitively, the reverse process of image diffusion often follows a coarse-to-fine trajectory: the overall layout appears first, then object contours, and finally textures and local detail. This feels natural because images have geometric structure: a continuous space, local neighborhoods, edges, textures, and a frequency spectrum. A blurred image is still an image, and a lightly noised image still keeps much of its spatial structure.

When we try to carry diffusion over to language, the analogy quickly becomes fragile.

What does it mean to *gradually destroy a sentence*? If we delete words, are we destroying syntax, semantics, or merely observability? If we perturb embeddings, are we moving along a direction of semantic abstraction, or simply moving through a continuous vector space? And if the model ultimately just reconstructs tokens, is it generating language, or filling in blanks?

So the real question is not whether language *can* be noised. Of course it can: we can mask tokens, substitute words, perturb embeddings, diffuse latent variables, or define transition processes over discrete distributions. The real question is whether that noising process corresponds to a *linguistically meaningful* form of degradation.

This leads to the central distinction of this note:

$$
\text{noise schedule} \neq \text{generation schedule}
$$

A **noise schedule** answers: *how much information should be destroyed?*

A **generation schedule** answers: *which kind of information should be generated first?*

Moreover, a schedule cannot exist apart from a space. Before discussing the generation schedule of language, we must ask a more basic question: ***in which space does language generation actually take place?*** This is where diffusion language models truly begin.

## 1. The implicit biases of language generation models

Every generative model carries an implicit coordinate system. Sometimes it is explicit, as in a latent variable model; sometimes it hides inside the factorization, the training objective, or the noising process.

Language modeling is not only about estimating $p(x)$; it is also about deciding *how* $x$ should be generated.

### 1.1 Autoregressive models: a surface-time coordinate system

An autoregressive language model defines the probability of a sequence through the factorization

$$
p(x)=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

This factorization gives language generation a very clear coordinate system:

$$
x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
$$

The model generates from left to right, and at every step the training target is the next token. This is one of the main reasons autoregressive (AR) models scale so reliably: the model never has to decide *what to generate next*, because the surface order of the data already answers that question.

Yet the strength of this coordinate system is also its limitation. **Left-to-right order is a surface-temporal order, not a semantic order.** When an AR model generates the first token of a long answer, it often already has to plan, implicitly, the global intent, the line of argument, the discourse structure, and even the final conclusion. It must predict the next token locally while keeping the whole semantically coherent.

We can write this as

$$
\underbrace{x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T}_{\text{surface order}}
\;\neq\;
\underbrace{m \rightarrow s \rightarrow x}_{\text{semantic order}}
$$

where $m$ denotes meaning or intent, $s$ an intermediate structure, and $x$ the final text. Autoregressive generation is certainly structured, but its structure is temporal rather than semantically hierarchical.

### 1.2 Chain-of-Thought: a hand-crafted generation schedule inside AR

Chain-of-Thought (CoT) offers an illuminating clue. A standard autoregressive model typically learns

$$
\text{question} \rightarrow \text{answer},
$$

whereas Chain-of-Thought rewrites this as

$$
\text{question} \rightarrow \text{reasoning path} \rightarrow \text{answer}.
$$

It is still autoregressive, but it inserts an intermediate structure before the final answer: the model is asked to produce a kind of reasoning skeleton first, and only then the output.

From the perspective of this note, **Chain-of-Thought can be read as a hand-crafted generation schedule embedded inside an AR model.**

It suggests that the goal of DLMs should not be parallel decoding or token denoising alone. A more meaningful goal is to learn the intermediate process by which *meaning gradually becomes text*.

### 1.3 Diffusion language models: noise is not automatically semantics

Diffusion language models try to bring iterative refinement into language generation. Depending on the design, the diffusion process may live in token space, mask space, embedding space, or a latent space.

In general form,

$$
q(z_t\mid z_0),\qquad p_\theta(z_{t-1}\mid z_t),
$$

where $z_0$ is a representation of the clean data and $z_t$ is a corrupted intermediate state.

If $z$ is a token sequence, diffusion is symbolic corruption; if $z$ is an embedding, diffusion is vector denoising; if $z$ is a latent variable, diffusion is refinement in a hidden space.

None of these choices automatically yields semantic generation. Tokens are not the basic units of meaning; the dimensions of an embedding need not correspond to semantic axes; a latent variable does not naturally become a discourse plan. And a denoising trajectory is not necessarily a path from meaning to text.

The common problem many current DLMs face can therefore be summarized as:

> ***A noise schedule does not automatically become a semantic schedule.***

This is not to say that DLMs are headed in the wrong direction, only that their central problem has not yet been adequately defined.

## 2. From generation space to generation schedule

Before discussing a generation schedule, we first need to ask which space it is defined in.

For image diffusion, the question of space is almost invisible. Images live naturally in pixel space or a visual latent space, and these spaces carry intuitive geometric meaning: distance, smoothness, locality, and frequency.

Language has no such natural generation space. The generation space of language is itself a modeling choice:

- If generation happens in **token space**, intermediate states are partially corrupted strings.
- If generation happens in **embedding space**, intermediate states are continuous vectors.
- If generation happens in a **semantic latent space**, intermediate states may correspond to intents, entities, relations, discourse plans, or reasoning skeletons.
- If generation happens in **edit space**, intermediate states are drafts and transitions are revisions.

We can introduce a representation map

$$
\phi:\mathcal{X}\rightarrow\mathcal{Z},
$$

where $\mathcal{X}$ is the space of observed text and $\mathcal{Z}$ is the generation space. The diffusion process is defined on $\mathcal{Z}$:

$$
q(z_t\mid z_0),\qquad z_0=\phi(x),
$$

and the reverse process is a trajectory:

$$
z_T \rightarrow z_{T-1} \rightarrow \cdots \rightarrow z_0.
$$

If $\mathcal{Z}$ is poorly chosen, the diffusion process may be mathematically sound and still carry no linguistic meaning. This gives two basic principles:

> ***The generation space defines what a state is.***
>
> ***The generation schedule defines how states should move.***

A language-native diffusion model needs both: a meaningful generation space *and* a meaningful generation trajectory. Without the right space, a schedule has nowhere to live; without the right schedule, a space has no direction of generation.

### 2.1 Why does language have no single frequency axis?

In images, frequency has a clear physical meaning: low frequencies correspond to overall layout, high frequencies to edges and fine detail.

In language, "frequency" has no single interpretation:

- **By semantic level**, low frequencies might correspond to topic and discourse structure, high frequencies to specific wording and punctuation.
- **By dependency distance**, low frequencies might correspond to long-range dependencies, high frequencies to local collocations.
- **By information content**, common function words usually carry little information, while rare technical terms may carry high semantic density, so the mapping can even run the other way.

This non-uniqueness is not a technical detail. It is an essential difference between language diffusion and image diffusion. **The spectrum of language does not exist directly in the token domain.** It has to be discovered, represented, or imposed through modeling.

### 2.2 The semantic spectrum

One possible direction is to define the frequencies of language not by surface position or token statistics, but by semantic invariance:

> ***The low-frequency structure of language is what stays invariant across paraphrases.***
>
> ***The high-frequency detail of language is what can vary without changing the core meaning.***

For example, the following sentences differ in surface form:

1. The model fails because its denoising trajectory is not semantically meaningful.
2. The failure comes from the fact that the reverse process does not follow a meaningful semantic path.
3. The problem is not denoising itself, but the lack of a semantic generation trajectory.
4. At its core, the model's problem is that it lacks a denoising path that carries meaning.

They differ in wording and syntax, yet share a similar semantic core. This suggests that language diffusion should not revolve around token recovery alone, but around the transition from semantic invariants to surface variants.

One possible trajectory is

$$
\text{intent} \rightarrow \text{entities and relations} \rightarrow \text{discourse structure} \rightarrow \text{syntactic organization} \rightarrow \text{surface wording}.
$$

This is not necessarily the only correct order. The point is that a meaningful generation schedule must say which content should stabilize first and which can be decided later. For language, the answer is very likely: **stabilize the meaning first, then refine the expression.**

## 3. What kind of DLM do we actually need?

If we accept the analysis above, an ideal DLM should not be defined merely as a diffusion model over tokens. That definition is too weak. A stronger one would be:

> ***A diffusion language model should learn a generation trajectory from meaning to text.***

That trajectory needs a space, and it needs a schedule.

### 3.1 It should generate in a semantic–structural space

A useful DLM should not treat language as a flat sequence of tokens. It should have some internal space in which to represent semantic and structural states. These states might include:

- communicative intent;
- topic and global semantics;
- entities, events, and relations;
- a discourse plan;
- a reasoning skeleton;
- a syntactic scaffold;
- a draft-level surface realization.

The point is not that these states must be annotated by hand, but that the model should be encouraged to learn intermediate representations that play similar roles.

A good diagnostic question is: ***if we stop the reverse diffusion process halfway, what do we see?***

If the answer is just "a corrupted string", the model is closer to token denoising. If the answer is "a rough semantic plan", "an unfinished draft", or "a structure whose wording is not yet settled", then it is closer to a language-native diffusion model.

### 3.2 It should learn what to stabilize first

Generation is not only producing content; it is also deciding which parts should remain uncertain.

In a reasonable language generation process, high-level meaning should usually stabilize before surface wording. The model should first decide *what to say*, then *how to say it*.

We can decompose the uncertainty of a text as

$$
u(x)=u_{\text{sem}}(x)+u_{\text{lex}}(x),
$$

where $u_{\text{sem}}$ is semantic uncertainty and $u_{\text{lex}}$ is lexical or stylistic uncertainty. An ideal generation schedule should reduce them in a structured way:

$$
u_{\text{sem}} \downarrow \quad \text{before} \quad u_{\text{lex}} \downarrow
$$

That is, **stabilize the meaning first, then refine the expression.** This is exactly what many token-level noise schedules struggle to express: they control the *strength* of corruption, but not semantic *priority*.

### 3.3 It should generate by iterating, not only by predicting

An autoregressive model generates by appending: it predicts the next token, one step at a time.

The potential advantage of a DLM should not be faster decoding alone, but a more natural ability to revise the whole. Human writing is rarely strictly left to right; it looks more like

$$
\text{idea} \rightarrow \text{outline} \rightarrow \text{draft} \rightarrow \text{revision} \rightarrow \text{final text}.
$$

This is closer to the spirit of diffusion than next-token prediction is. The model should be able to reorganize globally, polish locally, insert missing structure, remove redundancy, and adjust style.

So an ideal DLM should not ask only *what is the next token?* It should ask ***what is the next meaningful revision?***

### 3.4 Architecture provides the space; the objective defines the direction

Where do the generation space and the generation schedule come from? Very likely from both the architecture and the training objective.

The **architecture** determines what intermediate states the model can represent. It provides the space in which semantic skeletons, discourse plans, latent outlines, dependency relations, or draft states can exist.

The **training objective** determines how the model moves between those states. It decides whether the model is actually trained to move from high-level meaning toward low-level expression.

In short:

$$
\text{Architecture} = \text{where states can live},\qquad
\text{Objective} = \text{how states should move}.
$$

Architecture alone is not enough. A latent variable does not automatically become a semantic plan; without the right training pressure, it may encode only shallow statistics.

An objective alone is not enough either. Without a suitable internal representation space, the objective can only force the model to learn semantic organization at the token level, which is both difficult and unstable.

The core design principle for DLMs should therefore be:

> ***A structured generation space + a semantic generation schedule.***

This is what separates language-native diffusion from token denoising.

## 4. Conclusion

The original vision of DLMs was to bring the power of diffusion models to language generation. But perhaps the problem was framed too narrowly from the start.

The real question is not *how to diffuse tokens*, but ***in which space meaning should be diffused, and how it gradually becomes text.***

Image diffusion works partly because its generation space is naturally aligned with visual structure. Language does not provide such a space for free. Its structure is hidden among semantic abstraction, dependency relations, discourse organization, information density, and surface expression.

DLMs therefore cannot be reduced to token corruption, mask prediction, or embedding denoising. A good DLM should not only learn

$$
\text{noise} \rightarrow \text{tokens};
$$

it should learn

$$
\text{meaning} \rightarrow \text{structure} \rightarrow \text{text}.
$$

More fundamentally, it must also learn *where* this transformation should take place. The core claim of this note can be summarized as:

> ***A good DLM should not only learn how meaning becomes text.***
>
> ***It should first learn where meaning can become text.***

That may be the problem a language-native diffusion model truly needs to solve.
