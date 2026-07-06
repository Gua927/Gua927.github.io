---
title: "Encoder Series 1: From Signals to Internal Worlds"
date: 2026-06-24
excerpt: "A first-principles account of why encoders exist: intelligent systems cannot use raw signals directly, so they must construct compressed, structured internal worlds."
category: Research
tags: ["Encoder", "Representation"]
author: Runze Tian
affiliation: GenSI Lab, THU-AIR
bib: public/assets/blog/encoder-origins-and-mathematical-foundations/references.bib
---

> **Takeaway.**
>
> The encoder is the operation that turns signal into state. It exists because raw observations are not yet the right objects for prediction, reasoning, generation, science, or control. The central insight is not simply compression, but useful compression: keeping the structure that makes a future computation easier, and discarding what the system can afford to forget.

## The Representation Gap

The most basic question is not "which encoder architecture should we use?" It is:

> **Why does an intelligent system need an encoder at all?**

A system does not receive the world directly. It receives observations: pixels, tokens, waveforms, sensor streams, molecular strings, experimental measurements, interaction histories. These observations are high-dimensional, noisy, redundant, partial, and usually expressed in the wrong coordinate system for the task.

But the system does not want observations as such. It wants something else:

- a visual system wants objects, layout, geometry, and affordances;
- a language system wants contextual meaning;
- a scientific model wants variables, relations, symmetries, and laws;
- an agent wants a state from which the future can be predicted and actions can be chosen.

This mismatch is the representation gap.

$$
\text{world}
\to
\text{observation}
\xrightarrow{\quad E_\theta \quad}
\text{internal state}
\to
\text{prediction, reasoning, generation, control}.
$$

The encoder lives in the middle. Its job is not merely to reduce the size of the input. Its job is to construct the kind of internal object on which the rest of the system can operate.

In modern notation, we write:

$$
E_\theta: \mathcal{X} \to \mathcal{Z},
$$

where $\mathcal{X}$ is the observation space and $\mathcal{Z}$ is the representation space. The key object is not only the function $E_\theta$, but the space $\mathcal{Z}$ it creates. This space is where similarity, abstraction, memory, dynamics, and meaning become available to the model.

That is why this series starts here. Before studying image encoders, text encoders, scientific encoders, or world-model encoders, we need to understand the common pressure that creates all of them:

> **Principle.** Intelligence does not operate on raw signal. It operates on an internal world built from signal.

## The Core Insight

The encoder's core insight is often described as compression. That is true, but incomplete.

Compression alone is not enough. A zip file is compressed, but it is not a representation for reasoning. A blurred image is smaller, but may destroy the object boundary. A short text summary may preserve meaning, but lose syntax. A latent state may discard pixels, but preserve dynamics. What matters is not whether information is removed. What matters is which information is removed, which information survives, and what structure the surviving information takes.

So the better phrase is:

> **Core insight.** An encoder performs useful compression: it removes variation that is not needed for a future computation, while preserving and reorganizing the structure that is needed.

This gives the encoder three roles.

First, it **selects**. It decides what information should survive. A face encoder may ignore lighting. A semantic text encoder may ignore formatting. A robot state encoder may ignore background texture but keep object pose and velocity.

Second, it **structures**. It does not merely keep bits. It creates geometry, hierarchy, locality, relational structure, uncertainty, or temporal state. The output is a new coordinate system.

Third, it **enables**. The representation is valuable only because it makes another computation easier: classification, retrieval, reconstruction, prediction, planning, scientific inference, or generation.

The Information Bottleneck gives a compact mathematical form of this idea. If $X$ is the observation, $Z$ is the representation, and $Y$ is the relevant target, then a useful encoder should compress $X$ while preserving information about $Y$:

$$
\min_{p(z\mid x)}
I(X;Z) - \beta I(Z;Y).
$$

The first term penalizes how much of the input is carried into the representation. The second rewards information useful for the target [@tishby2000information]. This is not a complete theory of all encoders, but it captures the central tradeoff.

For a classifier, $Y$ may be a label. For a masked model, $Y$ may be the missing part. For a multimodal system, $Y$ may be another modality. For a world model, $Y$ is the future. For science, $Y$ may be a structure, law, trajectory, energy, or intervention outcome.

The same idea can be stated in the language of state. A representation is good when the original observation or history becomes unnecessary for the downstream question:

$$
Y \perp X \mid Z.
$$

In a world model, this becomes:

$$
p(o_{t+1:\infty}\mid o_{\le t}, a_{\le t})
\approx
p(o_{t+1:\infty}\mid z_t, a_t).
$$

The encoder compresses the past into a state that is sufficient enough for predicting and acting. This is the strongest version of the encoder idea: not a feature extractor, but a state constructor.

## Origins: Three Pressures

The encoder did not originate from one paper or one architecture. It emerged because three older problems kept producing the same need: communication needed codes, control needed state, and learning needed internal representations.

### Communication needed codes

The earliest technical meaning of encoding belongs to communication. A message must be transformed into a signal, sent through a channel, and recovered by a receiver:

$$
\text{message}
\to
\text{code}
\to
\text{channel}
\to
\text{decoded message}.
$$

Shannon's 1948 paper *A Mathematical Theory of Communication* made this problem mathematical [@shannon1948mathematical].[^shannon] Entropy measured uncertainty. Redundancy became something that could be quantified and exploited. Channel capacity described what could be transmitted reliably. Huffman's coding algorithm later showed how to construct efficient prefix codes from source probabilities [@huffman1952method].

The lasting lesson is not that neural encoders are source codes. They are not. The lesson is deeper:

> **A representation is good relative to a source, a channel, and a receiver.**

Modern machine learning changes the words but keeps the structure. The source is the data distribution. The channel is the architecture and training process. The receiver is the downstream computation. An image encoder, a text encoder, and a protein encoder differ because their sources have different structure and their receivers ask different questions.

### Control needed state

Communication explains how signals move. Control explains why internal state matters.

In *Cybernetics*, Wiener framed animals and machines through control and communication [@wiener1948cybernetics].[^wiener] A control system does not merely receive a signal and output a label. It observes, updates internal condition, acts, receives feedback, and updates again.

The important object is state:

$$
o_t
\xrightarrow{E_\theta}
z_t
\xrightarrow{\text{dynamics, policy}}
\hat{o}_{t+1}, a_t.
$$

If $z_t$ is wrong, control fails even if the observation is rich. If $z_t$ is good, the system can act under noise, delay, partial observability, and uncertainty.

This is why modern world models feel less like a new trick than a return to first principles. Ha and Schmidhuber framed world models as learning compressed spatial and temporal representations of environments [@ha2018world]. Dreamer-style agents then used latent imagination to learn behavior inside compact state spaces [@hafner2020dream].

The control view changes the encoder's objective. The goal is not to preserve the input. The goal is to preserve what can change the future.

### Learning needed internal representations

The neural origin of encoders begins with the question of whether perception can be implemented as internal transformation.

McCulloch and Pitts proposed a logical model of neural activity in which threshold units could implement computation [@mcculloch1943logical]. Rosenblatt's perceptron connected learning, perception, and pattern recognition in a concrete machine [@rosenblatt1958perceptron].[^perceptron] These systems were limited, but they introduced the idea that a network's internal activity can represent the input in a task-relevant way.

Backpropagation made this idea trainable. Rumelhart, Hinton, and Williams did not merely popularize an optimization method; their paper was titled *Learning representations by back-propagating errors* [@rumelhart1986learning]. The phrase matters. A hidden layer becomes useful because error signals teach it which distinctions should exist.

Deep autoencoders then made the encoder-decoder form explicit:

$$
x \xrightarrow{E_\theta} z \xrightarrow{D_\phi} \hat{x}.
$$

Hinton and Salakhutdinov showed that neural networks could learn compact nonlinear codes for high-dimensional data [@hinton2006reducing]. VAEs later made the encoder probabilistic, mapping an observation to an approximate posterior over latent variables [@kingma2014auto].

This is the point where encoding becomes representation learning. The code is no longer hand-designed. The internal world is learned.

## The Metaphysical Move

There is a more philosophical way to state the encoder's role:

> **An encoder is an ontology builder.**

It decides what kinds of things exist for the model.

Raw pixels do not contain "objects" as primitive entities. They contain intensities. A visual encoder makes edges, textures, parts, layouts, objects, and relations available. Raw token IDs do not contain meaning by themselves. A text encoder makes contextual concepts available. Raw experimental measurements do not directly expose laws. A scientific encoder tries to produce variables in which law-like structure becomes visible.

This is why $\mathcal{Z}$ is not a passive copy of $\mathcal{X}$. It is a constructed world:

$$
\mathcal{X}
\xrightarrow{\quad E_\theta \quad}
\mathcal{Z}.
$$

The arrow chooses what the rest of the model can see.

Every encoder therefore makes commitments.

It commits to a **geometry**. If two points are close in embedding space, the model treats them as similar. If image and text embeddings are aligned, the model can retrieve one through the other. If latent states interpolate smoothly, generation and planning become easier.

It commits to **invariances**. A representation may ignore lighting, translation, paraphrase, sensor noise, or irrelevant background. Forgetting is not merely failure. Often it is the point.

It commits to a **unit of thought**. CNNs encourage local visual features. Transformers encourage token interactions through attention [@vaswani2017attention]. Graph networks encourage relational message passing [@sanchez2020learning]. State-space world models encourage compact temporal state.

This is why the encoder is a metaphysical object, not only an engineering module. It defines the model's internal vocabulary of reality. Once the world has been encoded incorrectly, more computation downstream may not recover what was never represented.

## Modern Branches

Modern encoders can be understood as different answers to the same question:

> **What internal world should this system build?**

The branches differ less by whether they have an encoder block and more by what kind of representation they need.

### Image encoders build visual worlds

Image encoders turn pixels into visual structure: edges, textures, parts, objects, layouts, geometry, and semantics. CNNs introduced strong priors for locality and translation structure [@lecun1998gradient]. AlexNet showed how learned visual hierarchies could dominate large-scale recognition [@krizhevsky2012imagenet]. Vision Transformers reframed an image as a sequence of patches processed by attention [@dosovitskiy2021image]. Masked autoencoders made missing-patch prediction a scalable self-supervised route to visual representation [@he2022masked].

The development arc is:

$$
\text{pixels}
\to
\text{features}
\to
\text{semantic visual representation}.
$$

### Text encoders build semantic worlds

Text encoders turn token sequences into contextual meaning. A word is not represented once; it is represented as a function of its surrounding context. Transformer encoders made this practical through self-attention [@vaswani2017attention]. BERT showed that bidirectional masked prediction can pretrain broadly useful language representations [@devlin2019bert].

The development arc is:

$$
\text{symbols}
\to
\text{context}
\to
\text{meaning}.
$$

### Multimodal encoders build shared worlds

Multimodal encoders align different forms of observation. CLIP-style systems train image and text encoders into a shared embedding space where language can refer to vision and vision can be retrieved by language [@radford2021learning].

The important shift is that representation becomes an interface. The question is no longer only "what is in this image?" but "how can this image and this sentence point to the same concept?"

The development arc is:

$$
\text{modality-specific representations}
\to
\text{shared semantic geometry}.
$$

### Scientific encoders build law-bearing worlds

Scientific encoders should expose variables, relations, constraints, symmetries, and uncertainty. A protein model does not only need a sequence representation; it needs internal structure that supports 3D reasoning. AlphaFold is a landmark because it encodes biological sequence and evolutionary information into representations that support structure prediction [@jumper2021highly].

Graph simulators show a parallel direction in physics: encode particles, meshes, or objects as relational states, then learn dynamics through message passing [@sanchez2020learning].

The development arc is:

$$
\text{measurements}
\to
\text{structured state}
\to
\text{law-like prediction}.
$$

### World-model encoders build agent worlds

World-model encoders summarize sensory history into latent state. This state must preserve what is observable, infer what is hidden, track what changes, and support imagined futures. It is not only perception. It is state construction for agency.

The development arc is:

$$
\text{experience}
\to
\text{latent state}
\to
\text{prediction and control}.
$$

This branch makes the encoder's importance most explicit. An agent does not act in the raw observation space. It acts through the world it has encoded.

### Generative encoders build latent worlds

Generative encoders organize data for synthesis. Autoencoders and VAEs learn latent codes that support reconstruction or sampling [@kingma2014auto]. Masked modeling and denoising objectives learn representations by forcing the system to infer what is missing or corrupted. In these systems, generation and representation learning are tightly coupled: to generate well, the model must learn what structure the data has.

The development arc is:

$$
\text{data}
\to
\text{latent factors}
\to
\text{reconstruction or synthesis}.
$$

## A Compass for the Series

This series will use a consistent set of questions for each encoder family.

1. What is the observation space?
2. What is the representation space?
3. What structure does the encoder assume?
4. What information is preserved?
5. What information is forgotten?
6. What downstream computation becomes easier?
7. What failure modes follow from this representation?

These questions are more useful than listing architectures. Architectures matter, but they are implementations of representational commitments.

For image encoders, the commitment may be locality, patch structure, or visual-language alignment. For text encoders, it may be contextual semantics. For scientific encoders, it may be symmetry and relational structure. For world models, it may be predictive state. For generative encoders, it may be a latent space in which synthesis becomes tractable.

The central question will stay fixed:

> **What internal world does this encoder build, and what does that world make easy?**

## Closing

The origin of the encoder is a recurring pressure in intelligent systems. Communication needed codes. Control needed state. Neural learning needed internal representations. Science needed variables. Modern AI needs all of these at once.

That is why encoders are worth studying as a series rather than as a single module. They are where raw experience becomes structured possibility. They decide what the model can see, what it can ignore, what it can compare, what it can predict, and what it can imagine.

The thesis of this first note is therefore:

$$
\text{to build intelligence, first build the right internal world}.
$$

The rest of the series will examine how different encoder families attempt to do exactly that.

[^shannon]: Shannon worked at Bell Labs, where communication engineering, switching circuits, probability, and cryptography met. His 1948 paper is usually treated as the starting point of modern information theory.

[^wiener]: Wiener's cybernetics framed control and communication as a shared language for animals and machines. This made feedback, state, and self-regulation central ideas for later AI and systems theory.

[^perceptron]: Rosenblatt's perceptron should be read historically. Its importance is not that it solved modern representation learning, but that it connected learning, perception, and internal transformation in a concrete machine.
