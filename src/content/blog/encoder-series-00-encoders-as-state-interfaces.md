---
title: "Encoder Series 00 | Repre\u00ADsentation Before Intelligence"
date: 2026-07-07
excerpt: A first-principles introduction to the Encoder series, framing encoders as the state interface that compresses raw observations into structured representations for prediction, generation, retrieval, and action.
category: Research
tags: ["Encoder", "Representation"]
author: Runze Tian
affiliation: Talentlab, Westlake
affiliationHref: https://ai4s.lab.westlake.edu.cn/
bib: public/assets/blog/encoder-series-00-encoders-as-state-interfaces/references.bib
draft: false
---

<div class="blog-takeaway">
  <strong>Takeaway</strong>
  <p>Encoders are not merely front-end feature extractors. They are the state interface of intelligent systems: the mechanism that turns raw, noisy, partial observations into compact internal states that downstream models can compare, predict, generate from, and act upon. This series starts from that premise.</p>
</div>

## From World to State

An intelligent system can only reason over the world it has first made available to itself.

This is the starting point of the series. A language model does not encounter language as lived meaning; it receives token states. A vision model does not encounter a scene as a human observer does; it receives pixels, patches, feature maps, or visual tokens. A robot does not act on the physical world in its full richness; it acts through sensor streams, memory, and internal variables that stand in for the world.

Before there is reasoning, there is representation. Before there is generation, there is a latent space. Before there is planning, there is some internal answer to a more basic question:

**What state of the world does this system have?**

This note is the entrance to a longer series on encoders. I will not begin with BERT, ViT, CLIP, MAE, VAE, or any particular architecture. Those models matter, but they come after a more primitive constraint: the world must be transformed into state before a system can compute over it.

### Constraint 1: observation is not usable as-is

Raw observation is both too much and too little.

It is too much because the sensory world is high-dimensional, redundant, noisy, and full of accidental detail. A natural image contains lighting, pose, texture, background clutter, camera artifacts, and many pixel-level variations irrelevant to the task at hand. A sentence contains surface form, syntax, style, discourse context, and pragmatic hints. A molecular graph contains atom types, bonds, geometry, symmetries, and physical constraints. A video contains objects, motion, persistence, occlusion, and possible futures.

At the same time, raw observation is too little because the variables that matter are often not directly visible. A frame does not show the hidden side of an object. A word does not contain its context. A protein sequence does not explicitly expose its folded structure. A robot's camera does not directly identify which factors are controllable.

So the first problem is not simply input processing. It is state construction:

**How does an unstructured percept become a usable internal state?**

### Constraint 2: state must be smaller than world

No system can carry the world in full detail. It must compress.

Shannon's information theory gives us a language for uncertainty and coding, but intelligence requires more than efficient transmission [@shannon1948mathematical]. It requires states that make downstream computation easier. The information bottleneck view captures part of this tension: compress the input while preserving information relevant to a target [@tishby2000information]. Representation learning generalizes this into a broader program: learn transformations that expose useful factors of variation [@bengio2013representation].

This makes representation a selective act. A useful state must remove detail, but not remove the detail that matters. It must reduce the world without flattening the variables needed for prediction, comparison, control, retrieval, or generation.

In this sense, representation is not the opposite of forgetting. Representation is **controlled forgetting**: a disciplined loss of detail that makes abstraction possible.

![Grav filtering raw observations into a usable state](/assets/blog/encoder-series-00-encoders-as-state-interfaces/fig/observation-to-state-sieve.png "Figure 1. Raw observations become usable state through controlled forgetting: irrelevant detail is discarded while useful variables are carried forward.")

### Constraint 3: state must serve future computation

A state is not only a summary of the past. It is prepared for future use.

The system forms a state now because it expects to use that state later: to classify, retrieve, predict, generate, plan, or act. A representation that is compact but useless for the next computation is only compression. A representation that exposes future-relevant structure becomes intelligence-bearing state.

For this series, I will use a deliberately functional definition:

$$
z = f_{\mathrm{enc}}(x)
$$

An encoder maps an observation $x$ to a state $z$. But the formula hides the real question. The important issue is not whether $z$ is a vector, a sequence of tokens, a feature map, a graph embedding, or a distribution. The important issue is what kind of computational world $z$ creates.

Does it make similarity meaningful? Does it make prediction easier? Does it preserve controllable variables? Does it expose compositional structure? Does it discard nuisance variation while keeping enough detail for generation?

Observation is what arrives. State is what the system can use.

The encoder is the interface between the two.

## The Encoder Boundary

The vocabulary around encoders often makes them sound secondary. We call them feature extractors, backbones, embedding models, or front ends. These names are useful in engineering discussions, but they understate the conceptual role of the encoder. A feature extractor sounds like a tool that picks useful pieces from the input. A state interface is more fundamental: it defines the variables available to the decoder, retriever, planner, policy, or world model.

In many systems, the encoder has already made the most consequential decision: it has decided what kind of world the rest of the model gets to inhabit.

If the representation has no notion of object permanence, the planner cannot reason about hidden objects. If the representation erases geometry, the controller cannot recover spatial structure downstream. If the representation aligns image and text only at a coarse global level, a language model may speak fluently about an image while failing to ground its answer in the right visual evidence.

The downstream module may be large, expressive, and well trained. It can still only compute over the state it receives.

![Grav operating an encoder boundary that selects state variables](/assets/blog/encoder-series-00-encoders-as-state-interfaces/fig/encoder-boundary-gate.png "Figure 2. The encoder is a boundary: it selects which variables become available to downstream modules such as geometry, memory, and control.")

The encoder is therefore not merely an architectural block. It is a boundary. On one side is raw observation: high-dimensional, noisy, modal, local, and partial. On the other side is state: compressed, structured, comparable, reusable, and shaped by downstream needs. Encoders are central even when they are visually small in a diagram, because they determine what the system can know before it decides what to do.

### No neutral representation

Every representation carries a bias.

A convolutional encoder assumes locality and translation structure. A Transformer encoder assumes a tokenized world in which pairwise interactions can be globally mixed [@vaswani2017attention]. A graph encoder assumes relational structure. An equivariant encoder assumes that certain transformations should be respected rather than relearned. A masked modeling objective assumes that useful structure can be learned by recovering what is missing [@devlin2019bert; @he2022masked].

There is no neutral representation. Every representation is a claim about which differences matter.

This point is easy to miss in the age of scale. The bitter lesson is that general methods which scale with computation tend to beat handcrafted human knowledge in the long run [@sutton2019bitter]. But representation does not disappear. Its assumptions often move into architecture, objective, data distribution, augmentation, tokenization, masking policy, contrastive pairing, context length, and latent geometry.

Representation is where these choices become state.

## The Shape of Internal Worlds

Once observation has become state, the system no longer lives in the original world. It lives in the geometry of its representation.

This is the second first-principles step. A representation is not just a container of information. It is a space in which certain operations become easy, natural, or even possible.

### Geometry before semantics

One of the deepest intuitions in deep learning is that neural networks reshape data.

Chris Olah's essay on neural networks, manifolds, and topology makes this vivid: each layer transforms the data into a new representation, and the final classifier can be simple only because earlier layers have made the data easier to separate [@olah2014manifolds]. Learning is not merely assigning labels. It is the progressive deformation of a data manifold.

An encoder does not simply attach semantic labels to inputs. It builds a geometry in which some things become close, some things become far, some directions become meaningful, and some variations collapse. In a word embedding space, linguistic regularities can become vector relationships. In a vision embedding space, images with related content can become neighbors. In CLIP-like systems, images and text are pushed into a shared geometry where cross-modal retrieval becomes possible [@radford2021learning].

The tempting phrase is that the model has learned meaning. Sometimes that is a useful shorthand. A more precise statement is that the model has built a space where certain operations become possible.

Similarity search becomes nearest-neighbor lookup. Classification becomes linear probing. Retrieval becomes dot product ranking. Generation becomes movement through a latent manifold. Alignment becomes geometry across modalities. Control becomes action selection over compact state variables.

Semantics enters the system through geometry.

![Grav tracing a route through representation space](/assets/blog/encoder-series-00-encoders-as-state-interfaces/fig/representation-geometry-map.png "Figure 3. Once observation becomes state, the system works inside representation geometry: closeness, paths, and regions shape what future computation can do.")

### Prediction tests the state

If geometry is the shape of internal state, prediction is one of its strongest tests.

The future-facing view is especially clear in world models. In the World Models framework, visual observations are encoded into latent variables, a recurrent model predicts future latent states, and a controller acts using those compact states rather than raw pixels [@ha2018world]. The agent succeeds not because it reconstructs every detail of the environment, but because its latent state is sufficient for prediction and control.

Joint-Embedding Predictive Architectures push this idea further by predicting in representation space rather than reconstructing every pixel-level detail [@lecun2022path; @assran2023self]. This is not merely an efficiency trick. It is a statement about abstraction. If the world contains many unpredictable or irrelevant details, then forcing a model to predict all of them may waste capacity. A better state may preserve what is predictable and useful while discarding what cannot or should not be modeled at the current level.

This is where encoders become central to agency. Policies, planners, generative models, and multimodal systems all inherit the variables exposed by their state. The state is not passive memory. It is an interface to possible futures.

### Interpretation rebuilds the bridge

The representational world is not naturally written in human concepts. It is a mathematical object optimized for computational usefulness.

Distill's work on feature visualization and interpretability is valuable because it tries to build bridges from hidden objects back to human-understandable abstractions [@olah2017feature; @olah2018building]. The need for such bridges is itself revealing. A representation is not automatically a dictionary. It is a working internal world.

This is also where a philosophical risk appears. If representation is a map, the central danger is mistaking the map for the territory.

Every encoder produces a partial world. It may preserve variables rewarded by pretraining while ignoring variables that matter in deployment. It may align modalities globally while missing fine-grained grounding. It may encode shortcuts rather than causes. It may make representations linearly separable without making them semantically robust.

Many failures of intelligent systems can be read as representation failures:

- The model cannot reason about a relation because the relation was never represented.
- The model hallucinates because its language state is not grounded in the right external state.
- The robot fails because its perceptual state does not expose action-relevant affordances.
- The retrieval system fails because its embedding geometry captures surface similarity rather than task meaning.
- The multimodal model fails because alignment in a shared space is not the same as understanding.

The Platonic Representation Hypothesis gives this issue a provocative frame. It asks whether different models and modalities are converging toward a shared statistical model of reality as they scale [@huh2024platonic]. The question is fascinating, but it should be held with care. Even if representations converge geometrically, that does not imply that they are grounded, causal, controllable, or sufficient for reasoning.

A representation can be useful without being true. It can be aligned without being grounded. It can be predictive without being causal. It can be compact without being complete.

The map is necessary. The map is powerful. But the map is still a model.

## The Road Ahead

This series starts from a simple claim:

**Encoders are the state interface of intelligent systems.**

Everything else follows from that claim.

### Foundations

First, we will build the conceptual foundation: what a representation is, how information theory frames compression and relevance, how geometry shapes similarity, how latent variable models interpret hidden structure, how inductive bias constrains what can be learned, and how representations can be evaluated.

### Modalities

Then we will move through language encoders. The story begins with making discrete symbols continuous, passes through sequence encoders and attention, and arrives at contextual representations such as ELMo and BERT. The question will not only be "what architecture was used?" but "what state of language did the system learn to expose?"

Next comes vision. CNNs, patch tokenization, Vision Transformers, contrastive learning, masked image modeling, DINO-style features, and dense representations all answer the same question differently: how should pixels become visual state?

After that, we will discuss multimodal encoders and alignment. CLIP is a turning point because it shows how two modalities can enter a shared semantic space. But it also opens deeper questions: what is aligned, what remains modality-specific, and when does geometric proximity become grounding?

### Worlds and generation

The later parts of the series will move into scientific domains, graph and geometric encoders, world models, embodied intelligence, and generation. These areas make the encoder problem sharper. Molecules, proteins, materials, physical scenes, and action-conditioned environments are not just inputs; they are structured worlds. Their encoders must respect constraints that generic token sequences may not expose.

Finally, we will return to generation. Encoder-decoder models, autoencoders, VAEs, VQ-VAEs, latent diffusion, and representation autoencoders all show that generation often depends on first finding the right space in which to generate [@kingma2014auto; @hinton2006reducing].

That is the deeper reason to begin with representation.

Modern AI often looks like an output machine: it writes, draws, predicts, plans, and acts. But underneath those outputs is a quieter question that every system must answer:

**What state of the world do I have?**

The encoder is where that answer begins.
