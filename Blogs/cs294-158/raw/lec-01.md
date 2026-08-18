# Lecture 1: Deep Unsupervised Learning — Motivation & Landscape

---

## 1. Definition of Deep Unsupervised Learning

Deep Unsupervised Learning refers to **capturing rich patterns from raw, unannotated data using deep neural networks in a label-free manner**. 

### Two Primary Sub-Areas:
1. **Generative Modeling:**
   * Recreating the underlying high-dimensional data distribution $p(x)$.
   * Unlike standard supervised learning (which often maps inputs to a single deterministic label or low-dimensional categorical distribution), generative models must capture full, complex multi-modal probability distributions.
2. **Self-Supervised Learning (SSL):**
   * Formulating artificial "puzzle" tasks directly from unlabelled data (e.g., masking parts of an input, contrastive learning, temporal prediction).
   * Forces the network to learn rich semantic representations without requiring human-annotated labels.

---

## 2. Theoretical & Conceptual Motivations: Why Do We Care?

### A. The Biological / Parameter Scaling Argument (Geoffrey Hinton, 2014)
* **The Brain's Capacity vs. Experience:**
  * The human brain contains $\approx 10^{14}$ synapses (learnable parameters).
  * A human lives for approximately $10^9$ seconds ($\approx 30$ years).
  * Humans have vastly more parameters to tune than seconds of lived experience or direct supervision signals.
* **Perceptual Information Bandwidth:**
  * Sensory inputs (vision, audition, proprioception) supply on the order of $10^5$ dimensions/constraints per second.
  * To fully utilize the brain’s parameter capacity, learning must be predominantly **unsupervised/self-supervised**, extracting dense constraints from raw sensory perception rather than sparse external feedback.

---

### B. "LeCake" Analogy (Yann LeCun, 2016)
At his NeurIPS 2016 keynote, Yann LeCun proposed the famous cake analogy regarding machine intelligence:
* **The Cake (Unsupervised / Predictive Learning):**
  * The foundational bulk of intelligence.
  * Millions of bits of information per sample (predicting future video frames, missing words, spatial patches).
* **The Icing (Supervised Learning):**
  * $10$ to $10^4$ bits of information per sample (class labels, bounding boxes).
* **The Cherry on Top (Reinforcement Learning):**
  * A few bits of reward signal received sparingly once in a while.

> **Modern Validation:** Modern Large Language Models (LLMs) mirror this exact structure:
> 1. **Base Pre-training** (Unsupervised next-token prediction) $\to$ The Cake
> 2. **Supervised Fine-Tuning (SFT)** $\to$ The Icing
> 3. **RLHF (Reinforcement Learning from Human Feedback)** $\to$ The Cherry

---

### C. Intelligence as Optimal Compression (Kolmogorov Complexity & Solomonoff Induction)
* **Pattern Recognition = Compression:** Finding all regularities/patterns in data is equivalent to finding a compact program that can regenerate the data.
* **Kolmogorov Complexity:** The length of the shortest computer program that outputs a specific dataset.
  * In deep learning, **the neural network architecture and its learned weights constitute the program**.
  * A smaller, more compact neural representation that can accurately reconstruct the data implies a deeper understanding of the underlying data distribution.
* **Theoretical Grounding:**
  * **Solomonoff Induction:** Assigns higher prior probability to simpler explanations (shorter programs), yielding optimal inductive inference.
  * **AIXI:** An extension combining Solomonoff induction with sequential decision theory for optimal reinforcement learning agents.
* **Transfer Learning Perspective (Ilya Sutskever):**
  * If a model is pre-trained on distribution $\mathcal{D}_1$ and subsequently fine-tuned on $\mathcal{D}_2$:
  $$\text{Compressing } \mathcal{D}_2 \text{ conditioned on having already compressed } \mathcal{D}_1 \text{ requires fewer bits/effort than compressing } \mathcal{D}_2 \text{ from scratch.}$$
  * Unsupervised pre-training builds representations that make downstream task adaptation dramatically faster and more data-efficient.

---

## 3. Core Generative Paradigms & Historical Milestones

```
   [2006] Deep Belief Networks (DBNs)
              │
   [2013] Variational Autoencoders (VAEs)
              │
   [2014] Generative Adversarial Networks (GANs) ──► [2015] DCGAN ──► [2018] StyleGAN / BigGAN
              │
   [2016] Autoregressive Models (WaveNet, PixelCNN) ──► [2019+] GPT Series
              │
   [2020] Denoising Diffusion Probabilistic Models (DDPMs) ──► Latent Diffusion / DALL-E 2 / Imagen
```

### 1. Deep Belief Networks (DBNs, 2006)
* Early breakthrough in deep generative modeling (Hinton, Osindero, Teh).
* Demonstrated that multi-layer neural networks could learn to sample handwritten digits (MNIST), proving generative capabilities in neural architectures.

### 2. Variational Autoencoders (VAEs, 2013)
* Introduced explicit probabilistic latent variable modeling (Kingma & Welling).
* Maximizes the Evidence Lower Bound (ELBO).
* **Limitation:** Tends to produce blurry samples due to mode-averaging (reconstruction loss over pixel space).

### 3. Generative Adversarial Networks (GANs, 2014–2018)
* Introduced a game-theoretic framework: **Generator** vs. **Discriminator** (Goodfellow et al., 2014).
* **DCGAN (Radford et al., 2015):** Convolutional architectures enabling high-resolution image generation (e.g., bedrooms, faces).
* **BigGAN / StyleGAN (2018):** Achieved photo-realistic fidelity and smooth latent space interpolations.
* **Limitations of GANs:** Mode collapse (covering only some modes of the distribution while omitting others), unstable adversarial training dynamics, lack of explicit density evaluation.

### 4. Diffusion Models (DDPM, 2020–Present)
* Formulated as a progressive forward noisy process and a learned reverse denoising process (Ho, Jain, Abbeel, 2020).
* **Advantage over GANs:** Stable maximum-likelihood-style objective, eliminates mode collapse, full mode coverage across the data distribution, and exceptional fidelity.
* Serves as the backbone of modern text-to-image/video systems (DALL-E, Imagen, Stable Diffusion, Midjourney).

---

## 4. Multi-Modal Applications & Frontiers

### A. Image & Text-to-Image Generation
* Conditioning diffusion models on text embeddings (CLIP/T5) enables synthesis of out-of-distribution, compositionally novel concepts (e.g., *"a masterful oil painting of a Persian cat discovering crypto losses on its phone"*).
* Demonstrates compositional generalization rather than mere memorization of training instances.

---

### B. Audio & Speech
* **WaveNet (2016/2018):** Autoregressive raw waveform generation at high sampling rates ($16\text{–}24\text{ kHz}$), modeling sample-by-sample dependencies conditioned on previous samples.
* **Tacotron 2 (2018):** Two-stage text-to-speech: text $\to$ Mel-spectrogram $\to$ WaveNet neural vocoder $\to$ raw waveform.
* **AudioCraft / MusicGen / AudioGen (2023):** Discretizes audio into discrete tokens (neural audio codecs), then applies autoregressive language modeling over multi-stream tokens.
* **Voice Conversion / Cloning:** Mapping speaker identity embeddings while preserving linguistic content.

---

### C. Video Generation & World Modeling
* **Factorized Video Modeling (Emu Video, 2023):** Factorizing the joint distribution into generating keyframes first, followed by temporal interpolation.
* **Discrete Video Tokenization (VideoPoet, 2023):** Converting video frames and audio into discrete tokens and utilizing a unified autoregressive transformer.
* **Interactive World Simulators for Control (e.g., UniSim, 2023):**
  * Simulating long sequences of real-world environment interactions directly in video space.
  * Allows robots and autonomous agents to train policies in imagination/simulation before physical execution.

---

### D. Natural Language Processing (The Scaling & Prompting Revolution)
* **Character RNNs (2015, Karpathy):** Demonstrated that predicting the next character in unannotated text naturally learns syntax, grammar, Markdown/LaTeX formatting, and semantic structures.
* **Autoregressive Scaling (GPT-2, 2019 $\to$ GPT-4):**
  * Shifted from building bespoke models per NLP task to universal pre-training on massive corpora.
  * Discovered **in-context learning / zero-shot prompting**: conditioning the model with prefix instructions allows it to execute translation, summarization, arithmetic, and structured JSON output without gradient updates.

---

### E. 3D Generative Modeling
* **NeRF-based 3D Generative Models (GRAF 2020, GET3D 2022):**
  * Learning 3D generative distributions using only collections of unposed 2D images.
  * Explicitly models 3D camera projections and radiance fields, guaranteeing multi-view consistency.

---

### F. AI for Science (Biological Sequence Modeling)
* **Unsupervised Protein Modeling:**
  * Treating biological sequences (DNA, amino acids) as "text" in a biological language (e.g., TAPE benchmark, ESM).
  * Pre-training masked/autoregressive language models over evolutionary sequence databases uncovers secondary/tertiary structures and mutations.
  * Foundational component enabling breakthroughs like **AlphaFold** (Jumper et al., 2021).

---

## 5. Compression via Deep Generative Modeling

* **Lossless vs. Lossy Compression:**
  * **Entropy Bound (Shannon):** The minimum average bits required to encode data sampled from $p(x)$ is given by the cross-entropy / negative log-likelihood:
    $$\text{Bits per byte / dimension} \propto -\mathbb{E}_{x \sim p}[\log_2 q_\theta(x)]$$
  * A generative model with a better likelihood fit ($q_\theta \approx p$) directly yields tighter compression ratios.
* **Generative vs. Handcrafted Codecs:**
  * Handcrafted codecs (JPEG, JPEG2000, MP3) rely on fixed assumptions (e.g., spatial smoothness, frequency masking).
  * Learned generative codecs train optimal encoders/decoders directly on domain distributions, drastically outperforming standard codecs at very low bitrates.

---

## 6. Self-Supervised Learning in Vision: Changing Common Wisdom

A notable case study in how unsupervised learning paradigms evolved:

```
2015: Supervised pre-training on ImageNet dominates computer vision.
      (Efros vs. Malik "Gelato Bet" made: Can unsupervised beat supervised R-CNN?)
      │
2019: Contrastive Predictive Coding (CPC v2) achieves parity with supervised pre-training.
      │
2021: Masked Autoencoders (MAE, He et al.) decisively outperform supervised baselines.
```

### The Key Insight of MAE:
* **The Myth:** *"Masked language modeling (BERT) does not work for computer vision."*
* **The Breakthrough:** Unlike text (which has high information density per token), image pixels are heavily redundant.
  * Masking only $10\%\text{--}20\%$ of an image makes the reconstruction task trivial (pixels can be interpolated from immediate neighbors without semantic understanding).
  * **Masking $75\%\text{--}90\%$ of the image** forces the network to learn holistic scene semantics and structure, creating state-of-the-art visual representations.

---

## 7. Key Takeaways

1. **Scalability:** Unsupervised and self-supervised objectives scale monotonically with compute and dataset size because unlabelled data is virtually unlimited.
2. **Pre-train $\to$ Adapt Paradigm:** Pre-training large foundation models on raw distributions creates universal representations that transfer to downstream tasks (via fine-tuning or prompting) better than task-specific training from scratch.
3. **Fluidity of the Field:** Breakthroughs often occur when common assumptions are challenged (e.g., high masking ratios in vision, diffusion replacing GANs, next-token prediction replacing task-specific architectures).
