Here are the highly detailed, descriptive, blog-style course notes covering every aspect of the video snippet (~126:00 to 163:59), including the transition from VAEs to Diffusion Models, mathematical formulations, and all interactive student discussions.

---

# Course Blog: Transitioning from Variational Autoencoders (VAEs) to Denoising Diffusion Probabilistic Models (DDPMs)

Welcome back! In today’s session, we officially concluded our deep dive into **Variational Autoencoders (VAEs)** and crossed the bridge into one of the most exciting frameworks in modern generative AI: **Denoising Diffusion Probabilistic Models (DDPMs)**. 

Below is a comprehensive breakdown of the lecture, structured to preserve the exact narrative flow, pedagogical choices, mathematical steps, and interactive Q&A of our class.

---

## 1. Wrapping Up the VAE Pipeline: The Final Step

Before moving on, let’s recap how a trained VAE is utilized to generate brand-new data points. The final generation pipeline operates as follows:

1. **Latent Space Modeling:** We have a collection of latent vectors $\{Z_1, Z_2, \dots, Z_d\}$ mapped from our training data. We train a separate generative model (such as a Gaussian Mixture Model (GMM), a PixelCNN, or a simple diffusion model) on these representations.
2. **Sampling:** We sample a completely new latent vector, $\hat{Z}_{\text{new}}$, from this auxiliary generative model.
3. **Decoding:** We pass $\hat{Z}_{\text{new}}$ through our trained VAE Decoder to yield a newly synthesized, high-fidelity data point $\hat{X}_{\text{new}}$.

$$\hat{Z}_{\text{new}} \longrightarrow \boxed{\text{Dec}} \longrightarrow \hat{X}_{\text{new}}$$

This exact hierarchical paradigm—compressing high-dimensional data into a latent space and training a generator on top of those latents—is the fundamental blueprint for industry-scale architectures such as **Stable Diffusion** and Google's **Imagen**.

---

## 2. Introducing Denoising Diffusion Probabilistic Models (DDPMs)

In current computer vision and vision-language tasks, traditional VAEs and Generative Adversarial Networks (GANs) have largely been superseded by **Diffusion Models**. 

To complement our lectures, the instructor strongly recommended reading **Stanley Chan's "Tutorial on Diffusion Models for Imaging and Vision" (arXiv:2403.18103)**. This ~100-page document is highly structured and mirrors our class narrative:
* It begins with the fundamental building blocks of VAEs.
* It details the optimization of the Evidence Lower Bound (ELBO).
* It breaks down the reparameterization trick and high-dimensional KL divergence.
* It systematically transitions those concepts to define DDPMs.

---

## 3. DDPM as a "Special Case" of a VAE

The core thesis of our transition is elegant: **A Denoising Diffusion Probabilistic Model (DDPM) is essentially a special class of Hierarchical VAE.** 

To transform a standard VAE into a DDPM, we must enforce **three specific structural properties**:

### Property 1: Multiple Hierarchical Latent Spaces
* **Standard VAE:** Maps input data $X$ to a single latent space $Z$, and decodes it back:
  $$X \longrightarrow Z \longrightarrow X$$
* **DDPM:** Instead of a single projection, we map the input data through a sequential cascade of multiple latent spaces:
  $$X \longrightarrow Z_1 \longrightarrow Z_2 \longrightarrow Z_3 \longrightarrow \dots \longrightarrow Z_T \quad \text{(Encoding / Forward Process)}$$
  We then reconstruct the data by tracing our way backwards:
  $$Z_T \longrightarrow Z_{T-1} \longrightarrow \dots \longrightarrow Z_1 \longrightarrow X \quad \text{(Decoding / Reverse Process)}$$

> **🙋‍♂️ Class Doubt 1:** *Can you comment on the dimensionality of these latent spaces $Z_1, Z_2, Z_3, \dots$?*
>
> **👨‍🏫 Instructor's Answer:** *That brings us directly to our second property!*

### Property 2: Dimensionality Matching (No Bottleneck)
* **Standard VAE:** The latent space $Z$ acts as a bottleneck. It is highly compressed ($\text{dim}(Z) \ll \text{dim}(X)$) to capture only the most salient features.
* **DDPM:** There is no spatial bottleneck. The dimensionality of *every single* intermediate latent space is exactly identical to the dimensionality of the original data space:
  $$\text{dim}(Z_t) = \text{dim}(X) \quad \forall t \in \{1, \dots, T\}$$

### Property 3: A Fixed, Non-Learnable Encoding Process
* **Standard VAE:** The encoder $q_\phi(Z|X)$ has learnable weights $\phi$ that must be optimized using neural networks.
* **DDPM:** The encoding (forward) process $q(Z_t|Z_{t-1})$ is **completely fixed and non-learnable**. It is a predefined stochastic progression governed by a Markov process. Because we do not need to learn an encoder, our only task is to learn the parameters of the decoding (reverse) process.

---

## 4. Addressing Student Questions & Core Concepts

During the transition, several crucial student questions helped unpack these three properties:

### Q1: What exactly is a "Markov Process" in this context?
A **first-order Markov process** means that if we are at any intermediate latent step $t$, the state of our variable ($Z_t$) depends *only* on the state immediately preceding it ($Z_{t-1}$). It is completely independent of the older history ($Z_{t-2}, Z_{t-3}, \dots, X$).
Mathematically, the transition probability simplifies as:
$$q(Z_t \mid Z_{t-1}, Z_{t-2}, \dots, Z_1, X) = q(Z_t \mid Z_{t-1})$$

### Q2: Are we doing this one data point at a time instead of in batches?
In practice, we still train our models using **mini-batches** for parallelization and GPU efficiency. However, when writing out our mathematical proofs, the ELBO, or probability density functions, we analyze a **single, individual data point** first. Converting a single-point loss function to a batch-level loss function simply requires adding a summation/expectation over the batch size at the very end. To keep our mathematical notations clean, we will stick to single-point analysis.

### Q3: Is the encoding process still probabilistic?
Yes, it is entirely probabilistic because we inject random Gaussian noise at every single transition step. However, it is **fixed (non-learnable)** because the rules (the mean scaling and variance) governing how much noise we add at each step are hardcoded beforehand, rather than learned by an encoder network.

---

## 5. The Notational Shift in Diffusion Literature

The instructor raised an important warning regarding reading papers or Stanley Chan's tutorial: **the diffusion community uses a different set of notations than the VAE community**. 

To align with standard literature, we will adopt the following shift:
* We represent the original clean data point as **$x_0$** (instead of $X$).
* We represent our sequential latent spaces as **$x_1, x_2, \dots, x_T$** (instead of $z_1, z_2, \dots, z_T$).

| Concept | VAE Notation | DDPM Notation |
| :--- | :---: | :---: |
| **Original Clean Input** | $X$ | $x_0$ |
| **Intermediate Latent States** | $Z_1, Z_2, \dots$ | $x_1, x_2, \dots$ |
| **Final Terminating Latent State** | $Z_T$ | $x_T$ |

*Note: Do not confuse $x_1, \dots, x_T$ as different images in a batch! They represent the progressive, increasingly noisy states of the **same single image** $x_0$.*

---

## 6. Mathematical Formulation of the Forward (Encoding) Process

Let's write down the exact stochastic equations that define how we transition from our clean image $x_0$ to the final noisy state $x_T$.

The forward process progresses as:
1. **Step 1:** $x_1 = \alpha_1 x_0 + (1 - \alpha_1) \epsilon_1 \quad$ where $\epsilon_1 \sim \mathcal{N}(0, I)$
2. **Step 2:** $x_2 = \alpha_2 x_1 + (1 - \alpha_2) \epsilon_2 \quad$ where $\epsilon_2 \sim \mathcal{N}(0, I)$
3. **Step $t$:** $x_t = \alpha_t x_{t-1} + (1 - \alpha_t) \epsilon_t \quad$ where $\epsilon_t \sim \mathcal{N}(0, I)$

Where:
* $\alpha_1, \alpha_2, \dots, \alpha_T$ are **fixed, predetermined scalars** chosen in the range $[0, 1]$.
* $\epsilon_1, \dots, \epsilon_T$ represent independent standard Gaussian noise.
* All variables preserve identical dimensionality: $\text{dim}(x_0) = \text{dim}(x_t)$.

> **✏️ Class Correction Note:** *During the board derivation, the instructor initially indexed the first equation starting at $\alpha_0$ and $\epsilon_0$. A student pointed out that to stay consistent with the target state index $t$, the subscripts should match. The equations were updated to use $\alpha_t$ and $\epsilon_t$ for state $x_t$, which aligns with standard formulations.*

---

## 7. Visualizing the Forward Diffusion Chain

To build an intuition of what is happening under the hood, imagine we start with a clean image of a dog ($x_0$):

1. At $x_1$, we scale the image down slightly by $\alpha_1$ and add a tiny bit of Gaussian noise. The dog is still clearly visible, just slightly pixelated.
2. At $x_2$, we take the noisy $x_1$, scale it down further, and inject more noise.
3. As we repeat this process over $T$ steps (e.g., $T = 1000$):
   
$$x_0 \longrightarrow x_1 \longrightarrow x_2 \longrightarrow \dots \longrightarrow x_T$$

4. Eventually, all structural information of the dog is completely destroyed. The final latent state $x_T$ reaches its **stationary distribution**, which is pure isotropic Gaussian noise:
   
$$x_T \sim \mathcal{N}(0, I)$$

---

## 8. Summary & Looking Ahead

By structuring DDPM as a Hierarchical VAE with matched dimensions and a fixed Markovian encoder, we have established our starting point. 

Our goal is now clear: since the forward path of adding noise is entirely fixed, we must train a neural network (typically a U-Net) to learn the **reverse (decoding) path** $p_\theta(x_{t-1} \mid x_t)$. This decoder will take a block of pure noise $x_T \sim \mathcal{N}(0, I)$ and gradually denoise it step-by-step back into a clean image $x_0$.

In the next class, we will write down the **ELBO** specifically for this multi-latent setup, perform the algebraic expansion of the KL terms, and derive the final loss functions used to train DDPMs!

# Lecture Notes: Deep Generative Models — Lecture 9: Denoising Diffusion Probabilistic Models (DDPMs)

---

## 1. Classroom Warm-up: The $\beta$-VAE Quiz Controversy
Before diving into the mathematics of diffusion models, the class spent some time discussing a quiz question on $\beta$-Variational Autoencoders ($\beta$-VAEs). This discussion highlights the delicate balance between latent disentanglement and reconstruction quality in generative modeling.

### The Core Discussion
*   **The Question:** What happens to disentanglement and reconstruction quality as we increase the weight ($\beta$) on the Kullback-Leibler (KL) divergence penalty in the VAE loss?
*   **The Loss Function:** 
    $$\mathcal{L}_{\beta\text{-VAE}} = \text{Reconstruction Loss} + \beta \cdot D_{KL}(q_\phi(z|x) \parallel p(z))$$

### Student Debates & Teacher Clarifications

> **Student Doubt (Sanchit/Suhas):** *"Reconstruction quality reduces when we increase $\beta$, right? That is definite. But what do we mean by entanglement? If the posterior becomes normal, does entanglement increase or decrease?"*

*   **The Explanation:** Entanglement refers to how mixed or correlated the underlying generative factors are within the latent space. 
    *   If $\beta$ is tuned appropriately ($\beta > 1$), it forces the latent dimensions to align with independent, orthogonal factors of variation, which *increases disentanglement* (a good thing for interpretability).
    *   However, if $\beta$ is made excessively large, the KL divergence term dominates. This forces the posterior $q_\phi(z|x)$ to match the prior $p(z) = \mathcal{N}(0, I)$ completely, regardless of the input $x$. 
    *   When the posterior collapses to the prior, the latent code becomes uninformative. Every input maps to the exact same standard Gaussian. This represents **maximum entanglement (or representation collapse)**, and the reconstruction quality drops significantly because the decoder has no input-specific information to work with.

*   **Resolution:** Because the term "disentanglement" can be interpreted differently depending on whether one refers to the independent alignment of latent factors or the complete collapse/overlap of the latent space under extreme prior regularisation, the professor decided to bypass the ambiguity:
    > *"There is some confusion on how this question can be interpreted. To make things easy, we will ignore this question on the quiz and add one mark to everyone's score. Let's move on to today's topic."*

---

## 2. Introduction to Denoising Diffusion Probabilistic Models (DDPMs)
Denoising Diffusion Probabilistic Models (DDPMs) can be viewed as a special class of Hierarchical Variational Autoencoders. To understand how they differ from standard VAEs, we can outline three core structural properties:

1.  **Multiple Latent Spaces (Hierarchy):** While a standard VAE has a single latent space $Z$ mapping to data space $X$, a DDPM defines a chain of latent spaces. We denote the data as $x_0$, and the latent variables as $x_1, x_2, \dots, x_T$.
2.  **Dimensionality Preservation:** Unlike VAEs which often compress data into a lower-dimensional bottleneck, in DDPMs, the dimensionality of all latent spaces is identical to the data space:
    $$\text{dim}(x_0) = \text{dim}(x_1) = \text{dim}(x_2) = \dots = \text{dim}(x_T)$$
3.  **Fixed (Non-Learnable) Encoder:** In a VAE, we learn both the encoder $q_\phi(z|x)$ and the decoder $p_\theta(x|z)$. In DDPMs, the encoding process (forward process) is completely fixed and defined by a Markov chain that systematically adds Gaussian noise to the data until it becomes pure noise. We only learn the decoding process (reverse process) which reconstructs the data by denoising.

---

## 3. The Forward (Encoding) Process
The forward process is a Markov chain that starts at the data distribution $x_0$ and sequentially adds small amounts of Gaussian noise at each step $t \in [1, T]$ according to a predefined noise schedule $\alpha_1, \alpha_2, \dots, \alpha_T$ (where $\alpha_t \in (0, 1)$).

The transition at step $t$ is mathematically defined as:
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t, \quad \text{where } \epsilon_t \sim \mathcal{N}(0, I)$$

This transition yields the conditional Gaussian distribution:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)I)$$

```
Forward Process (Noising - Fixed):
   [ x_0 (Data) ] ---> [ x_1 ] ---> [ x_2 ] ---> ... ---> [ x_T (Pure Noise) ]
```

### Student Debates & Teacher Clarifications

> **Student Doubt (Sarvesh):** *"Sir, in the encoding process, where did the square root come from? In the previous class, wasn't it written as $x_t = \alpha_t x_{t-1} + (1-\alpha_t)\epsilon_t$?"*

*   **The Explanation:** The square roots are part of the *variance-preserving* formulation. If we assume that $x_{t-1}$ has unit variance, we want to ensure $x_t$ also maintains unit variance after adding noise. 
    Let's compute the variance of $x_t$:
    $$\text{Var}(x_t) = \text{Var}(\sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t)$$
    Since $x_{t-1}$ and $\epsilon_t$ are independent:
    $$\text{Var}(x_t) = (\sqrt{\alpha_t})^2 \text{Var}(x_{t-1}) + (\sqrt{1 - \alpha_t})^2 \text{Var}(\epsilon_t)$$
    $$\text{Var}(x_t) = \alpha_t (1) + (1 - \alpha_t) (1) = 1$$
    Using the square root coefficients ensures that the scale of the data does not explode or collapse as we traverse the chain up to $T$ steps.

---

## 4. The Reverse (Decoding) Process
The reverse process is also a Markov chain, but it runs in the opposite direction. It starts from a standard Gaussian prior $p(x_T) = \mathcal{N}(0, I)$ and sequentially denoises the latent variable to reconstruct the data $x_0$. 

```
Reverse Process (Denoising - Learnable):
   [ x_0 (Data) ] <--- [ x_1 ] <--- [ x_2 ] <--- ... <--- [ x_T (Prior) ]
```

Because the reverse step $q(x_{t-1}|x_t)$ is mathematically intractable to compute directly (as it requires integrating over the entire data distribution), we approximate it using a parameterized Gaussian transition $p_\theta(x_{t-1}|x_t)$:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Where the mean $\mu_\theta$ and covariance $\Sigma_\theta$ are parameterized by a deep neural network (such as a U-Net) that we must train.

---

## 5. Mathematical Derivation of the DDPM ELBO
To train our reverse model parameters $\theta$, we maximize the evidence lower bound (ELBO) on the log-likelihood of the data $x_0$. Let's derive this step-by-step.

We define the approximate posterior (forward process) conditioned on $x_0$ as:
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

The joint distribution of the generative (reverse) process is:
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$

Using the standard variational lower bound setup:
$$\log p_\theta(x_0) \ge \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right]$$

Let us analyze the log-fraction inside the expectation:
$$\text{ELBO} = \mathbb{E}_{q} \left[ \log \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^T q(x_t|x_{t-1})} \right]$$

To make this objective tractable, we isolate the first step ($t=1$) of both products:
$$\text{ELBO} = \mathbb{E}_{q} \left[ \log \frac{p(x_T) \cdot p_\theta(x_0|x_1) \prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_1|x_0) \prod_{t=2}^T q(x_t|x_{t-1})} \right]$$

### The Conditioning Trick (Bayes' Rule)
The forward transitions $q(x_t|x_{t-1})$ go in the forward direction. We want to reformulate them to match the backward transitions. Since the process is Markovian, conditioning on $x_0$ doesn't alter the transition $q(x_t|x_{t-1}) = q(x_t|x_{t-1}, x_0)$. 

By applying Bayes' rule to the terms conditioned on $x_0$, we can write:
$$q(x_t | x_{t-1}, x_0) = \frac{q(x_{t-1} | x_t, x_0) \cdot q(x_t | x_0)}{q(x_{t-1} | x_0)}$$

Now, let's substitute this expansion back into the denominator product for $t \ge 2$:
$$\prod_{t=2}^T q(x_t|x_{t-1}) = \prod_{t=2}^T \frac{q(x_{t-1} | x_t, x_0) \cdot q(x_t | x_0)}{q(x_{t-1} | x_0)}$$

If we expand this product, we see a telescoping sequence of fractions:
$$\prod_{t=2}^T \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} = \frac{q(x_2|x_0)}{q(x_1|x_0)} \cdot \frac{q(x_3|x_0)}{q(x_2|x_0)} \cdot \frac{q(x_4|x_0)}{q(x_3|x_0)} \dots \frac{q(x_T|x_0)}{q(x_{T-1}|x_0)}$$

Notice how all intermediate terms cancel out:
$$\prod_{t=2}^T \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} = \frac{q(x_T|x_0)}{q(x_1|x_0)}$$

Therefore, the entire forward denominator product simplifies to:
$$\prod_{t=2}^T q(x_t|x_{t-1}) = \frac{q(x_T|x_0)}{q(x_1|x_0)} \prod_{t=2}^T q(x_{t-1}|x_t, x_0)$$

We can now plug this simplified denominator back into our ELBO equation:
$$\text{ELBO} = \mathbb{E}_{q} \left[ \log \frac{p(x_T) \cdot p_\theta(x_0|x_1) \prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_1|x_0) \cdot \left( \frac{q(x_T|x_0)}{q(x_1|x_0)} \prod_{t=2}^T q(x_{t-1}|x_t, x_0) \right)} \right]$$

Notice that $q(x_1|x_0)$ in the denominator cancels out:
$$\text{ELBO} = \mathbb{E}_{q} \left[ \log \frac{p(x_T) \cdot p_\theta(x_0|x_1) \prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_T|x_0) \prod_{t=2}^T q(x_{t-1}|x_t, x_0)} \right]$$

Now, using the laws of logarithms, we split the fraction into three distinct terms:
$$\text{ELBO} = \mathbb{E}_{q} \left[ \log p_\theta(x_0|x_1) + \log \frac{p(x_T)}{q(x_T|x_0)} + \sum_{t=2}^T \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t, x_0)} \right]$$

By utilizing the properties of expectations (marginalizing out variables that do not appear in specific terms), we arrive at the final three-term objective:
$$\text{ELBO} = \mathbb{E}_{q(x_1|x_0)} \left[ \log p_\theta(x_0|x_1) \right] - D_{KL}(q(x_T|x_0) \parallel p(x_T)) - \sum_{t=2}^T \mathbb{E}_{q(x_t|x_0)} \left[ D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_\theta(x_{t-1}|x_t)) \right]$$

---

## 6. Deconstructing the Three Terms of the DDPM ELBO

The final ELBO formula is composed of three beautifully structured, intuitive terms:

```
                  =======================================================
                  =               THE DDPM ELBO OBJECTIVE               =
                  =======================================================
                                             |
       --------------------------------------+-------------------------------------
       |                                     |                                    |
       v                                     v                                    v
[ RECONSTRUCTION ]                    [ PRIOR MATCHING ]                 [ DENOISING MATCHING ]
E_q [ log p_θ(x_0|x_1) ]               -D_KL( q(x_T|x_0) || p(x_T) )      -Σ E_q [ D_KL( q(x_t-1|x_t,x_0) || p_θ ) ]
       |                                     |                                    |
       v                                     v                                    v
Reconstructs the original             Measures how close final noised     Compares the learnable step
data point x_0 from its               distribution is to prior normal.    with the tractable ground-truth
one-step noised version x_1.          No θ parameters; ignored during     posterior transition.
                                      training.
```

### Term 1: The Reconstruction Term
$$\mathbb{E}_{q(x_1|x_0)} \left[ \log p_\theta(x_0|x_1) \right]$$
*   **Meaning:** This term measures how well the model can reconstruct the original clean data point $x_0$ from its slightly noised version $x_1$ at the very end of the reverse process. 
*   **Role:** This is equivalent to the reconstruction term in a standard VAE.

### Term 2: The Prior Matching Term
$$- D_{KL}(q(x_T|x_0) \parallel p(x_T))$$
*   **Meaning:** This term compares the distribution of the final noised state $q(x_T|x_0)$ with our standard Gaussian prior $p(x_T) = \mathcal{N}(0, I)$.
*   **Role:** Since the forward process is fixed and $T$ is chosen to be sufficiently large (e.g., $T = 1000$), $q(x_T|x_0)$ is guaranteed to converge to a standard normal distribution. This term does not depend on our learnable parameters $\theta$. Therefore, **it is a constant during training and can be ignored**.

### Term 3: The Denoising Matching / Consistency Terms
$$- \sum_{t=2}^T \mathbb{E}_{q(x_t|x_0)} \left[ D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_\theta(x_{t-1}|x_t)) \right]$$
*   **Meaning:** This is the core of diffusion training. Instead of trying to force $p_\theta(x_{t-1}|x_t)$ to match the intractable forward step $q(x_t|x_{t-1})$ directly, we match it to the **ground-truth posterior transition** $q(x_{t-1}|x_t, x_0)$.
*   **Role:** Because we condition on the clean data $x_0$, the distribution $q(x_{t-1}|x_t, x_0)$ is a **tractable Gaussian** that can be calculated analytically. We are training our network $p_\theta(x_{t-1}|x_t)$ to perform a single denoising step that matches this analytically known transition.

---

## 7. Deep-Dive: Key Student Doubts & Clarifications

### Why is the encoding process fixed?

> **Student Doubt (Abhitosh):** *"Why do we fix the encoding process instead of learning it? In VAEs, we learn the encoder parameters $\phi$. Why is it better to keep it fixed?"*

*   **The Explanation:** In standard VAEs, learning both the encoder $q_\phi(z|x)$ and decoder $p_\theta(x|z)$ simultaneously can lead to optimization issues like posterior collapse or unstable training dynamics. 
    In DDPMs, we bypass the need to learn an encoder by defining a simple physical process: adding noise. By using a Markov chain with Gaussian transitions, statistical theory guarantees that the final state $x_T$ will converge to a standard Gaussian prior. 
    This means we get a mathematically structured latent space for free, allowing us to focus the neural network's capacity entirely on learning the reverse process (denoising).

---

### Why must we condition on $x_0$ to make the reverse process tractable?

> **Student Doubt (Sanchit):** *"If we did not condition on $x_0$ in Bayes' rule, could we still derive a backward distribution $q(x_{t-1}|x_t)$ directly without $x_0$?"*

*   **The Explanation:** If we do not condition on $x_0$, Bayes' rule gives us:
    $$q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1}) \cdot q(x_{t-1})}{q(x_t)}$$
    To compute this, we would need the marginal distributions $q(x_{t-1})$ and $q(x_t)$. These marginals depend on the overall data distribution $q(x_0)$:
    $$q(x_t) = \int q(x_t|x_0) \cdot q(x_0) \, dx_0$$
    Since $q(x_0)$ is our unknown, highly complex real-world data distribution (e.g., natural images), this integral is completely intractable.
    However, if we condition everything on a single data point $x_0$, the marginal distributions $q(x_t|x_0)$ and $q(x_{t-1}|x_0)$ are **simple Gaussians** that we can compute analytically! 
    Thus, conditioning on $x_0$ is the mathematical key that makes the training objective computable.

---

### Correcting the Reverse Indexing

> **Student Doubt (Lokesh):** *"Sir, in the joint distribution $p_\theta(x_{0:T})$, shouldn't the sequence of decoding transitions go from $x_T$ down to $x_0$ rather than the other way around? Why is the last term $p_\theta(x_0|x_1)$?"*

*   **The Explanation:** Lokesh is correct. The generative process is a reverse Markov chain that begins with the prior $p(x_T)$ and moves backward to $x_0$. 
    By applying the chain rule of probability in reverse, we write:
    $$p_\theta(x_{0:T}) = p(x_T) \cdot p_\theta(x_{T-1}|x_T) \cdot p_\theta(x_{T-2}|x_{T-1}) \dots p_\theta(x_0|x_1)$$
    $$\text{Or compactly: } p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$
    So indeed, the final step in generating clean data from noise is taking $x_1$ and using the decoder transition $p_\theta(x_0|x_1)$ to reconstruct the final clean data point $x_0$.

---

## 8. Summary & Next Steps
We have derived the tractable ELBO objective for DDPMs. By utilizing a fixed forward Markov chain and applying Bayes' rule conditioned on $x_0$, we transformed an intractable hierarchical VAE objective into a series of simple KL divergence terms between Gaussian distributions. 

In the next lecture, we will look at how we can analytically calculate the mean and variance of $q(x_{t-1}|x_t, x_0)$ and see how this reduces the training objective to a simple **mean-squared error loss** on the predicted noise.

# Deep Dive into DDPMs: Reparameterization, Bayes' Posterior, and the Noise-Prediction Objective

---

## 1. Introduction: The Core Objective

In our study of generative models, Denoising Diffusion Probabilistic Models (DDPMs) stand out for their stability and sample quality. At the heart of a DDPM's training objective is the **Evidence Lower Bound (ELBO)**. When we expand the ELBO for diffusion models, it decomposes into three main components:
1. A reconstruction term at step $t=1$.
2. A prior matching term at the final step $T$ (which has no trainable parameters if the prior is fixed).
3. A sum of **consistency terms** (or **denoising matching terms**) for all intermediate steps:

$$\sum_{t=2}^{T} \mathbb{E}_{q(x_t | x_0)} \left[ D_{KL} \left( q(x_{t-1} | x_t, x_0) \,||\, p_\theta(x_{t-1} | x_t) \right) \right]$$

This consistency term acts as our training target. It forces the learnable reverse transitions $p_\theta(x_{t-1} | x_t)$ to match the true mathematical denoising transition $q(x_{t-1} | x_t, x_0)$ when the original clean data point $x_0$ is known. 

To compute and optimize this KL divergence, we must derive analytical expressions for both terms. This requires calculating the true posterior distribution $q(x_{t-1} | x_t, x_0)$ and showing that we can jump directly to any arbitrary time step $t$ during the forward process without iterative sampling.

---

## 2. Denoising Posterior via Bayes' Rule

To compute $q(x_{t-1} | x_t, x_0)$, we apply Bayes' rule to write the conditional probability in a tractable form:

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \, q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

Due to the Markov property of the forward process, the state $x_t$ depends only on $x_{t-1}$. Knowing $x_0$ provides no additional information if $x_{t-1}$ is already given. Therefore, we can simplify $q(x_t | x_{t-1}, x_0)$ to $q(x_t | x_{t-1})$:

$$q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}) \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

While $q(x_t | x_{t-1})$ is defined directly by our forward transition schedules, the terms $q(x_{t-1} | x_0)$ and $q(x_t | x_0)$ require marginalizing over all intermediate steps. Performing this marginalization iteratively is computationally expensive. Instead, we use a recursive reparameterization trick to compute these marginals in closed form.

---

## 3. Direct Step Marginalization via Recursion

The forward process adds Gaussian noise at each step according to:

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1}, \quad \text{where } \epsilon_{t-1} \sim \mathcal{N}(0, I)$$

Here, $\alpha_t = 1 - \beta_t$, and $\beta_t$ represents the noise variance schedule. To express $x_t$ directly as a function of the clean starting image $x_0$, we recursively unfold this definition.

### Step 1: Unfolding One Step
Substitute the definition of $x_{t-1}$ into the equation for $x_t$:

$$x_t = \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2} \right) + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$

Distributing the scaling factors yields:

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$

### Step 2: Combining Gaussian Variables
Both $\epsilon_{t-2}$ and $\epsilon_{t-1}$ are independent, standard normally distributed random variables. The sum of two independent Gaussians is also a Gaussian distribution:

$$X \sim \mathcal{N}(0, \sigma_1^2 I), \quad Y \sim \mathcal{N}(0, \sigma_2^2 I) \implies X + Y \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2) I)$$

Applying this rule to our noise terms:
* The variance of the first scaled noise term is: $\sigma_1^2 = \alpha_t(1 - \alpha_{t-1})$
* The variance of the second scaled noise term is: $\sigma_2^2 = 1 - \alpha_t$

Summing these variances simplifies to:

$$\sigma_{\text{combined}}^2 = \alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \alpha_{t-1}$$

This allows us to collapse the two noise terms into a single standard Gaussian noise term $\epsilon^* \sim \mathcal{N}(0, I)$:

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \epsilon^*$$

---

### Student Doubt Interlude

> **Student (Raghavendra):** *Did we just prove that a Gaussian of a Gaussian is also a Gaussian?*
>
> **Instructor:** *To be precise, we are using the property that a linear combination of independent Gaussian random variables is also Gaussian. If we scale and add independent standard normal samples, the result remains normally distributed. We compute the new mean by scaling the means, and the new variance by summing the squared scaling factors. This is the foundation of the reparameterization trick, allowing us to combine any sequence of Gaussian noise additions into a single step.*

---

### Step 3: Generalizing to $x_0$
By repeating this recursive substitution down to $x_0$, we obtain the general form:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \bar{\alpha}_t = \prod_{i=1}^t \alpha_i \text{ and } \epsilon \sim \mathcal{N}(0, I)$$

This derivation yields the closed-form marginal distribution of $x_t$ given $x_0$:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

This is a critical property of DDPMs: **we can sample any arbitrary noisy state $x_t$ directly from the clean data $x_0$ in a single step**, bypassing the need to generate the intermediate states $x_1, x_2, \dots, x_{t-1}$.

---

## 4. Analytical Derivation of the Posterior

Having derived the closed-form expressions for the marginals, we can evaluate the Bayes' rule expansion:

$$q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}) \frac{q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

We substitute the respective Gaussian probability density functions into this formula:
1. $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I) \propto \exp \left( -\frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{2\beta_t} \right)$
2. $q(x_{t-1} | x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1 - \bar{\alpha}_{t-1}) I) \propto \exp \left( -\frac{\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2}{2(1 - \bar{\alpha}_{t-1})} \right)$
3. $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I) \propto \exp \left( -\frac{\|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2}{2(1 - \bar{\alpha}_t)} \right)$

We collect the exponential terms and focus on the quadratic forms involving $x_{t-1}$:

$$\text{Exponent} = -\frac{1}{2} \left[ \frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{\beta_t} + \frac{\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2}{1 - \bar{\alpha}_{t-1}} - \frac{\|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2}{1 - \bar{\alpha}_t} \right]$$

Since $x_t$ and $x_0$ are conditioned (and treated as constants with respect to $x_{t-1}$), we expand these quadratic terms and group by the coefficients of $x_{t-1}^2$ and $x_{t-1}$:

$$\text{Exponent} = -\frac{1}{2} \left[ \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) x_{t-1}^2 - 2 \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right) x_{t-1} + C \right]$$

where $C$ is a constant term independent of $x_{t-1}$.

By completing the square, we identify this as a Gaussian distribution $\mathcal{N}(x_{t-1}; \mu_q(x_t, x_0), \Sigma_q(t))$:

### 1. Denoising Variance ($\Sigma_q(t)$)
The inverse of the variance is the coefficient of $x_{t-1}^2$:

$$\frac{1}{\tilde{\beta}_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})}$$

Substituting $\alpha_t = 1 - \beta_t$ into the numerator:

$$\alpha_t - \alpha_t \bar{\alpha}_{t-1} + \beta_t = 1 - \beta_t - \bar{\alpha}_t + \beta_t = 1 - \bar{\alpha}_t$$

Thus, the variance $\tilde{\beta}_t$ is:

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

### 2. Denoising Mean ($\mu_q(x_t, x_0)$)
The mean is the coefficient of $x_{t-1}$ divided by the coefficient of $x_{t-1}^2$:

$$\mu_q(x_t, x_0) = \tilde{\beta}_t \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right)$$

Substituting $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ yields:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$$

Using $\beta_t = 1 - \alpha_t$, we get the standard expression for the true posterior mean:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} x_0$$

---

## 5. Simplifying the KL Divergence Target

During training, we want our parameterized reverse transition $p_\theta(x_{t-1} | x_t)$ to match the true posterior $q(x_{t-1} | x_t, x_0)$. We model the reverse process as a Gaussian transition:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(t))$$

In standard DDPM implementations, we fix the covariance $\Sigma_\theta(t) = \sigma_q^2(t) I$ to be equal to the true posterior covariance $\tilde{\beta}_t I$ (or simply the forward noise schedule $\beta_t I$). 

This choice simplifies the KL divergence calculation. For two multivariate Gaussians sharing the same covariance matrix, the KL divergence simplifies to the squared $L_2$ distance between their means:

$$D_{KL} \left( q(x_{t-1} | x_t, x_0) \,||\, p_\theta(x_{t-1} | x_t) \right) = \frac{1}{2\sigma_q^2(t)} \|\mu_q(x_t, x_0) - \mu_\theta(x_t, t)\|^2$$

This demonstrates a key property of DDPMs: **the KL divergence objective simplifies to a Mean Squared Error (MSE) loss between the true posterior mean $\mu_q$ and our model's predicted mean $\mu_\theta$.**

---

### Student Doubt Interlude

> **Student (Sanchit):** *Why are we assuming the decoding distribution is Gaussian? Isn't that a highly restrictive assumption for modeling complex real-world data like natural images?*
>
> **Instructor:** *We are not assuming that the final data distribution $p(x_0)$ is Gaussian. We only assume that the conditional transition from $x_t$ to $x_{t-1}$ is Gaussian. If the step sizes $\beta_t$ are sufficiently small, this Gaussian conditional assumption holds mathematically. The marginal distribution at the end of the chain, $p_\theta(x_0)$, is a deep hierarchical mixture model (similar to an infinitely deep VAE). By chaining together hundreds or thousands of simple Gaussian transitions, the model can generate highly non-Gaussian, complex data distributions.*

---

## 6. The Noise-Prediction Objective

The objective function currently targets mean matching:

$$\mathcal{L}_{\text{simple}} = \mathbb{E} \left[ \|\mu_q(x_t, x_0) - \mu_\theta(x_t, t)\|^2 \right]$$

However, we can parameterize this target differently. We know the forward process defines $x_t$ as:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \implies x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right)$$

We substitute this expression for $x_0$ back into our formulation for $\mu_q(x_t, x_0)$:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} \left[ \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon \right) \right]$$

Since $\sqrt{\bar{\alpha}_t} = \sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}$, we simplify the coefficients:

$$\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

This formulation shows that the true posterior mean is simply a scaled version of $x_t$, adjusted by subtracting a scaled version of the added noise $\epsilon$. 

To match this structure, we parameterize our model's mean $\mu_\theta(x_t, t)$ to predict the noise instead of the raw mean:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

where $\epsilon_\theta(x_t, t)$ is a neural network trained to predict the noise $\epsilon$.

Substituting these reparameterized means back into our MSE loss causes the shared terms to cancel, simplifying the objective function to:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right] = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)\|^2 \right]$$

This is the standard training objective for DDPMs. Instead of learning to generate an image directly, the neural network learns to predict the noise added to a clean image at any given timestep.

---

## 7. Network Architecture and the Training Protocol

```
Forward Process (Fixed):
x_0 (Clean) -------> [Add Noise] -------> x_t (Noisy Image)
                         ^
                         | (Sampled t, ε)

Reverse Process (Learnable):
x_t (Noisy Image) -------> [ U-Net ] -------> ε_θ (Predicted Noise)
                             ^
                             |
                           t (Timestep Embedding)
```

In practice, we implement this training objective using a **U-Net** architecture. This network takes both the noisy image $x_t$ and a scalar timestep $t$ as inputs. The timestep $t$ is projected into a high-dimensional embedding space (similar to positional encodings in Transformers) and injected into the U-Net's residual blocks.

### Step-by-Step Training Algorithm

For each training iteration:
1. **Sample clean data:** Draw a clean image $x_0 \sim q(x_0)$ from your dataset.
2. **Sample a timestep:** Uniformly select a random timestep $t \sim \mathcal{U}(\{1, \dots, T\})$.
3. **Sample random noise:** Generate a noise vector $\epsilon \sim \mathcal{N}(0, I)$ of the same shape as $x_0$.
4. **Construct the noisy input:** Compute the corrupted image $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
5. **Predict the noise:** Pass $x_t$ and $t$ through the U-Net to obtain the noise estimate $\epsilon_\theta(x_t, t)$.
6. **Optimize:** Compute the gradient of the MSE loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ and update the network weights $\theta$.

By framing the model to predict the noise $\epsilon$ rather than the absolute mean $\mu$, the neural network acts as a score-matching model. This focus on high-frequency noise prediction simplifies the optimization landscape, contributing to the training stability of DDPMs.