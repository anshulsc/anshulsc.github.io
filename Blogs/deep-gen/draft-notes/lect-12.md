# Understanding Normalizing Flows: Motivations, The Big Idea, and Essential Foundations

Welcome to this comprehensive, detailed guide on **Normalizing Flows**, a class of probabilistic generative models designed to address two of the most fundamental challenges in machine learning and statistics: **exact density estimation** and **efficient sampling**. 

While deep generative models like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) have gained immense popularity, they often compromise on either direct density evaluation or sampling tractability. Normalizing Flows offer an elegant mathematical framework to achieve both simultaneously.

---

## 1. The Two Core Challenges of Generative Modeling

At the heart of probabilistic machine learning lie two fundamental questions:

1. **Density Estimation:** *To which distribution does my data belong?* Given a set of observed data points, can we identify the underlying probability density function (PDF) and its parameters?
2. **Sampling:** *Can I generate new, novel data points from this distribution?* Once we have captured the distribution, can we easily draw new samples that look like they belong to the same dataset?

Let’s unpack why solving these two challenges is so valuable.

### Challenge 1: Why Do We Need Density Estimation?
Knowing the exact probability density of a data point is incredibly useful for two main reasons:
* **Exploration:** It allows us to map out the entire space of the data. We can understand the relationship between different variables, identify modes (peaks) in the distribution, and see how the density changes if we modify the underlying generation process.
* **Decision Making:** Density estimation enables exact uncertainty computation. If a system encounters a data point with an extremely low probability density, it can flag it as an outlier or novelty. This is critical for raising alerts in real-world safety systems (such as medical diagnosis or autonomous driving).

> **Traditional Machine Learning Limitations:** Traditional ML classification or regression algorithms are not typically designed with exact density computation in mind. While some classifiers can output a "score" or "confidence," these are often uncalibrated and do not represent a valid, normalized probability density.

---

### 🙋 Student Doubt: Why can't we just use classic density estimators like histograms or kernel density estimation?
> **Answer:** Classic non-parametric methods like histograms or Kernel Density Estimation (KDE) suffer heavily from the **curse of dimensionality**. In high-dimensional spaces (e.g., images with thousands of pixels), the amount of data required to populate the space grows exponentially. Additionally, computing KDE over millions of points at test time becomes prohibitively expensive. This is why we must turn to parametric methods that leverage deep neural networks.

---

### Challenge 2: Why Do We Need Tractable Sampling?
Being able to generate new data points from a learned distribution has massive practical benefits:
* **Simplifying Computation:** In Bayesian inference, we often need to compute the posterior distribution $p(z|x) = \frac{p(x|z)p(z)}{p(x)}$. However, the denominator (the normalizing constant or marginal likelihood) requires solving an integral over the latent space:
  $$\int p(x|z)p(z) dz$$
  In high dimensions, this integral is mathematically intractable. If we can easily sample from our distributions, we can approximate these complex integrals using Monte Carlo methods.
* **Latent Variable Models:** Models like VAEs use latent variables to represent underlying concepts in data. Traditionally, to make these models mathematically workable, researchers assume very simple latent distributions (like a standard Gaussian). While this makes sampling easy, it restricts the model's capacity to represent highly complex data.
* **Synthetic Data Generation:** Collecting and labeling real-world datasets is expensive and time-consuming. Furthermore, privacy regulations increasingly restrict the use of sensitive real-world datasets. Being able to sample high-quality synthetic data from a sophisticated, learned distribution helps us train deep learning models safely and cheaply.

---

### 🙋 Student Doubt: If we already have sampling algorithms like Markov Chain Monte Carlo (MCMC), why invent something new?
> **Answer:** MCMC methods (such as Metropolis-Hastings or Hamiltonian Monte Carlo) are mathematically robust and work in high dimensions, but they are highly iterative. They require a "burn-in" period and sequential sampling, making them extremely slow. For industrial applications with real-time constraints—especially on edge or embedded devices—MCMC is computationally impractical. Normalizing Flows seek to make both sampling and density estimation extremely fast and direct.

---

## 2. Placing Normalizing Flows in the Generative Landscape

To appreciate the design of Normalizing Flows, let us briefly compare them with other prominent deep generative models.

```
                         Deep Generative Models
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  Generative Adversarial      Variational Autoencoders    Normalizing Flows
    Networks (GANs)                (VAEs)                  (e.g., RealNVP, Glow)
  • Great sampling quality    • Explicit density proxy    • Exact density estimation
  • Implicit distribution     • Approximate inference     • Exact, tractable sampling
  • No density estimation     • Rigid latent assumptions  • Invertible mappings
```

### Generative Adversarial Networks (GANs)
GANs are highly effective at sampling. By setting up a game between a Generator and a Discriminator, they learn to generate incredibly realistic data. However, GANs only learn the distribution *implicitly*. They do not define an explicit probability density function. You cannot feed an arbitrary image into a standard GAN and get back a number representing its exact likelihood.

### Variational Autoencoders (VAEs)
VAEs define an explicit probabilistic model using latent variables. They are trained by maximizing the **Evidence Lower Bound (ELBO)**, which is a variational proxy for the true log-likelihood. While VAEs allow for tractable sampling, they struggle with two main limitations:
1. They rely on an *approximation* of the density (ELBO), not the exact density.
2. They typically require rigid, simple assumptions (like Gaussian priors) on both the latent space and the target space, which can limit their capacity to capture complex data manifolds.

---

## 3. The Big Idea: Change of Variable

To understand how Normalizing Flows resolve these limitations, let's start with a simple mathematical concept from high school algebra: the **change of variables**.

Suppose you are asked to solve this intimidating sixth-degree polynomial equation:
$$x^6 - 9x^3 + 8 = 0$$

Directly finding the roots of a sixth-degree equation is difficult. However, by changing our perspective (our variables), we can simplify the problem. Notice that we can rewrite the equation as:
$$(x^3)^2 - 9(x^3) + 8 = 0$$

By substituting $u = x^3$, the complex sixth-degree equation transforms into a simple quadratic equation:
$$u^2 - 9u + 8 = 0$$

We can easily solve this quadratic equation to find the roots for $u$:
$$(u-1)(u-8) = 0 \implies u = 1 \quad \text{and} \quad u = 8$$

Since we know the exact relationship between our complex variable ($x$) and our simple variable ($u$), we can map these solutions back to find the roots of $x$:
$$x^3 = 1 \implies x = 1$$
$$x^3 = 8 \implies x = 2$$

This simple algebraic trick represents the core philosophy of Normalizing Flows: **When faced with a highly complex mathematical space, transform it into a simpler, well-understood space, solve the problem there, and then map the solution back using the known transformation.**

---

## 4. Applying Change of Variables to Probability Distributions

In the context of generative modeling, our variables are not algebraic unknowns; they are **random variables** governed by probability distributions.

Imagine we have two distributions:
1. **The Sophisticated/Target Distribution ($q(x)$):** This represents our complex, real-world data (e.g., a highly intricate 2D spiral distribution, or the space of natural human faces). We can neither easily compute its exact density nor draw samples from it directly.
2. **The Simple/Base Distribution ($p(z)$):** This is a simple, tractable distribution (such as a 2D Standard Normal distribution). We can instantly draw samples from it and compute the exact density of any point within it.

```
      Sophisticated Space (x)                      Base Space (z)
   ┌───────────────────────────┐                ┌───────────────────┐
   │                           │    g = f⁻¹     │     ■   ■  ■      │
   │      ╭━━━━━━━╮            │ ─────────────> │   ■  ■■■■■  ■     │
   │      ┃ Spiral┃            │                │    ■  ■■■  ■      │
   │      ╰━━━━━━━╯            │ <────────━━━━  │      ■   ■        │
   │                           │       f        │                   │
   └───────────────────────────┘                └───────────────────┘
        Target: q(x) (Complex)                     Base: p(z) (Gaussian)
```

Normalizing Flows construct a bijective (one-to-one), invertible, and differentiable function $f$ that maps the simple base space to the complex target space:
$$x = f(z) \quad \text{and} \quad z = f^{-1}(x) = g(x)$$

### How this enables Sampling:
To generate a new complex sample $x$:
1. Sample $z$ from the simple base distribution (e.g., draw a sample from a Standard Gaussian).
2. Pass it through the forward transformation: $x = f(z)$.
3. Because $f$ is designed to map the high-probability regions of $p(z)$ to the high-probability regions of $q(x)$, the resulting $x$ will look like a valid sample from our target distribution.

### How this enables Density Estimation:
To compute the exact probability density of an observed data point $x$:
1. Pass $x$ through the inverse transformation to map it back to the base space: $z = f^{-1}(x)$.
2. Compute the probability density of $z$ under the simple base distribution $p(z)$.
3. Adjust this value to account for how the transformation $f$ stretched or squished the probability space.

---

## 5. The Mathematics of Normalizing Flows

Let’s derive the exact relationship between the density of our target space $q(x)$ and our base space $p(z)$.

### 5.1 The 1D (Univariate) Case
By the law of conservation of probability, the total probability mass in a tiny interval $dz$ in the base space must equal the total probability mass in the corresponding transformed interval $dx$ in the target space:
$$\int_{z}^{z+dz} p(z') dz' = \int_{x}^{x+dx} q(x') dx'$$

For infinitesimally small intervals, this simplifies to:
$$p(z)|dz| = q(x)|dx|$$

We use absolute values because probability density must always be positive, regardless of whether the transformation function is increasing or decreasing. Rearranging this equation to solve for our target density $q(x)$ yields:
$$q(x) = p(z) \left| \frac{dz}{dx} \right|$$

Substituting $z = f^{-1}(x)$:
$$q(x) = p(f^{-1}(x)) \left| \frac{d}{dx}f^{-1}(x) \right|$$

To prevent numerical underflow when multiplying very small probabilities in computer systems, we work in the log-space:
$$\log q(x) = \log p(f^{-1}(x)) + \log \left| \frac{d}{dx}f^{-1}(x) \right|$$

Alternatively, since $x = f(z)$, we can express this as:
$$\log q(x) = \log p(f^{-1}(x)) - \log \left| \frac{d}{dz}f(z) \right|$$

Here, the term $\left| \frac{dz}{dx} \right|$ acts as a **correction factor** that accounts for how the transformation stretches or squishes the local space.

---

### 🙋 Student Doubt: Why do we need the absolute value on the derivative?
> **Answer:** Suppose our transformation is $x = -2z$. If we sample $z$ from an interval $[0, 1]$, it maps to $[-2, 0]$ in $x$. The derivative $\frac{dx}{dz} = -2$, which is negative because the mapping flips the orientation of the space. However, probability density must remain positive. Taking the absolute value $|-2| = 2$ ensures our transformed density remains mathematically valid.

---

### 5.2 The Multi-Dimensional (Multivariate) Case
Real-world data is rarely one-dimensional. In multiple dimensions, our random variables $\mathbf{x}$ and $\mathbf{z}$ are vectors of dimension $D$. 

Instead of a simple scalar derivative, we must evaluate how our transformation changes volume in a multi-dimensional space. This change in volume is captured by the **Jacobian matrix** $J(\mathbf{z})$, which is the matrix of all first-order partial derivatives of the vector-valued function $f$:

$$J(f(\mathbf{z})) = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_D} \\
\vdots & \ddots & \cdots \\
\frac{\partial f_D}{\partial z_1} & \cdots & \frac{\partial f_D}{\partial z_D}
\end{bmatrix}$$

The **determinant of the Jacobian** ($\det J$) tells us exactly how much a tiny volume in the base space is scaled (stretched or squished) when mapped to the target space. 

Thus, the multivariate change of variables formula is:
$$q(\mathbf{x}) = p(f^{-1}(\mathbf{x})) \left| \det J(f^{-1}(\mathbf{x})) \right|$$

In log-space, this becomes:
$$\log q(\mathbf{x}) = \log p(f^{-1}(\mathbf{x})) + \log \left| \det J(f^{-1}(\mathbf{x})) \right|$$

---

## 6. Composing Mappings: Building the "Flow"

If we use a single, highly complex neural network as our transformation function $f$, we run into two major bottlenecks:
1. **Invertibility:** General deep neural networks are not easily invertible.
2. **Computational Complexity:** Computing the determinant of a large, arbitrary $D \times D$ Jacobian matrix has a worst-case computational complexity of $\mathcal{O}(D^3)$, which is prohibitively slow for high-dimensional data (e.g., an image with $D = 100,000$ pixels).

To solve this, we break our complex, monolithic transformation into a sequence of simpler, highly tractable, and invertible steps:

$$f = f_n \circ f_{n-1} \circ \dots \circ f_1$$

This sequence of transformations is what we call a **Normalizing Flow**.

```
    Base Space                                                    Target Space
     ╭━━━━━━╮      f₁      ╭━━━━━━╮      f₂      ╭━━━━━━╮      fn      ╭━━━━━━╮
     ┃  z₀  ┃ ───────────> ┃  z₁  ┃ ───────────> ┃  z₂  ┃ ───...━━━> ┃  zn  ┃ (x)
     ╰━━━━━━╯ <━━━━━━━━━━━ ╰━━━━━━╯ <━━━━━━━━━━━ ╰━━━━━━╯ <━━━━━━━━━━━ ╰━━━━━━╯
                   g₁            g₂                            gn
```

By composing functions this way:
1. **The forward mapping** is simple: $\mathbf{x} = \mathbf{z}_n = f_n(f_{n-1}(\dots(f_1(\mathbf{z}_0))\dots))$.
2. **The inverse mapping** remains mathematically clean: $g = f^{-1} = f_1^{-1} \circ f_2^{-1} \circ \dots \circ f_n^{-1}$.
3. **The Jacobian determinant** simplifies beautifully. Thanks to the chain rule, the Jacobian determinant of a composed function is simply the product of the Jacobian determinants of each individual step:
   $$\det J(f^{-1}(\mathbf{x})) = \prod_{i=1}^{n} \det J(g_i(\mathbf{z}_i))$$
   
   In log-space, this product turns into a highly efficient sum:
   $$\log q(\mathbf{x}) = \log p(\mathbf{z}_0) + \sum_{i=1}^{n} \log \left| \det J(g_i(\mathbf{z}_i)) \right|$$

By designing each individual step $f_i$ such that its Jacobian is easy to compute (for instance, a triangular matrix, where the determinant is simply the product of the diagonal elements), we can scale Normalizing Flows to incredibly high-dimensional datasets.

---

## 7. Learning and Optimization

How do we train a Normalizing Flow model to capture a target distribution? We use standard machine learning optimization techniques, depending on what we know about our target distribution.

### Scenario A: We have a dataset of samples, but we don't know the true distribution PDF.
This is the most common scenario (e.g., we have a dataset of face images). We want to find the parameters $\theta$ of our flow functions $f_\theta$ that maximize the likelihood of our observed data.
* **Loss Function:** We minimize the **Negative Log-Likelihood (NLL)** over our training dataset:
  $$\mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log p(\mathbf{z}_0) + \sum_{i=1}^{n} \log \left| \det J(g_{i, \theta}(\mathbf{z}_i)) \right| \right]$$
* **Method:** We split our data into mini-batches and use gradient descent (e.g., Adam optimizer) to update the parameters $\theta$.

### Scenario B: We can compute the target density, but we cannot easily sample from it.
Suppose we have a mathematically defined unnormalized target distribution (often the case in physics or chemistry). We want our flow model to learn to sample from it.
* **Loss Function:** We minimize the **Kullback-Leibler (KL) Divergence** between our flow's distribution and the target distribution.
* **Method:** We draw samples from our simple base distribution, pass them through the flow, evaluate their discrepancy from the target distribution, and update our parameters to close the gap.

---

## Summary of Famous Normalizing Flow Architectures

As you explore this field further, you will encounter several landmark architectures:

| Model | Primary Advantage | Primary Limitation | Common Use Case |
| :--- | :--- | :--- | :--- |
| **Inverse Autoregressive Flow (IAF)** | Incredibly fast sampling. | Slow density estimation (requires sequential step-by-step calculation). | Real-time speech synthesis, VAE latent spaces. |
| **Masked Autoregressive Flow (MAF)** | Incredibly fast density estimation. | Slow sampling (sequential generation). | Density estimation, anomaly detection. |
| **RealNVP / Glow** | Decent, parallelizable speeds for *both* sampling and density estimation. | Less expressive per layer; requires many composed steps. | High-quality image generation (e.g., human face synthesis). |

Normalizing Flows stand out as a mathematically rigorous and highly versatile tool in generative modeling, bridging the gap between exact probabilistic reasoning and deep representation learning.

# Lecture Notes: CS 480/680 Lecture 23 — Normalizing Flows
**Instructor:** Priyank Jaini (Guest Lecture for Pascal Poupart)  
**Date:** July 24, 2019  
**University of Waterloo**  

---

## 1. Introduction & The Core Problem: Density Estimation

In unsupervised learning, we are often tasked with understanding the underlying structure of unlabeled data [2]. The most fundamental formulation of this task is **density estimation** [2]. 

Given a dataset:
$$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\} \sim q(\mathbf{x})$$

Our goal is to construct an estimate of the true probability density function $q(\mathbf{x})$ from which these data points were generated [2].

### Why Density Estimation Matters
Having an explicit density function $q(\mathbf{x})$ is a crucial bottleneck step for several classical and modern machine learning tasks [2]:
* **Importance Sampling:** When calculating expectations under a complex target distribution, we use a proposal distribution [2]. If our proposal is not close to the target, the variance of our estimator can explode to infinity [2]. Accurate density estimation allows us to construct highly efficient proposal distributions.
* **Bayesian Inference:** Calculating the posterior distribution $p(\theta \mid \mathcal{D})$ is often intractable due to the normalizing constant (evidence) $p(\mathcal{D})$ [2]. Density estimation methods allow us to approximate these complex posteriors [2].
* **Generative Modeling (Image & Audio Synthesis):** Creating realistic samples (e.g., PixelRNN, WaveNet, or Glow) relies on accurately capturing the data distribution [2].
* **Downstream Applications:** It is also used in network routing, sleep-stage classification using neurological data, and human activity recognition [2].

---

## 2. Comparing Generative Models: VAEs, GANs, and Normalizing Flows

Before diving into Normalizing Flows, let's establish how they differ from the generative architectures you have already covered in this course: Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) [4].

```
           [ GANs & VAEs ]                               [ Normalizing Flows ]
   ---------------------------------              ---------------------------------
   - Implicit density models                      - Explicit density models
   - Learn to generate samples                    - Learn the exact likelihood
   - Latent space can be lower-dim                - Latent space must match data-dim
```

### Generative Adversarial Networks (GANs)
* **Mechanism:** You sample noise $\mathbf{z}$ from a simple prior $p(\mathbf{z})$ (usually a standard normal) and pass it through a Generator network $G_\phi$ to produce synthetic data $\hat{q}_{\text{synth}}$ [4]. A Discriminator $D_\theta$ is trained in a zero-sum game to distinguish real data from synthetic data [4].
* **Limitation:** GANs are **implicit density models** [4]. While they excel at generating highly sharp samples, they do not provide a mathematical formula to evaluate the exact probability density $q(\mathbf{x})$ of a given data point [4].

### Variational Autoencoders (VAEs)
* **Mechanism:** An Encoder network $f_\theta$ maps a data point $\mathbf{x}$ to a distribution over a latent space $\mathbf{z}$ [4]. A Decoder $g_\phi$ then reconstructs the data point from a sample $\mathbf{z} \sim p(\mathbf{z})$ [4]. The network is trained by maximizing the Evidence Lower Bound (ELBO), which balances reconstruction error and the KL-divergence of the latent distribution [4].
* **Limitation:** Like GANs, VAEs do not give us an **explicit, tractable representation** of the density function $q(\mathbf{x})$ [4]. We can approximate it, but we cannot evaluate the exact likelihood of a test point directly.

### The Normalizing Flows Agenda
Normalizing Flows aim to achieve **explicit representation of density functions** [5]. We want to find a *deterministic, invertible, and differentiable* mathematical mapping $T$ that transforms a simple, known source distribution $p(\mathbf{z})$ into a highly complex target distribution $q(\mathbf{x})$ [5, 8]. Because the mapping is invertible, we can compute the exact probability density of any point in the data space [8].

---

## 3. The Mathematics of Flow: Change of Variables

The entire framework of Normalizing Flows is built upon the fundamental probabilistic principle of the **change of variables** [7].

### 3.1 The Univariate Case
Let's build intuition using a simple 1D example [6]. Suppose we have a uniform random variable $z \sim \text{Uniform}(0, 1)$ with density $p(z) = 1$ [6]. We apply a deterministic transformation:
$$x = T(z) = 3z + 1$$

This maps the interval $[0, 1]$ to the interval $[1, 4]$ [6]. Because probability mass must be conserved (i.e., the total area under the probability density curve must equal 1), stretching the support of the distribution from a length of $1$ to a length of $3$ forces the height of the density function to scale down by a factor of $1/3$ [6]. Thus, the target density is $q(x) = 1/3$ [6].

Mathematically, the conservation of probability mass over an infinitesimal region requires [6, 7]:
$$p(z) \, |dz| = q(x) \, |dx|$$

Rewriting this in terms of $q(x)$ gives us the univariate change of variables formula [6, 7]:
$$q(x) = p(z) \left| \frac{dz}{dx} \right| = p(z) \left| \frac{dT(z)}{dz} \right|^{-1}$$

where $z = T^{-1}(x)$ [7, 8].

---

### 🙋 Student Doubt 1: Why do we have the absolute value sign?
**Lecturer's Response:**  
The absolute value sign is necessary because probability densities must always be non-negative. If we had a decreasing monotonic transformation, such as $x = -3z + 1$, the derivative $\frac{dT(z)}{dz}$ would be $-3$. The absolute value ensures that the scaling factor remains positive ($|-3|^{-1} = 1/3$), preserving a valid, positive probability density.

---

### 3.2 The Multivariate Case
When working in higher dimensions ($\mathbf{z}, \mathbf{x} \in \mathbb{R}^d$), the scaling of the infinitesimal volume is determined by the determinant of the Jacobian matrix of the transformation $T$ [7].

Let $T: \mathbb{R}^d \to \mathbb{R}^d$ be a bijective, differentiable mapping [7]. The multivariate change of variables formula is [7]:
$$q(\mathbf{x}) = p(\mathbf{z}) \left| \det \nabla_{\mathbf{z}} T(\mathbf{z}) \right|^{-1}$$

where $\mathbf{z} = T^{-1}(\mathbf{x})$ [7, 8].

To train a model using this formula, we maximize the log-likelihood of our dataset $\mathcal{D}$ [8]. The objective function becomes [8]:
$$\hat{T} = \arg\max_T \sum_{i=1}^n \left( \log p(T^{-1}(\mathbf{x}_i)) - \log \left| \det \nabla_{\mathbf{z}} T(\mathbf{z}_i) \right| \right)$$

This reveals two massive computational bottlenecks [8]:
1. **The Inverse Bottleneck:** We must be able to compute $T^{-1}(\mathbf{x}_i)$ efficiently to find the corresponding latent points $\mathbf{z}_i$ [8].
2. **The Determinant Bottleneck:** Computing the determinant of an arbitrary $d \times d$ Jacobian matrix $\nabla_{\mathbf{z}} T$ is incredibly expensive, scaling cubically with the dimension: $\mathcal{O}(d^3)$ [8]. For high-dimensional data (like a $256 \times 256$ pixel image, where $d \approx 196,000$), an $\mathcal{O}(d^3)$ operation is completely intractable.

---

## 4. The Breakthrough Solution: Increasing Triangular Maps

To solve the determinant bottleneck, we restrict the transformation $T$ to the family of **increasing triangular maps** [9].

Let $T: \mathbb{R}^d \to \mathbb{R}^d$ be structured such that each output component $x_j$ depends only on the latent variables up to index $j$ [9]:
$$x_1 = T_1(z_1)$$
$$x_2 = T_2(z_1, z_2)$$
$$x_3 = T_3(z_1, z_2, z_3)$$
$$\dots$$
$$x_d = T_d(z_1, z_2, \dots, z_d)$$

Because of this specific dependency structure, the Jacobian matrix $\nabla_{\mathbf{z}} T$ becomes lower triangular [9]:
$$\nabla_{\mathbf{z}} T = \begin{bmatrix} 
\frac{\partial T_1}{\partial z_1} & 0 & \dots & 0 \\ 
\frac{\partial T_2}{\partial z_1} & \frac{\partial T_2}{\partial z_2} & \dots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial T_d}{\partial z_1} & \frac{\partial T_d}{\partial z_2} & \dots & \frac{\partial T_d}{\partial zd} 
\end{bmatrix}$$

The determinant of any triangular matrix is simply the product of its diagonal entries [10]:
$$\det \nabla_{\mathbf{z}} T = \prod_{j=1}^d \frac{\partial T_j}{\partial z_j}$$

By using a triangular map, we reduce the complexity of the determinant computation from $\mathcal{O}(d^3)$ to **$\mathcal{O}(d)$** [10]!

* **Increasing Condition:** To guarantee that the map is invertible and monotonic, we require each diagonal element of the Jacobian to be strictly positive: $\frac{\partial T_j}{\partial z_j} > 0$ [9].

---

### 🙋 Student Doubt 2: Does $T$ have to be bijective/invertible?
**Lecturer's Response:**  
Absolutely. For the change of variables formula to hold, the mapping $T$ must be a bijection (both one-to-one and onto). If it weren't invertible, we wouldn't be able to map a data point $\mathbf{x}$ back to its unique latent representation $\mathbf{z} = T^{-1}(\mathbf{x})$, which is required to evaluate the density $p(\mathbf{z})$ [8]. The "increasing" condition ($\frac{\partial T_j}{\partial z_j} > 0$) guarantees this bijectivity [9].

---

### The Key Theorem
> **Theorem (Bogachev et al., 2005):** There always exists a unique *increasing triangular map* that transforms any smooth source density $p(\mathbf{z})$ into any smooth target density $q(\mathbf{x})$ (for a fixed ordering of variables) [9].

This theorem ensures that our mathematical restriction to triangular maps does not limit our model's representational capacity; we can still theoretically represent any target distribution [9].

---

## 5. Classic Normalizing Flow Architectures

Let's explore how researchers have parameterized these triangular maps over the years [11].

```
                     [ Normalizing Flow Timeline ]
 NADE/MADE (2011/15) ---> IAF & MAF (2016/17) ---> Real-NVP (2017) ---> NAFs (2018) ---> SOS Flows (2019)
```

### 5.1 Autoregressive Models with Gaussian Conditionals
Recall that any joint density can be decomposed using the chain rule of probability [12]:
$$q(\mathbf{x}) = \prod_{j=1}^d q_j(x_j \mid \mathbf{x}_{<j})$$

If we assume the conditional distributions are Gaussian, we write [13]:
$$q_j(x_j \mid \mathbf{x}_{<j}) = \mathcal{N}\left(x_j; \, \mu_j(\mathbf{x}_{<j}), \, \sigma_j^2(\mathbf{x}_{<j})\right)$$

We can generate this distribution by applying an affine transformation to a standard normal noise variable $z_j \sim \mathcal{N}(0, 1)$ [13]:
$$x_j = \sigma_j(\mathbf{x}_{<j}) \cdot z_j + \mu_j(\mathbf{x}_{<j})$$

This is an increasing triangular map [13]:
* $x_1 = \sigma_1 \cdot z_1 + \mu_1$ [13]
* $x_2 = \sigma_2(x_1) \cdot z_2 + \mu_2(x_1)$ [13]
* $x_d = \sigma_d(\mathbf{x}_{<d}) \cdot z_d + \mu_d(\mathbf{x}_{<d})$ [13]

Here, $\mu_j$ and $\sigma_j$ are parameterized by deep neural networks that take previous variables as inputs [13].

---

### 5.2 Masked Autoregressive Flows (MAFs)
To build highly expressive models, we stack multiple simple autoregressive layers [14]:
$$\mathbf{z} \longrightarrow \boxed{\text{AR}^{(1)}} \longrightarrow \mathbf{z}_1 \longrightarrow \boxed{P} \longrightarrow \boxed{\text{AR}^{(2)}} \longrightarrow \mathbf{z}_2 \longrightarrow \dots \longrightarrow \mathbf{x}$$

where $P$ is a permutation matrix [14].

---

### 🙋 Student Doubt 3: What does the permutation matrix $P$ do? Why do we need it?
**Lecturer's Response:**  
When we define a single triangular map, we enforce a strict, rigid hierarchy of variable dependencies (e.g., $x_1$ depends on nothing, $x_2$ only on $z_1$, and so on) [9]. In real-world data, the actual causal dependencies are rarely aligned with this arbitrary ordering. 

By inserting a permutation matrix $P$ between the stacked autoregressive layers, we shuffle the variables [14]. This allows subsequent layers to learn different dependency orderings, breaking the rigid structure of a single layer and allowing the joint network to capture much more complex, high-dimensional distributions [14]. Since the determinant of a permutation matrix is always $\pm 1$, its log-determinant is $0$, meaning it adds zero computational cost to our Jacobian calculation [14].

---

### 5.3 Real-NVP (Real-valued Non-Volume Preserving Flows)
Instead of fully autoregressive transformations (where every step depends on all prior variables), Real-NVP splits the latent vector into two halves [15]:
$$\mathbf{z}_1 = [z_1, \dots, z_{l-1}] \quad \text{and} \quad \mathbf{z}_2 = [z_l, \dots, z_d]$$

The transformation is defined as [15]:
$$\mathbf{x}_1 = \mathbf{z}_1$$
$$\mathbf{x}_2 = \mathbf{z}_2 \odot \exp\left(\alpha(\mathbf{z}_1)\right) + \mu(\mathbf{z}_1)$$

Since $\mathbf{x}_1$ is identical to $\mathbf{z}_1$, the Jacobian of this transformation has a block-triangular structure [15]:
$$J = \begin{bmatrix} 
I & 0 \\ 
\frac{\partial \mathbf{x}_2}{\partial \mathbf{z}_1} & \text{diag}(\exp(\alpha(\mathbf{z}_1))) 
\end{bmatrix}$$

The determinant is simply the product of the diagonal elements of the bottom-right block, which is incredibly cheap to compute: $\prod \exp(\alpha(\mathbf{z}_1))$.

---

### 5.4 Neural Autoregressive Flows (NAFs)
What if we replace the simple affine (linear) transformation in our autoregressive step with a non-linear one [16]? NAFs pass each $z_j$ through a dedicated deep neural network (DNN) [16]:
$$x_j = \text{DNN}(z_j; \, \mathbf{w}_j(\mathbf{z}_{<j}))$$

To ensure that the DNN defines a valid, invertible, and increasing triangular map, we must guarantee that its derivative with respect to $z_j$ is strictly positive [16]. NAFs achieve this by [16]:
1. Enforcing **strictly positive weights** across the network [16].
2. Using **strictly monotonic activation functions** (such as Sigmoid or Tanh) [16].

While NAFs are universal approximators of triangular maps, constraining neural network weights to be strictly positive makes them highly difficult to train in practice [16].

---

## 6. Sum-of-Squares (SOS) Polynomial Flows

To address the training difficulties of NAFs, Priyank Jaini and his supervisor Yaoliang Yu introduced **Sum-of-Squares (SOS) Polynomial Flows** (ICML 2019) [17].

Instead of using constrained neural networks, they utilized polynomials to define the transformations [17].

### The Core Idea
An increasing function must have a strictly positive derivative: $\frac{\partial T_j}{\partial z_j} > 0$ [9]. From algebra, we know that any non-negative polynomial can be represented as (or approximated by) a **sum of squares of polynomials** [17].

Thus, we can define the derivative of our transformation as a sum of squares of polynomials [17]:
$$\frac{\partial T_j}{\partial z_j} = \sum_{k=1}^K \mathcal{P}_{r, k}^2(z_j)$$

where $\mathcal{P}_{r, k}(z_j) = \sum_{l=0}^r a_{l, k} z_j^l$ is a polynomial of degree $r$ [17].

Because the derivative is a sum of squared terms, it is guaranteed to be non-negative for any real-valued coefficients $a_{l, k}$ [17]. We obtain the transformation $T_j$ simply by integrating this expression [17]:
$$x_j = T_j(z_j) = \int \left( \sum_{k=1}^K \mathcal{P}_{r, k}^2(z_j) \right) dz_j + C$$

### Key Benefits of SOS Flows
1. **Unconstrained Optimization:** Unlike NAFs, there are **no constraints on the coefficients** $a_{l, k}$ [18]. We can train them using standard backpropagation and unconstrained optimization algorithms [18].
2. **Interpretability:** The coefficients of the polynomials directly control the higher-order statistical moments of the target distribution (such as skewness and kurtosis) [18].
3. **Universality:** It is proven to be a universal representational framework capable of approximating any target density [18].

---

## 7. Applications: Invertible 1x1 Convolutions & Glow

One of the most visually stunning applications of Normalizing Flows is **Glow** (Kingma & Dhariwal, NeurIPS 2018), which introduced **invertible 1x1 convolutions** [19].

```
       [ Glow Step of Flow ]
 z ---> [ Actnorm ] ---> [ Invertible 1x1 Conv ] ---> [ Affine Coupling Layer ] ---> x
```

* **Invertible 1x1 Convolution:** A 1x1 convolution over an image tensor with $C$ channels is equivalent to multiplying the channel vector at each pixel by a $C \times C$ weight matrix $W$ [19].
* **LU Decomposition:** To make computing the determinant of $W$ cheap, Glow parameterizes $W$ using LU decomposition ($W = P L U$), where $L$ is lower triangular with ones on the diagonal, and $U$ is upper triangular [19]. This allows us to compute the determinant of the convolution step in $\mathcal{O}(C)$ time [19].

---

### 🙋 Student Doubt 4: How do latent space operations like image interpolation work?
**Lecturer's Response:**  
Let's write this out on the blackboard to see the exact mathematics behind it.

```
 [ Image Space X ]                                   [ Latent Space Z ]
   Real Face (x)   --- (Forward Flow T^-1) --->        Latent Vector (z)
                                                              |
                                                    + alpha * v (Attribute Vector)
                                                              |
   Modified Face   <--- (Inverse Flow T) ------       New Latent Vector (z')
```

1. **Defining the Attribute Vector:**  
   First, we label our dataset based on the attribute we want to manipulate. For example, let's say we want to make faces look "older" [3]:
   * Let $X_{\text{old}} = \{\mathbf{x}_1, \dots, \mathbf{x}_m\}$ be images of old people.
   * Let $X_{\text{young}} = \{\mathbf{y}_1, \dots, \mathbf{y}_k\}$ be images of young people.

2. **Mapping to the Latent Space:**  
   Using our trained invertible flow $T^{-1}$, we map all these images into their latent representations:
   $$\mathbf{z}_i^{\text{old}} = T^{-1}(\mathbf{x}_i) \quad \text{and} \quad \mathbf{z}_j^{\text{young}} = T^{-1}(\mathbf{y}_j)$$

3. **Computing the Attribute Direction:**  
   We calculate the average latent vector for both groups:
   $$\bar{\mathbf{z}}_{\text{old}} = \frac{1}{m} \sum_{i=1}^m \mathbf{z}_i^{\text{old}} \quad \text{and} \quad \bar{\mathbf{z}}_{\text{young}} = \frac{1}{k} \sum_{j=1}^k \mathbf{z}_j^{\text{young}}$$
   
   The difference vector $\mathbf{v} = \bar{\mathbf{z}}_{\text{old}} - \bar{\mathbf{z}}_{\text{young}}$ represents the **"aging direction"** in our latent space.

4. **Manipulating a New Image:**  
   Given a new image of a young person $\mathbf{x}_{\text{test}}$, we map it to the latent space:
   $$\mathbf{z}_{\text{test}} = T^{-1}(\mathbf{x}_{\text{test}})$$
   
   We shift this latent vector along our aging direction:
   $$\mathbf{z}_{\text{new}} = \mathbf{z}_{\text{test}} + \alpha \mathbf{v}$$
   
   Finally, we map it back to the image space using the inverse mapping $T$:
   $$\mathbf{x}_{\text{modified}} = T(\mathbf{z}_{\text{new}})$$
   
   This yields a synthetic image of the same person, modified to look older [3].

---

### 🙋 Student Doubt 5: What guarantees that this operation preserves the person's identity instead of mapping them to a completely different person?
**Lecturer's Response:**  
This is a great question. In a well-trained normalizing flow, the latent space $\mathbf{z}$ is structured such that different factors of variation are disentangled into orthogonal directions. The vector $\mathbf{z}_{\text{test}}$ encodes all the features of the person, including their identity, gender, and hair color. 

By applying a localized shift $+ \alpha \mathbf{v}$ along the aging direction, we only perturb the specific latent dimensions that encode age-related features. Because the shift is local and orthogonal to the identity-encoding dimensions, the person's unique identity is preserved when we project back to the image space using the generator mapping $T$.

---

## 8. The Catch: Drawbacks of Normalizing Flows

Normalizing Flows provide explicit, tractable density estimation and produce highly sharp samples [19]. However, this comes with a major architectural drawback:

> **The Dimensionality Constraint:** The latent space $\mathbf{z}$ must have the exact same dimensionality as the data space $\mathbf{x}$ ($\mathbf{z}, \mathbf{x} \in \mathbb{R}^d$) [7].

Unlike VAEs or GANs—which can compress a high-dimensional image (e.g., $256 \times 256 \times 3 \approx 196,000$ dimensions) into a tiny, low-dimensional bottleneck latent space (e.g., $128$ dimensions)—**Normalizing Flows cannot compress data**. 

If your input is a high-resolution image, your latent space must also be 196,000-dimensional. Managing, storing, and training networks with such high-dimensional latent spaces requires an immense amount of parameters and memory, making Normalizing Flows computationally heavy to deploy for very large-scale datasets.


# Masterclass Lecture Notes: Generative Modeling, Normalizing Flows, and GANs

This set of detailed notes covers the mathematical, architectural, and conceptual foundations of generative models. It is designed for those encountering these topics for the first time, starting from basic probability density modeling up to advanced deep architectures like Real-NVP, Glow, StyleGAN, and Pix2Pix.

---

## Part 1: The Foundations of Generative Modeling

### 1.1 What is Generative Modeling?
At its core, generative modeling is about learning the probability distribution of a given dataset [1, 2]. Suppose we have a set of data samples (such as images of faces, audio waves, or medical scans) [2]:
$$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\} \sim p(\mathbf{x})$$

We assume all these samples are drawn from an underlying, unknown probability distribution $p(\mathbf{x})$ [2]. Our goals are:
1. **Density Estimation:** Construct an approximate model $q(\mathbf{x})$ that behaves like the true distribution $p(\mathbf{x})$ [2]. This allows us to evaluate how likely any arbitrary input $\mathbf{x}$ is under our model [8, 13].
2. **Sampling:** Generate completely new, realistic data points $\hat{\mathbf{x}}$ by sampling from our learned model $q(\mathbf{x})$ [2].

### 1.2 Why is this "Self-Supervised"?
Generative modeling is classified as **self-supervised** (or unsupervised) learning [3]:
* **No Manual Labels:** The training process requires only raw data $\mathbf{x}$ [3]. We do not need annotators to label images as "car," "cat," or "dog" [3].
* **Hidden Structure:** The model must discover the underlying, latent factors of variation (the hidden structure) to represent the data [3].
* **Feature Transfer:** The representations learned by solving these generative tasks can be transferred to downstream tasks like classification, regression, or anomaly detection [3].

---

### 1.3 A Map of the Generative Landscape (Taxonomy)
Different families of generative models strike different trade-offs between sample quality, exact likelihood computation, and training stability [43]:

```
                                     [ Generative Models ]
                                               |
                     -----------------------------------------------------
                    |                                                     |
            [ Explicit Density ]                                  [ Implicit Density ]
                    |                                                     |
         -------------------------                             -------------------------
        |                         |                           |                         |
   [ Tractable ]           [ Approximate ]                [ Direct ]               [ Indirect ]
     - Autoregressive        - Variational Autoencoder      - GANs                   - Diffusion Models
     - Normalizing Flows       (VAE)
```

1. **Explicit Density Models:** These models explicitly define a probability density function $q(\mathbf{x})$ [43].
   * **Tractable Density:** We can compute the exact probability $q(\mathbf{x})$ of any data point in a single forward pass [43]. Examples include **Autoregressive Models** and **Normalizing Flows** [43].
   * **Approximate Density:** The exact density is mathematically intractable, but we can optimize a lower bound (ELBO) or a variational approximation [4, 43]. Examples include **Variational Autoencoders (VAEs)** [43].
2. **Implicit Density Models:** These models do not define an explicit probability density function [43]. Instead, they focus purely on sampling [43].
   * **Direct Sampling:** We can generate samples directly in a single step (e.g., **Generative Adversarial Networks (GANs)**) [43].
   * **Indirect/Iterative Sampling:** Samples are generated via an iterative denoising process (e.g., **Diffusion Models**) [43].

---

## Part 2: Non-Parametric Density Estimation (Kernel Density Estimation)

To understand deep parametric models, we must first look at classical, non-parametric approaches [2, 5]. The most famous of these is **Kernel Density Estimation (KDE)** [4, 5].

### 2.1 The Intuition of KDE
Imagine dropping a single grain of sand at every data point $\mathbf{x}_i$ in our training set [6]. Where points are clustered tightly, the sand piles up [6]. If we smooth out these piles of sand using a bell-shaped curve (a kernel), the height of the sand pile at any coordinate $x$ represents the probability density [6].

$$\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$

Let's break down every variable in this formula [6]:
* **$\hat{f}_h(x)$**: The estimated probability density at a query point $x$ [6].
* **$n$**: The total number of training samples [6].
* **$x_i$**: The $i$-th training sample from our dataset [6].
* **$K(u)$**: The **Kernel function** [6]. It is a symmetric, non-negative function that integrates to 1 (making $\hat{f}_h(x)$ a valid probability density).
* **$h$**: The **bandwidth** [6]. It controls the width of each kernel, acting as a smoothing parameter [6].

### 2.2 Common Kernels
Different kernel shapes can be used to smooth the data points [6]:

```
       [ Gaussian ]              [ Tophat / Box ]             [ Epanechnikov ]
          _---_                       _____                         _-_
        /       \                    |     |                       /   \
       /         \                   |     |                      /     \
  ____/           \____         _____|_____|_____            ____/       \____
```

When using a **Gaussian Kernel**, the estimator becomes [7]:
$$\hat{f}_h(x) = \frac{1}{n} \sum_{i=1}^n \frac{1}{h\sqrt{2\pi}} \exp\left( \frac{-(x-x_i)^2}{2h^2} \right)$$

This places a small Gaussian distribution centered at every training data point $x_i$ and averages them [7].

---

### 💡 Concept Spotlight: The Impact of Bandwidth ($h$)
The choice of bandwidth $h$ governs the bias-variance trade-off in non-parametric density estimation:
* **If $h$ is too small (Under-smoothing):** The density estimate becomes a collection of highly isolated, sharp spikes centered exactly at the training points. This results in **overfitting**; the model fails to generalize to the space between points.
* **If $h$ is too large (Over-smoothing):** The kernels become extremely wide and flat. This results in **underfitting**.

---

### 📝 Quiz Question 1: KDE and Generalization
**Scenario:** You have observed the following 1D training samples: $\{2.0, 2.1, 2.2, 8.0, 8.1, 8.2\}$. You fit a KDE with a **very large** kernel bandwidth $h$. What is the likely consequence?

* **Correct Answer:** **The density will become smoother and may place a high probability between the two groups.**
* **Explanation:** A very large bandwidth makes each individual Gaussian kernel extremely wide. As a result, the density estimates from the distinct cluster around $2.0$ and the cluster around $8.0$ will heavily overlap in the middle. The model will assign a high probability density to the region around $5.0$, where no training data actually exists.

---

### 2.3 Sampling from a KDE
How do we generate new data points from a KDE? The process is direct:
1. Randomly select a data point $x_i$ from the training set.
2. Sample noise $\epsilon$ from the distribution defined by the kernel $K$ (e.g., if using a Gaussian kernel, sample $\epsilon \sim \mathcal{N}(0, h^2)$).
3. Output the new sample: $\hat{x} = x_i + \epsilon$.

### 2.4 Why KDE Fails in High Dimensions (The Curse of Dimensionality)
KDE is highly effective in 1D, 2D, or 3D spaces (such as mapping animal habitats on a geographical map) [4, 5]. However, it is entirely impractical for high-dimensional data, such as images [2, 4].

---

### 📝 Quiz Question 2: KDE as a Generative Model
**Scenario:** Suppose you want to generate realistic human face images. Which is the strongest argument against using KDE directly?

* **Correct Answer:** **KDE scales poorly in very high-dimensional spaces.**
* **Explanation:** Face images are extremely high-dimensional (e.g., $256 \times 256$ pixels $\approx 65,536$ dimensions). Due to the **curse of dimensionality**, the volume of a space scales exponentially with its dimension. In such massive spaces, training points become incredibly sparse, requiring an astronomically large dataset to estimate a meaningful density or generate coherent images.

---

## Part 3: From Representation to Generation: Autoencoders

An Autoencoder (AE) is a neural network designed to learn a compressed representation of data in an unsupervised manner [10].

```
   [ Input Space X ]            [ Latent Space Z ]           [ Output Space X' ]
    Real Image (x)  --> [ Encoder ] -->  z  --> [ Decoder ] --> Reconstructed Image (x')
```

* **Encoder ($f_\theta$):** Maps a high-dimensional input $\mathbf{x}$ to a low-dimensional latent code $\mathbf{z}$ [10].
* **Decoder ($g_\phi$):** Maps the latent code $\mathbf{z}$ back to the original space to reconstruct $\mathbf{x}' \approx \mathbf{x}$ [10].

---

### 📝 Quiz Question 3: Autoencoder vs. Flow
**Scenario:** Suppose an autoencoder reconstructs images perfectly. Can it automatically be used as a good generative model?

* **Correct Answer:** **No, because reconstruction quality does not guarantee that latent codes follow a simple distribution.**
* **Explanation:** Perfect reconstruction simply means the decoder can invert the encoder's mapping for the training data. However, a standard autoencoder does not constrain the distribution of the latent space $\mathbf{z}$. The latent codes might be distributed in a highly irregular, fractured, and sparse manner. If you attempt to generate a new image by sampling a random latent vector $\mathbf{z} \sim \mathcal{N}(0, I)$, the sampled vector will likely land in an "empty" region of the latent space that the decoder was never trained on, resulting in meaningless, garbled outputs.

---

## Part 4: Normalizing Flows (NFs)

To solve the limitations of standard autoencoders, GANs, and VAEs, we turn to **Normalizing Flows** [9, 10]. They are designed to do three things simultaneously:
1. Generate realistic, high-quality samples [24].
2. Compute the **exact likelihood** $p(\mathbf{x})$ of any data point [24].
3. Learn a **meaningful, invertible latent space** [24].

```
                   g (Forward / Generation)
   [ Latent Space Z ]  ========================>  [ Data Space X ]
   Simple Prior p(z)   <========================  Complex Density p(x)
                     f (Inverse / Normalizing)
```

### 4.1 The Core Idea: Bijective Networks
Instead of compressing the data into a lower-dimensional bottleneck, a Normalizing Flow uses a **bijective network** [9]. A bijective function $g: \mathcal{Z} \to \mathcal{X}$ is:
* **Invertible:** For every $\mathbf{x}$, there is a unique, computable $\mathbf{z} = g^{-1}(\mathbf{x})$ [10, 11].
* **Differentiable:** Both $g$ and its inverse $f = g^{-1}$ are differentiable [10, 11].
* **Dimension-preserving:** The latent space $\mathbf{z}$ has the **exact same dimensionality** as the data space $\mathbf{x}$ [10].

### 4.2 The Mathematics of Flow: Change of Variables
Normalizing Flows use the multivariate change of variables formula to map a simple prior distribution $p_z(\mathbf{z})$ (such as an isotropic Gaussian) to a complex data distribution $p_x(\mathbf{x})$ [9, 11]:

$$p_x(\mathbf{x}) = p_z(g^{-1}(\mathbf{x})) \cdot \left| \det \left( \frac{\partial g^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right) \right|$$

Let's write this in the more common notation where $f(\mathbf{x}) = \mathbf{z} = g^{-1}(\mathbf{x})$ is the "normalizing" direction [10, 11]:
$$p_x(\mathbf{x}) = p_z(f(\mathbf{x})) \cdot \left| \det J_f(\mathbf{x}) \right|$$

Taking the logarithm gives us the log-likelihood objective function [11]:
$$\log p_x(\mathbf{x}) = \log p_z(f(\mathbf{x})) + \log \left| \det J_f(\mathbf{x}) \right|$$

Let's unpack every component of this equation [11]:
* **$\log p_x(\mathbf{x})$**: The exact log-probability of our data point $\mathbf{x}$ under the model [11].
* **$\log p_z(f(\mathbf{x}))$**: The log-probability of the normalized latent vector $\mathbf{z} = f(\mathbf{x})$ under our simple Gaussian prior [11]. This term penalizes latent codes that stray too far from the origin.
* **$J_f(\mathbf{x}) = \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}$**: The **Jacobian matrix** of the transformation $f$ [11]. It contains all first-order partial derivatives, capturing how the transformation distorts space [11].
* **$\log \left| \det J_f(\mathbf{x}) \right|$**: The log-determinant of the Jacobian [11]. It measures how much the local volume expands or contracts [11]. Intuitively, it ensures that the total probability integrates to 1 by accounting for the warping of space.

### 4.3 Composing Multiple Layers
A single simple transformation cannot map a Gaussian to a complex image distribution. However, we can compose many invertible, bijective layers together [12]:
$$\mathbf{x} \xrightarrow{\quad f_T \quad } \mathbf{z}_{T-1} \xrightarrow{\quad f_{T-1} \quad } \dots \xrightarrow{\quad f_1 \quad } \mathbf{z}_0 = \mathbf{z} \sim p_z(\mathbf{z})$$

The overall transformation is $f = f_1 \circ f_2 \circ \dots \circ f_T$ [12]. By the chain rule, the log-determinant of the composed transformation is simply the sum of the log-determinants of the individual layers [12]:
$$\log p_x(\mathbf{x}) = \log p_z(\mathbf{z}_0) + \sum_{t=1}^T \log \left| \det J_{f_t}(\mathbf{z}_t) \right|$$

This composition allows us to warp a simple distribution step-by-step into an incredibly complex, multi-modal shape [12].

---

### 📝 Quiz Question 4: Generative Model Family
**Scenario:** Suppose you have a model that can generate high-quality images, but you cannot compute the probability of an image under the model. Which generative family was largely designed to solve this issue?

* **Correct Answer:** **Normalising Flows**
* **Explanation:** GANs can generate realistic images but do not provide an explicit probability density function. Normalizing Flows were designed specifically to allow exact, tractable likelihood computation while remaining generative models.

---

## Part 5: Real-NVP and Affine Coupling Layers

Calculating the determinant of an arbitrary Jacobian matrix scales cubically with the dimensions: $\mathcal{O}(d^3)$ [8, 19]. To make this calculation efficient, **Real-NVP** (Dinh et al., 2017) introduced **Affine Coupling Layers** [14, 15].

```
               [ Affine Coupling Step ]
               
             /--->  x_{1:d}  --------==========-------->  y_{1:d} = x_{1:d}
            /                 |         |
   Input x                     v         v
            \              [ s(.) ]  [ t(.) ]
             \                |         |
              \---> x_{d+1:D} ---> (x) ---- (+) -------> y_{d+1:D} = x_{d+1:D} * e^{s} + t
```

### 5.1 The Forward Propagation
An affine coupling layer splits the input dimensions into two halves [15]:
1. The first $d$ dimensions are passed through unchanged [15]:
   $$\mathbf{y}_{1:d} = \mathbf{x}_{1:d}$$
2. The remaining dimensions (from $d+1$ to $D$) undergo an affine transformation (scaling and shifting) based on the first half [15]:
   $$\mathbf{y}_{d+1:D} = \mathbf{x}_{d+1:D} \odot \exp\left(s(\mathbf{x}_{1:d})\right) + t(\mathbf{x}_{1:d})$$

Here, $s$ and $t$ are scale and translation functions, which are parameterized by arbitrary neural networks (e.g., standard residual networks) [15].

### 5.2 The Inverse Propagation (Sampling)
During generation, we need to map from $\mathbf{y}$ back to $\mathbf{x}$ [15]. What makes affine coupling layers highly effective is that **the neural networks $s$ and $t$ do not need to be invertible** [15]. We can invert the entire layer analytically [15]:

1. Because the first part was passed through unchanged, we recover it directly [15]:
   $$\mathbf{x}_{1:d} = \mathbf{y}_{1:d}$$
2. We can then pass this recovered $\mathbf{x}_{1:d}$ through the exact same networks $s$ and $t$ to invert the second half [15]:
   $$\mathbf{x}_{d+1:D} = \left(\mathbf{y}_{d+1:D} - t(\mathbf{y}_{1:d})\right) \odot \exp\left(-s(\mathbf{y}_{1:d})\right)$$

This allows us to use highly complex, non-invertible neural networks inside our flow architecture while maintaining exact, analytic invertibility for the overall system [15].

### 5.3 Why the Jacobian is Incredibly Cheap
Let's look at the Jacobian matrix for this transformation [15]:

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}^T} = \begin{bmatrix} 
I_d & 0 \\ 
\frac{\partial \mathbf{y}_{d+1:D}}{\partial \mathbf{x}_{1:d}} & \text{diag}\left(\exp\left(s(\mathbf{x}_{1:d})\right)\right) 
\end{bmatrix}$$

Because the top-right block is $0$ (the first half $\mathbf{y}_{1:d}$ has zero dependency on the second half $\mathbf{x}_{d+1:D}$), this matrix is **block lower-triangular** [15]. 

The determinant of a block triangular matrix is simply the product of the diagonal blocks' determinants [15]. Since the top-left block is the identity matrix $I_d$ (determinant of 1) and the bottom-right block is a diagonal matrix, its determinant is trivial to compute [15]:

$$\det J = \prod_{j=d+1}^D \exp\left(s(\mathbf{x}_{1:d})\right)_j$$

Taking the log-determinant reduces to a simple sum over our scale network's output [15]:
$$\log \left| \det J \right| = \sum_{j} s(\mathbf{x}_{1:d})_j$$

This completely bypasses the expensive $\mathcal{O}(d^3)$ determinant computation, reducing it to an $\mathcal{O}(d)$ vector sum [13, 15]!

---

### 📝 Quiz Question 5: Affine Coupling Layers
**Scenario:** Suppose a flow transformation is extremely expressive, but computing its inverse requires expensive numerical optimization. Why is this problematic?

* **Correct Answer:** **Sampling and likelihood computation become difficult.**
* **Explanation:** Normalizing Flows require both forward and inverse operations to be fast. If inverting the model requires iterative numerical optimization, sampling (which runs in the inverse direction) becomes extremely slow and computationally impractical.

---

## Part 6: Stacking & Dimensionality reduction

### 6.1 The Shared-Dimension Bottleneck
In a standard Normalizing Flow, the latent space $\mathbf{z}$ must have the same dimension as the input $\mathbf{x}$ [16]. For a high-resolution image, this means the latent space contains the same number of dimensions, which is computationally expensive to model and forces the prior to capture fine-grained pixel noise alongside global structures [16].

### 6.2 Factorization to the Rescue
To solve this, Real-NVP utilizes a **multi-scale architecture with factorization** [16]. 

```
   [ Input x ] ---> [ Coupling Layers ] ---> [ Partition ] ===> Output z_1 directly to prior
                                                   |
                                                   v
                                             [ Lower-dim z_2 ] ---> [ Next Flow Block ]
```

At intermediate stages, the latent vector is partitioned [16]. A portion of the latent variables (e.g., half of the channels) is "factored out" and evaluated directly against a Gaussian prior, while only the remaining variables are passed deeper into the next blocks of the flow [16]. This drastically reduces the computational load and memory footprint at deeper levels [16].

---

## Part 7: Glow (Generative Flow with Invertible 1x1 Convolutions)

Glow (Kingma & Dhariwal, 2018) improved on Real-NVP by introducing learnable **invertible 1x1 convolutions** [18, 19].

### 7.1 What is an Invertible 1x1 Convolution?
Instead of shuffing channels using fixed permutations, Glow uses a learnable 1x1 convolution [19]. A 1x1 convolution is equivalent to multiplying the channel vector at each spatial position by a weight matrix $W \in \mathbb{R}^{C \times C}$, where $C$ is the number of channels [19].

For a tensor of shape $H \times W \times C$, the log-determinant is [19]:
$$\log \left| \det \left(\text{1x1 Conv}\right) \right| = H \cdot W \cdot \log \left| \det W \right|$$

### 7.2 LU Decomposition
If the number of channels $C$ is large, computing $\det W$ scales as $\mathcal{O}(C^3)$ [19]. Glow parameterizes $W$ directly in its **LU decomposition** to keep this cheap [19]:
$$W = P L \left( U + \text{diag}(s) \right)$$

where:
* $P$ is a permutation matrix [19].
* $L$ is lower triangular with ones on the diagonal [19].
* $U$ is upper triangular with zeros on the diagonal [19].
* $s$ is a vector containing the diagonal elements [19].

The log-determinant of $W$ simplifies directly to the sum of the log of the elements in $s$ [19]:
$$\log \left| \det W \right| = \sum_{i} \log |s_i|$$

This reduces the complexity from $\mathcal{O}(C^3)$ to **$\mathcal{O}(C)$** [19].

---

### ⚠️ Training Alert: Matrix Collapse
A notable danger of this parameterization is that $W$ can collapse to a non-invertible matrix ($\det W = 0$) during training if any element $s_i$ approaches zero [19]. This must be monitored, and training should be reset to the last valid checkpoint if it occurs [19].

---

### 7.3 Latent Space Operations: Vector Arithmetic
Since Normalizing Flows learn a bijective mapping to a smooth Gaussian prior, we can perform semantic vector arithmetic directly in the latent space [20].

```
  [ Latent Space Z ]                                    [ Image Space X ]
  
  z_A (smiling)  \                                      x_A (smiling)
                  -->  z_mid = 0.5 * (z_A + z_B) ---->  x_mid (subtle smile)
  z_B (neutral)  /                                      x_B (neutral)
```

1. **Linear Interpolation:** To morph image $x_A$ smoothly into $x_B$, we map them to $z_A = f(x_A)$ and $z_B = f(x_B)$ [20]. We interpolate linearly in the latent space:
   $$z_{\text{interp}} = (1 - \alpha)z_A + \alpha z_B$$
   Projecting this back, $x_{\text{interp}} = g(z_{\text{interp}})$, yields a highly realistic, smooth transition [20].
2. **Attribute Manipulation:** We can isolate specific attributes by finding direction vectors [5]. For instance, by calculating the average latent vector of smiling faces $\bar{z}_{\text{smile}}$ and neutral faces $\bar{z}_{\text{neutral}}$, we obtain the "smiling vector":
   $$\mathbf{v}_{\text{smile}} = \bar{z}_{\text{smile}} - \bar{z}_{\text{neutral}}$$
   We can add this vector to a new, neutral face's latent code to make them smile [5]:
   $$z_{\text{new}} = z_{\text{neutral\_test}} + \gamma \mathbf{v}_{\text{smile}}$$

---

## Part 8: Conditional Normalizing Flows (CNFs)

Conditional Normalizing Flows (Winkler et al., 2019) extend flows to model conditional distributions $p(\mathbf{y} \mid \mathbf{x})$, enabling tasks like super-resolution, colorization, or structured prediction [21, 22].

```
   [ Latent Space Z ]  ==================================>  [ Target Space Y ]
   Cond. Prior p(z|x)  <== [ Flow f(y; x) conditioned on x ]  Real Image (y)
```

### 8.1 Conditioning the Network
Both our simple prior and our coupling layers are conditioned on an external variable $\mathbf{x}$ [22]:
* **Conditional Prior:** The latent variable $\mathbf{z}$ is drawn from a prior parameterized by $\mathbf{x}$ [22]:
  $$p(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\left(\mathbf{z}; \, \mu(\mathbf{x}), \, \sigma^2(\mathbf{x})\right)$$
* **Conditional Coupling Layers:** The translation and scale networks now accept $\mathbf{x}$ as an additional input [22]:
  $$\mathbf{y}_2 = \mathbf{z}_2 \odot \exp\left(s(\mathbf{z}_1, \mathbf{x})\right) + t(\mathbf{z}_1, \mathbf{x})$$

### 8.2 Application: Super-Resolution
In super-resolution, we want to generate a high-resolution image $\mathbf{y}$ conditioned on a low-resolution input $\mathbf{x}$ [23]. CNFs are highly suited for this [21]. They don't just output a single blurred average; they capture the multi-modal distribution of all possible high-resolution details that could fit the low-resolution input [21, 23].

---

## Part 9: Generative Adversarial Networks (GANs)

In contrast to the explicit density estimation of Normalizing Flows, GANs are implicit models that focus purely on sampling high-quality, realistic outputs [25, 43].

```
  z ~ p_z  --> [ Generator G ] --> Generated Image G(z)  ----\
                                                               v
                                                      [ Discriminator D ] --> Real (1) / Fake (0)
                                                             ^
  Real Image x ----------------------------------------------/
```

### 9.1 The Minimax Game
GANs are trained using a two-player minimax game [27, 28]:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log D(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log \left(1 - D(G(\mathbf{z}))\right) \right]$$

Let's unpack every term of this equation [27, 28]:
* **$D(\mathbf{x})$**: The probability assigned by the Discriminator that real data $\mathbf{x}$ is indeed real [26, 27].
* **$G(\mathbf{z})$**: The synthetic sample produced by the Generator from noise vector $\mathbf{z}$ [25, 26].
* **$D(G(\mathbf{z}))$**: The Discriminator's assessment of the synthetic sample [26, 27].
* **$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log D(\mathbf{x}) \right]$**: The Discriminator wants to maximize this term (predicting $D(\mathbf{x}) = 1$ for real data) [28].
* **$\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log \left(1 - D(G(\mathbf{z}))\right) \right]$**: The Discriminator wants to maximize this (predicting $D(G(\mathbf{z})) = 0$, making $\log(1) = 0$) [28]. Conversely, the Generator wants to minimize this term (fooling the discriminator into predicting $D(G(\mathbf{z})) = 1$) [26, 29].

### 9.2 Pros and Cons of GANs
* **Pros:** Exceptionally sharp, highly realistic image quality [40].
* **Cons:** Unstable training (susceptible to vanishing gradients), lack of an explicit loss curve to monitor convergence, and a common failure mode called **Mode Collapse** [5, 40].

---

### 📝 Quiz Question 8: GAN Failure Modes
**Scenario:** A GAN generates very realistic images, but every generated face looks almost identical. What is the most likely problem?

* **Correct Answer:** **Mode Collapse**
* **Explanation:** Mode collapse occurs when the Generator discovers a small subset (a single "mode") of the data distribution that successfully fools the Discriminator. Instead of covering the entire diverse distribution (different genders, poses, and skin tones), the Generator outputs nearly identical images because it is the safest way to win the minimax game [5].

---

## Part 10: StyleGAN and Disentanglement

StyleGAN (Karras et al., 2019) completely redesigned the Generator architecture of GANs to enable control over image synthesis at different levels of detail [30, 31].

```
  [ Latent z ] --> [ Mapping Network f ] --> [ Style w ] 
                                                   |
                                                   v  (Controls coarse, middle, and fine styles)
  Const 4x4x512 ---------------------------> [ Synthesis Network g ] ---> Output Image
```

### 10.1 Key Architectural Elements
1. **Mapping Network ($f$):** Instead of feeding the raw latent code $\mathbf{z} \in \mathcal{Z}$ directly into the generator, StyleGAN passes it through an 8-layer fully connected network to produce an intermediate style vector $\mathbf{w} \in \mathcal{W}$ [31].
2. **Synthesis Network ($g$):** The generation process starts from a learned constant $4 \times 4 \times 512$ tensor, rather than a raw noise vector [31]. 
3. **Adaptive Instance Normalization (AdaIN):** At each convolutional layer, the feature maps are normalized and scaled/shifted using styles derived from the vector $\mathbf{w}$ [31]:
   $$\text{AdaIN}(\mathbf{x}_i, \mathbf{w}) = y_{s,i} \left( \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} \right) + y_{b,i}$$
4. **Noise Injection:** Stochastic, fine-grained details (like hair strands, skin pores, and freckles) are injected by adding per-pixel Gaussian noise directly to each layer [31, 34].

---

### 📝 Quiz Question 9: StyleGAN and Noise
**Scenario:** Suppose you remove all stochastic noise inputs from StyleGAN. Which effect is most likely?

* **Correct Answer:** **Fine details such as freckles, pores, and hair texture become less diverse.**
* **Explanation:** In StyleGAN, global identity and pose are controlled by the intermediate style vector $\mathbf{w}$. Per-pixel noise injection is designed specifically to capture fine stochastic details (like the exact curl of hair or the pattern of skin pores) [34]. Removing the noise causes these details to look flat and repetitive ("painterly"), while the person's core identity remains intact [34].

---

### 10.2 What is Disentanglement?
In standard latent spaces, changing one coordinate of $\mathbf{z}$ might simultaneously alter multiple semantic attributes (e.g., changing hair color also shifts gender and age) [33]. This is an **entangled** space [33].

We want a **disentangled representation**, where individual coordinates of our latent space correspond to distinct, independent semantic attributes [33]:

```
  [ Entangled Space Z ]                      [ Disentangled Space W ]
  
        Age                                        Age
       /                                            |
  z_1 --- Gender                             w_1 ---------> (Controls ONLY age)
       \                                            |
        Hair Color                                 w_2 ---------> (Controls ONLY gender)
```

By mapping $\mathcal{Z} \to \mathcal{W}$ through a Mapping Network, StyleGAN allows $\mathcal{W}$ to untangle the factors of variation [30]. This enables a "SIMS Character Creator" experience where developers can change one slider without affecting others [33].

---

## Part 11: Pix2Pix (Conditional GANs for Image Translation)

Pix2Pix (Isola et al., 2017) applies Conditional GANs to translate input images from one domain to another (such as turning a hand-drawn sketch of a shoe into a photorealistic product image) [4, 36].

```
  [ Sketch Input x ] ---> [ Generator G ] ---> [ Generated Shoe G(x) ]
                                 |
                                 v
                             + L1 Loss (Encourages pixel-level similarity to ground truth)
```

### 11.1 The Conditional Objective
In a Conditional GAN, both the Generator and the Discriminator receive the conditioning image $\mathbf{x}$ as an input [38]. To ensure the generated output is structurally accurate, Pix2Pix combines the adversarial loss with a standard **$L_1$ reconstruction loss** [38]:

$$G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L_1}(G)$$

where:
$$\mathcal{L}_{L_1}(G) = \mathbb{E}_{\mathbf{x}, \mathbf{y}, \mathbf{z}} \left[ \|\mathbf{y} - G(\mathbf{x}, \mathbf{z})\|_1 \right]$$

### Why the $L_1$ Loss is Crucial
* **The Adversarial Loss** $\mathcal{L}_{cGAN}$ forces the generator to produce sharp, realistic textures and details.
* **The $L_1$ Loss** forces the generator to maintain global structural coherence and keep the output aligned with the input sketch [38]. Without the $L_1$ loss, the generator would "hallucinate" random plausible shoes that have no relationship to the input sketch.

---

### 📝 Quiz Question 10: Conditional GAN
**Scenario:** Suppose you train a GAN to convert semantic segmentation maps into street-view images. Why is a Conditional GAN more appropriate than a standard GAN?

* **Correct Answer:** **It can use the segmentation map as additional information.**
* **Explanation:** A standard GAN generates images starting from random noise, meaning you have no control over the layout of the output street scene. A Conditional GAN accepts the semantic segmentation map as an explicit input, using it as a blueprint to guide the generation of cars, roads, and buildings in their exact designated positions [4, 38].

---

## Summary Comparison: Choosing the Right Generative Model

There is no single best generative model; the correct choice depends entirely on your specific engineering and scientific requirements [7]:

| Model Family | Tractable Likelihood $p(\mathbf{x})$? | Sample Quality | Training Stability | Latent Representation | Key Drawback |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KDE** [7] | Yes (Exact) | Poor (Blurry) | N/A (Non-parametric) | None [2, 5] | Scales terribly in high dimensions [2, 4]. |
| **VAE** [43] | No (Approximate via ELBO) | Good (Can be slightly blurry) | Stable | Yes (Continuous, compressed) | Intractable exact likelihood. |
| **Normalizing Flows** [43] | **Yes (Exact & Tractable)** | Very Good [19] | Moderate | **Yes (Invertible, same-dim)** | Latent space dimension must equal data dimension [24]. |
| **GANs** [43] | **No** [4] | **Outstanding** [40] | Unstable (Minimax game) | Yes (Can be highly disentangled) | No explicit density, prone to Mode Collapse [5]. |

---

### 📝 Quiz Question 6: Choosing the Right Generative Model
**Scenario:** You are building a medical imaging system. The doctors require:
1. Generation of realistic samples [3].
2. The ability to compute exact likelihoods (to catch anomalies or out-of-distribution inputs) [3].
3. A meaningful latent representation (to interpolate between different disease states) [3].

Which model family is the most suitable?

* **Correct Answer:** **Normalising Flow** [3]
* **Explanation:** GANs cannot evaluate exact likelihoods, and KDE scales poorly on high-dimensional image data [4]. Only Normalizing Flows satisfy all three requirements simultaneously [3].

# Masterclass Lecture Notes: Generative Modeling, Normalizing Flows, and GANs

This set of detailed notes covers the mathematical, architectural, and conceptual foundations of generative models. It is designed for those encountering these topics for the first time, starting from basic probability density modeling up to advanced deep architectures like Real-NVP, Glow, StyleGAN, and Pix2Pix.

---

## Part 1: The Foundations of Generative Modeling

### 1.1 What is Generative Modeling?
At its core, generative modeling is about learning the probability distribution of a given dataset [1, 2]. Suppose we have a set of data samples (such as images of faces, audio waves, or medical scans) [2]:
$$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\} \sim p(\mathbf{x})$$

We assume all these samples are drawn from an underlying, unknown probability distribution $p(\mathbf{x})$ [2]. Our goals are:
1. **Density Estimation:** Construct an approximate model $q(\mathbf{x})$ that behaves like the true distribution $p(\mathbf{x})$ [2]. This allows us to evaluate how likely any arbitrary input $\mathbf{x}$ is under our model [8, 13].
2. **Sampling:** Generate completely new, realistic data points $\hat{\mathbf{x}}$ by sampling from our learned model $q(\mathbf{x})$ [2].

### 1.2 Why is this "Self-Supervised"?
Generative modeling is classified as **self-supervised** (or unsupervised) learning [3]:
* **No Manual Labels:** The training process requires only raw data $\mathbf{x}$ [3]. We do not need annotators to label images as "car," "cat," or "dog" [3].
* **Hidden Structure:** The model must discover the underlying, latent factors of variation (the hidden structure) to represent the data [3].
* **Feature Transfer:** The representations learned by solving these generative tasks can be transferred to downstream tasks like classification, regression, or anomaly detection [3].

---

### 1.3 A Map of the Generative Landscape (Taxonomy)
Different families of generative models strike different trade-offs between sample quality, exact likelihood computation, and training stability [43]:

```
                                     [ Generative Models ]
                                               |
                     -----------------------------------------------------
                    |                                                     |
            [ Explicit Density ]                                  [ Implicit Density ]
                    |                                                     |
         -------------------------                             -------------------------
        |                         |                           |                         |
   [ Tractable ]           [ Approximate ]                [ Direct ]               [ Indirect ]
     - Autoregressive        - Variational Autoencoder      - GANs                   - Diffusion Models
     - Normalizing Flows       (VAE)
```

1. **Explicit Density Models:** These models explicitly define a probability density function $q(\mathbf{x})$ [43].
   * **Tractable Density:** We can compute the exact probability $q(\mathbf{x})$ of any data point in a single forward pass [43]. Examples include **Autoregressive Models** and **Normalizing Flows** [43].
   * **Approximate Density:** The exact density is mathematically intractable, but we can optimize a lower bound (ELBO) or a variational approximation [4, 43]. Examples include **Variational Autoencoders (VAEs)** [43].
2. **Implicit Density Models:** These models do not define an explicit probability density function [43]. Instead, they focus purely on sampling [43].
   * **Direct Sampling:** We can generate samples directly in a single step (e.g., **Generative Adversarial Networks (GANs)**) [43].
   * **Indirect/Iterative Sampling:** Samples are generated via an iterative denoising process (e.g., **Diffusion Models**) [43].

---

## Part 2: Non-Parametric Density Estimation (Kernel Density Estimation)

To understand deep parametric models, we must first look at classical, non-parametric approaches [2, 5]. The most famous of these is **Kernel Density Estimation (KDE)** [4, 5].

### 2.1 The Intuition of KDE
Imagine dropping a single grain of sand at every data point $\mathbf{x}_i$ in our training set [6]. Where points are clustered tightly, the sand piles up [6]. If we smooth out these piles of sand using a bell-shaped curve (a kernel), the height of the sand pile at any coordinate $x$ represents the probability density [6].

$$\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$

Let's break down every variable in this formula [6]:
* **$\hat{f}_h(x)$**: The estimated probability density at a query point $x$ [6].
* **$n$**: The total number of training samples [6].
* **$x_i$**: The $i$-th training sample from our dataset [6].
* **$K(u)$**: The **Kernel function** [6]. It is a symmetric, non-negative function that integrates to 1 (making $\hat{f}_h(x)$ a valid probability density).
* **$h$**: The **bandwidth** [6]. It controls the width of each kernel, acting as a smoothing parameter [6].

### 2.2 Common Kernels
Different kernel shapes can be used to smooth the data points [6]:

```
       [ Gaussian ]              [ Tophat / Box ]             [ Epanechnikov ]
          _---_                       _____                         _-_
        /       \                    |     |                       /   \
       /         \                   |     |                      /     \
  ____/           \____         _____|_____|_____            ____/       \____
```

When using a **Gaussian Kernel**, the estimator becomes [7]:
$$\hat{f}_h(x) = \frac{1}{n} \sum_{i=1}^n \frac{1}{h\sqrt{2\pi}} \exp\left( \frac{-(x-x_i)^2}{2h^2} \right)$$

This places a small Gaussian distribution centered at every training data point $x_i$ and averages them [7].

---

### 💡 Concept Spotlight: The Impact of Bandwidth ($h$)
The choice of bandwidth $h$ governs the bias-variance trade-off in non-parametric density estimation:
* **If $h$ is too small (Under-smoothing):** The density estimate becomes a collection of highly isolated, sharp spikes centered exactly at the training points. This results in **overfitting**; the model fails to generalize to the space between points.
* **If $h$ is too large (Over-smoothing):** The kernels become extremely wide and flat. This results in **underfitting**.

---

### 📝 Quiz Question 1: KDE and Generalization
**Scenario:** You have observed the following 1D training samples: $\{2.0, 2.1, 2.2, 8.0, 8.1, 8.2\}$. You fit a KDE with a **very large** kernel bandwidth $h$. What is the likely consequence?

* **Correct Answer:** **The density will become smoother and may place a high probability between the two groups.**
* **Explanation:** A very large bandwidth makes each individual Gaussian kernel extremely wide. As a result, the density estimates from the distinct cluster around $2.0$ and the cluster around $8.0$ will heavily overlap in the middle. The model will assign a high probability density to the region around $5.0$, where no training data actually exists.

---

### 2.3 Sampling from a KDE
How do we generate new data points from a KDE? The process is direct:
1. Randomly select a data point $x_i$ from the training set.
2. Sample noise $\epsilon$ from the distribution defined by the kernel $K$ (e.g., if using a Gaussian kernel, sample $\epsilon \sim \mathcal{N}(0, h^2)$).
3. Output the new sample: $\hat{x} = x_i + \epsilon$.

### 2.4 Why KDE Fails in High Dimensions (The Curse of Dimensionality)
KDE is highly effective in 1D, 2D, or 3D spaces (such as mapping animal habitats on a geographical map) [4, 5]. However, it is entirely impractical for high-dimensional data, such as images [2, 4].

---

### 📝 Quiz Question 2: KDE as a Generative Model
**Scenario:** Suppose you want to generate realistic human face images. Which is the strongest argument against using KDE directly?

* **Correct Answer:** **KDE scales poorly in very high-dimensional spaces.**
* **Explanation:** Face images are extremely high-dimensional (e.g., $256 \times 256$ pixels $\approx 65,536$ dimensions). Due to the **curse of dimensionality**, the volume of a space scales exponentially with its dimension. In such massive spaces, training points become incredibly sparse, requiring an astronomically large dataset to estimate a meaningful density or generate coherent images.

---

## Part 3: From Representation to Generation: Autoencoders

An Autoencoder (AE) is a neural network designed to learn a compressed representation of data in an unsupervised manner [10].

```
   [ Input Space X ]            [ Latent Space Z ]           [ Output Space X' ]
    Real Image (x)  --> [ Encoder ] -->  z  --> [ Decoder ] --> Reconstructed Image (x')
```

* **Encoder ($f_\theta$):** Maps a high-dimensional input $\mathbf{x}$ to a low-dimensional latent code $\mathbf{z}$ [10].
* **Decoder ($g_\phi$):** Maps the latent code $\mathbf{z}$ back to the original space to reconstruct $\mathbf{x}' \approx \mathbf{x}$ [10].

---

### 📝 Quiz Question 3: Autoencoder vs. Flow
**Scenario:** Suppose an autoencoder reconstructs images perfectly. Can it automatically be used as a good generative model?

* **Correct Answer:** **No, because reconstruction quality does not guarantee that latent codes follow a simple distribution.**
* **Explanation:** Perfect reconstruction simply means the decoder can invert the encoder's mapping for the training data. However, a standard autoencoder does not constrain the distribution of the latent space $\mathbf{z}$. The latent codes might be distributed in a highly irregular, fractured, and sparse manner. If you attempt to generate a new image by sampling a random latent vector $\mathbf{z} \sim \mathcal{N}(0, I)$, the sampled vector will likely land in an "empty" region of the latent space that the decoder was never trained on, resulting in meaningless, garbled outputs.

---

## Part 4: Normalizing Flows (NFs)

To solve the limitations of standard autoencoders, GANs, and VAEs, we turn to **Normalizing Flows** [9, 10]. They are designed to do three things simultaneously:
1. Generate realistic, high-quality samples [24].
2. Compute the **exact likelihood** $p(\mathbf{x})$ of any data point [24].
3. Learn a **meaningful, invertible latent space** [24].

```
                   g (Forward / Generation)
   [ Latent Space Z ]  ========================>  [ Data Space X ]
   Simple Prior p(z)   <========================  Complex Density p(x)
                     f (Inverse / Normalizing)
```

### 4.1 The Core Idea: Bijective Networks
Instead of compressing the data into a lower-dimensional bottleneck, a Normalizing Flow uses a **bijective network** [9]. A bijective function $g: \mathcal{Z} \to \mathcal{X}$ is:
* **Invertible:** For every $\mathbf{x}$, there is a unique, computable $\mathbf{z} = g^{-1}(\mathbf{x})$ [10, 11].
* **Differentiable:** Both $g$ and its inverse $f = g^{-1}$ are differentiable [10, 11].
* **Dimension-preserving:** The latent space $\mathbf{z}$ has the **exact same dimensionality** as the data space $\mathbf{x}$ [10].

### 4.2 The Mathematics of Flow: Change of Variables
Normalizing Flows use the multivariate change of variables formula to map a simple prior distribution $p_z(\mathbf{z})$ (such as an isotropic Gaussian) to a complex data distribution $p_x(\mathbf{x})$ [9, 11]:

$$p_x(\mathbf{x}) = p_z(g^{-1}(\mathbf{x})) \cdot \left| \det \left( \frac{\partial g^{-1}(\mathbf{x})}{\partial \mathbf{x}} \right) \right|$$

Let's write this in the more common notation where $f(\mathbf{x}) = \mathbf{z} = g^{-1}(\mathbf{x})$ is the "normalizing" direction [10, 11]:
$$p_x(\mathbf{x}) = p_z(f(\mathbf{x})) \cdot \left| \det J_f(\mathbf{x}) \right|$$

Taking the logarithm gives us the log-likelihood objective function [11]:
$$\log p_x(\mathbf{x}) = \log p_z(f(\mathbf{x})) + \log \left| \det J_f(\mathbf{x}) \right|$$

Let's unpack every component of this equation [11]:
* **$\log p_x(\mathbf{x})$**: The exact log-probability of our data point $\mathbf{x}$ under the model [11].
* **$\log p_z(f(\mathbf{x}))$**: The log-probability of the normalized latent vector $\mathbf{z} = f(\mathbf{x})$ under our simple Gaussian prior [11]. This term penalizes latent codes that stray too far from the origin.
* **$J_f(\mathbf{x}) = \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}$**: The **Jacobian matrix** of the transformation $f$ [11]. It contains all first-order partial derivatives, capturing how the transformation distorts space [11].
* **$\log \left| \det J_f(\mathbf{x}) \right|$**: The log-determinant of the Jacobian [11]. It measures how much the local volume expands or contracts [11]. Intuitively, it ensures that the total probability integrates to 1 by accounting for the warping of space.

### 4.3 Composing Multiple Layers
A single simple transformation cannot map a Gaussian to a complex image distribution. However, we can compose many invertible, bijective layers together [12]:
$$\mathbf{x} \xrightarrow{\quad f_T \quad } \mathbf{z}_{T-1} \xrightarrow{\quad f_{T-1} \quad } \dots \xrightarrow{\quad f_1 \quad } \mathbf{z}_0 = \mathbf{z} \sim p_z(\mathbf{z})$$

The overall transformation is $f = f_1 \circ f_2 \circ \dots \circ f_T$ [12]. By the chain rule, the log-determinant of the composed transformation is simply the sum of the log-determinants of the individual layers [12]:
$$\log p_x(\mathbf{x}) = \log p_z(\mathbf{z}_0) + \sum_{t=1}^T \log \left| \det J_{f_t}(\mathbf{z}_t) \right|$$

This composition allows us to warp a simple distribution step-by-step into an incredibly complex, multi-modal shape [12].

---

### 📝 Quiz Question 4: Generative Model Family
**Scenario:** Suppose you have a model that can generate high-quality images, but you cannot compute the probability of an image under the model. Which generative family was largely designed to solve this issue?

* **Correct Answer:** **Normalising Flows**
* **Explanation:** GANs can generate realistic images but do not provide an explicit probability density function. Normalizing Flows were designed specifically to allow exact, tractable likelihood computation while remaining generative models.

---

## Part 5: Real-NVP and Affine Coupling Layers

Calculating the determinant of an arbitrary Jacobian matrix scales cubically with the dimensions: $\mathcal{O}(d^3)$ [8, 19]. To make this calculation efficient, **Real-NVP** (Dinh et al., 2017) introduced **Affine Coupling Layers** [14, 15].

```
               [ Affine Coupling Step ]
               
             /--->  x_{1:d}  --------==========-------->  y_{1:d} = x_{1:d}
            /                 |         |
   Input x                     v         v
            \              [ s(.) ]  [ t(.) ]
             \                |         |
              \---> x_{d+1:D} ---> (x) ---- (+) -------> y_{d+1:D} = x_{d+1:D} * e^{s} + t
```

### 5.1 The Forward Propagation
An affine coupling layer splits the input dimensions into two halves [15]:
1. The first $d$ dimensions are passed through unchanged [15]:
   $$\mathbf{y}_{1:d} = \mathbf{x}_{1:d}$$
2. The remaining dimensions (from $d+1$ to $D$) undergo an affine transformation (scaling and shifting) based on the first half [15]:
   $$\mathbf{y}_{d+1:D} = \mathbf{x}_{d+1:D} \odot \exp\left(s(\mathbf{x}_{1:d})\right) + t(\mathbf{x}_{1:d})$$

Here, $s$ and $t$ are scale and translation functions, which are parameterized by arbitrary neural networks (e.g., standard residual networks) [15].

### 5.2 The Inverse Propagation (Sampling)
During generation, we need to map from $\mathbf{y}$ back to $\mathbf{x}$ [15]. What makes affine coupling layers highly effective is that **the neural networks $s$ and $t$ do not need to be invertible** [15]. We can invert the entire layer analytically [15]:

1. Because the first part was passed through unchanged, we recover it directly [15]:
   $$\mathbf{x}_{1:d} = \mathbf{y}_{1:d}$$
2. We can then pass this recovered $\mathbf{x}_{1:d}$ through the exact same networks $s$ and $t$ to invert the second half [15]:
   $$\mathbf{x}_{d+1:D} = \left(\mathbf{y}_{d+1:D} - t(\mathbf{y}_{1:d})\right) \odot \exp\left(-s(\mathbf{y}_{1:d})\right)$$

This allows us to use highly complex, non-invertible neural networks inside our flow architecture while maintaining exact, analytic invertibility for the overall system [15].

### 5.3 Why the Jacobian is Incredibly Cheap
Let's look at the Jacobian matrix for this transformation [15]:

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}^T} = \begin{bmatrix} 
I_d & 0 \\ 
\frac{\partial \mathbf{y}_{d+1:D}}{\partial \mathbf{x}_{1:d}} & \text{diag}\left(\exp\left(s(\mathbf{x}_{1:d})\right)\right) 
\end{bmatrix}$$

Because the top-right block is $0$ (the first half $\mathbf{y}_{1:d}$ has zero dependency on the second half $\mathbf{x}_{d+1:D}$), this matrix is **block lower-triangular** [15]. 

The determinant of a block triangular matrix is simply the product of the diagonal blocks' determinants [15]. Since the top-left block is the identity matrix $I_d$ (determinant of 1) and the bottom-right block is a diagonal matrix, its determinant is trivial to compute [15]:

$$\det J = \prod_{j=d+1}^D \exp\left(s(\mathbf{x}_{1:d})\right)_j$$

Taking the log-determinant reduces to a simple sum over our scale network's output [15]:
$$\log \left| \det J \right| = \sum_{j} s(\mathbf{x}_{1:d})_j$$

This completely bypasses the expensive $\mathcal{O}(d^3)$ determinant computation, reducing it to an $\mathcal{O}(d)$ vector sum [13, 15]!

---

### 📝 Quiz Question 5: Affine Coupling Layers
**Scenario:** Suppose a flow transformation is extremely expressive, but computing its inverse requires expensive numerical optimization. Why is this problematic?

* **Correct Answer:** **Sampling and likelihood computation become difficult.**
* **Explanation:** Normalizing Flows require both forward and inverse operations to be fast. If inverting the model requires iterative numerical optimization, sampling (which runs in the inverse direction) becomes extremely slow and computationally impractical.

---

## Part 6: Stacking & Dimensionality reduction

### 6.1 The Shared-Dimension Bottleneck
In a standard Normalizing Flow, the latent space $\mathbf{z}$ must have the same dimension as the input $\mathbf{x}$ [16]. For a high-resolution image, this means the latent space contains the same number of dimensions, which is computationally expensive to model and forces the prior to capture fine-grained pixel noise alongside global structures [16].

### 6.2 Factorization to the Rescue
To solve this, Real-NVP utilizes a **multi-scale architecture with factorization** [16]. 

```
   [ Input x ] ---> [ Coupling Layers ] ---> [ Partition ] ===> Output z_1 directly to prior
                                                   |
                                                   v
                                             [ Lower-dim z_2 ] ---> [ Next Flow Block ]
```

At intermediate stages, the latent vector is partitioned [16]. A portion of the latent variables (e.g., half of the channels) is "factored out" and evaluated directly against a Gaussian prior, while only the remaining variables are passed deeper into the next blocks of the flow [16]. This drastically reduces the computational load and memory footprint at deeper levels [16].

---

## Part 7: Glow (Generative Flow with Invertible 1x1 Convolutions)

Glow (Kingma & Dhariwal, 2018) improved on Real-NVP by introducing learnable **invertible 1x1 convolutions** [18, 19].

### 7.1 What is an Invertible 1x1 Convolution?
Instead of shuffing channels using fixed permutations, Glow uses a learnable 1x1 convolution [19]. A 1x1 convolution is equivalent to multiplying the channel vector at each spatial position by a weight matrix $W \in \mathbb{R}^{C \times C}$, where $C$ is the number of channels [19].

For a tensor of shape $H \times W \times C$, the log-determinant is [19]:
$$\log \left| \det \left(\text{1x1 Conv}\right) \right| = H \cdot W \cdot \log \left| \det W \right|$$

### 7.2 LU Decomposition
If the number of channels $C$ is large, computing $\det W$ scales as $\mathcal{O}(C^3)$ [19]. Glow parameterizes $W$ directly in its **LU decomposition** to keep this cheap [19]:
$$W = P L \left( U + \text{diag}(s) \right)$$

where:
* $P$ is a permutation matrix [19].
* $L$ is lower triangular with ones on the diagonal [19].
* $U$ is upper triangular with zeros on the diagonal [19].
* $s$ is a vector containing the diagonal elements [19].

The log-determinant of $W$ simplifies directly to the sum of the log of the elements in $s$ [19]:
$$\log \left| \det W \right| = \sum_{i} \log |s_i|$$

This reduces the complexity from $\mathcal{O}(C^3)$ to **$\mathcal{O}(C)$** [19].

---

### ⚠️ Training Alert: Matrix Collapse
A notable danger of this parameterization is that $W$ can collapse to a non-invertible matrix ($\det W = 0$) during training if any element $s_i$ approaches zero [19]. This must be monitored, and training should be reset to the last valid checkpoint if it occurs [19].

---

### 7.3 Latent Space Operations: Vector Arithmetic
Since Normalizing Flows learn a bijective mapping to a smooth Gaussian prior, we can perform semantic vector arithmetic directly in the latent space [20].

```
  [ Latent Space Z ]                                    [ Image Space X ]
  
  z_A (smiling)  \                                      x_A (smiling)
                  -->  z_mid = 0.5 * (z_A + z_B) ---->  x_mid (subtle smile)
  z_B (neutral)  /                                      x_B (neutral)
```

1. **Linear Interpolation:** To morph image $x_A$ smoothly into $x_B$, we map them to $z_A = f(x_A)$ and $z_B = f(x_B)$ [20]. We interpolate linearly in the latent space:
   $$z_{\text{interp}} = (1 - \alpha)z_A + \alpha z_B$$
   Projecting this back, $x_{\text{interp}} = g(z_{\text{interp}})$, yields a highly realistic, smooth transition [20].
2. **Attribute Manipulation:** We can isolate specific attributes by finding direction vectors [5]. For instance, by calculating the average latent vector of smiling faces $\bar{z}_{\text{smile}}$ and neutral faces $\bar{z}_{\text{neutral}}$, we obtain the "smiling vector":
   $$\mathbf{v}_{\text{smile}} = \bar{z}_{\text{smile}} - \bar{z}_{\text{neutral}}$$
   We can add this vector to a new, neutral face's latent code to make them smile [5]:
   $$z_{\text{new}} = z_{\text{neutral\_test}} + \gamma \mathbf{v}_{\text{smile}}$$

---

## Part 8: Conditional Normalizing Flows (CNFs)

Conditional Normalizing Flows (Winkler et al., 2019) extend flows to model conditional distributions $p(\mathbf{y} \mid \mathbf{x})$, enabling tasks like super-resolution, colorization, or structured prediction [21, 22].

```
   [ Latent Space Z ]  ==================================>  [ Target Space Y ]
   Cond. Prior p(z|x)  <== [ Flow f(y; x) conditioned on x ]  Real Image (y)
```

### 8.1 Conditioning the Network
Both our simple prior and our coupling layers are conditioned on an external variable $\mathbf{x}$ [22]:
* **Conditional Prior:** The latent variable $\mathbf{z}$ is drawn from a prior parameterized by $\mathbf{x}$ [22]:
  $$p(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\left(\mathbf{z}; \, \mu(\mathbf{x}), \, \sigma^2(\mathbf{x})\right)$$
* **Conditional Coupling Layers:** The translation and scale networks now accept $\mathbf{x}$ as an additional input [22]:
  $$\mathbf{y}_2 = \mathbf{z}_2 \odot \exp\left(s(\mathbf{z}_1, \mathbf{x})\right) + t(\mathbf{z}_1, \mathbf{x})$$

### 8.2 Application: Super-Resolution
In super-resolution, we want to generate a high-resolution image $\mathbf{y}$ conditioned on a low-resolution input $\mathbf{x}$ [23]. CNFs are highly suited for this [21]. They don't just output a single blurred average; they capture the multi-modal distribution of all possible high-resolution details that could fit the low-resolution input [21, 23].

---

## Part 9: Generative Adversarial Networks (GANs)

In contrast to the explicit density estimation of Normalizing Flows, GANs are implicit models that focus purely on sampling high-quality, realistic outputs [25, 43].

```
  z ~ p_z  --> [ Generator G ] --> Generated Image G(z)  ----\
                                                               v
                                                      [ Discriminator D ] --> Real (1) / Fake (0)
                                                             ^
  Real Image x ----------------------------------------------/
```

### 9.1 The Minimax Game
GANs are trained using a two-player minimax game [27, 28]:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log D(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log \left(1 - D(G(\mathbf{z}))\right) \right]$$

Let's unpack every term of this equation [27, 28]:
* **$D(\mathbf{x})$**: The probability assigned by the Discriminator that real data $\mathbf{x}$ is indeed real [26, 27].
* **$G(\mathbf{z})$**: The synthetic sample produced by the Generator from noise vector $\mathbf{z}$ [25, 26].
* **$D(G(\mathbf{z}))$**: The Discriminator's assessment of the synthetic sample [26, 27].
* **$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log D(\mathbf{x}) \right]$**: The Discriminator wants to maximize this term (predicting $D(\mathbf{x}) = 1$ for real data) [28].
* **$\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}} \left[ \log \left(1 - D(G(\mathbf{z}))\right) \right]$**: The Discriminator wants to maximize this (predicting $D(G(\mathbf{z})) = 0$, making $\log(1) = 0$) [28]. Conversely, the Generator wants to minimize this term (fooling the discriminator into predicting $D(G(\mathbf{z})) = 1$) [26, 29].

### 9.2 Pros and Cons of GANs
* **Pros:** Exceptionally sharp, highly realistic image quality [40].
* **Cons:** Unstable training (susceptible to vanishing gradients), lack of an explicit loss curve to monitor convergence, and a common failure mode called **Mode Collapse** [5, 40].

---

### 📝 Quiz Question 8: GAN Failure Modes
**Scenario:** A GAN generates very realistic images, but every generated face looks almost identical. What is the most likely problem?

* **Correct Answer:** **Mode Collapse**
* **Explanation:** Mode collapse occurs when the Generator discovers a small subset (a single "mode") of the data distribution that successfully fools the Discriminator. Instead of covering the entire diverse distribution (different genders, poses, and skin tones), the Generator outputs nearly identical images because it is the safest way to win the minimax game [5].

---

## Part 10: StyleGAN and Disentanglement

StyleGAN (Karras et al., 2019) completely redesigned the Generator architecture of GANs to enable control over image synthesis at different levels of detail [30, 31].

```
  [ Latent z ] --> [ Mapping Network f ] --> [ Style w ] 
                                                   |
                                                   v  (Controls coarse, middle, and fine styles)
  Const 4x4x512 ---------------------------> [ Synthesis Network g ] ---> Output Image
```

### 10.1 Key Architectural Elements
1. **Mapping Network ($f$):** Instead of feeding the raw latent code $\mathbf{z} \in \mathcal{Z}$ directly into the generator, StyleGAN passes it through an 8-layer fully connected network to produce an intermediate style vector $\mathbf{w} \in \mathcal{W}$ [31].
2. **Synthesis Network ($g$):** The generation process starts from a learned constant $4 \times 4 \times 512$ tensor, rather than a raw noise vector [31]. 
3. **Adaptive Instance Normalization (AdaIN):** At each convolutional layer, the feature maps are normalized and scaled/shifted using styles derived from the vector $\mathbf{w}$ [31]:
   $$\text{AdaIN}(\mathbf{x}_i, \mathbf{w}) = y_{s,i} \left( \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} \right) + y_{b,i}$$
4. **Noise Injection:** Stochastic, fine-grained details (like hair strands, skin pores, and freckles) are injected by adding per-pixel Gaussian noise directly to each layer [31, 34].

---

### 📝 Quiz Question 9: StyleGAN and Noise
**Scenario:** Suppose you remove all stochastic noise inputs from StyleGAN. Which effect is most likely?

* **Correct Answer:** **Fine details such as freckles, pores, and hair texture become less diverse.**
* **Explanation:** In StyleGAN, global identity and pose are controlled by the intermediate style vector $\mathbf{w}$. Per-pixel noise injection is designed specifically to capture fine stochastic details (like the exact curl of hair or the pattern of skin pores) [34]. Removing the noise causes these details to look flat and repetitive ("painterly"), while the person's core identity remains intact [34].

---

### 10.2 What is Disentanglement?
In standard latent spaces, changing one coordinate of $\mathbf{z}$ might simultaneously alter multiple semantic attributes (e.g., changing hair color also shifts gender and age) [33]. This is an **entangled** space [33].

We want a **disentangled representation**, where individual coordinates of our latent space correspond to distinct, independent semantic attributes [33]:

```
  [ Entangled Space Z ]                      [ Disentangled Space W ]
  
        Age                                        Age
       /                                            |
  z_1 --- Gender                             w_1 ---------> (Controls ONLY age)
       \                                            |
        Hair Color                                 w_2 ---------> (Controls ONLY gender)
```

By mapping $\mathcal{Z} \to \mathcal{W}$ through a Mapping Network, StyleGAN allows $\mathcal{W}$ to untangle the factors of variation [30]. This enables a "SIMS Character Creator" experience where developers can change one slider without affecting others [33].

---

## Part 11: Pix2Pix (Conditional GANs for Image Translation)

Pix2Pix (Isola et al., 2017) applies Conditional GANs to translate input images from one domain to another (such as turning a hand-drawn sketch of a shoe into a photorealistic product image) [4, 36].

```
  [ Sketch Input x ] ---> [ Generator G ] ---> [ Generated Shoe G(x) ]
                                 |
                                 v
                             + L1 Loss (Encourages pixel-level similarity to ground truth)
```

### 11.1 The Conditional Objective
In a Conditional GAN, both the Generator and the Discriminator receive the conditioning image $\mathbf{x}$ as an input [38]. To ensure the generated output is structurally accurate, Pix2Pix combines the adversarial loss with a standard **$L_1$ reconstruction loss** [38]:

$$G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L_1}(G)$$

where:
$$\mathcal{L}_{L_1}(G) = \mathbb{E}_{\mathbf{x}, \mathbf{y}, \mathbf{z}} \left[ \|\mathbf{y} - G(\mathbf{x}, \mathbf{z})\|_1 \right]$$

### Why the $L_1$ Loss is Crucial
* **The Adversarial Loss** $\mathcal{L}_{cGAN}$ forces the generator to produce sharp, realistic textures and details.
* **The $L_1$ Loss** forces the generator to maintain global structural coherence and keep the output aligned with the input sketch [38]. Without the $L_1$ loss, the generator would "hallucinate" random plausible shoes that have no relationship to the input sketch.

---

### 📝 Quiz Question 10: Conditional GAN
**Scenario:** Suppose you train a GAN to convert semantic segmentation maps into street-view images. Why is a Conditional GAN more appropriate than a standard GAN?

* **Correct Answer:** **It can use the segmentation map as additional information.**
* **Explanation:** A standard GAN generates images starting from random noise, meaning you have no control over the layout of the output street scene. A Conditional GAN accepts the semantic segmentation map as an explicit input, using it as a blueprint to guide the generation of cars, roads, and buildings in their exact designated positions [4, 38].

---

## Summary Comparison: Choosing the Right Generative Model

There is no single best generative model; the correct choice depends entirely on your specific engineering and scientific requirements [7]:

| Model Family | Tractable Likelihood $p(\mathbf{x})$? | Sample Quality | Training Stability | Latent Representation | Key Drawback |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KDE** [7] | Yes (Exact) | Poor (Blurry) | N/A (Non-parametric) | None [2, 5] | Scales terribly in high dimensions [2, 4]. |
| **VAE** [43] | No (Approximate via ELBO) | Good (Can be slightly blurry) | Stable | Yes (Continuous, compressed) | Intractable exact likelihood. |
| **Normalizing Flows** [43] | **Yes (Exact & Tractable)** | Very Good [19] | Moderate | **Yes (Invertible, same-dim)** | Latent space dimension must equal data dimension [24]. |
| **GANs** [43] | **No** [4] | **Outstanding** [40] | Unstable (Minimax game) | Yes (Can be highly disentangled) | No explicit density, prone to Mode Collapse [5]. |

---

### 📝 Quiz Question 6: Choosing the Right Generative Model
**Scenario:** You are building a medical imaging system. The doctors require:
1. Generation of realistic samples [3].
2. The ability to compute exact likelihoods (to catch anomalies or out-of-distribution inputs) [3].
3. A meaningful latent representation (to interpolate between different disease states) [3].

Which model family is the most suitable?

* **Correct Answer:** **Normalising Flow** [3]
* **Explanation:** GANs cannot evaluate exact likelihoods, and KDE scales poorly on high-dimensional image data [4]. Only Normalizing Flows satisfy all three requirements simultaneously [3].