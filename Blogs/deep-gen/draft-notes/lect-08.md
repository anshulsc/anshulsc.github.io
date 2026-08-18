# Deep Generative Models: Latent Variable Models & Variational Autoencoders

---

## Introduction to Latent Variable Models (LVMs)

When modeling real-world data, the observed variables $x$ are often high-dimensional and complex (e.g., images, audio signals, or text). Generative modeling aims to find a probability distribution $P_\theta(x)$ that closely approximates the true data-generating distribution $P_x$. 

Directly modeling $P_\theta(x)$ is often challenging. **Latent Variable Models (LVMs)** address this by introducing unobserved, hidden random variables $z$ that capture the underlying factors of variation in the data.

### Mathematical Formulation
Given a dataset:
$$D = \{x_1, x_2, \dots, x_n\} \quad \text{where} \quad x_i \overset{\text{i.i.d.}}{\sim} P_x$$

We define a model parameterized by $\theta$. In an LVM, the marginal distribution of the observed data $P_\theta(x)$ is defined by integrating or summing out the latent variables $z$ from the joint distribution $P_\theta(x, z)$:

*   **If $z$ is discrete:**
    $$P_\theta(x) = \sum_{z} P_\theta(x, z)$$
*   **If $z$ is continuous:**
    $$P_\theta(x) = \int_{z} P_\theta(x, z) \, dz$$

Here, $z$ acts as a hidden/unobserved representation. For every observed data point $x_i$, there exists an associated unobserved latent variable $z_i$.

---

### Discrete vs. Continuous Latent Spaces

Depending on the nature of the latent variable $z$, LVMs can be categorized into two paradigms:

#### 1. Discrete Latent Spaces (Clustering Paradigm)
When $z$ is discrete, it can take one of $M$ categories: $z \in \{1, 2, \dots, M\}$. Given a data point $x_i \in D$, the conditional distribution $z_i \mid x_i$ tells us which of the $M$ categories or clusters $x_i$ belongs to. 
*   **Examples:** Gaussian Mixture Models (GMMs) and $K$-means clustering.

#### 2. Continuous Latent Spaces (Dimensionality Reduction Paradigm)
When $z$ is continuous (e.g., $z \in \mathbb{R}^K$), it acts as a low-dimensional embedding or feature vector representing $x$.
*   **Examples:** Principal Component Analysis (PCA), Autoencoders, and Variational Autoencoders (VAEs).

---

## The Optimization Challenge & The Derivation of ELBO

The primary objective of learning an LVM is to estimate the parameters $\theta$ such that the Kullback-Leibler (KL) divergence between the true data distribution $P_x$ and our model $P_\theta$ is minimized:

$$\min_\theta D_{\text{KL}}(P_x \parallel P_\theta)$$

Minimizing this KL divergence is mathematically equivalent to maximizing the expected log-likelihood of the observed data:

$$\theta^* = \arg\max_\theta \mathbb{E}_{P_x} [\log P_\theta(x)]$$

### The Intractability Bottleneck
Let us write down the log-likelihood for a single data point $x$:

$$l(\theta) = \log P_\theta(x) = \log \int_{z} P_\theta(x, z) \, dz$$

Computing this marginal log-likelihood directly is generally **intractable** for complex models. If $z$ is continuous and high-dimensional, or if the relationship between $z$ and $x$ is highly non-linear (as parameterized by a deep neural network), we cannot compute the integral over $z$ analytically, nor can we easily compute the true posterior distribution:

$$P_\theta(z \mid x) = \frac{P_\theta(x, z)}{P_\theta(x)}$$

### Introducing the Variational Distribution $q(z \mid x)$
To resolve this intractability, we introduce an arbitrary, tractable variational distribution $q(z \mid x)$ (often called the *variational latent posterior* or *encoder*) to approximate the true intractable posterior $P_\theta(z \mid x)$. 

We rewrite the marginal log-likelihood by multiplying and dividing the integrand by $q(z \mid x)$:

$$l(\theta) = \log \int_{z} P_\theta(x, z) \frac{q(z \mid x)}{q(z \mid x)} \, dz$$

Expressing this as an expectation over the distribution $q(z \mid x)$:

$$l(\theta) = \log \mathbb{E}_{q(z \mid x)} \left[ \frac{P_\theta(x, z)}{q(z \mid x)} \right]$$

### Applying Jensen's Inequality
Since the logarithm is a concave function, we can apply **Jensen's Inequality** ($\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$) to establish a lower bound:

$$l(\theta) \ge \mathbb{E}_{q(z \mid x)} \left[ \log \frac{P_\theta(x, z)}{q(z \mid x)} \right] = \mathcal{F}_\theta(q)$$

This lower bound, $\mathcal{F}_\theta(q)$, is known as the **Evidence Lower Bound (ELBO)**. 

$$\log P_\theta(x) \ge \text{ELBO}$$

Instead of maximizing the intractable log-likelihood $l(\theta)$ directly, we optimize this lower bound over both the model parameters $\theta$ and the variational parameters of $q$.

---

### Student Doubt 1: Expectations and the Variational Distribution

> **Raghavendra (Student):** *"This $q(z \mid x)$ is also unknown, right? So how do we compute the expectation?"*

**Instructor's Explanation:**
At the beginning, both the model parameters $\theta$ and the variational distribution $q(z \mid x)$ are indeed unknown. We handle this by setting up a joint optimization problem over both variables:

$$\theta^*, q^* = \arg\max_{\theta, q} \text{ELBO}(\theta, q)$$

Depending on the complexity of the model, we can solve this in different ways. 
* In simpler models (like GMMs), we can alternate optimization using the **Expectation-Maximization (EM)** algorithm, where the optimal $q$ can be solved analytically at each step.
* In deep neural network models (like VAEs), we parameterize $q$ using a second neural network with parameters $\phi$, writing it as $q_\phi(z \mid x)$, and optimize $\theta$ and $\phi$ jointly using gradient-based methods.

---

## Classical Latent Models: Expectation-Maximization (EM) & GMMs

When the latent space is discrete and the conditional distributions are simple, we can compute the true posterior analytically. This is the foundation of the **Gaussian Mixture Model (GMM)** trained via the **Expectation-Maximization (EM)** algorithm.

### Gaussian Mixture Models (GMM)
A GMM models the data as a linear combination of $M$ different Gaussian distributions:

$$P_\theta(x) = \sum_{z=1}^{M} P_\theta(z) P_\theta(x \mid z)$$

$$\text{Where } P_\theta(z = j) = \alpha_j \quad \text{and} \quad P_\theta(x \mid z = j) = \mathcal{N}(x; \mu_j, \Sigma_j)$$

Thus:
$$P_\theta(x) = \sum_{j=1}^{M} \alpha_j \mathcal{N}(x; \mu_j, \Sigma_j)$$

The complete set of parameters is $\theta = \{\alpha_1, \dots, \alpha_M, \mu_1, \dots, \mu_M, \Sigma_1, \dots, \Sigma_M\}$, subject to $\sum_j \alpha_j = 1$ and $\alpha_j \ge 0$.

### The EM Optimization Loop
The EM algorithm alternates between two steps to maximize the ELBO:

1.  **E-Step (Expectation):** Fix $\theta$ and find the optimal $q$ that maximizes the ELBO. It can be mathematically shown that the optimal variational distribution $q^*(z \mid x)$ is exactly the true posterior distribution $P_\theta(z \mid x)$:
    $$q^*(z = j \mid x) = P_\theta(z = j \mid x)$$
    Using Bayes' rule, this is highly tractable for GMMs:
    $$P_\theta(z = j \mid x) = \frac{P_\theta(x \mid z = j) P_\theta(z = j)}{\sum_{k=1}^M P_\theta(x \mid z = k) P_\theta(z = k)} = \frac{\alpha_j \mathcal{N}(x; \mu_j, \Sigma_j)}{\sum_{k=1}^M \alpha_k \mathcal{N}(x; \mu_k, \Sigma_k)}$$
2.  **M-Step (Maximization):** Fix $q$ and maximize the ELBO with respect to $\theta$:
    $$\theta_{t+1} = \arg\max_\theta \mathbb{E}_{q^*(z \mid x)} [\log P_\theta(x, z)]$$

---

### Student Doubt 2: Tractability of $P_\theta(x \mid z)$

> **Harish (Student):** *"But is $P_\theta(x \mid z)$ also unknown?"*

**Instructor's Explanation:**
No, $P_\theta(x \mid z)$ is defined by design. In a Gaussian Mixture Model, once you specify which mixture component $z = j$ you are looking at, the distribution over $x$ is simply a single-component Gaussian $\mathcal{N}(x; \mu_j, \Sigma_j)$. Because the form of this distribution is chosen by the modeler, its likelihood is fully known and easily computed given the parameters $\mu_j$ and $\Sigma_j$.

---

## Moving to Deep Latent Models: Variational Autoencoders (VAEs)

While EM is powerful, it relies on being able to compute the true posterior $P_\theta(z \mid x)$ analytically. In deep learning, we often want to model complex, highly non-linear relationships. For example, we might define:

$$P_\theta(x \mid z) = \mathcal{N}(x; g_\theta(z), \sigma^2 I)$$

where $g_\theta(z)$ is a deep neural network (often called the **Decoder**). 

When $g_\theta(z)$ is non-linear, the posterior $P_\theta(z \mid x)$ is no longer analytical or tractable. Thus, the classical EM algorithm cannot be applied. This limitation is what motivated Kingma & Welling (2013) to introduce the **Variational Autoencoder (VAE)** framework in their seminal paper, *Auto-Encoding Variational Bayes*.

### Three Main Goals of a Latent Variable Model (per VAE Paper)
The VAE framework is designed to simultaneously solve three challenges:

1.  **Efficient Parameter Estimation ($\theta$):** Learning the generative parameters $\theta$ to maximize the marginal likelihood over large datasets.
2.  **Efficient Posterior Inference:** Approximating the true latent posterior distribution $P_\theta(z \mid x)$ given an observed value $x$. This is useful for representation learning or dimensionality reduction.
3.  **Efficient Marginal Inference:** Allowing us to perform tasks like generation (sampling new data points), denoising, inpainting, and super-resolution.

---

## Decomposing the ELBO Loss Function

To train both the generative model $P_\theta(x \mid z)$ and the variational model $q_\phi(z \mid x)$, we rewrite the ELBO formulation to highlight its structural components. 

Recall the ELBO:
$$\mathcal{F}_\theta(q) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{P_\theta(x, z)}{q_\phi(z \mid x)} \right]$$

Since $P_\theta(x, z) = P_\theta(x \mid z) P(z)$, where $P(z)$ is the prior distribution over the latent variables (typically assumed to be a standard normal distribution $\mathcal{N}(0, I)$), we can write:

$$\mathcal{F}_{\theta, \phi}(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{P_\theta(x \mid z) P(z)}{q_\phi(z \mid x)} \right]$$

Using the properties of logarithms:
$$\mathcal{F}_{\theta, \phi}(x) = \mathbb{E}_{q_\phi(z \mid x)} [ \log P_\theta(x \mid z) ] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{P(z)}{q_\phi(z \mid x)} \right]$$

The second term can be recognized as the negative KL divergence:
$$\mathcal{F}_{\theta, \phi}(x) = \mathbb{E}_{q_\phi(z \mid x)} [ \log P_\theta(x \mid z) ] - D_{\text{KL}}(q_\phi(z \mid x) \parallel P(z))$$

Thus, the VAE training objective (maximizing the ELBO) consists of two competing terms:

$$\max_{\theta, \phi} \left( \underbrace{\mathbb{E}_{q_\phi(z \mid x)} [ \log P_\theta(x \mid z) ]}_{\text{Reconstruction Term}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \parallel P(z))}_{\text{Regularization Term}} \right)$$

*   **Reconstruction Term:** Encourages the decoder to reconstruct the input $x$ accurately from its latent representation $z$.
*   **Regularization Term:** Forces the approximated posterior $q_\phi(z \mid x)$ to be close to the prior $P(z)$, preventing the latent space from collapsing or overfitting to individual data points.

---

### Student Doubt 3: Understanding the KL Divergence Notation

> **Harish (Student):** *"For this to be KL divergence, shouldn't there be a $q_\phi(z \mid x)$ term also before the logarithm?"*

**Instructor's Explanation:**
Yes, the expectation operator $\mathbb{E}_{q_\phi(z \mid x)}$ implicitly contains that term. Let us write down the expectation explicitly as an integral:

$$\mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{P(z)}{q_\phi(z \mid x)} \right] = \int_{z} q_\phi(z \mid x) \log \left( \frac{P(z)}{q_\phi(z \mid x)} \right) \, dz$$

If we pull a negative sign out of the logarithm:

$$- \int_{z} q_\phi(z \mid x) \log \left( \frac{q_\phi(z \mid x)}{P(z)} \right) \, dz = - D_{\text{KL}}(q_\phi(z \mid x) \parallel P(z))$$

So the $q_\phi$ term is indeed present—it is provided by the definition of the expectation over $q_\phi$.

---

## Representing Distributions via Neural Networks

How do neural networks (which are deterministic mapping functions) represent probability distributions? There are two primary paradigms:

```
                  ┌────────────────────────────────────────┐
                  │          PROBABILISTIC WAY             │
                  │  NN outputs distribution parameters    │
                  │  e.g., x ──► [ NN ] ──► (μ, σ)         │
                  └────────────────────────────────────────┘

                  ┌────────────────────────────────────────┐
                  │          DETERMINISTIC WAY             │
                  │  NN acts as direct mapping function    │
                  │  e.g., (z ~ N(0,I)) ──► [ NN ] ──► x̂   │
                  └────────────────────────────────────────┘
```

### 1. The Probabilistic Paradigm
Instead of predicting a value directly, the neural network predicts the *parameters* (like the mean $\mu$ and covariance $\Sigma$) of a family of distributions.
*   **Encoder ($q_\phi(z \mid x)$):** Takes $x$ as input, and outputs $\mu_\phi(x)$ and $\sigma_\phi(x)$. We then define our distribution over the latent space as:
    $$q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

### 2. The Deterministic Paradigm
The neural network itself acts as a mapping function that transforms a random noise vector into a sample.
*   **Decoder ($P_\theta(x \mid z)$) / GAN Generator:** Takes a sampled latent vector $z$ and directly maps it to an output sample $\hat{x} = g_\theta(z)$. 

---

### Student Doubt 4: Stochasticity in Reconstructed Outputs

> **Lokesh (Student):** *"For a given $x_i$, will the reconstructed output $\hat{x}_i$ from the decoder be different every time we run it? Because we sample $z_i$ in between, which might change."*

**Instructor's Explanation:**
Yes, that is correct. Because $z_i$ is sampled stochastically from the distribution $q_\phi(z \mid x_i)$, different forward passes for the exact same input $x_i$ will yield slightly different latent vectors $z_i$. Consequently, the reconstructed output $\hat{x}_i = g_\theta(z_i)$ will also vary slightly between forward passes.

---

### Student Doubt 5: Executing the Stochastic Sampling Step

> **Harish (Student):** *"How is this sampling of $z$ from $q$ practically executed? Is it like calling a random function?"*

**Instructor's Explanation:**
Yes, under the hood we use standard pseudo-random number generators (like `torch.randn` or `numpy.random.normal`) to sample from our distribution. 

However, doing this directly introduces a major problem: **sampling is a non-differentiable operation**. If we sample $z$ directly from $\mathcal{N}(\mu, \Sigma)$, we cannot compute gradients with respect to $\phi$ (the encoder parameters) because the stochastic node blocks the backpropagation path. 

To solve this, we must use a technique called the **Reparameterization Trick**, which isolates the stochasticity and allows us to train the entire network end-to-end. We will cover this in detail right after the break.

---

## Architectural Summary of a Variational Autoencoder (VAE)

Below is a conceptual layout of the VAE training pipeline:

```
   Observed Input (x)
         │
         ▼
 ┌───────────────┐
 │    ENCODER    │  Parameterized by neural network weights φ
 └───────────────┘
    │         │
    ▼         ▼
  Mean(μ)  Variance(σ)
    │         │
    └────┬────┘
         ▼
   Sampling Step ──◄── Stochastic Noise (ε) sampled from N(0, I)
         │
         ▼
   Latent Rep. (z)
         │
         ▼
 ┌───────────────┐
 │    DECODER    │  Parameterized by neural network weights θ
 └───────────────┘
         │
         ▼
 Reconstructed Output (x̂)
```

1.  **Encoder:** Evaluates the input $x$ and outputs the parameters $\mu(x)$ and $\sigma(x)$.
2.  **Sampling Step:** A latent vector $z$ is sampled from $\mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$.
3.  **Decoder:** Takes the sampled $z$ and reconstructs the data point as $\hat{x}$.

During optimization, the parameters $\theta$ (decoder) and $\phi$ (encoder) are jointly updated via gradient descent to maximize the ELBO.

## Variational Autoencoders (VAEs): Reparameterization, ELBO Optimization, and Training Dynamics

In deep generative modeling, training variational latent variable models requires optimizing the Evidence Lower Bound (ELBO) with respect to both generative parameters ($\theta$) and variational parameters ($\phi$). This post covers the mathematical formulation of ELBO gradients, the necessity of the reparameterization trick, the practical implementation of VAE training, and an in-depth recap of the student-teacher interactions from the lecture.

---

### 1. The Core Objective: Computing ELBO Gradients

The Evidence Lower Bound (ELBO), denoted as $\mathcal{F}_{\theta}(q_{\phi})$, serves as our optimization objective. It is expressed as:

$$\mathcal{F}_{\theta}(q_{\phi}) = \mathbb{E}_{q_{\phi}(z|x)} [ \log p_{\theta}(x|z) ] - D_{KL}(q_{\phi}(z|x) \parallel p(z))$$

To optimize this objective using gradient-based methods, we must compute the gradients of $\mathcal{F}_{\theta}(q_{\phi})$ with respect to the encoder parameters $\phi$ and the decoder parameters $\theta$. 

The gradient with respect to the decoder parameters $\theta$ only affects the first term (the reconstruction term), as the Kullback-Leibler (KL) divergence term does not depend on $\theta$:

$$\nabla_{\theta} \mathcal{F}_{\theta}(q_{\phi}) = \mathbb{E}_{q_{\phi}(z|x)} [ \nabla_{\theta} \log p_{\theta}(x|z) ]$$

This expectation can be easily approximated using Monte Carlo sampling. However, computing the gradient with respect to the encoder parameters $\phi$ is significantly more challenging because the parameter $\phi$ determines the distribution over which the expectation is taken.

---

### 2. Why Naive Gradients Fail

To understand why a naive gradient computation fails, let us abstract the first term of the ELBO. Let the parameter of interest be $\psi$ (where $\psi = \phi$), the random variable be $v$ (where $v = z$), and the function of the random variable be $f_{\psi}(v)$ (where $f_{\psi}(v) = \log p_{\theta}(x|z)$). We wish to compute the gradient:

$$\nabla_{\psi} \mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ]$$

Expanding the expectation into its integral form:

$$\nabla_{\psi} \mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] = \nabla_{\psi} \int_{v} p_{\psi}(v) f_{\psi}(v) \, dv$$

Applying the Leibniz rule (moving the derivative inside the integral under standard regularity conditions) and the product rule for differentiation, we obtain:

$$\nabla_{\psi} \int_{v} p_{\psi}(v) f_{\psi}(v) \, dv = \int_{v} \left( \nabla_{\psi} p_{\psi}(v) \cdot f_{\psi}(v) + p_{\psi}(v) \cdot \nabla_{\psi} f_{\psi}(v) \right) \, dv$$

We can split this integral into two distinct terms:

$$\nabla_{\psi} \mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] = \int_{v} p_{\psi}(v) \nabla_{\psi} f_{\psi}(v) \, dv + \int_{v} f_{\psi}(v) \nabla_{\psi} p_{\psi}(v) \, dv$$

1. **The First Term:** $\int_{v} p_{\psi}(v) \nabla_{\psi} f_{\psi}(v) \, dv = \mathbb{E}_{p_{\psi}(v)} [ \nabla_{\psi} f_{\psi}(v) ]$. This is a standard expectation and can be easily approximated using Monte Carlo samples drawn from $p_{\psi}(v)$.
2. **The Second Term:** $\int_{v} f_{\psi}(v) \nabla_{\psi} p_{\psi}(v) \, dv$. Because the gradient operator is acting directly on the probability density function $p_{\psi}(v)$ rather than a function of $v$, this integral cannot be directly expressed as an expectation over $p_{\psi}(v)$. 

While we can apply the score function estimator (or log-derivative trick, also known as the REINFORCE algorithm in reinforcement learning) to rewrite this as $\mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) \nabla_{\psi} \log p_{\psi}(v) ]$, the resulting gradient estimator suffers from extremely high variance, making it impractical for stable deep neural network training.

---

### 3. The Savior: The Reparameterization Trick

The **Reparameterization Trick** bypasses this high-variance issue by expressing the stochastic latent variable $v$ as a deterministic function of a parameter-free noise variable $\epsilon$ and the distribution parameters.

Assume we can find an auxiliary noise variable $\epsilon$ distributed according to some base distribution $p_{\epsilon}$ (which does not depend on $\psi$), and a deterministic, differentiable mapping $g(\epsilon, \psi)$ such that:

$$v = g(\epsilon, \psi), \quad \epsilon \sim p_{\epsilon}$$

Using the **Law of the Unconscious Statistician (LOTUS)**, we can rewrite the expectation over $p_{\psi}(v)$ as an expectation over $p_{\epsilon}$:

$$\mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] = \mathbb{E}_{p_{\epsilon}} [ f_{\psi}(g(\epsilon, \psi)) ]$$

Now, when we take the gradient with respect to $\psi$:

$$\nabla_{\psi} \mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] = \nabla_{\psi} \mathbb{E}_{p_{\epsilon}} [ f_{\psi}(g(\epsilon, \psi)) ]$$

Since the expectation is now taken over the parameter-free distribution $p_{\epsilon}$, we can move the gradient operator inside the expectation:

$$\nabla_{\psi} \mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] = \mathbb{E}_{p_{\epsilon}} [ \nabla_{\psi} f_{\psi}(g(\epsilon, \psi)) ]$$

Applying the chain rule for differentiation, we can compute this gradient directly. This expectation can be approximated using Monte Carlo sampling:

$$\nabla_{\psi} \mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] \approx \frac{1}{M} \sum_{j=1}^{M} \nabla_{\psi} f_{\psi}(g(\epsilon_j, \psi)), \quad \epsilon_j \sim p_{\epsilon}$$

---

### 4. Reparameterization in Variational Autoencoders

In a standard VAE, we assume the approximate posterior (encoder distribution) is a multivariate Gaussian with a diagonal covariance matrix:

$$q_{\phi}(z|x) = \mathcal{N}(z; \mu_{\phi}(x), \Sigma_{\phi}(x))$$

Using the reparameterization trick, we choose our base distribution to be a standard normal distribution:

$$\epsilon \sim \mathcal{N}(0, I)$$

We can then express the latent representation $z$ deterministically as:

$$z = g(\epsilon, \phi) = \mu_{\phi}(x) + \Sigma_{\phi}^{1/2}(x) \odot \epsilon$$

where $\odot$ denotes element-wise multiplication. This formulation isolates the stochasticity in the parameter-free variable $\epsilon$, allowing gradients to flow back from the decoder through $z$ to the encoder parameters $\phi$ via standard backpropagation.

#### Alternative Reparameterization: The Inverse CDF Method
While Gaussian reparameterization is the most common approach, the lecture also noted the **Inverse Cumulative Distribution Function (Inverse CDF) Method** for completeness. 

If $u \sim \text{Unif}[0, 1]$, and $F^{-1}(u)$ is the inverse CDF of the target distribution, then the random variable $x = F^{-1}(u)$ will follow the target distribution. Here, the noise variable is $\epsilon = u \sim \text{Unif}[0, 1]$, and our transformation function is $g(u) = F^{-1}(u)$. This method requires the CDF of our target distribution to be easily invertible, which is why Gaussian reparameterization remains more popular in practical VAE implementations.

---

### 5. Training the VAE: Step-by-Step

A VAE is trained by alternating or jointly performing gradient updates on both the encoder ($\phi$) and the decoder ($\theta$).

```
  +---------+             +-----------------------+              +---------+
  |         |      x      |    Encoder (q_phi)    |  mu, sigma   |         |
  |  Input  |------------>|                       |------------->|  Loss   |
  |         |             |  Outputs: mu, sigma   |              |  (KL)   |
  +---------+             +-----------------------+              +---------+
       |                              |                               ^
       |                              | (Reparameterize)              |
       |                              v                               |
       |                            +---+                             |
       |                            | z |                             |
       |                            +---+                             |
       |                              |                               |
       |                              v                               |
       |                  +-----------------------+                   |
       |                  |    Decoder (p_theta)  |                   |
       |                  |                       |                   |
       |                  |  Output: log p(x|z)   |                   |
       |                  +-----------------------+                   |
       |                              |                               |
       +----------------------------->v-------------------------------+
                                    Loss 
                               (Reconstruction)
```

#### A. The Reconstruction Loss (Decoder Training)
To compute the reconstruction loss, we make a parametric assumption on the decoder output. Let us assume the generative distribution $p_{\theta}(x|z)$ is Gaussian:

$$p_{\theta}(x|z) = \mathcal{N}(x; \mathcal{D}_{\theta}(z), I)$$

where $\mathcal{D}_{\theta}(z)$ is the output of the decoder network. Taking the log of this distribution:

$$\log p_{\theta}(x|z) = \log \left[ \frac{1}{(2\pi)^{d/2}} \exp \left( -\frac{1}{2} \|x - \mathcal{D}_{\theta}(z)\|^2_2 \right) \right] \propto -\|x - \mathcal{D}_{\theta}(z)\|^2_2$$

Thus, maximizing the log-likelihood term is equivalent to minimizing the squared error reconstruction loss between the input $x$ and the reconstructed output $\mathcal{D}_{\theta}(z)$.

To train the decoder parameters $\theta$:
1. Pass $x_i$ through the encoder to obtain $\mu_{\phi}(x_i)$ and $\Sigma_{\phi}(x_i)$.
2. Sample $\epsilon_j \sim \mathcal{N}(0, I)$ and construct $z_j = \mu_{\phi}(x_i) + \Sigma_{\phi}^{1/2}(x_i) \odot \epsilon_j$.
3. Pass $z_j$ through the decoder to compute the reconstruction $\mathcal{D}_{\theta}(z_j)$.
4. Compute the gradient of the reconstruction loss with respect to $\theta$:
   $$\nabla_{\theta} \mathcal{F}_{\theta}(q_{\phi}) \approx -\nabla_{\theta} \left[ \frac{1}{M} \sum_{j=1}^{M} \|x_i - \mathcal{D}_{\theta}(z_j)\|^2_2 \right]$$
5. Backpropagate the gradients to update $\theta$ while keeping $\phi$ fixed.

#### B. The KL Divergence Loss (Encoder Training)
The second term in the ELBO is the KL divergence between the approximate posterior and the prior: $D_{KL}(q_{\phi}(z|x) \parallel p(z))$. 

Assuming $q_{\phi}(z|x) = \mathcal{N}(z; \mu_{\phi}(x), \Sigma_{\phi}(x))$ and the prior $p(z) = \mathcal{N}(0, I)$, this term has a closed-form analytical solution:

$$D_{KL}(q_{\phi}(z|x) \parallel p(z)) = -\frac{1}{2} \left[ \log |\Sigma_{\phi}(x)| + d - \text{tr}(\Sigma_{\phi}(x)) - \|\mu_{\phi}(x)\|^2_2 \right]$$

Because this term has an analytical solution, we can compute its gradient with respect to $\phi$ directly without using the reparameterization trick.

To train the encoder parameters $\phi$:
1. The encoder is updated using gradients from two sources:
   - **The Reconstruction Gradient:** This is backpropagated from the reconstruction loss, through the decoder, and back to the encoder outputs ($\mu_{\phi}, \Sigma_{\phi}$) via the reparameterized $z_j$.
   - **The KL Divergence Gradient:** This is calculated analytically directly from the encoder outputs.
2. Combine these gradients to update $\phi$ while keeping $\theta$ fixed:
   $$\nabla_{\phi} \mathcal{F}_{\theta}(q_{\phi}) \approx \nabla_{\phi} \left[ \frac{1}{M} \sum_{j=1}^{M} \log p_{\theta}(x_i | g(\epsilon_j, \phi)) \right] - \nabla_{\phi} D_{KL}(q_{\phi}(z|x_i) \parallel p(z))$$

---

### 6. Interactive Q&A: Addressing Common Doubts

Several critical conceptual questions were asked by students during the lecture. Their resolutions clarify the nuances of VAE training.

#### Doubt 1: Why introduce a generic parameter $\psi$ instead of $\phi$?
* **Student's Question:** "Why don't we just use $\phi$ directly instead of introducing $\psi$? It might make things less confusing."
* **Instructor's Answer:** The instructor explained that using $\psi$ was purely for algebraic generality. In the general proof, we want to understand how to differentiate the expectation of *any* function $f_{\psi}(v)$ where the distribution itself is parameterized by $\psi$. Mapping this general proof back to our VAE parameters:
  - The parameter $\psi$ corresponds to the encoder parameters $\phi$.
  - The random variable $v$ corresponds to our latent variable $z$.
  - The function $f$ corresponds to $\log p_{\theta}(x|z)$.
  By using a general notation ($\psi, v, f$), we can cleanly prove the theorem without getting bogged down in VAE-specific notations.

#### Doubt 2: How many samples $M$ should we draw for each data point?
* **Student's Question:** "In the equations, we have an average over $M$ samples. Does that mean we pass $M$ different $z$'s through the decoder for a single input $x_i$?"
* **Instructor's Answer:** Yes, that is correct. For a single input image $x_i$, the encoder produces a single mean $\mu$ and covariance $\Sigma$. However, we can sample $M$ different noise vectors $\epsilon_j \sim \mathcal{N}(0, I)$ to generate $M$ different latent representations $z_j$. This highlights the power of probabilistic encoders over deterministic encoders (such as those in standard autoencoders or models like BERT, which output exactly one deterministic embedding per input). 
  - **In practice:** For computational efficiency, most naive VAE implementations set $M = 1$ during training (a single-sample approximation). While $M$ is a tunable hyperparameter, a single sample per data point is often sufficient when training over mini-batches because the stochasticity averages out across the batch.

#### Doubt 3: Are we eliminating sampling during backpropagation?
* **Student's Question:** "By using the reparameterization trick, are we converting a stochastic process into a deterministic one so we can flow gradients?"
* **Instructor's Answer:** No, we are not eliminating sampling. The process remains stochastic because we still sample $\epsilon$ randomly. Instead, we have *re-routed* the path of the gradient. 
  - In the naive formulation, the sampling process itself depended on the parameters we wanted to update ($\phi$), which blocked backpropagation.
  - In the reparameterized formulation, the random sampling comes from a constant distribution ($\mathcal{N}(0, I)$) that has no dependency on $\phi$. The latent variable $z$ is then constructed via a deterministic, differentiable transformation. This allows us to backpropagate through the parameters $\mu$ and $\sigma$ without issues.

#### Doubt 4: VAEs vs. GANs Convergence
* **Student's Question:** "How does the training stability of VAEs compare to GANs?"
* **Instructor's Answer:** VAEs are known to converge much more stably and easily than Generative Adversarial Networks (GANs). 
  - In GANs, training is a minimax game (a saddle-point problem) where the generator and discriminator act as adversaries. This competitive setup can easily lead to mode collapse or training instability.
  - In VAEs, there is no adversary. Both the encoder and decoder work collaboratively to maximize a single joint objective (the ELBO). Since there is no adversarial game, the training dynamics are significantly more stable and robust.

#### Doubt 5: Exam Pattern & Practice Expectations
* **Student's Question:** "Will the upcoming exam contain purely theoretical questions, or will there be mathematical derivations?"
* **Instructor's Answer:** The exam will not focus on purely descriptive, rote-learning questions (e.g., "describe the difference between GANs and VAEs"). Instead, it will be thought-provoking and heavily mathematically oriented. Students are expected to understand the underlying mathematics and be comfortable with derivations, similar to those presented on the blackboard during lectures.

---

### 7. Homework Problems

To prepare for the next lecture and the upcoming exam, complete the following proofs on your own:

1. **LOTUS (Law of the Unconscious Statistician):** Prove that $\mathbb{E}_{p_{\psi}(v)} [ f_{\psi}(v) ] = \mathbb{E}_{p_{\epsilon}} [ f_{\psi}(g(\epsilon, \psi)) ]$ under the change of variables formula for probability densities.
2. **Analytical Gaussian KL Divergence:** Derive the analytical closed-form solution for the KL divergence between $q_{\phi}(z|x) = \mathcal{N}(z; \mu, \Sigma)$ (where $\Sigma$ is diagonal) and $p(z) = \mathcal{N}(0, I)$. Show your step-by-step integration.