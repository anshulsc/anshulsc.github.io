# Study Notes: Deep Generative Models — Lecture on $f$-GANs: Variational Divergence Minimization

---

## Introduction: The Core Challenge of Generative Modeling

In generative modeling, our primary goal is to learn an unknown data distribution, $P_x$, from which we only have a finite set of observed samples:

$$D = \{x_1, x_2, \dots, x_n\} \sim P_x$$

We want to construct a model—a generative sampler—parameterized by $\theta$, which we denote as $g_\theta(z)$. This generator takes an arbitrary, easy-to-sample source of noise (such as a standard Gaussian distribution, $z \sim \mathcal{N}(0, I)$) and maps it to the data space:

$$\hat{x} = g_\theta(z)$$

The distribution of these generated samples is denoted by $P_\theta$ (or $P_g$). Ideally, we want to update the parameters $\theta$ such that $P_\theta$ becomes as close as possible to $P_x$. 

To do this, we need a mathematical framework to measure the distance—or **divergence**—between two probability distributions:

$$\theta^* = \arg\min_\theta D(P_x \parallel P_\theta)$$

---

## Part 1: Measuring Closeness & The Intractability of $f$-Divergences

### What is an $f$-Divergence?
An $f$-divergence is a general family of divergence measures between two probability density functions, $P_x$ and $P_\theta$. Given a convex, lower semi-continuous function $f: \mathbb{R}^+ \to \mathbb{R}$ such that $f(1) = 0$, the $f$-divergence is defined as:

$$D_f(P_x \parallel P_\theta) = \int_{\mathcal{X}} P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$$

By plugging in different convex functions for $f(u)$, we recover well-known divergence metrics:
*   **Kullback-Leibler (KL) Divergence:** $f(u) = u \log u$
*   **Jensen-Shannon (JS) Divergence:** $f(u) = \frac{1}{2} \left( u \log u - (u + 1) \log \frac{u + 1}{2} \right)$
*   **Total Variation Distance:** $f(u) = \frac{1}{2} |u - 1|$

---

### 💡 Student Doubt & Interactive Q&A

**Student (Indrajit):** *Does the generated sample $\hat{x}$ depend on the random input $z$ we provide? And does it matter what distribution we use for $z$?*

**Professor:** Yes, it absolutely depends on $z$. The generator $g_\theta(z)$ is a deterministic function mapping a random variable $z$ to a generated sample $\hat{x}$. Thus, different values of $z$ will map to different generated outputs (for instance, different human faces). 

Regarding the input distribution: you can choose any simple distribution (like a Uniform or Gaussian) as long as you **keep it consistent**. If you train your model using $z \sim \text{Uniform}[0, 1]$, you must use $z \sim \text{Uniform}[0, 1]$ during inference. Changing the input noise distribution at test time will yield poor results because the neural network's mapping is specifically optimized for the training input distribution.

---

**Student (Mukesh):** *If the model is supposed to generate human faces, is there a chance it outputs weird anomalies—like a face with three or four eyes?*

**Professor:** This depends entirely on how well your model is trained. If you successfully minimize the divergence $D_f(P_x \parallel P_\theta)$ to a very low value, your model's distribution $P_\theta$ will closely align with the true data distribution $P_x$. Since human faces in $P_x$ do not have three or four eyes, a well-trained model will not produce such anomalies. However, if your training has not converged or has failed (i.e., the divergence remains high), you will indeed see visual artifacts and unrealistic features.

---

### The Intractability Bottleneck
In practice, we cannot compute the integral for $D_f(P_x \parallel P_\theta)$ directly. Doing so requires knowing the analytical probability density functions $P_x(x)$ and $P_\theta(x)$. In real-world applications, $P_x$ is an unknown, highly complex distribution (e.g., the space of natural images), and $P_\theta$ is implicitly defined by our generator network.

We only have **samples** from these distributions:
*   Samples from $P_x$: Our empirical training dataset $D$.
*   Samples from $P_\theta$: Generated outputs obtained by sampling $z \sim \mathcal{N}(0, I)$ and passing them through $g_\theta(z)$.

### Why Can't We Just Use the Law of Large Numbers (LLN)?
The Law of Large Numbers states that we can approximate an expectation using sample means:

$$\int_{\mathcal{X}} h(x) P_x(x) \, dx = \mathbb{E}_{x \sim P_x}[h(x)] \approx \frac{1}{N} \sum_{i=1}^N h(x_i)$$

However, the definition of $f$-divergence contains the density ratio $\frac{P_x(x)}{P_\theta(x)}$ inside the convex function $f$:

$$\int_{\mathcal{X}} P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$$

Because the densities themselves are embedded within the non-linear function $f$, this expression cannot be written as a simple expectation over $P_x$ or $P_\theta$ that we can directly estimate using only samples. This is where we need a mathematical workaround: **Variational Bounding**.

---

## Part 2: Variational Bounding via Convex Conjugates

To make the $f$-divergence estimable from samples, we leverage the concept of the **Fenchel Conjugate** (also known as the convex conjugate).

### Conjugate of a Convex Function
For any convex, lower semi-continuous function $f(u)$, its convex conjugate $f^*(t)$ is defined as:

$$f^*(t) = \sup_{u \in \text{dom } f} \{ ut - f(u) \}$$

By the property of duality (since the conjugate of a conjugate of a convex function is the function itself), we can write the inverse relationship:

$$f(u) = \sup_{t \in \text{dom } f^*} \{ ut - f^*(t) \}$$

### 💡 Student Doubt & Interactive Q&A

**Student (Indrajit):** *If we draw the graph of the conjugate of a convex function, what does it look like? Will it be the inverse of the original function?*

**Professor:** It is not the algebraic inverse (like $f^{-1}(x)$), but rather a dual representation. Geometrically, it represents the maximum discrepancy between the linear function $ut$ and the function $f(u)$. Crucially, the conjugate function $f^*(t)$ is always convex, regardless of whether $f(u)$ is convex, though here we deal specifically with convex $f(u)$.

---

### Deriving the Variational Lower Bound
Let us substitute this dual representation of $f(u)$ back into the definition of $f$-divergence by setting $u = \frac{P_x(x)}{P_\theta(x)}$:

$$D_f(P_x \parallel P_\theta) = \int_{\mathcal{X}} P_\theta(x) \left( \sup_{t} \left\{ \frac{P_x(x)}{P_\theta(x)} t - f^*(t) \right\} \right) dx$$

Instead of finding a single scalar $t$ for the supremum, we can define $t$ as a function of $x$. Let $T(x)$ be an arbitrary function mapping from our data space $\mathcal{X}$ to $\mathbb{R}$. By replacing the scalar supremum with a supremum over a class of functions $\mathcal{T}$, we establish a lower bound:

$$D_f(P_x \parallel P_\theta) \geq \sup_{T \in \mathcal{T}} \int_{\mathcal{X}} P_\theta(x) \left\{ \frac{P_x(x)}{P_\theta(x)} T(x) - f^*(T(x)) \right\} dx$$

Distributing the $P_\theta(x)$ term inside the integrand yields:

$$D_f(P_x \parallel P_\theta) \geq \sup_{T \in \mathcal{T}} \left\{ \int_{\mathcal{X}} P_x(x) T(x) \, dx - \int_{\mathcal{X}} P_\theta(x) f^*(T(x)) \, dx \right\}$$

This is a breakthrough! Both integrals are now expressed as expectations over our distributions $P_x$ and $P_\theta$:

$$D_f(P_x \parallel P_\theta) \geq \sup_{T \in \mathcal{T}} \left\{ \mathbb{E}_{x \sim P_x}[T(x)] - \mathbb{E}_{\hat{x} \sim P_\theta}[f^*(T(\hat{x}))] \right\}$$

Because the objective is formulated entirely in terms of expectations, we can use the Law of Large Numbers to estimate this lower bound using only samples.

---

### 💡 Student Doubt & Interactive Q&A

**Student (Aditya):** *Professor, on the blackboard in the previous steps, there seems to be a minor typo in the parenthetical grouping for the integration terms before the expectation. Shouldn't the closing bracket for the integrand be placed before the $dx$?*

**Professor:** Excellent catch! Yes, the integration variable $dx$ must apply to the entire boundary terms. The corrected line should read:

$$\int_{\mathcal{X}} P_\theta(x) T(x) \frac{P_x(x)}{P_\theta(x)} dx - \int_{\mathcal{X}} P_\theta(x) f^*(T(x)) dx$$

Which simplifies to:

$$\int_{\mathcal{X}} P_x(x) T(x) dx - \int_{\mathcal{X}} P_\theta(x) f^*(T(x)) dx$$

Thank you for pointing that out.

---

**Student (Harish):** *Does the function $T(x)$ itself have to be a convex function?*

**Professor:** No, $T(x)$ does not need to be convex. Only the function $f(u)$ from our $f$-divergence definition must be convex to guarantee the Fenchel duality holds. $T(x)$ is simply a variational function which we will parameterize using a neural network. It can be any arbitrary continuous function.

---

## Part 3: The Adversarial Minimax Saddle-Point Game

Now that we have a lower bound expressed as expectations, we can implement it using neural networks. We parameterize two separate networks:

1.  **The Generator:** $g_\theta(z)$, which implicitly defines $P_\theta$ by mapping $z \sim P_z$ to $\hat{x}$.
2.  **The Discriminator (or Critic):** $T_w(x)$, which parameterizes our variational function class $T \in \mathcal{T}$.

Replacing the arbitrary function class $\mathcal{T}$ with our neural network $T_w$, our lower bound objective function $V(\theta, w)$ is written as:

$$V(\theta, w) = \mathbb{E}_{x \sim P_x}[T_w(x)] - \mathbb{E}_{z \sim P_z}[f^*(T_w(g_\theta(z)))]$$

To get the tightest possible lower bound on the divergence, we want to maximize this objective with respect to the discriminator's parameters $w$:

$$\max_w V(\theta, w)$$

To minimize the divergence between our generated distribution and the true data distribution, we want to minimize this objective with respect to the generator's parameters $\theta$:

$$\theta^* = \arg\min_\theta \max_w V(\theta, w)$$

This formulation is a **Saddle-Point Optimization Problem** (commonly referred to as a minimax game).

```
                      Cost V(θ, w)
                           ▲
                           │     ▲ (Maximize w - Discriminator)
                           │    ╱ ╲
                           │   ╱   ╲
     ──────────────────────┼──╱─────╲────────────────► w
                           │  ╲     ╱
                           │   ╲   ╱
                           │    ▼ (Minimize θ - Generator)
                           │
```

---

### 💡 Student Doubt & Interactive Q&A

**Student (Siddharth):** *If we parameterize our variational function class using a neural network $T_w(x)$, how does that actually work? For example, if the target function we want to represent is a cubic polynomial like $y = 2x^3 + 3x + 5$, how does the neural network represent it?*

**Professor:** That comes down to the **Universal Approximation Theorem**. Deep neural networks with non-linear activation functions (like ReLU or Sigmoid) are universal function approximators. 

By adjusting the weights $w$ and biases through training, the network can approximate any continuous function—including a cubic polynomial—to any arbitrary degree of accuracy over a compact domain. You would optimize those parameters by minimizing the mean squared error between the network's predictions and the true polynomial values.

---

**Student (Raghavendra):** *What is the output dimensionality of this discriminator network $T_w(x)$?*

**Professor:** The output dimension of $T_w(x)$ depends on the choice of the $f$-divergence, but it is typically a scalar (a real number). For example, in the original GAN formulation (which corresponds to the Jensen-Shannon divergence), the discriminator outputs a scalar value representing the classification probability (or logit) of whether the input sample $x$ is real or generated.

---

## Part 4: Behind the Scenes — Why Math Matters

### A Pedagogical Note from the Professor
During the lecture, the professor addressed some student feedback regarding the heavy emphasis on mathematics in the course:

> *"I recently received anonymous student feedback where someone noted that they are paying a high credit fee (around 20,000 rupees per credit at IISc) and expected more practical coding rather than mathematical derivations. While I completely understand the desire for practical application, you must realize that **there is no meaningful practice without theory**.*
>
> *Everyone and their grandmother is a 'data scientist' these days. What actually differentiates a highly skilled professional from someone who just copies APIs is a deep, fundamental understanding of **why** these algorithms work. Math provides that grounded foundation. Rest assured, all your assignments will be practical coding tasks, but in this class, we will always build our understanding from first principles."*

### Chronology of GANs vs. $f$-GANs
The historical timeline of these models illustrates why this mathematical grounding is so valuable:
*   **2014:** The original Generative Adversarial Network (GAN) was proposed. It relied heavily on intuitive, empirical formulations (competing networks playing a cat-and-mouse game) with specific proofs tailored mostly to the Jensen-Shannon divergence.
*   **2016:** The $f$-GAN paper was published. By introducing Fenchel duality and variational divergence minimization, the authors provided a unified mathematical framework. This proved that the original GAN was merely a single instance of a much larger family of variational divergence estimators.

By understanding the unified math of $f$-GANs, we gain the ability to swap in any convex function $f$ and design customized generative objectives tailored to specific tasks, rather than being limited to the original 2014 heuristic.

---

## Summary Cheat Sheet for Recall

| Concept                     | Mathematical Formula / Definition                                                                                                     | Purpose                                                                                 |
| :----------------------------| :--------------------------------------------------------------------------------------------------------------------------------------| :----------------------------------------------------------------------------------------|
| **$f$-Divergence**          | $D_f(P_x \parallel P_\theta) = \int P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$                                          | Quantifies the distance between two probability distributions.                          |
| **Convex Conjugate**        | $f^*(t) = \sup_u \{ ut - f(u) \}$                                                                                                     | Provides a dual representation to linearize the density ratio inside $f(u)$.            |
| **Variational Lower Bound** | $D_f(P_x \parallel P_\theta) \geq \sup_{T} \{ \mathbb{E}_{x \sim P_x}[T(x)] - \mathbb{E}_{\hat{x} \sim P_\theta}[f^*(T(\hat{x}))] \}$ | Converts an intractable integral into expectations estimable via sample averages (LLN). |
| **Minimax Game Objective**  | $\min_\theta \max_w \left( \mathbb{E}_{x \sim P_x}[T_w(x)] - \mathbb{E}_{z \sim P_z}[f^*(T_w(g_\theta(z)))] \right)$                  | Formulates GAN training as a saddle-point optimization problem.                         |
# Deep Generative Models: From $f$-Divergence to Variational GAN Training

## Introduction: The Unified Framework of Generative Adversarial Networks

In this lecture, we bridge the gap between abstract statistical distances—specifically $f$-divergences—and the highly practical, adversarial world of Generative Adversarial Networks (GANs). We show how the classic "vanilla" GAN is not an isolated heuristic, but rather a mathematically principled special case of a broader family of generative models called **$f$-GANs**. 

Historically, GANs were introduced as a minimax game where a generator and a discriminator compete. Here, we demonstrate that by carefully choosing our generator function $f(u)$ within the $f$-divergence framework, we naturally recover the exact loss functions, network architectures, and training dynamics of vanilla adversarial learning.

---

## 1. An Instance of $f$-GAN: Deriving the Vanilla GAN Loss

To instantiate a concrete GAN model from the general $f$-divergence formulation, we choose a specific convex function $f(u)$. This function serves as the generator of our divergence metric.

### The Mathematical Formulation

Let us define the convex, lower-semicontinuous function $f(u)$ as:
$$f(u) = u \log u - (u+1) \log(u+1)$$

In the general $f$-GAN framework, the adversarial objective function $J(\theta, \omega)$ is expressed using the Fenchel conjugate $f^*$:
$$J(\theta, \omega) = \mathbb{E}_{x \sim P_x}[T_\omega(x)] - \mathbb{E}_{\hat{x} \sim P_\theta}[f^*(T_\omega(x))]$$

where:
* $x \sim P_x$ represents real samples from our data distribution.
* $\hat{x} \sim P_\theta$ represents fake/generated samples produced by our generator model $G_\theta(z)$.
* $T_\omega(x)$ is the variational function (the critic network) parameterized by weights $\omega$.

By plugging the convex conjugate $f^*(t) = -\log(1 - e^t)$ associated with our choice of $f(u)$ into the objective, we obtain the objective function of the vanilla GAN:
$$J(\theta, \omega) = \mathbb{E}_{x \sim P_x} [\log D_\omega(x)] - \mathbb{E}_{\hat{x} \sim P_\theta} [\log(1 - D_\omega(\hat{x}))]$$

Here, we define a helper function $D_\omega(x)$—which acts as our discriminator—by applying a sigmoid activation function $\sigma(\cdot)$ over the raw scalar output of our critic network $T_\omega(x)$:
$$D_\omega(x) = \sigma(T_\omega(x)) = \frac{1}{1 + e^{-T_\omega(x)}}$$

---

### Student Doubts & Clarifications on $f(u)$

During the derivation, several crucial points were raised by the class regarding the mathematical validity of the divergence:

> **Student (Lokesh):** "Sir, shouldn't $f(1) = 0$ for this to be a mathematically valid $f$-divergence?"

**Professor's Explanation:** 
Lokesh is absolutely correct. By definition, a standard $f$-divergence requires $f(1) = 0$ so that the divergence between identical distributions evaluates to $0$. 

If we evaluate our initial candidate $f(1) = 1 \log 1 - (1+1)\log(1+1)$, it yields $-2 \log 2 \neq 0$. To rectify this, the true generator function corresponding to the normalized Jensen-Shannon (JS) divergence must be defined as:
$$f(u) = u \log u - (u+1) \log \left(\frac{u+1}{2}\right)$$

Now, evaluating $f(1)$ gives:
$$f(1) = 1 \log 1 - 2 \log \left(\frac{2}{2}\right) = 0 - 0 = 0$$

This brings us to a key conceptual insight: **The vanilla GAN loss is not a mathematically strict $f$-divergence.** Rather, it is shifted by a constant factor of $2 \log 2$ (or $\log 4$). Because this offset is constant, it does not alter the gradients during optimization. The standard GAN formulation is simply a scaled and shifted version of the true Jensen-Shannon divergence.

---

> **Student (Raghavendra):** "Sir, can you please repeat why we need this final sigmoid activation for $D_\omega(x)$?"

```
               CRITIC / DISCRIMINATOR NETWORK (D_w)
             ┌──────────────────────────────────┐
Real (x)  ──>│  Raw Neural Output   ─> Sigmoid  ├─> Output Score in [0, 1]
or           │      ( T_w(x) )        ( σ(•) )  │
Fake (x̂)  ──>│                                  │
             └──────────────────────────────────┘
```

**Professor's Explanation:** 
The variational function $T_\omega(x)$ cannot output arbitrary values. Mathematically, its range must respect the domain of the convex conjugate function $f^*(t)$. 

For the JS divergence pair, the domain constraints require us to wrap the raw neural network output $T_\omega(x)$ with a sigmoid activation $\sigma(\cdot)$. This maps the network's output space cleanly to the probability range $[0, 1]$, allowing the network to be interpreted as a classifier (discriminator). 

Different choices of $f$-divergences in the $f$-GAN framework dictate different mathematically optimal output activations $g_f$ (such as linear, exponential, or tanh) to respect their corresponding conjugate domains.

---

## 2. Formulating the Practical Minimax Optimization

In practice, we cannot compute analytical expectations over complex data distributions. Instead, we approximate these expectations using finite sample estimates from our dataset (applying the Law of Large Numbers). 

Our objective is to solve the minimax problem:
$$\theta^*, \omega^* = \arg \min_\theta \max_\omega \hat{J}(\theta, \omega)$$

We define our empirical objective estimator $\hat{J}(\theta, \omega)$ as:
$$\hat{J}(\theta, \omega) = \frac{1}{N} \sum_{i=1}^N \log D_\omega(x_i) - \frac{1}{M} \sum_{j=1}^M \log(1 - D_\omega(\hat{x}_j))$$

Where:
* Real data mini-batch: $\{x_1, x_2, \dots, x_N\} \sim P_x$ is sampled directly from our dataset.
* Latent noise mini-batch: $\{z_1, z_2, \dots, z_M\} \sim \mathcal{N}(0, I)$ is passed through the generator to yield fake samples $\{\hat{x}_1, \hat{x}_2, \dots, \hat{x}_M\}$, where $\hat{x}_j = G_\theta(z_j)$.

---

> **Student (Sarvesh):** "Sir, is there a specific reason for choosing $N$ and $M$ to be different, or can they be the same?"

**Professor's Explanation:** 
In training pipelines, $N$ and $M$ represent the batch sizes of your real data and fake generated data respectively. While they can mathematically be different, in practice, we almost always set them to be equal (i.e., $N = M$) for balanced mini-batch stochastic gradient descent (SGD). This is a standard batch-processing configuration; we divide the entire dataset into batches of size $N$ and iterate until we complete one full epoch.

---

## 3. Step-by-Step GAN Training Mechanics

Training a GAN is an alternating two-step optimization process: updating the Generator parameters ($\theta$) and the Discriminator/Critic parameters ($\omega$).

### Phase A: Generator Training (Update $\theta$, Keep $\omega$ Fixed)

We update the generator's parameters $\theta$ by taking a step in the direction of the negative gradient of our objective:
$$\theta^{t+1} \leftarrow \theta^t - \alpha \nabla_\theta \hat{J}(\theta^t, \omega^t)$$

Evaluating the gradient with respect to $\theta$:
$$\nabla_\theta \hat{J}(\theta, \omega) = \nabla_\theta \left[ \frac{1}{N} \sum_{i=1}^N \log D_\omega(x_i) - \frac{1}{M} \sum_{j=1}^M \log(1 - D_\omega(G_\theta(z_j))) \right]$$

Because the first term (real data) does not depend on the generator's parameters $\theta$, its gradient is zero. The optimization simplifies to:
$$\nabla_\theta \hat{J}(\theta, \omega) = - \nabla_\theta \left[ \frac{1}{M} \sum_{j=1}^M \log(1 - D_\omega(G_\theta(z_j))) \right]$$

```
                   GENERATOR TRAINING FLOW (ω is Fixed)
                  ┌─────────────────────────────────────┐
  z_j ───────────>│  Generator Network: G_θ(z)          │
                  └─────────┬───────────────────────────┘
                            │  Forward: x̂_j = G_θ(z_j)
                            ▼
                  ┌─────────────────────────────────────┐
  x̂_j ───────────>│  Discriminator Network: D_ω(x̂)      │
                  └─────────┬───────────────────────────┘
                            │  Forward: D_ω(x̂_j)
                            ▼
                     Loss Computation:
                     Loss = -1/M * Σ log(1 - D_ω(x̂_j))
                            │
                            │  Backward Pass (Backprop)
                            ▼
  Gradient ◄────────  Through D_ω (No parameter updates!)
  Flows     ────────> 
  Backward  ◄────────  Through G_θ (Update G weights θ)
```

#### Backpropagation Pipeline for Phase A:
1. **Forward Pass:** Sample latent noise $z_j \sim \mathcal{N}(0, I)$ and generate $\hat{x}_j = G_\theta(z_j)$. Pass $\hat{x}_j$ through the discriminator to compute $D_\omega(\hat{x}_j)$.
2. **Backward Pass:** Compute the gradient of the loss with respect to $D_\omega(\hat{x}_j)$. Propagate the gradients **backward through the discriminator network $D_\omega$** to obtain the gradients at the generated inputs $\hat{x}_j$. Continue backpropagating these gradients **through the generator network $G_\theta$** to update the weights $\theta$.
3. **Crucial Rule:** The discriminator parameters $\omega$ are kept **fixed** during this phase. The discriminator network is solely used as a differentiable pathway to backpropagate gradients to the generator.

*Note on Network Architecture:* During the lecture, the generator was initially drawn narrowing from $z$ to $x$. As pointed out by a student, the latent space dimension is typically much smaller than the data space dimension (e.g., $z \in \mathbb{R}^{32}$ while $x \in \mathbb{R}^{10000}$). Thus, the generator network must be **expanding** in size, not narrowing.

---

### Phase B: Discriminator / Critic Training (Update $\omega$, Keep $\theta$ Fixed)

We update the discriminator's parameters $\omega$ to maximize the objective (using gradient ascent, or gradient descent on the negative of the objective):
$$\omega^{t+1} \leftarrow \omega^t + \beta \nabla_\omega \hat{J}(\theta^t, \omega^t)$$

```
                 DISCRIMINATOR TRAINING FLOW (θ is Fixed)
                  ┌─────────────────────────────────────┐
  x_i (Real) ────>│                                     │
                  │  Discriminator Network: D_ω(x)      ├──> Forward: D_ω(x_i)
  x̂_j (Fake) ────>│                                     │    & D_ω(x̂_j)
                  └─────────┬───────────────────────────┘
                            │
                            ▼
                     Loss Computation:
                     Loss = - [ 1/N * Σ log D_ω(x_i) 
                               - 1/M * Σ log(1 - D_ω(x̂_j)) ]
                            │
                            │  Backward Pass (Backprop)
                            ▼
                     Compute Gradient w.r.t ω
                     Update discriminator weights ω
                     (Gradient propagation STOPS at D's inputs)
```

#### Backpropagation Pipeline for Phase B:
1. **Forward Pass:**
   * Pass real samples $x_i$ through $D_\omega$ to compute $D_\omega(x_i)$.
   * Generate fake samples $\hat{x}_j$ using the **fixed** generator. Pass them through $D_\omega$ to compute $D_\omega(\hat{x}_j)$.
2. **Backward Pass:** Compute the loss and propagate the gradients through the discriminator to calculate the gradient with respect to $\omega$. 
3. **Crucial Rule:** The backpropagation path **stops at the input layer of the discriminator**. Gradients do not flow back into the generator network because the generator's parameters $\theta$ are kept **fixed**.

---

## 4. Class Q&A and Practical Considerations

### Non-Image Data & The Limits of GANs

> **Student (Sushil):** "Sir, is $x$ always an image here, or can it be any kind of high-dimensional data like text or audio? And do we use GANs for text generation?"

**Professor's Explanation:** 
In our formulation, $x$ is general and can represent images, audio, or text. However, **GANs are rarely used for text generation.** 

The reason is fundamental to backpropagation: text generation involves sampling discrete tokens (words or characters) from a probability distribution. Discrete sampling is a step function and has zero gradient almost everywhere, making it non-differentiable. Because the backpropagation path for the generator must go through the generated samples (as shown in Phase A), we cannot propagate gradients from the discriminator back into the generator when the generated tokens are discrete. We will discuss alternative solutions to this restriction in future classes.

---

### Reference Materials & Deep Dives

> **Student (Aastik):** "Is there some reference material we can look at if we want to go deeper into the mathematics of this?"

**Professor's Explanation:** 
For those looking to explore these concepts further, my handwritten lecture notes (available on our shared Google Drive) contain a complete list of background readings and references at the end. 

Specifically, you should review:
1. **The $f$-GAN Paper:** *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization* by Sebastian Nowozin et al. (NIPS 2016).
2. **Foundational Course on ML:** For a rigorous foundation in classical machine learning, I highly recommend my advisor Professor Shastry's NPTEL course, **Pattern Recognition and Neural Networks (PRNN)**. It provides an excellent theoretical grounding.

---

## Classroom Intermission: A Light-hearted Tale of Cultural Training

To break up the dense mathematics, the professor shared a humorous anecdote contrasting how we discipline children with how we "abuse" or train our generative networks:

> *"I was talking to a senior colleague from IIT Delhi, Professor Bhim Singh. He told me about a Punjabi collaborator in Canada who went to work and left his 10-year-old son home with his grandparents. The grandfather scolded the child for some minor mischief. Having been raised in Canada, the child immediately called the police to complain about grandfather's 'abuse'. In Canada and Sweden, child protection laws are so strict that the parents were penalized and the child was placed in government care. 
> 
> A few months later, the family visited India. The moment they stepped off the plane at the Delhi airport, the father immediately began beating the child! When the airport security guards came over to ask what was happening, the father explained the Canadian police incident. The Indian policemen immediately replied, 'Oh, is that so? We will also join in and beat him! Beat him more!'"*

A student added to the fun:
> *"Sir, I had a geography teacher in school who used to ask us to bend down and touch our own feet. The moment we did, he would smack us on our backs with a wooden stick! It was very common in our generation, but with Gen Z, you cannot even scold them anymore."*

While a humorous distraction, this serves as a metaphorical reminder of adversarial training: the discriminator acts as the strict "parental" critic, imposing penalizing losses on the generator's outputs until the generator learns to produce flawless, indistinguishable data.

---

## References

[1] Nowozin, S., Cseke, B., & Tomioka, R. (2016). *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*. Advances in Neural Information Processing Systems (NIPS 2016).