# Deep Generative Models: Adversarial Generative Models & $f$-Divergences

Welcome to these comprehensive, blog-style lecture notes for **Lecture 4: Adversarial Generative Models**. This post reconstructs the entire session, including the mathematical formulations, pedagogical discussions, real-world examples, and student-teacher interactions. 

---

## 1. Course Logistics & The Core Philosophy

Before diving into the technical details, let's establish the context of the course. The course relies on a strong foundation in linear algebra, probability, and machine learning basics (specifically referencing Chapters 2, 3, 5, and 6 of the Ian Goodfellow *Deep Learning* textbook). The evaluation scheme is structured around two exams, a series of quizzes, and highly practical coding assignments.

### Recap: The Generative Modeling Goal
At its core, generative modeling aims to solve a fundamental problem:
Given a dataset $D = \{x_1, x_2, \dots, x_n\}$ consisting of samples drawn from an unknown true probability distribution $P_X$, we want to:
1. **Estimate** the underlying true distribution $P_X$.
2. **Sample** new, unseen data points from our estimate.

To do this, we previously introduced the framework of **Divergence Minimization**. The philosophy is straightforward:
* We assume a parametric family of probability density functions, denoted by $P_\theta$, where $\theta$ represents the model parameters we can adjust.
* We define a distance or divergence metric $D(P_X \parallel P_\theta)$ to quantify how different our model's distribution is from the true data distribution.
* We adjust/compute the parameters $\theta$ such that this divergence is minimized:
$$\theta^* = \arg\min_{\theta} D(P_X \parallel P_\theta)$$

---

## 2. Unconditional Generative Models: The Power of the Random Seed

In **Adversarial Generative Models** (and specifically Generative Adversarial Networks, or GANs), we do not explicitly define a mathematical formula for the density $P_\theta(x)$. Instead, we represent $P_\theta$ implicitly using a **Deep Neural Network (DNN)**, which we call the generator: $g_\theta(z)$.

### How the Generator Works
Instead of directly calculating a probability for a given $x$, we draw a sample $z$ from a simple, known distribution (like a standard multivariate Gaussian, $z \sim \mathcal{N}(0, I)$) and pass it through the deterministic neural network $g_\theta(z)$. 

```
  [ Simple Noise Seed ] -----> [ Generator Network ] -----> [ Generated Sample ]
     z ~ N(0, I)                   g_θ(z)                     x_hat ~ P_θ
```

Because $z$ is a random variable, passing it through the deterministic function $g_\theta$ transforms it into a new random variable, $\hat{x} = g_\theta(z)$. The distribution of this output random variable is what we define as our model distribution, $P_\theta$.

---

### Student Q&A: Understanding the Seed $z$ and Dimensionality

During the lecture, several students raised crucial questions regarding the nature of the seed $z$ and how we structure the network to map low dimensions to high dimensions.

#### Vivek's Doubt: Is $z$ representative of the data?
* **Question**: Is the seed $z$ representative of the data, or was some transformation done to it?
* **Instructor's Response**: No, $z$ has zero connection to the data. It is simply an input random seed (or latent vector) that we start with. It is completely independent of our training dataset. It acts as a blank canvas of noise that the network learns to paint into structured data.

#### Harish's Doubt: Why a Normal Distribution?
* **Question**: Why do we choose a normal distribution for $z$? Can we use other distributions?
* **Instructor's Response**: Yes, you can technically use other distributions, such as a uniform distribution. However, the normal distribution is generally preferred because it has **infinite support**. This means its probability density is non-zero over the entire real space ($\mathbb{R}^k$). A uniform distribution has bounded support, which can restrict the mathematical flexibility of the generator when mapping to highly complex, high-dimensional target spaces.
* **Follow-up Question**: Does the dimension $k$ of $z$ depend on the data size, or is it arbitrary?
* **Instructor's Response**: It is a hyperparameter. Typically, we choose $k$ to be small (e.g., $16, 32, 100, 128$). While it is technically a design choice, research shows that there is often an optimal latent dimensionality for a given dataset, but in practice, it is kept fixed during training.

#### Astik's Doubt: Low to High Dimensionality
* **Question**: In standard classifiers, we reduce dimensionality (e.g., image to class label). Here, we are starting with a low-dimensional seed $z$ and going to a high-dimensional image $\hat{x}$. How does that work?
* **Instructor's Response**: Excellent observation. Classifiers perform *dimensionality reduction* (mapping high-dimensional data to a low-dimensional class index). Generative models do the opposite: they perform **dimensionality expansion** ($d \gg k$). We start with a low-dimensional noise vector $z \in \mathbb{R}^k$ and expand it to match the dimensionality of our data $\hat{x} \in \mathbb{R}^d$ (where $d$ is the number of pixels, e.g., $200 \times 200 = 40,000$). To implement this with convolutional networks, we use specialized layers like **upsampling** and **transpose convolutions**, which will be covered in detail during the TA sessions.

---

### Mathematical Example: Transformation of Variables
To build intuition for how a deterministic function changes a distribution, consider a simple 1D example:
* Let the input random variable be uniform: $z \sim U[0, 1]$.
* Let our "generator" function be: $g(z) = z^2$.
* The output variable is $\hat{x} = z^2$. 

Even though our input $z$ is perfectly uniform, the output $\hat{x}$ is **not** uniform. Its density is skewed toward 0 because the squaring operation compresses the values near 0 and stretches them near 1. By choosing a highly complex, multi-layered neural network $g_\theta$, we can warp a simple Gaussian distribution into incredibly complex distributions (such as the distribution of natural images).

---

## 3. Inside the Generator Network Architecture ($g_\theta$)

Let's look at the mathematical structure of a simple 4-layer fully-connected feedforward neural network generator:

$$g_\theta(z) = W_4 \sigma\left( W_3 \sigma\left( W_2 \sigma\left( W_1 z \right) \right) \right)$$

Where:
* $z \in \mathbb{R}^{k \times 1}$ is our input seed vector.
* $\sigma(\cdot)$ is an element-wise non-linear activation function (such as the Sigmoid or ReLU function).
* $W_1, W_2, W_3, W_4$ are the weight matrices (parameters) of the network.

### Analyzing the Shapes of the Weight Matrices
To ensure the output dimensionality expands from $k$ to $d$, we must carefully design the shapes of our weight matrices:
* $W_1 \in \mathbb{R}^{L_1 \times k}$ $\implies$ Output of first layer is $L_1 \times 1$.
* $W_2 \in \mathbb{R}^{L_2 \times L_1}$ $\implies$ Output of second layer is $L_2 \times 1$.
* $W_3 \in \mathbb{R}^{L_3 \times L_2}$ $\implies$ Output of third layer is $L_3 \times 1$.
* $W_4 \in \mathbb{R}^{d \times L_3}$ $\implies$ Final output $\hat{x}$ is $d \times 1$.

Since we want to expand the dimensionality, we typically design the network such that:
$$k < L_1 < L_2 < L_3 < d$$

---

### Student Q&A: Clarifying Network Layers

#### Sachin's Doubt: Typo in Matrix Dimensions
* **Question**: In the expression for $W_3$, shouldn't its shape be $L_3 \times L_2$ instead of $L_3 \times L_1$?
* **Instructor's Response**: Yes, that is correct. It was a typo on my part. Since the output of the second layer has dimension $L_2$, the matrix $W_3$ must have $L_2$ columns to perform the matrix multiplication. Its shape is indeed $L_3 \times L_2$. Thank you for pointing that out.

#### Raghavendra's Doubt: Activation on the Final Layer
* **Question**: Is there no activation function $\sigma$ applied after the final weight matrix $W_4$?
* **Instructor's Response**: That depends entirely on your design choices. If your target data has arbitrary values in $\mathbb{R}^d$, you can leave the final layer linear (no activation). If your target data consists of image pixels normalized between $0$ and $1$, you might apply a Sigmoid activation function at the very end to constrain the outputs.

---

### Unconditional vs. Conditional Generation

#### Nirmit's Doubt: Generating Specific Images from Text
* **Question**: If we want to generate a specific image—for example, if we input the text "a cup on a table", will this model output an image of a cup on a table?
* **Instructor's Response**: No. What you are describing is **conditional generative modeling**, where the generation is guided by an external condition (like a text prompt or class label). Right now, we are studying **unconditional generative modeling**. In this setup, the network simply takes random noise $z$ and outputs a random sample from the dataset. You have no control over *which* specific category or image the network generates—it just outputs a highly realistic sample that fits the overall distribution. We will cover conditional generative models later in the course.

#### Real-World Example: Unconditional Generation in Action
The website [thispersondoesnotexist.com](https://thispersondoesnotexist.com) is a perfect example of unconditional generation. 
* Every time you refresh the page, a random number generator creates a random seed $z \sim \mathcal{N}(0, I)$.
* This seed is fed into a highly complex, pre-trained generator network $g_\theta(z)$.
* The output is a high-resolution, photorealistic image of a human face. 
* There is no text prompt guiding the creation; the network is simply mapping random noise to the learned manifold of human faces.

---

## 4. The Core Challenge: Divergence Minimization with Only Samples

#### Sushil's Doubt: Where is the Data Used?
* **Question**: If we only pass the random seed $z$ through our generator $g_\theta(z)$, where do we actually use our training data $D = \{x_1, \dots, x_n\}$? How do we calculate the loss?
* **Instructor's Response**: This is the central question of generative modeling. We use the training data to define our objective. We want the distribution of our generated samples, $P_\theta$, to match the true data distribution, $P_X$.
But here is the catch: **We do not have the analytical, closed-form equations for either $P_X$ or $P_\theta$.** 
* For $P_X$, we only have a finite set of data samples: $D = \{x_1, \dots, x_n\}$.
* For $P_\theta$, we only have the generator network $g_\theta$, which allows us to *produce* samples $\hat{x}$ by drawing different random seeds $z$.

So, how do we compute a divergence metric $D(P_X \parallel P_\theta)$ and optimize our parameters $\theta$ when we only have access to *samples* from both distributions? 
Different generative frameworks solve this in different ways. Generative Adversarial Networks (GANs) solve this using an adversarial game, while Variational Autoencoders (VAEs) use variational inference, and other models use Maximum Likelihood Estimation. To understand how we compare these distributions, we must first mathematically define families of divergence metrics.

---

## 5. Mathematically Defining the $f$-Divergence Family

To compare two arbitrary probability density functions $P_X$ (true distribution) and $P_\theta$ (model distribution), we can define a generalized class of divergences called **$f$-divergences**:

$$D_f(P_X \parallel P_\theta) = \int P_\theta(x) f\left( \frac{P_X(x)}{P_\theta(x)} \right) dx$$

### Mathematical Requirements for $f(u)$
For $D_f(P_X \parallel P_\theta)$ to behave as a valid divergence metric, the generator function $f: \mathbb{R}^+ \to \mathbb{R}$ must satisfy two conditions:
1. $f(u)$ must be a **convex function**.
2. $f(1) = 0$.

If these conditions are met, then by Jensen's Inequality, we can guarantee that:
$$D_f(P_X \parallel P_\theta) \ge 0$$
And $D_f(P_X \parallel P_\theta) = 0$ if and only if $P_X = P_\theta$ almost everywhere.

---

### Step-by-Step Derivations of Classic Divergences

By choosing different convex functions for $f(u)$, we can recover several classic probability distance metrics. Let's derive three of them step-by-step.

#### 1. Kullback-Leibler (KL) Divergence
Let's choose the convex function:
$$f(u) = u \log u$$
*(Check: $f(1) = 1 \log(1) = 0$. Check passed.)*

Now, let's substitute $f(u) = u \log u$ into our general $f$-divergence formula where our variable $u = \frac{P_X(x)}{P_\theta(x)}$:

$$D_f(P_X \parallel P_\theta) = \int P_\theta(x) \left[ \left( \frac{P_X(x)}{P_\theta(x)} \right) \log\left( \frac{P_X(x)}{P_\theta(x)} \right) \right] dx$$

The $P_\theta(x)$ terms outside and inside the brackets cancel out:

$$D_f(P_X \parallel P_\theta) = \int P_X(x) \log\left( \frac{P_X(x)}{P_\theta(x)} \right) dx$$

This is exactly the mathematical definition of the **KL-Divergence**:
$$D_{KL}(P_X \parallel P_\theta)$$

---

#### 2. Jensen-Shannon (JS) Divergence
Let's choose the convex function:
$$f(u) = \frac{1}{2} \left( u \log u - (u + 1) \log \left(\frac{u + 1}{2}\right) \right)$$
*(Check: $f(1) = \frac{1}{2} (0 - 2 \log(1)) = 0$. Check passed.)*

By substituting this $f(u)$ into the $f$-divergence equation and simplifying the algebra, we recover the **Jensen-Shannon Divergence**:
$$D_{JS}(P_X \parallel P_\theta) = \frac{1}{2} D_{KL}\left(P_X \parallel \frac{P_X + P_\theta}{2}\right) + \frac{1}{2} D_{KL}\left(P_\theta \parallel \frac{P_X + P_\theta}{2}\right)$$

---

#### 3. Total Variation (TV) Distance
Let's choose the convex function:
$$f(u) = \frac{1}{2} |u - 1|$$
*(Check: $f(1) = \frac{1}{2} |1 - 1| = 0$. Check passed.)*

Substituting this into our general equation:

$$D_f(P_X \parallel P_\theta) = \int P_\theta(x) \left[ \frac{1}{2} \left| \frac{P_X(x)}{P_\theta(x)} - 1 \right| \right] dx$$

Bring the positive scalar term $P_\theta(x)$ inside the absolute value brackets:

$$D_f(P_X \parallel P_\theta) = \frac{1}{2} \int \left| P_\theta(x) \cdot \frac{P_X(x)}{P_\theta(x)} - P_\theta(x) \right| dx$$

$$D_f(P_X \parallel P_\theta) = \frac{1}{2} \int \left| P_X(x) - P_\theta(x) \right| dx$$

This is the standard definition of the **Total Variation Distance**.

---

## 6. Summary Cheat Sheet for Recall

Use this table to quickly recall the core details of today's lecture when studying for quizzes or exams:

| Concept / Variable       | Physical / Mathematical Meaning                   | Key Takeaway / Hyperparameter                                                         |
| :-------------------------| :--------------------------------------------------| :--------------------------------------------------------------------------------------|
| **$P_X$**                | True underlying data distribution (unknown).      | We only have samples $D = \{x_1, \dots, x_n\}$ from it.                               |
| **$P_\theta$**           | Model's generated distribution.                   | Represented implicitly by passing $z \sim \mathcal{N}(0, I)$ through $g_\theta(z)$.   |
| **$z \in \mathbb{R}^k$** | Low-dimensional random seed vector.               | Independent of data; sampled easily using standard random generators (e.g., `randn`). |
| **$x \in \mathbb{R}^d$** | High-dimensional data space (e.g., image pixels). | Generative networks expand dimensionality ($d \gg k$).                                |
| **$g_\theta(z)$**        | Generator Deep Neural Network.                    | Transforms random noise into structured samples.                                      |
| **$\theta$**             | Model parameters (weights of the network).        | Optimized by minimizing the divergence $D(P_X \parallel P_\theta)$.                   |
| **$f$-Divergence**       | Generalized family of divergence metrics.         | Requires a convex function $f(u)$ where $f(1) = 0$.                                   |


# Mastering $f$-GANs: How to Estimate Divergences from Samples (Without Knowing the Densities)

In generative modeling, particularly when training Generative Adversarial Networks (GANs), we often face a fundamental challenge: **How do we minimize the distance between two probability distributions when we only have access to their samples?** 

This post provides a comprehensive set of lecture notes detailing how we can leverage the **Legendre-Fenchel Conjugate** and the **Law of Large Numbers** to derive a variational lower bound for $f$-divergences. This formulation allows us to bypass the need to know the analytical probability density functions of our data and instead compute divergence bounds purely using samples.

---

## 1. Setting the Stage: The Core Goal and a Critical Correction

Before diving into the mathematical framework, let us clarify the fundamental properties of $f$-divergences and resolve an important constraint discussed in the lecture.

### The Objective
Suppose we have:
*   $P_x$: The true data distribution (unknown analytical form, but we have samples $x \sim P_x$).
*   $P_\theta$: The model distribution parameterized by $\theta$ (unknown analytical form, but we can generate samples $x \sim P_\theta$).

**Goal:** Compute the $f$-divergence $D_f(P_x \parallel P_\theta)$ using only these samples.

---

### Student Doubt 1: Correcting the Divergence Constraints
During the recap of $f$-divergences, a key property of the generator function $f(u)$ was discussed. 

*   **Initial slide note:** The slide initially stated that $f(u)$ is a convex, lower semi-continuous function mapping $\mathbb{R}^+ \to \mathbb{R}$ such that $f(0) = 1$.
*   **Student Intervention (Astik):** *"Sir, you mentioned some properties about this $f$ function for $f$-divergences. But this $f(0) = 1$ is not following for the three functions we wrote, right? It should be $f(1) = 0$."*
*   **Professor's Response:** *"Ah, yes! Thanks for pointing that out. It should evaluate to $0$ at $1$. That is, $f(1) = 0$."*

#### Why $f(1) = 0$?
An $f$-divergence is defined as:
$$D_f(P_x \parallel P_\theta) = \int_{\mathcal{X}} P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$$

If the two distributions are identical ($P_x = P_\theta$), the ratio of their densities is $1$ everywhere. For the divergence to be zero when the distributions match, the function $f(1)$ must equal $0$. 

---

## 2. The Naive Approach and Why It Fails

To appreciate the variational approach, we must first understand why the standard definition of $f$-divergence cannot be directly approximated using standard sampling techniques.

### The Law of Large Numbers (LLN)
If we want to evaluate an integral of the form:
$$I = \int_{\mathcal{X}} h(x) P_x(x) dx = \mathbb{E}_{x \sim P_x}[h(x)]$$

And we do not know the density $P_x(x)$ but have $n$ independent samples $x_1, x_2, \dots, x_n \sim P_x$, the Law of Large Numbers states that we can approximate this expectation as a sample mean:
$$\mathbb{E}_{x \sim P_x}[h(x)] \approx \frac{1}{n} \sum_{i=1}^n h(x_i)$$

As $n \to \infty$, this sample average converges to the true expectation.

#### Example: Expectation of a Random Variable
If we define $h(x) = x$, then:
$$\int_{\mathcal{X}} x P_x(x) dx = \mathbb{E}_{P_x}[X] \approx \frac{1}{n} \sum_{i=1}^n x_i$$
This is the standard sample mean used daily in statistics.

---

### Student Doubt 2: Why Can't We Just Apply LLN Directly to $D_f$?
*   **Student:** *"If we don't know $P_x$ and $P_\theta$, but we have samples from them, why can't we just directly apply the Law of Large Numbers to compute the divergence (like KL or JS divergence)?"*
*   **Professor:** *"Because the ratio of the densities is inside the non-linear function $f(u)$. Let's write down the definition again:*
    $$D_f(P_x \parallel P_\theta) = \int_{\mathcal{X}} P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$$
    *If we try to write this as an expectation over $P_\theta$, we get:*
    $$\mathbb{E}_{x \sim P_\theta}\left[f\left(\frac{P_x(x)}{P_\theta(x)}\right)\right]$$
    *To evaluate this using samples $x_i \sim P_\theta$, we would need to calculate $f\left(\frac{P_x(x_i)}{P_\theta(x_i)}\right)$ for each sample. But we do not know the analytical density values $P_x(x_i)$ or $P_\theta(x_i)$! Thus, we cannot compute this ratio, making a direct application of the LLN impossible."*

---

## 3. The Mathematical Savior: Convex Conjugates

To resolve this bottleneck, we introduce a powerful tool from convex analysis: the **Legendre-Fenchel Conjugate** (often called the convex conjugate).

### Definition of the Fenchel Conjugate
For any convex function $f(u)$, its convex conjugate $f^*(t)$ is defined as:
$$f^*(t) = \sup_{u \in \text{dom } f} \{ ut - f(u) \}$$

*(Note: In the lecture, the professor writes $\max$ instead of $\sup$ for simplicity).*

### Geometric Intuition
Let us visualize what $f^*(t)$ represents:

```
    f(u) ^
         |          . f(u)
         |         /
         |        /  .  <- Maximum vertical distance is f*(t)
         |       /  . /
         |      / .  /   <- Line: ut - f*(t)
         |     /.   /
         |    /    /
         +------------------------> u
```

1.  For a fixed slope $t$, the term $ut$ represents a line passing through the origin.
2.  The term $ut - f(u)$ represents the vertical distance between the line $ut$ and the convex curve $f(u)$.
3.  The conjugate value $f^*(t)$ is the **maximum vertical distance** between the linear function $ut$ and the convex function $f(u)$.
4.  By definition, this supremum implies:
    $$f^*(t) \ge ut - f(u) \quad \forall u, t$$
    Rearranging this inequality yields:
    $$f(u) \ge ut - f^*(t) \quad \forall u, t$$

---

### Student Doubt 3: Clarifying the Geometry of the Conjugate
*   **Student:** *"Is it the value of $u$ that maximizes the term $ut - f(u)$?"*
*   **Professor:** *"Yes, exactly. For a fixed slope $t$, we search over all $u$ in the domain of $f$ to find the point that maximizes the vertical gap. At this maximizing point, the line with slope $t$ is tangent to $f(u)$."*

---

### The Double Conjugate Property (Duality)
Because $f(u)$ is a convex and lower semi-continuous function, it is equal to its own double conjugate:
$$(f^*)^* = f$$

This yields the dual representation of $f(u)$:
$$f(u) = \sup_{t \in \text{dom } f^*} \{ tu - f^*(t) \}$$

This identity is the key to transforming our divergence. It allows us to represent a non-linear convex function $f(u)$ as a supremum over a family of linear bounds.

---

## 4. Deriving the Variational Lower Bound on $f$-Divergence

Now we can "connect the dots" to see how this dual representation transforms the $f$-divergence definition into something we can estimate.

### Step 1: Substitute the Dual Representation
Recall the definition of $f$-divergence:
$$D_f(P_x \parallel P_\theta) = \int_{\mathcal{X}} P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$$

Let $u = \frac{P_x(x)}{P_\theta(x)}$. Applying the double conjugate property to $f(u)$:
$$f\left(\frac{P_x(x)}{P_\theta(x)}\right) = \sup_{t} \left\{ t \frac{P_x(x)}{P_\theta(x)} - f^*(t) \right\}$$

Substituting this back into the integral:
$$D_f(P_x \parallel P_\theta) = \int_{\mathcal{X}} P_\theta(x) \left[ \sup_{t} \left\{ t \frac{P_x(x)}{P_\theta(x)} - f^*(t) \right\} \right] dx$$

---

### Step 2: From Scalar $t$ to Function $T(x)$
In the expression above, the supremum over $t$ is computed **point-wise** for every $x \in \mathcal{X}$. This means the optimal $t$ can be different for different values of $x$. 

To represent this mathematically, we can replace the scalar $t$ with a function $T(x)$ that maps from the input space $\mathcal{X}$ to the domain of the conjugate function $f^*$. Let $\mathcal{T}$ be the set of all arbitrary functions $T: \mathcal{X} \to \mathbb{R}$.

Because the supremum is taken point-wise, we can write:
$$D_f(P_x \parallel P_\theta) = \sup_{T \in \mathcal{T}} \int_{\mathcal{X}} P_\theta(x) \left[ T(x) \frac{P_x(x)}{P_\theta(x)} - f^*(T(x)) \right] dx$$

---

### Step 3: Distribute the Densities
Let us expand the term inside the integral:
$$D_f(P_x \parallel P_\theta) = \sup_{T \in \mathcal{T}} \int_{\mathcal{X}} \left[ T(x) P_\theta(x) \frac{P_x(x)}{P_\theta(x)} - P_\theta(x) f^*(T(x)) \right] dx$$

Assuming $P_\theta(x) > 0$ over the support, the terms cancel out:
$$D_f(P_x \parallel P_\theta) = \sup_{T \in \mathcal{T}} \int_{\mathcal{X}} \left[ T(x) P_x(x) - f^*(T(x)) P_\theta(x) \right] dx$$

Splitting the integral into two terms:
$$D_f(P_x \parallel P_\theta) = \sup_{T \in \mathcal{T}} \left\{ \int_{\mathcal{X}} T(x) P_x(x) dx - \int_{\mathcal{X}} f^*(T(x)) P_\theta(x) dx \right\}$$

Using the definition of expectation, we get our final, elegant objective:
$$D_f(P_x \parallel P_\theta) = \sup_{T \in \mathcal{T}} \left\{ \mathbb{E}_{x \sim P_x}[T(x)] - \mathbb{E}_{x \sim P_\theta}[f^*(T(x))] \right\}$$

---

### Step 4: Creating a Practical Lower Bound
If we could optimize over the space of *all possible functions* $\mathcal{T}$, the equation above would hold with strict equality. However, in practice, we cannot search over an infinite-dimensional space of arbitrary functions. 

Instead, we restrict $T(x)$ to a parameterized family of functions $\mathcal{T}_\omega$ (such as a neural network with parameters $\omega$). Restricting the search space to a subset $\mathcal{T}_\omega \subset \mathcal{T}$ yields a **lower bound**:
$$D_f(P_x \parallel P_\theta) \ge \max_{T_\omega} \left\{ \mathbb{E}_{x \sim P_x}[T_\omega(x)] - \mathbb{E}_{x \sim P_\theta}[f^*(T_\omega(x))] \right\}$$

We can now approximate these expectations using samples and the Law of Large Numbers! If we have samples $\{x_i\}_{i=1}^{n} \sim P_x$ and $\{y_j\}_{j=1}^{m} \sim P_\theta$:
$$D_f(P_x \parallel P_\theta) \ge \max_{T_\omega} \left\{ \frac{1}{n} \sum_{i=1}^n T_\omega(x_i) - \frac{1}{m} \sum_{j=1}^m f^*(T_\omega(y_j)) \right\}$$

---

## 5. Student Debates and Conceptual Refinements

Several students raised critical questions during the latter half of the lecture regarding the utility and limitations of this lower bound.

### Student Doubt 4: Why is a Loose Lower Bound Dangerous?
*   **Student (Lokesh):** *"Sir, if we are minimizing the divergence by minimizing its lower bound, does that actually guarantee we are improving the model? What if we find a smaller lower bound but the true divergence is still high?"*
*   **Professor:** *"That is an exceptionally important point! If the lower bound is loose, minimizing it does not guarantee that we are minimizing the actual $f$-divergence. This is why we need a **tight** lower bound. The tighter the bound, the more accurate our approximation is.*
    
    *How do we make the bound tight? We must make our function class $\mathcal{T}_\omega$ as expressive as possible. In a GAN framework, $T_\omega(x)$ is represented by the **discriminator** (or critic) network. We train the discriminator to maximize this bound (making it as tight as possible), while the generator tries to minimize it. If the discriminator is weak, the bound is loose, and the generator's updates will be meaningless."*

---

### Student Doubt 5: Confirming the Structure of the Bound
*   **Student:** *"So the lower bound is always in the form of a difference between an expectation over $P_x$ and an expectation over $P_\theta$?"*
*   **Professor:** *"Yes! It will always take the form:*
    $$\mathbb{E}_{P_x}[T(x)] - \mathbb{E}_{P_\theta}[f^*(T(x))]$$
    *Because it is expressed entirely as expectations over our two distributions, we can evaluate it directly using our data samples and generated samples. We no longer need to know the analytical forms of $P_x$ and $P_\theta$."*

---

## 6. Summary of Key Concepts

| Concept | Mathematical Form | Practical Meaning in Generative Modeling |
| :--- | :--- | :--- |
| **$f$-Divergence** | $\int P_\theta(x) f\left(\frac{P_x(x)}{P_\theta(x)}\right) dx$ | The distance metric we want to minimize between data and model. Uncomputable directly from samples. |
| **Convex Conjugate** | $f^*(t) = \sup_{u} \{ ut - f(u) \}$ | Linear dual representation of the convex function $f(u)$. |
| **Variational Lower Bound** | $\ge \mathbb{E}_{P_x}[T(x)] - \mathbb{E}_{P_\theta}[f^*(T(x))]$ | A computable approximation of the divergence. The function $T(x)$ acts as the **Discriminator**. |
| **The GAN Game** | $\min_\theta \max_\omega \left( \mathbb{E}_{P_x}[T_\omega(x)] - \mathbb{E}_{P_\theta}[f^*(T_\omega(x))] \right)$ | The min-max objective of $f$-GANs. The discriminator maximizes to tighten the bound; the generator minimizes to reduce divergence. |

By translating the uncomputable density ratios of $f$-divergences into a dual optimization problem of expectations, we unlock the ability to train deep generative models on complex, high-dimensional datasets using nothing but raw samples. In the next lecture, we will look at how specific choices of the generator function $f(u)$ recover familiar objectives like the classical GAN, JS-GAN, and Wasserstein GAN.