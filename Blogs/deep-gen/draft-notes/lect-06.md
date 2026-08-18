# Lecture Notes: Interpreting Naive GANs as Classifier-Guided Generative Models and Practical Implementations

---

## Section 1: Foundations and Mathematical Setup

To understand Generative Adversarial Networks (GANs) from first principles, we can formulate them as a variational minimization of an $f$-divergence between the true data distribution $P_x$ and the model's generated distribution $P_\theta$. 

Recall that the variational lower bound of any $f$-divergence is given by:

$$J(\theta, \omega) = \mathbb{E}_{x \sim P_x} [T_\omega(x)] - \mathbb{E}_{\tilde{x} \sim P_\theta} [f^*(T_\omega(\tilde{x}))]$$

where $T_\omega$ is a variational function parameterized by $\omega$, and $f^*$ is the Fenchel conjugate of the generator function $f$. 

For the specific case of the **Jensen-Shannon (JS) divergence**, the generator function is defined as:

$$f(u) = u \log u - (u+1) \log \left(\frac{u+1}{2}\right)$$

Through algebraic manipulation of the conjugate function $f^*(t)$ and setting the output of our variational network to be parameterized via a sigmoid activation function, we can rewrite the objective. If we define our classifier/discriminator $D_\omega(x)$ as:

$$D_\omega(x) = \frac{1}{1 + e^{-T_\omega(x)}} = \sigma(T_\omega(x))$$

the general variational objective $J(\theta, \omega)$ simplifies to the familiar naive GAN loss function:

$$J(\theta, \omega) = \mathbb{E}_{x \sim P_x} [\log D_\omega(x)] + \mathbb{E}_{\tilde{x} \sim P_\theta} [\log(1 - D_\omega(\tilde{x}))]$$

This mathematical derivation provides a direct link between $f$-divergence minimization and binary cross-entropy. Below, we explore the conceptual interpretation of this framework as a classifier-guided generation process.

---

## Section 2: Naive GAN as a Classifier-Guided Generative Model

Let us establish the practical setup of a generative model.

### The Setup
* **True Data:** We have a dataset $D = \{x_1, x_2, \dots, x_n\}$ consisting of samples drawn from an underlying, unknown true data distribution $P_x$.
* **The Generator:** We define a neural network $g_\theta(z)$, parameterized by $\theta$. It takes a simple latent noise vector $z$ (typically drawn from a standard multivariate Gaussian distribution $z \sim \mathcal{N}(0, I)$) and transforms it into a synthetic sample $\hat{x} \sim P_\theta$.
* **The Goal:** Our objective is to tweak the parameters $\theta$ of the generator such that the generated distribution $P_\theta$ matches the true data distribution $P_x$ as closely as possible.

### The Classifier
Suppose we introduce a binary classifier $D_\omega$ (parameterized by $\omega$) designed to differentiate between real and generated samples:

$$D_\omega(x \text{ or } \hat{x}) = \begin{cases} 
1 & \text{if } x \text{ is classified as coming from } P_x \\ 
0 & \text{if } \hat{x} \text{ is classified as coming from } P_\theta 
\end{cases}$$

---

### 🙋‍♂️ Student Doubt 1 (Sanchit)
> **Sanchit:** *"We are saying that this classifier will output $1$ if $x$ comes from $P_x$. What if $x$ does *not* come from $P_x$?"*

**Instructor's Clarification:**  
It will output $0$. The classifier $D_\omega$ is explicitly trained on a binary classification task. It takes either a real sample $x$ or a fake sample $\hat{x}$ as its input. 

> **Sanchit:** *"What if the sample $x$ or $\hat{x}$ that we have drawn does not come from *either* $P_x$ or $P_\theta$?"*

**Instructor's Clarification:**  
By construction, the input space of the classifier is restricted solely to the union of the supports of $P_x$ and $P_\theta$. In this setup, we only feed the classifier samples that we have actively drawn from our true dataset $D$ or generated from our model $g_\theta(z)$. Thus, it acts strictly as a binary classifier between these two specific sets of samples.

---

## Section 3: The Core Philosophical Question

With this binary classifier $D_\omega$ in hand, we ask: **Can this classifier be used to 'tweak' $\theta$ such that $P_\theta$ becomes close to $P_x$?**

At a high level, the intuitive answer is: **Keep changing $\theta$ until the classifier completely fails.**

If the classifier can no longer distinguish between samples from $P_x$ and $P_\theta$, it implies that the generated samples have become indistinguishable from the real samples, meaning $P_\theta \approx P_x$.

---

### 🙋‍♂️ Student Doubt 2 (Sanchit)
> **Sanchit:** *"Since samples drawn from $P_x$ give a $1$, and generated samples $\hat{x}$ give a $0$, can we try to maximize the cumulative sum of the classifier's outputs?"*

**Instructor's Clarification:**  
Maximize with respect to what? Remember, we must distinguish our "handles" (the parameters we can actively change). If the classifier $D_\omega$ is kept *fixed*, the only parameters we can tweak to change the generated samples are the generator parameters $\theta$. Simply maximizing the outputs of a fixed classifier will not work, as we will see in the counter-example below.

---

### 🙋‍♂️ Student Doubt 3 (Divyang)
> **Divyang:** *"Can we use backpropagation here to help us?"*

**Instructor's Clarification:**  
Of course. Whenever we train neural networks, backpropagation is the underlying mechanism we use to compute gradients. However, let us first look at this from a high-level, philosophical perspective before diving into the calculus of backpropagation.

---

### 🙋‍♂️ Student Doubt 4 (Abhilesh)
> **Abhilesh:** *"If the classifier has already approximated $P_x$ during classification training, can we use the classifier to provide feedback inputs directly to the generator?"*

**Instructor's Clarification:**  
Yes, that is the general idea, but we must be highly precise about *how* that feedback is utilized. Let us look at why a static, fixed classifier is insufficient for this task.

---

## Section 4: Why a Fixed Classifier Fails (The Counter-Example)

It is highly tempting to assume that if a classifier fails to distinguish between two distributions, those distributions must be identical. However, **classifier failure does not necessarily imply $P_\theta = P_x$.**

Let us construct a geometric counter-example in a two-dimensional feature space:

```
                  Feature Y
                      ^
                      |       (Top-Right)
     [True Cluster]   |       [Generated Cluster at P_θ2]
         (P_x)        |                  (P_θ2)
           X X X      |                    O O O
          X X X X     |                   O O O O
                      |
----------------------*----------------------> Feature X
                      |      \ Decision Boundary (D_ω)
                      |       \
                      |        \
                      |         \  [Generated Cluster at P_θ1]
                      |          \           (P_θ1)
                      |           \            O O O
                      |            \          O O O O
                      |             \
```

1. **Initial State ($P_{\theta_1}$):** 
   The true data distribution $P_x$ is clustered in the top-left quadrant. The initial generated distribution $P_{\theta_1}$ is clustered in the bottom-right quadrant.
2. **The Fixed Classifier ($D_\omega$):** 
   We train a linear classifier $D_\omega$. It draws a decision boundary (the diagonal line). 
   * It classifies everything *above and to the left* of the boundary as **Real ($1$)**.
   * It classifies everything *below and to the right* of the boundary as **Fake ($0$)**.
3. **Tweaking the Generator ($\theta$):**
   If we keep the classifier $D_\omega$ fixed and tweak $\theta$ to fool it, the generator's objective is simply to shift its generated samples across the decision boundary so that the classifier outputs $1$ (Real).
4. **The Failure State ($P_{\theta_2}$):**
   The generator shifts its cluster to the top-right quadrant ($P_{\theta_2}$). Because the top-right quadrant lies above the fixed decision boundary, the classifier outputs $1$ for all these samples. 
   
   The classifier now completely fails to identify these samples as fake. However, **$P_{\theta_2}$ is clearly not equal to $P_x$**. They occupy entirely different regions of the feature space!

---

### 🙋‍♂️ Student Doubt 5 (Raghavendra)
> **Raghavendra:** *"Are we assuming at this stage that the classifier is perfect—that it is able to distinguish between real and fake samples all the time?"*

**Instructor's Clarification:**  
When we initialize the process, we assume the classifier is reasonably trained and capable of distinguishing the two clusters. However, this counter-example demonstrates that even if you start with a highly capable classifier, simply keeping it fixed and moving the generator until the classifier fails is mathematically insufficient to force $P_\theta$ to converge to $P_x$.

---

### 🙋‍♂️ Student Doubt 6 (Indrajit)
> **Indrajit:** *"When we move the generated samples $\hat{x}$ above the line, won't the classifier adjust its decision boundary to differentiate them again?"*

**Instructor's Clarification:**  
Precisely! That is the key. In our counter-example, we assumed the classifier was static. To prevent the generator from exploiting static gaps in the decision boundary, **the classifier must be updated dynamically**. It must adjust its boundary to catch the generator's new position. 

This introduces a dynamic, alternating training process—a **cat-and-mouse game**.

---

### 🙋‍♂️ Student Doubt 7 (Harish)
> **Harish:** *"Our ultimate goal is to make $P_\theta = P_x$. Why do we even need a classifier for this? Can't we just set this up as a standard regression problem, find an $L_2$ loss between the samples, and train the generator directly?"*

**Instructor's Clarification:**  
What would you regress over? If you try to pair every generated sample $\hat{x}_i$ to a specific true sample $x_i$ using an $L_2$ pixel-to-pixel or feature-to-feature loss, two major issues arise:
1. **No direct alignment:** There is no natural, one-to-one mapping between a random noise vector $z_i$ and a specific real data point $x_i$.
2. **Regression to the mean:** Minimizing $L_2$ distances across unaligned distributions forces the generator to output the average of all possible data points. In image generation, this results in extremely blurry, highly unrealistic images that represent the statistical mean rather than distinct samples from the distribution.

We do not want to match samples point-by-pixel; **we want to match distributions at a global, distributional level.** This is why we must use an adversarial setup.

---

## Section 5: The Minimax Game and Mode Collapse

To resolve the issue of the generator exploiting a static classifier, we must train both networks interactively.

### The Cat-and-Mouse Dynamics
1. Start with generator $g_\theta$ producing $P_{\theta_1}$ and train a classifier $D_{\omega_1}$ to separate it from $P_x$.
2. Tweak the generator to fool $D_{\omega_1}$, shifting the distribution to $P_{\theta_2}$.
3. Retrain the classifier to obtain a new boundary $D_{\omega_2}$ that successfully separates $P_{\theta_2}$ from $P_x$.
4. Tweak the generator again to shift the distribution to $P_{\theta_3}$ to fool $D_{\omega_2}$.
5. Repeat this process iteratively.

### The Risk of Mode Collapse
This alternating setup introduces a major training instability known as **Mode Collapse**. 

If the generator only has enough capacity to model a single cluster (a "mode") of the target distribution, it will continuously jump its cluster from one region to another to escape the newly updated decision boundaries of the classifier. It plays a perpetual game of hide-and-seek without ever spreading out to cover the entire true distribution $P_x$.

### Mathematical Formulation
To formalize this adversarial game, we construct a minimax objective function. Let $D_\omega(x) \in [0, 1]$ denote the classifier's predicted probability that a sample $x$ is real.

1. **The Classifier's Goal ($\max_\omega$):**  
   The classifier wants to maximize the probability of correctly labeling both real and fake samples.
   * For real data $x \sim P_x$, it maximizes $\mathbb{E}_{x \sim P_x}[\log D_\omega(x)]$.
   * For fake data $\hat{x} \sim P_\theta$, it maximizes $\mathbb{E}_{\hat{x} \sim P_\theta}[\log(1 - D_\omega(\hat{x}))]$.

2. **The Generator's Goal ($\min_\theta$):**  
   The generator wants to minimize the classifier's ability to distinguish between the two, which is equivalent to minimizing the exact same objective function.

This leads directly to the formulation of the **Saddle Point / Minimax Problem**:

$$\min_\theta \max_\omega J(\theta, \omega) = \min_\theta \max_\omega \left( \mathbb{E}_{x \sim P_x}[\log D_\omega(x)] + \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D_\omega(g_\theta(z)))] \right)$$

---

## Section 6: Post-Training Inference (Generation)

Once the minimax optimization converges to an equilibrium (ideally where $P_\theta = P_x$ and $D_\omega(x) = 0.5$ everywhere), how do we use the trained model?

```
During Inference:
   z ~ N(0, I)  -----> [ Trained Generator g_θ* ] -----> Generated Sample x_cap ~ P_x
                         (Discriminator discarded)
```

At this stage, the discriminator has completed its job. 

> *"The discriminator/classifier is like a school teacher. Once you have fully learned the material from them, you can graduate and go on your way—the teacher is no longer needed during active generation."*

To generate new data:
1. Sample a latent vector $z \sim \mathcal{N}(0, I)$.
2. Pass it through the trained generator $g_{\theta^*}(z)$ to obtain a synthetic sample $\hat{x}$.
3. Discard the discriminator $D_{\omega^*}$ entirely.

Ideally, the generated samples $\hat{x}_1, \dots, \hat{x}_k$ are novel; they do not exist in the training dataset $D$, but they look as though they were drawn directly from the true distribution $P_x$.

---

## Section 7: From Multi-Layer Perceptrons to Deep Convolutional GANs (DCGANs)

When implementing GANs practically, the choice of neural network architecture significantly impacts the quality of the generated outputs, especially for spatial data like images.

### The Flattening Problem in MLP-GANs
In a standard Multi-Layer Perceptron (MLP) GAN:
* The latent vector $z \in \mathbb{R}^{16}$ is mapped through fully connected (FC) layers to a flat vector of size, say, $10,000$.
* This flat vector is then reshaped into a $100 \times 100$ image.
* Similarly, the discriminator flattens the $100 \times 100$ image back into a $10,000$-dimensional vector before feeding it through fully connected layers.

This approach discards crucial spatial correlations and local structures, making it highly difficult for the model to generate coherent images.

### The DCGAN Architecture
Deep Convolutional GANs (DCGANs) preserve spatial structures by replacing fully connected layers with convolutional operations:

1. **The Discriminator:**  
   A standard Convolutional Neural Network (CNN) (such as a ResNet variant) that takes a spatial image (e.g., $100 \times 100 \times 3$) and downsamples it through successive convolutional layers to output a single scalar probability in $[0, 1]$.

2. **The Generator:**  
   Instead of downsampling, the generator must upscale a low-dimensional vector $z \in \mathbb{R}^{16}$ into a high-dimensional image (e.g., $100 \times 100 \times 3$). This upscaling is achieved using **Transpose Convolutions** (sometimes referred to as fractionally-strided convolutions or deconvolutional layers).

```
MLP-GAN Pipeline:
z (16) --[Fully Connected]--> flat (10,000) --[Reshape]--> image (100x100)

DCGAN Pipeline (Spatial):
z (16x1x1) --[Transpose Conv]--> (32x8x8) --[Transpose Conv]--> image (3/100x100)
```

*For a detailed mathematical treatment of how strides, padding, and kernels behave in reverse, students are highly encouraged to read the classic reference paper:* **"A guide to convolution arithmetic for deep learning."**

---

## Section 8: Variational Analysis of f-Divergences

### 🙋‍♂️ Student Doubt 8 (Raghavendra)
> **Raghavendra:** *"You mentioned that this classifier-guided interpretation holds true specifically for the Jensen-Shannon divergence. Why is that the case? Why doesn't it apply to other f-divergences like Kullback-Leibler (KL) or Pearson $\chi^2$?"*

**Instructor's Clarification:**  
This is an excellent and subtle point. The classification interpretation depends entirely on the **domain of the Fenchel conjugate $f^*(t)$** and the resulting range of the variational function $T(x)$.

Let us look at the recommended output activations and domains for different divergences:

| Divergence Name | Generator $f(u)$ | Conjugate $f^*(t)$ | Domain of $f^*(t)$ | Variational Activation $g_f(s)$ |
| :--- | :--- | :--- | :--- | :--- |
| **Kullback-Leibler (KL)** | $u \log u$ | $e^{t-1}$ | $\mathbb{R}$ | $s$ (Linear/No activation) |
| **Pearson $\chi^2$** | $(u-1)^2$ | $t + \frac{t^2}{4}$ | $\mathbb{R}$ | $s$ (Linear/No activation) |
| **Jensen-Shannon (JS)** | $u \log u - (u+1) \log \left(\frac{u+1}{2}\right)$ | $-\log(2 - e^t)$ | $t < \log 2$ | $-\log(1 + e^{-s})$ |
| **Naive GAN** | $u \log u - (u+1) \log (u+1)$ | $-\log(1 - e^t)$ | $t < 0$ | $-\log(1 + e^{-s})$ |

### Why Only JS / Naive GAN Yields a Binary Classifier:
1. **Bounded Outputs:** For the JS and Naive GAN formulations, the output activation function maps the variational network's output to a strictly bounded domain (specifically, after applying a sigmoid transformation, it maps directly to $[0, 1]$). This allows us to interpret the output of the network $D_\omega(x)$ directly as a probability (likelihood of being real).
2. **Unbounded Regressors:** For other divergences like KL or Pearson $\chi^2$, the domain of the conjugate function $f^*(t)$ is the entire real line ($\mathbb{R}$). The optimal variational function $T^*(x)$ is not a bounded probability, but rather a ratio of densities (or a function of it) that can take any real value from $-\infty$ to $+\infty$. 
   
   In those cases, the variational network acts as a **regressor** rather than a binary classifier. There is no simple way to map its unbounded outputs to a standard "real ($1$) vs. fake ($0$)" classification framework.

  Thus, while the general principle of updating a variational network to tighten the lower bound of an $f$-divergence remains identical across all GANs, **interpreting the discriminator specifically as a binary classifier is a unique property of the Jensen-Shannon / Naive GAN formulation.**


# Lecture Notes: Deep Generative Models—Conditional GANs & Unpaired Image-to-Image Translation (CycleGAN)

In the study of generative adversarial networks (GANs), we often begin with unconditional models such as DC-GAN. Unconditional models attempt to learn the marginal distribution of our target data, $P_X$, by mapping an arbitrary noise vector $z \sim \mathcal{N}(0, I)$ to a sample $x$. While this approach can generate realistic samples, it lacks control: we cannot specify *what* class or variation of data the model should generate. 

To address this limitation, the deep learning community introduced conditional formulations and cross-domain mapping techniques [1703.10593]. These notes cover the theoretical foundations, objective functions, architectural designs, and common student inquiries regarding Conditional GANs (C-GANs) and Cycle-Consistent GANs (CycleGANs).

---

## 1. Conditional GANs (C-GAN)

### 1.1 Objective and Setup
In a standard, unconditional GAN, the generator models the data distribution $P_X$ using a mapping function $G_\theta(z)$. In a Conditional GAN, our objective transitions from modeling the marginal distribution $P_X$ to modeling the **conditional distribution** $P(X | Y)$ [1411.1784]. Here, the random variable $Y$ represents a conditioning variable, such as a class label or a textual embedding [1411.1784].

```
Unconditional GAN:  z ~ N(0, I)  --------> [  G_θ  ] --------> x ~ P_X
Conditional GAN:    z ~ N(0, I)  ---\
                                     =======> [  G_θ  ] --------> x_hat ~ P(X|Y)
                    y (Condition) --/
```

We assume our training dataset consists of paired samples:
$$\mathcal{D} = \{ (x_1, y_1), (x_2, y_2), \dots, (x_n, y_n) \} \sim p_{XY}$$

Where:
*   $x_i \in \mathcal{X}$ is the target data (e.g., an image) [1411.1784].
*   $y_i \in \mathcal{Y}$ is the conditioning variable (e.g., class label, text embedding) [1411.1784].

---

### 💬 Deep-Dive Doubt: What can $X$ and $Y$ represent?
During the lecture, **Sanchit** asked:
> *"Is $X$ limited only to images, or can it be something else?"*

**Instructor's Explanation:**
$X$ is not restricted to images. GANs are general-purpose implicit density estimators. For instance, in **Text-to-Speech (TTS)**, the conditioning variable $Y$ is a text sequence, and the output $X$ is a speech waveform. The same conditional framework applies; we simply condition the speech generator on the text features to synthesize class-consistent audio waves.

**Sanchit** followed up:
> *"How is this different from Transformer algorithms (like GPT) where we also provide a text prompt to generate outputs?"*

**Instructor's Explanation:**
Models like GPT are **autoregressive models**. They generate output sequentially, token by token, by modeling the joint probability distribution via the chain rule of probability: $P(x_1, x_2, \dots) = \prod P(x_i | x_{<i})$. C-GANs, on the other hand, are **adversarially trained models**. They produce the entire output (e.g., an image or a signal) in a single parallel forward pass through a generator, trained via a minimax game against a discriminator. The optimization principles and architectural backbones are fundamentally different.

---

### 1.2 Mathematical Formulation and Loss Function
To train a C-GAN, both the generator $G_\theta$ and the discriminator $D_w$ must receive the conditioning variable $y$ [1411.1784]. The minimax objective function $J(\theta, w)$ is written as:

$$\min_{\theta} \max_w J(\theta, w) = \mathbb{E}_{(x, y) \sim p_{XY}} \left[ \log D_w(x, y) \right] + \mathbb{E}_{y \sim p_Y, \, z \sim p_z} \left[ \log \left( 1 - D_w(G_\theta(z, y), y) \right) \right]$$

#### Breaking Down the Expectations:
1.  **First Term (Real Data)**: We sample actual pairs of data and conditions $(x, y)$ from our empirical joint distribution $p_{XY}$. The discriminator $D_w(x, y)$ outputs the probability that $x$ is real *under the condition $y$* [1411.1784].
2.  **Second Term (Generated Data)**: We sample a condition $y$ from the marginal distribution $p_Y$ and a noise vector $z$ from our prior $p_z$ (typically a standard Gaussian $\mathcal{N}(0, I)$) [1411.1784]. The generator synthesizes a candidate $\hat{x} = G_\theta(z, y)$ [1411.1784]. The discriminator evaluates this sample as $D_w(G_\theta(z, y), y)$, determining whether the generated sample matches the condition $y$ [1411.1784].

---

### 💬 Deep-Dive Doubt: Handling the conditioning variable $Y$
**Kartik Kumar** raised an important implementation question:
> *"While training, do we give all classes? How do we handle $Y$ during training versus inference?"*

**Instructor's Explanation:**
Yes, during training, the dataset contains samples from all classes (e.g., all digits $0\text{--}9$ in MNIST) [1411.1784]. The model learns the joint distribution $P(X, Y)$ by experiencing varied pairings. 

At **inference time (post-training)**, the conditioning variable $Y$ acts as a **control switch**. If we want to generate a digit '5', we explicitly set $y$ to represent '5', sample $z \sim \mathcal{N}(0, I)$, and pass them to the frozen generator $G_{\theta^*}(z, y)$. The generator will output a synthetic '5'. This allows us to direct the generation process explicitly.

---

### 1.3 Architectural Implementation
The conditioning variable $y$ can be represented in different formats depending on its type:
*   **Discrete Classes**: Typically represented as a **one-hot vector** [1411.1784]. For a 10-class dataset (like MNIST), $y \in \{0, 1\}^{10}$.
*   **Text / Continuous Attributes**: Represented as a continuous **embedding vector** (e.g., TF-IDF, BERT embeddings, or CLIP embeddings) [1411.1784].

```
Generator Block Diagram:
z (Noise Vector) ---------\
                           [ Concatenation / Projection ] ---> [ Generator Network G_θ ] ---> x_hat
y (One-hot / Embedding) --/

Discriminator Block Diagram:
x (Real x or Fake x_hat) --\
                           [ Concatenation / Projection ] ---> [ Discriminator Network D_w ] ---> Scalar [0, 1]
y (One-hot / Embedding) ---/
```

---

### 💬 Deep-Dive Doubt: Vector Concatenation
**Vivek** asked:
> *"How do we concatenate or join the image vector $x$ and the conditioning vector $y$ in the discriminator?"*

**Instructor's Explanation:**
In simple Multilayer Perceptrons (MLPs), we flatten the image $x$ into a $d_x$-dimensional vector and concatenate it with the $d_y$-dimensional conditioning vector $y$ to form a single $(d_x + d_y)$-dimensional input vector [1411.1784]. 

In Convolutional Neural Networks (CNNs), we can pass $y$ through a linear layer to map it to the spatial dimensions of the image (or feature maps) as an additional channel [1411.1784]. For example, if a feature map has shape $H \times W \times C$, we can tile the conditioning vector $y$ to shape $H \times W \times C_y$ and concatenate them along the channel dimension to get $H \times W \times (C + C_y)$.

**Sushil** also asked:
> *"What is the core difference between C-GAN and normal GANs?"*

**Instructor's Explanation:**
In an unconditional GAN, the mapping is from a latent space directly to the data space: $G: \mathcal{Z} \to \mathcal{X}$. There is no way to enforce semantic meaning onto specific regions of $\mathcal{Z}$ without post-hoc latent space exploration. C-GAN enforces a conditional structure $G: \mathcal{Z} \times \mathcal{Y} \to \mathcal{X}$, giving us direct control over the semantic output by feeding $y$ explicitly [1411.1784].

---

## 2. Unpaired Image-to-Image Translation (CycleGAN)

Conditional GANs perform exceptionally well when we have paired training data, $\mathcal{D} = \{(x_i, y_i)\}$. However, for many tasks, paired data is extremely difficult or impossible to obtain [1703.10593]. For example, we cannot easily obtain photos of the exact same scene in both summer and winter, or a portrait of the same person as both a human and a horse [1703.10593].

```
Paired Data (e.g., Pix2Pix):       [ Sketch of Shoe ] <======== Exact 1-to-1 Mapping ========> [ Real Shoe Photo ]
Unpaired Data (e.g., CycleGAN):    { Set of Horse Photos } <--- No 1-to-1 Mapping ---> { Set of Zebra Photos }
```

### 2.1 Problem Setup
Let us define two independent domains with empirical datasets [1703.10593]:
*   Source domain samples: $\{s_1, s_2, \dots, s_N\} \sim P_S$ (e.g., domain of Horses) [1703.10593].
*   Target domain samples: $\{t_1, t_2, \dots, t_M\} \sim P_T$ (e.g., domain of Zebras) [1703.10593].

We do not have any pairing information indicating which horse $s_i$ corresponds to which zebra $t_j$ [1703.10593]. Our objective is to learn a mapping $G_1: S \to T$ such that the distribution of $G_1(s)$ is indistinguishable from $P_T$ [1703.10593].

---

### 💬 Deep-Dive Doubt: Mapping without Pairing
**Sarvesh** asked:
> *"Shouldn't there be a mapping from which $S$ is mapped to which $T$?"*

**Instructor's Explanation:**
In paired frameworks like Pix2Pix, yes, we require a explicit $s_i \leftrightarrow t_i$ pair during training. But the entire purpose of **CycleGAN** is to learn this translation **without paired data** [1703.10593]. We only need two unpaired collections of images. CycleGAN will learn to translate a horse to a zebra while preserving the background structure, purely by utilizing **cycle consistency** [1703.10593].

**Harish** added:
> *"What if we do not know the underlying distributions $P_S$ and $P_T$?"*

**Instructor's Explanation:**
We almost never know the analytical form of $P_S$ or $P_T$. We only have empirical samples (images of horses and zebras). GANs circumvent the need for analytical probability density functions by using a discriminator as a surrogate of divergence [1703.10593]. The discriminator learns to distinguish between the empirical samples of the target domain and the generated samples [1703.10593].

---

### 2.2 Why Adversarial Loss Alone Fails
If we only use a standard adversarial loss for our mapping $G_1: S \to T$ with a discriminator $D_T$, the generator $G_1$ is free to map any source image $s$ to *any* realistic image in domain $T$ [1703.10593]. For instance, it could map every single horse image to the exact same zebra image, or scramble the semantic layout completely while still outputting a photo that looks like a valid zebra [1703.10593]. This is a form of mode collapse and semantic drift [1703.10593]. 

To force the model to preserve the structural and semantic properties of the input image $s$, we must constrain the mapping to be **invertible** [1703.10593].

---

### 2.3 Cycle Consistency
The core contribution of CycleGAN is the **Cycle Consistency Loss** [1703.10593]. It states that if we translate an image from domain $S$ to domain $T$, and then translate it back to domain $S$, we should arrive back at our original image [1703.10593]:

$$G_2(G_1(s)) \approx s \quad \text{and} \quad G_1(G_2(t)) \approx t$$

To achieve this, we train two generators and two discriminators simultaneously [1703.10593]:
1.  **Generators**:
    *   $G_1: S \to T$ [1703.10593]
    *   $G_2: T \to S$ [1703.10593]
2.  **Discriminators**:
    *   $D_T$: Distinguishes between real target images $t$ and generated target images $G_1(s)$ [1703.10593].
    *   $D_S$: Distinguishes between real source images $s$ and generated source images $G_2(t)$ [1703.10593].

```
Forward Cycle:  s (Source) ---> [ G_1 ] ---> t_hat (Target) ---> [ G_2 ] ---> s_tilde ≈ s (Reconstructed)
Backward Cycle: t (Target) ---> [ G_2 ] ---> s_hat (Source) ---> [ G_1 ] ---> t_tilde ≈ t (Reconstructed)
```

---

### 💬 Deep-Dive Doubt: Autoencoders vs. CycleGAN
**Vivek** observed:
> *"In this case, the mapping $G_2(G_1(s))$ looks like an autoencoder, right?"*

**Instructor's Explanation:**
The mathematical formulation resembles an autoencoder because we map an input to a latent/intermediate representation and reconstruct it: $s \to G_1(s) \to G_2(G_1(s)) \approx s$ [1703.10593]. 

However, there is a major difference. A standard autoencoder compresses the input through a low-dimensional bottle-neck (latent space $\mathcal{Z}$) to prevent identity mapping. In CycleGAN, the intermediate representation $G_1(s)$ is not a low-dimensional vector; it is a **high-dimensional image in another domain** (e.g., a fully-detailed zebra image) [1703.10593]. We prevent the trivial identity mapping by enforcing that $G_1(s)$ must satisfy the adversarial loss of domain $T$ [1703.10593]. It forces the intermediate representation to be a valid translation rather than arbitrary code [1703.10593].

---

### 2.4 Mathematical Loss Functions of CycleGAN
The total loss function of CycleGAN is a combination of adversarial losses and the cycle consistency loss [1703.10593]:

$$\mathcal{L}(G_1, G_2, D_S, D_T) = \mathcal{L}_{GAN}(G_1, D_T, S, T) + \mathcal{L}_{GAN}(G_2, D_S, T, S) + \lambda \mathcal{L}_{cyc}(G_1, G_2)$$

Where $\lambda$ is a hyperparameter that controls the relative importance of the cycle consistency loss [1703.10593].

#### 1. Adversarial Losses
For the mapping $G_1: S \to T$ and its discriminator $D_T$ [1703.10593]:
$$\mathcal{L}_{GAN}(G_1, D_T, S, T) = \mathbb{E}_{t \sim P_T}[\log D_T(t)] + \mathbb{E}_{s \sim P_S}[\log(1 - D_T(G_1(s)))]$$

For the mapping $G_2: T \to S$ and its discriminator $D_S$ [1703.10593]:
$$\mathcal{L}_{GAN}(G_2, D_S, T, S) = \mathbb{E}_{s \sim P_S}[\log D_S(s)] + \mathbb{E}_{t \sim P_T}[\log(1 - D_S(G_2(t)))]$$

#### 2. Cycle Consistency Loss
This loss measures the reconstruction error at the pixel level, usually using the L1 norm to promote sharper reconstructions [1703.10593]:

$$\mathcal{L}_{cyc}(G_1, G_2) = \mathbb{E}_{s \sim P_S} \left[ \| G_2(G_1(s)) - s \|_1 \right] + \mathbb{E}_{t \sim P_T} \left[ \| G_1(G_2(t)) - t \|_1 \right]$$

---

### 💬 Deep-Dive Doubt: Loss Computation
A **student** asked:
> *"Are the differences between the reconstructed images computed at the pixel level?"*

**Instructor's Explanation:**
Yes, the cycle consistency loss is computed at the pixel level. We compute the L1 norm (the absolute pixel-by-pixel difference) between the original image $s$ and the reconstructed image $G_2(G_1(s))$ [1703.10593]. This pixel-level constraint acts as a regularizer, forcing the generators to maintain the spatial structure, edges, and semantic content of the input image [1703.10593].

**Sushil** followed up on a broader evaluation question:
> *"How do we calculate or identify the performance of these GANs to know if they are performing well on the data?"*

**Instructor's Explanation:**
Evaluating generative models is inherently difficult because we do not have explicit likelihoods. While we will cover evaluation metrics in detail later, the current standard in the literature is to use the **Fréchet Inception Distance (FID)**. FID passes both real and generated images through a pre-trained ImageNet classifier (like Inception-V3), extracts feature representations from an intermediate layer, and calculates the Wasserstein-2 distance between the fitted multivariate Gaussians of those features. A lower FID score generally correlates with better visual quality and conditional consistency.

---

## 3. Summary of Key Concepts

| Concept / Metric | Conditional GAN (C-GAN) | Cycle-Consistent GAN (CycleGAN) |
| :--- | :--- | :--- |
| **Data Requirement** | Paired data $\{(x_i, y_i)\} \sim P_{XY}$ [1411.1784]. | Unpaired data $S \sim P_S$ and $T \sim P_T$ [1703.10593]. |
| **Generators** | One generator: $G(z, y)$ [1411.1784]. | Two generators: $G_1: S \to T$ and $G_2: T \to S$ [1703.10593]. |
| **Discriminators** | One discriminator: $D(x, y)$ [1411.1784]. | Two discriminators: $D_T$ and $D_S$ [1703.10593]. |
| **Objective Constraints** | Minimax objective conditioned on class/text [1411.1784]. | Minimax objective + Cycle Consistency Loss ($L_1$ norm) [1703.10593]. |
| **Primary Use Case** | Text-to-Image, class-conditioned generation [1411.1784]. | Unpaired style transfer, domain adaptation [1703.10593]. |

---

## 4. Practical Implementation Checklist (For Assignments)

If you are implementing the class-conditional generation task from your assignment (e.g., generating animal faces from 10 or 20 subclasses out of 90 classes):

1.  **Define the switch ($y$)**: Convert your class IDs into $C$-dimensional one-hot vectors.
2.  **Generator Inputs**: Sample $z \sim \mathcal{N}(0, I)$. Project/reshape $y$ and concatenate it with $z$ before passing it to the generator conv-transpose blocks.
3.  **Discriminator Inputs**: Concatenate the conditioning variable $y$ (reshaped/tiled to match spatial dimensions) with the image channels ($3 + C$) at the input of the discriminator.
4.  **Minimax Optimization**: Remember to update the discriminator and generator alternately. Ensure the discriminator is trained with both real pairs $(x, y)$ and fake pairs $(\hat{x}, y)$ [1411.1784].

*This concludes our overview of Conditional GANs and CycleGANs. In the next lecture, we will discuss Autoregressive Models (such as PixelCNN) and move towards Diffusion Models.*