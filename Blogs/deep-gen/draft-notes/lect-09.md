# Deep Generative Models (E1 286s): Lecture Notes on VAE Inference, the $\beta$-VAE Trade-off, and VQ-VAEs

---

## Introduction: Post-Training VAE Workflow

Once the training of a Variational Autoencoder (VAE) is complete, we are left with a optimized encoder network $q_{\phi^*}(z|x)$ and a decoder network $p_{\theta^*}(x|z)$ (where the asterisks denote the final trained parameters). In the deployment phase, we utilize these two halves of the network for two distinct primary tasks: **Posterior Inference** (mapping data to the latent space) and **Data Generation** (sampling new data from the latent space). 

This guide covers the mechanics of both tasks, discusses critical training trade-offs such as *posterior collapse*, and provides a conceptual transition into Vector Quantized VAEs (VQ-VAEs).

---

## 1. Posterior Inference (Obtaining Embeddings)

Posterior inference answers the question: *Given a novel, unseen test datapoint $x_{\text{test}}$, what is its corresponding representation in the lower-dimensional latent space?*

```
             ┌────────────────────────┐
             │       Encoder          │     Mean: μ_φ*(x_test)
x_test ───►  │  q_φ*(z | x_test)      ├────────► [Embedding Option 1]
             │                        │
             └──────────┬─────────────┘     Variance/Std Dev: Σ_φ*(x_test)
                        │
                        ▼
                [Reparameterization] ─────────► Sampled Latent z_test
                z_test = μ + ε * Σ              [Embedding Option 2]
```

### The Mathematical Pipeline
We pass $x_{\text{test}}$ through our trained encoder. Since the encoder in a VAE is probabilistic, it does not directly output a single vector $z_{\text{test}}$. Instead, it outputs the parameters of a conditional Gaussian distribution:
1. **The Latent Mean**: $\mu_{\phi^*}(x_{\text{test}})$
2. **The Latent Diagonal Covariance/Standard Deviation**: $\Sigma_{\phi^*}(x_{\text{test}})$

To obtain a concrete latent vector $z_{\text{test}}$ (also called the **embedding**), we have two main operational choices:

#### Option A: Deterministic Mean Embedding (Most Common for Downstream Tasks)
$$\text{Embedding}(x_{\text{test}}) = \mu_{\phi^*}(x_{\text{test}})$$
In this option, we ignore the variance and use the mean of the predicted distribution directly as the embedding. This is typically preferred for downstream tasks because it is deterministic and represents the central tendency of the encoded data point.

#### Option B: Reparameterized Sampled Embedding
$$\text{Embedding}(x_{\text{test}}) = z_{\text{test}} = \mu_{\phi^*}(x_{\text{test}}) + \epsilon_t \odot \Sigma_{\phi^*}(x_{\text{test}}), \quad \text{where } \epsilon_t \sim \mathcal{N}(0, I)$$
Here, we sample a random noise vector $\epsilon_t$ from a standard normal distribution and scale/shift it using the encoder's outputs. This preserves the stochastic nature of the VAE.

---

### Student Seminar Discussion & Doubts

#### **Doubt 1 (Shirisha):** 
> *"When you say $z_{\text{test}}$ can be used for classification, what kind of classification do you mean? How does it compare to standard approaches?"*

**Explanation:**  
If we train a VAE on the MNIST dataset, the raw input dimension of each image is $28 \times 28 = 784$. A standard classification pipeline might feed these 784-dimensional pixel vectors into a Convolutional Neural Network (CNN) to predict digits ($0-9$).

With a trained VAE, we can leverage the low-dimensional latent space (e.g., $d = 10$). For every image $x_i$, we pass it through the encoder to extract its low-dimensional representation $z_i = \mu_{\phi^*}(x_i)$. Because the VAE is trained to group similar semantic structures together, the latent space is often well-clustered. 

We can then train a significantly smaller and simpler classifier—such as a Multi-Layer Perceptron (MLP) or even a linear classifier—directly on the 10-dimensional space of $z$ rather than the 784-dimensional pixel space. In your course assignments, you are expected to compare these two paradigms:
1. Training a full CNN classifier directly on the high-dimensional data space.
2. Training a simple MLP on the VAE-derived embeddings ($z$) and comparing the classification accuracy and parameter efficiency.

#### **Doubt 2 (Sarvesh):** 
> *"In the reparameterization formula $z_{\text{test}} = \mu + \epsilon_t \cdot \Sigma$, shouldn't we raise the $\Sigma$ term to the power of $1/2$ (take the square root) because we need the standard deviation rather than the variance?"*

**Explanation:**  
This depends entirely on the design of the encoder network's output heads:
* If your neural network is configured to output the **variance** ($\sigma^2$), then yes, you must take the square root ($\Sigma = \sqrt{\sigma^2}$) to obtain the standard deviation for the reparameterization trick.
* In practice, to avoid dealing with negative numbers (since variance must be strictly positive), networks are often designed to output $\log(\sigma^2)$ or directly output the **standard deviation** $\sigma$ via an activation function like Softplus. If the output of your network is already representing standard deviation, taking the square root is unnecessary. It is simply a matter of parameterization.

#### **Doubt 3 (Sanchit):** 
> *"Is this process technically what is meant by 'variational inference'?"*

**Explanation:**  
Yes, this is specifically referred to as **Variational Posterior Inference**. 

In Bayesian modeling, we are interested in finding the true posterior distribution of the latent variables given the data: $p_{\theta^*}(z|x)$. Because calculating this true posterior is mathematically intractable (requiring integration over the entire latent space), we approximate it. 

We define a family of tractable distributions, $q_\phi(z|x)$ (our encoder), and find the parameters $\phi$ that bring this approximation as close as possible to the true posterior. We do this by minimizing the Kullback-Leibler (KL) divergence:
$$D_{\text{KL}}\left(q_\phi(z|x) \,\|\, p_\theta(z|x)\right)$$

Since $p_{\theta}(z|x)$ is unknown, we cannot minimize this term directly. Instead, we manipulate this relationship using Bayes' rule to derive the Evidence Lower Bound (ELBO):
$$\log p_\theta(x) - D_{\text{KL}}\left(q_\phi(z|x) \,\|\, p_\theta(z|x)\right) = \text{ELBO}(\theta, \phi)$$

Because the KL divergence is non-negative ($D_{\text{KL}} \ge 0$), maximizing the ELBO directly maximizes the log-likelihood of our data while implicitly minimizing the divergence between our variational posterior $q_\phi(z|x)$ and the true posterior $p_\theta(z|x)$.

---

## 2. Data Generation & Sampling

Data generation is the process of synthesizing completely new, realistic data points that did not exist in our training set.

```
          Sample Prior
          z ~ N(0, I)
               │
               ▼
     ┌──────────────────┐
     │     Decoder      │
     │   p_θ*(x | z)    ├────────► Generated Datapoint: x̂
     │                  │
     └──────────────────┘
```

### The Generation Pipeline
1. **Sample from the Prior**: We draw a random latent vector $z$ directly from our known prior distribution:
   $$z \sim \mathcal{N}(0, I)$$
2. **Decode**: We pass this sampled vector through our trained decoder network $p_{\theta^*}(x|z)$ to produce a synthesized data point:
   $$\hat{x} \sim p_{\theta^*}(x|z)$$

---

### Student Seminar Discussion & Doubts

#### **Doubt 4 (Sanchit / Ravindra):** 
> *"Why does sampling $z$ from $\mathcal{N}(0, I)$ actually work? During training, the decoder only saw latent representations produced by the encoder $q_{\phi}(z|x)$. Why can we suddenly use a generic Gaussian at test time?"*

**Explanation:**  
This is the fundamental reason VAEs are capable of generation, whereas naive Autoencoders are not. 

In a naive Autoencoder, the encoder can map data points to arbitrary, highly isolated regions of the latent space. There is no constraint forcing these embeddings to populate the latent space in a coherent or continuous manner. As a result, the latent space is full of "voids"—regions where the decoder has never seen any data. If you sample a random $z$ from an unregularized latent space, the decoder will output meaningless noise.

In a VAE, the loss function includes the KL divergence regularization term:
$$D_{\text{KL}}\left(q_\phi(z|x) \,\|\, \mathcal{N}(0, I)\right)$$

This term acts as a force pull, penalizing the encoder if its output distributions stray away from a standard normal distribution. Consequently, the encoder is forced to pack all latent representations near the center of the standard normal space. 

Because of this constraint, the decoder is thoroughly trained to reconstruct valid, realistic data points from any latent vector sampled from the $\mathcal{N}(0, I)$ neighborhood.

#### **Doubt 5 (Indrajit):** 
> *"Can you explicitly clarify the difference between Reconstruction and Generation?"*

**Explanation:**  
Let us define these two tasks clearly:

* **Reconstruction**: You start with an *existing* data point $x$ from your dataset. You pass it through the encoder to get $z$, and then pass that $z$ through the decoder to get $\hat{x}$. The objective is to make $\hat{x}$ an exact replica of $x$. This is an autoencoding task:
  $$x \longrightarrow \text{Encoder} \longrightarrow z \longrightarrow \text{Decoder} \longrightarrow \hat{x} \approx x$$
* **Generation (Sampling)**: You do *not* have an input data point. You start by drawing a random vector $z$ from the prior distribution $\mathcal{N}(0, I)$ and feed it directly into the decoder to generate a brand-new data point $\hat{x}$. There is no encoder involved in this path:
  $$z \sim \mathcal{N}(0, I) \longrightarrow \text{Decoder} \longrightarrow \hat{x}_{\text{new}}$$

---

## 3. The Reconstruction vs. Regularization Trade-off ($\beta$-VAE)

In a standard VAE, we optimize the ELBO, which can be expressed as a loss minimization objective:
$$\mathcal{L}_{\text{VAE}} = \text{Reconstruction Loss} + D_{\text{KL}}\left(q_\phi(z|x) \,\|\, \mathcal{N}(0, I)\right)$$

This setup reveals a natural tension between its two components:
1. **Reconstruction Loss**: Demands that the encoder preserve as much detail about $x$ as possible, which encourages unique, distinct latent representations (high variance in latent space).
2. **KL Regularization**: Demands that the latent representations for *all* inputs match the prior $\mathcal{N}(0, I)$. If this term dominates, the encoder is pressured to map every input to the exact same standard normal distribution, losing all unique information about individual inputs.

### Posterior Collapse
If the regularization term is too dominant or if the decoder network is too powerful (e.g., a highly autoregressive model), the network can experience **posterior collapse**. In this scenario, the model discovers that the easiest way to minimize the loss is to let the encoder completely ignore the input $x$ and output the prior:
$$q_\phi(z|x) \approx \mathcal{N}(0, I) \quad \forall \, x$$
When this happens, the KL divergence term drops to zero, but the latent space becomes completely useless for representing data because $z$ no longer contains any information about $x$.

### The $\beta$-VAE Formulation
To manage this trade-off, Higgins et al. introduced the **$\beta$-VAE**, which scales the influence of the KL regularization term using a hyperparameter $\beta \ge 0$:
$$\mathcal{L}_{\beta\text{-VAE}} = \|x - \hat{x}\|_2^2 + \beta \, D_{\text{KL}}\left(q_\phi(z|x) \,\|\, \mathcal{N}(0, I)\right)$$

By adjusting the "knob" of $\beta$, we can observe distinct behaviors in the model:

| $\beta$ Value | Optimization Focus | Latent Space Behavior | Visual Consequences |
| :--- | :--- | :--- | :--- |
| **$\beta = 0$** | 100% Reconstruction | Unregularized. Becomes a naive Autoencoder. | Sharp reconstructions; generation is impossible due to an unregularized latent space. |
| **$0 < \beta < 1$** | Heavy Reconstruction | Latent distribution deviates from Gaussian; retains fine-grained data details. | Highly detailed, sharp reconstructions; generated samples may look unrealistic due to gaps in the latent space. |
| **$\beta = 1$** | Balanced (Standard VAE) | Standard VAE trade-off. | Decent balance of generation and reconstruction. |
| **$\beta > 1$** | Heavy Regularization | Forces the latent space to be strictly Gaussian. Disentangles latent factors. | Blurry reconstructions due to information loss; generated samples are highly coherent but lack fine details (*potential posterior collapse*). |

---

### Student Seminar Discussion & Doubts

#### **Doubt 6 (Vivek):** 
> *"Does setting $\beta = 0$ truly turn the VAE back into a standard, non-probabilistic Autoencoder?"*

**Explanation:**  
Yes, absolutely. If you set $\beta = 0$, you completely eliminate the KL divergence penalty. The network will find that the easiest way to reduce the reconstruction error to its absolute minimum is to map the latent distributions to highly concentrated delta-functions (zero variance) in arbitrary locations of the latent space. 

Without the KL regularization term, the probabilistic nature of the VAE serves no purpose, and the model mathematically collapses back into a naive Autoencoder.

---

## 4. Vector Quantized VAE (VQ-VAE)

To resolve the trade-off limitations of continuous VAEs (such as blurriness and posterior collapse), van den Oord et al. introduced the **Vector Quantized VAE (VQ-VAE)**, which utilizes a discrete latent space rather than a continuous Gaussian one.

### The Core Motivation
Many natural structures are inherently discrete. For instance, human speech is composed of a finite set of distinct phonemes. Similarly, images can be decomposed into discrete components (textures, shapes, objects).

Instead of representing these structures with a continuous Gaussian latent distribution, VQ-VAE represents data using a discrete **Latent Dictionary** (also known as a **Codebook**) consisting of $M$ vectors, each of dimension $K$:
$$\mathbf{C} = \{z_1, z_2, \dots, z_M\}, \quad z_j \in \mathbb{R}^K$$

```
             Continuous Latent                      Nearest-Neighbor Lookup
             from Encoder: z_e(x)                   in Codebook (Quantization)
x ───► Encoder ───────► [ z_e(x) ] ───► [Nearest Codebook Vector] ───► Decoder ───► x̂
                                                    ▲
                                                    │
                                           Codebook: {z_1, z_2, ..., z_M}
```

### The VQ-VAE Pipeline
1. **Encode**: The encoder $E(x)$ takes an input $x$ and outputs a continuous latent vector $z_e(x)$.
2. **Quantize (Vector Quantization)**: We map the continuous vector $z_e(x)$ to its nearest representative in the codebook. This is done by finding the codebook vector $z_j$ that minimizes the Euclidean distance:
   $$z_q(x) = z_{j^*} \quad \text{where} \quad j^* = \arg\min_{j \in \{1, \dots, M\}} \|z_e(x) - z_j\|_2$$
3. **Decode**: The quantized discrete representation $z_q(x)$ is then fed directly into the decoder $D(z_q(x))$ to reconstruct the input.

This quantization process replaces the continuous Gaussian sampling step of standard VAEs. It completely bypasses the need for a KL divergence regularization term, thereby avoiding the issue of posterior collapse entirely.

---

### Student Seminar Discussion & Doubts

#### **Doubt 7 (Raghavendra):** 
> *"If we limit the codebook to a small, discrete set of vectors (e.g., $M = 100$), is that representation expressive enough to reconstruct complex datasets like ImageNet?"*

**Explanation:**  
The key is that the codebook vectors are not used to represent the entire image as a single vector. 

Instead, the encoder maps a high-dimensional image (e.g., $256 \times 256 \times 3$) to a grid of continuous latent vectors (e.g., a $32 \times 32$ spatial grid, where each grid cell is a $K$-dimensional vector). We then perform vector quantization *separately* for each cell in this grid against our codebook of size $M$.

Consequently, the image is represented by a $32 \times 32$ grid of discrete indexes. Even with a modest codebook size of $M = 512$, the total number of unique latent configurations the model can represent is:
$$512^{32 \times 32} = 512^{1024}$$

This discrete grid representation provides an exceptionally large capacity, allowing VQ-VAEs to reconstruct highly complex, high-resolution datasets with remarkable clarity.

---

## 5. Pedagogical Summary: The Philosophy of E1 286s

As we conclude this discussion on Variational Autoencoders, it is important to reflect on the core design principles of these models. In modern generative modeling, we do not simply want models that can copy and paste training data. Our objective is to build systems that construct a structured, meaningful latent space—a compressed internal representation of the real world.

Standard VAEs achieve this by regularizing a continuous latent space with a Gaussian prior. While elegant, this introduces a delicate balancing act (the $\beta$-parameter) to prevent the model from ignoring the latent space entirely (posterior collapse). VQ-VAEs elegant bypass this by discretizing the latent space into a learnable codebook. 

Understanding these foundational relationships prepares you to explore more advanced architectures, such as Diffusion Models and discrete Autoregressive Models, which build directly upon these latent representation principles.