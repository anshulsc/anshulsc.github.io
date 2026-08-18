
# Deep Generative Models: A Comprehensive Guide to DDPM ELBO, Score Matching, and Classifier-Guided Diffusion

Welcome to this highly detailed study guide and lecture notes companion for **Lecture 10: DDPMs/DDIMs** of the *Deep Generative Models* course (August 2024 Term). 

In this session, we systematically unpack the mathematical foundation of Denoising Diffusion Probabilistic Models (DDPMs), focusing heavily on the simplification of the Evidence Lower Bound (ELBO). We will cover how the reverse process is parameterized, how the network is trained in practice, and the ingenious way time step $t$ is embedded. We will also touch upon Tweedie’s formula to connect diffusion models with score matching, before concluding with the mathematical derivation of classifier-guided diffusion for conditional generation.

---

## Section 1: Simplifying the DDPM ELBO

In DDPMs, our goal is to model the reverse process $p_\theta(x_{t-1} | x_t)$ using a neural network. From the variational formulation of diffusion models, the training objective (the Evidence Lower Bound, or ELBO) contains a Kullback-Leibler (KL) divergence term that compares the posterior distribution of the forward process (conditioned on the clean image $x_0$) with our parameterized reverse step:

$$D_{KL}\Big(q(x_{t-1} | x_t, x_0) \;\parallel\; p_\theta(x_{t-1} | x_t)\Big)$$

To compute this term analytically, we exploit the fact that both distributions can be formulated as multivariate Gaussians.

### 1.1 The Forward Posterior Distribution $q(x_{t-1} | x_t, x_0)$
The forward process posterior, conditioned on $x_0$, is a Gaussian distribution:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\Big(x_{t-1}; \mu_q(x_t, x_0), \Sigma_q(t)\Big)$$

Where the analytical mean $\mu_q(x_t, x_0)$ is a weighted combination of the noisy image $x_t$ and the original clean image $x_0$:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} x_0$$

Here:
*   $\alpha_t = 1 - \beta_t$, where $\beta_t$ is the noise schedule at step $t$.
*   $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ represents the cumulative noise scaling factor.

### 1.2 Reparameterizing the Mean $\mu_q$
During training, we do not want our network to directly predict the clean image $x_0$ from a highly noisy $x_t$, as this is a highly non-linear and difficult mapping. Instead, we can reparameterize $x_0$ using the forward diffusion equation:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\bar{\alpha}}_t} \epsilon_t \quad \text{where } \epsilon_t \sim \mathcal{N}(0, I)$$

Rearranging this equation to solve for $x_0$ gives:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t}{\sqrt{\bar{\alpha}_t}}$$

Now, we substitute this expression for $x_0$ back into our formula for $\mu_q(x_t, x_0)$:

$$\mu_q(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} \left[ \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t}{\sqrt{\bar{\alpha}_t}} \right]$$

Using the identity $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1} \implies \sqrt{\bar{\alpha}_{t-1}} / \sqrt{\bar{\alpha}_t} = 1 / \sqrt{\alpha_t}$, we can simplify this expression. Grouping the terms by $x_t$ and $\epsilon_t$, we get:

$$\mu_q(x_t, x_0) = \left[ \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) + \frac{1 - \alpha_t}{\sqrt{\alpha_t}} \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \right] x_t - \left[ \frac{1 - \alpha_t}{\sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t}} \right] \epsilon_t$$

Simplifying the coefficient of $x_t$ yields:

$$\frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + 1 - \alpha_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1}{\sqrt{\alpha_t}}$$

Thus, the mean of the forward posterior simplifies beautifully to:

$$\mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right)$$

This tells us that the mean step backward is simply a scaled version of the current image $x_t$ minus a scaled version of the noise $\epsilon_t$ that was added to it.

### 1.3 Parameterizing the Reverse Process mean $\mu_\theta$
Since $\mu_q(x_t, x_0)$ depends on the added noise $\epsilon_t$, we can parameterize our neural network's mean $\mu_\theta(x_t, t)$ to match this exact structure. Instead of predicting the mean directly, we train a neural network $\hat{\epsilon}_\theta(x_t, t)$ to predict the noise $\epsilon_t$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon}_\theta(x_t, t) \right)$$

Since both distributions $q$ and $p_\theta$ are Gaussians sharing the same variance $\Sigma_q(t)$, their KL divergence simplifies directly to the squared L2-norm distance between their means:

$$D_{KL}\Big(q(x_{t-1} | x_t, x_0) \;\parallel\; p_\theta(x_{t-1} | x_t)\Big) \propto \|\mu_q(x_t, x_0) - \mu_\theta(x_t, t)\|^2_2$$

Substituting our parameterized formulations for both means:

$$\|\mu_q - \mu_\theta\|^2_2 = \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon}_\theta(x_t, t) \right) \right\|^2_2$$

This simplifies dramatically to:

$$\|\mu_q - \mu_\theta\|^2_2 \propto \left\| \epsilon_t - \hat{\epsilon}_\theta(x_t, t) \right\|^2_2$$

Hence, the complex variational training objective of a DDPM simplifies to a straightforward **noise-prediction regression task**.

---

## Section 2: Practical Training Implementation

Let's translate this theory into a concrete training routine. The standard algorithm for training a DDPM on a dataset is structured as follows:

```python
# Pseudo-code for DDPM Training Loop
for epoch in range(num_epochs):
    for x0_batch in dataloader:  # Clean data samples from the dataset
        # 1. Sample random time steps uniformly for the batch
        t = sample_uniform(low=1, high=T, size=batch_size) 
        
        # 2. Sample standard Gaussian noise
        epsilon = sample_normal(mean=0, variance=I, size=x0_batch.shape)
        
        # 3. Compute noisy images x_t at their respective time steps
        xt_batch = sqrt(alpha_bar[t]) * x0_batch + sqrt(1 - alpha_bar[t]) * epsilon
        
        # 4. Predict the noise using our U-Net model
        predicted_noise = model(xt_batch, t)
        
        # 5. Calculate MSE Loss
        loss = mean_squared_error(epsilon, predicted_noise)
        
        # 6. Backpropagation and Optimizer Step
        loss.backward()
        optimizer.step()
```

---

### 🙋‍♂️ Interactive Student Q&A: Part 1 (Training & Mechanics)

During this part of the lecture, several students raised crucial questions regarding the training dynamics of DDPMs.

> **Student Doubt 1:** *Sir, when we are taking $x_0$, does the time step $t$ have to be of a fixed length? That is, do we have to calculate the forward process for all steps (from $1$ to $1000$) for every single image during training?*
> 
> **Instructor Answer:** No! This is one of the most elegant aspects of DDPM training. If we had to calculate $x_1, x_2, \dots, x_{1000}$ sequentially for every image in every batch, training would be incredibly slow. Because of the closed-form recursion formula:
> 
> $$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$
> 
> we can jump directly to any arbitrary time step $t$ in a single step. For each image in our batch, we randomly select a different $t \in [1, T]$ uniformly. This means the model learns to denoise across all different noise levels simultaneously in a single training step.

> **Student Doubt 2:** *Is it one or more than one $t$ for a given $x_0$ in a single batch?*
> 
> **Instructor Answer:** In a single training step, we sample exactly **one** random time step $t$ for each image $x_0$ in the batch. For instance, if your batch size is 64, image 1 might be sampled at $t=400$, image 2 at $t=12$, image 3 at $t=950$, etc. Over many epochs and iterations, the network will see every image at various different time steps. This stochastic sampling provides an unbiased estimator of the true objective while keeping the computational cost per iteration minimal.

---

## Section 3: Architecting the U-Net and Embedding Time Step $t$

The neural network used to predict the noise is typically a **U-Net** architecture. 

```
Input: x_t (d dimensions) ──> [ Downsampling / Contraction ] ──┐
                                                                │  Skip Connections
                                                           [Bottleneck d']
                                                                │
Input: t (scalar) ─> [Embedding] ─> [ Upsampling / Expansion  ] <─┘ ──> Output: Predicted Noise (d dimensions)
```

The U-Net consists of:
1.  **Contraction (Downsampling) Path:** Processes the spatial input image $x_t$ (dimensionality $d$) and progressively reduces its spatial resolution while increasing channel depth, down to a low-dimensional bottleneck ($d'$).
2.  **Expansion (Upsampling) Path:** Progressively increases the spatial resolution back to the original image dimensions $d$.
3.  **Skip Connections:** Links corresponding layers from the contraction path to the expansion path, preserving high-frequency spatial details.

### 3.1 The Time-Embedding Challenge
Our model $\hat{\epsilon}_\theta(x_t, t)$ must be conditioned on both the noisy image $x_t$ and the current time step $t$. However, $x_t$ is a high-dimensional tensor (e.g., $3 \times 64 \times 64 = 12288$ dimensions), while $t$ is a simple scalar. 

If we simply concatenate $t$ as a single scalar channel to the image, the neural network’s convolution layers will completely ignore it because its signal is washed out by the high-dimensional spatial data. To prevent this, we must project the scalar $t$ into a high-dimensional vector representation. This process is called **Positional Embedding** or **Time Embedding**.

### 3.2 Sinusoidal Time Embedding
We use a sinusoidal embedding scheme, identical to the positional encodings used in Transformers:

$$\hat{t} = \left[ \sin(\omega_1 t), \cos(\omega_1 t), \sin(\omega_2 t), \cos(\omega_2 t), \dots, \sin(\omega_{D/2} t), \cos(\omega_{D/2} t) \right]$$

Where the frequencies are exponentially spaced:

$$\omega_i = \frac{1}{10000^{\frac{2i}{D}}}$$

This maps the scalar time step $t$ into a $D$-dimensional vector $\hat{t}$.

---

### 🙋‍♂️ Interactive Student Q&A: Part 2 (Time Embeddings)

At this point, the students pushed deeper into the mechanics of positional embeddings and neural network conditioning.

> **Student Doubt 3:** *How do we actually give this $D$-dimensional time embedding vector $\hat{t}$ as an input to the U-Net?*
> 
> **Instructor Answer:** We don't just concatenate it at the input layer. Instead, we feed the $D$-dimensional vector $\hat{t}$ into a small Multi-Layer Perceptron (MLP) to project it. At various residual blocks inside the U-Net (both in the downsampling and upsampling paths), we project this embedding to match the channel dimension of that layer's feature map. We then add or multiply this projected vector to the feature map. This ensures that the time step information is actively infused throughout all levels of the network.

> **Student Doubt 4:** *Do the embedding frequencies have to be harmonic or can they be any frequency?*
> 
> **Instructor Answer:** In practice, we use exponentially scaled harmonic frequencies (such as the geometric progression used in the Transformer positional encoding). This is crucial because it allows the model to learn both high-frequency details (for small time steps $t$ where tiny noise adjustments are made) and low-frequency structures (for large $t$ where global image features are formed).

> **Student Doubt 5:** *When we concatenate or add $t$ to other layers of the U-Net, is its dimension less than the spatial dimension $d$? How do we handle spatial mismatch?*
> 
> **Instructor Answer:** Excellent question. The feature maps inside a U-Net are 3D tensors of shape $(\text{Channels}, \text{Height}, \text{Width})$. The time embedding $\hat{t}$ is a 1D vector of shape $(\text{Embedding Dimension})$. To inject $\hat{t}$ into a feature map, we pass it through a linear projection layer to change its dimension to match the channel size. Then, we spatially broadcast this 1D vector across the height and width dimensions (effectively replicating the vector for every pixel location) so that we can perform element-wise addition or scaling.

> **Student Doubt 6:** *In conditional GANs, we concatenate a one-hot vector of class labels. Is this time embedding similar to that class conditioning?*
> 
> **Instructor Answer:** Conceptually, yes—both are conditioning signals. However, remember that right now we are discussing *unconditional* generation. The time step $t$ is NOT a class label; it is a parameter of the diffusion process. The network *must* know $t$ to determine how much noise to remove. If we don't provide $t$, the network won't know whether it is looking at a slightly noisy image ($t=10$) or pure white noise ($t=990$), leading to poor generation. Later, we will see how we can add actual class conditioning $y$ on top of this.

---

## Section 4: Score-Based Diffusion and Tweedie’s Formula

Let's now establish a deep connection between DDPMs and another major family of generative models: **Score-Based Generative Models**.

### 4.1 Tweedie’s Formula
Tweedie's formula is a classic statistical result. Suppose we observe a noisy variable $z$ generated by adding Gaussian noise to an underlying mean $\mu_z$:

$$z \sim \mathcal{N}(z; \mu_z, \Sigma_z)$$

Tweedie's formula states that the true, posterior expectation of the clean mean $\mu_z$ given the noisy observation $z$ is:

$$\mathbb{E}[\mu_z | z] = z + \Sigma_z \nabla_z \log p(z)$$

Where $\nabla_z \log p(z)$ is the **score function**—the gradient of the log-probability density of the noisy data with respect to the noisy data itself.

### 4.2 Connecting Tweedie's Formula to Diffusion
In the context of diffusion models, at time step $t$, our noisy image $x_t$ is generated from the clean image $x_0$ via:

$$x_t \sim \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I \right)$$

Applying Tweedie's formula directly where $z = x_t$, $\mu_z = \sqrt{\bar{\alpha}_t} x_0$, and $\Sigma_z = (1 - \bar{\alpha}_t) I$:

$$\mathbb{E}\left[ \sqrt{\bar{\alpha}_t} x_0 \;\middle|\; x_t \right] = x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)$$

Rearranging this to solve for the expected clean image $x_0$:

$$\mathbb{E}[x_0 | x_t] = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}$$

At the same time, we have our reparameterization formula:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t}{\sqrt{\bar{\alpha}_t}}$$

Equating these two expressions:

$$\frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t}{\sqrt{\bar{\alpha}_t}} = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)}{\sqrt{\bar{\alpha}_t}}$$

$$\implies -\sqrt{1 - \bar{\alpha}_t} \epsilon_t = (1 - \bar{\alpha}_t) \nabla_{x_t} \log p(x_t)$$

Which yields the fundamental identity:

$$\nabla_{x_t} \log p(x_t) = -\frac{\epsilon_t}{\sqrt{1 - \bar{\alpha}_t}}$$

### 4.3 Score Matching Equivalence
This identity is profound. It demonstrates that the score of the data distribution at noise level $t$ is directly proportional to the added noise $\epsilon_t$. 

If we substitute this identity into our DDPM noise-prediction loss:

$$\mathcal{L}_{simple}(\theta) = \left\| \epsilon_t - \hat{\epsilon}_\theta(x_t, t) \right\|^2_2 = \left\| -\sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p(x_t) - \hat{\epsilon}_\theta(x_t, t) \right\|^2_2$$

This reveals that training a DDPM to predict noise is mathematically identical to training a model to estimate the score of the data distribution (Denoising Score Matching)!

---

## Section 5: Classifier-Guided Diffusion (Conditional Generation)

Thus far, we have only discussed unconditional generation. What if we want to generate images belonging to a specific class $y$ (e.g., "generate a cat")? This requires sampling from the conditional distribution $p(x_t | y)$. 

By establishing the connection to score matching, we can easily derive how to condition our model using **Classifier Guidance**.

### 5.1 The Mathematical Derivation
We want to find the conditional score function $\nabla_{x_t} \log p(x_t | y)$. Using Bayes' Rule:

$$p(x_t | y) = \frac{p(x_t) p(y | x_t)}{p(y)}$$

Taking the logarithm of both sides:

$$\log p(x_t | y) = \log p(x_t) + \log p(y | x_t) - \log p(y)$$

Now, we take the gradient with respect to the image $x_t$:

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t) - \nabla_{x_t} \log p(y)$$

Since $p(y)$ does not depend on the image $x_t$, its gradient $\nabla_{x_t} \log p(y) = 0$. This leaves us with:

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

This decomposition is incredibly beautiful. It splits the conditional score into two separate, modular parts:
1.  **Unconditional Score ($\nabla_{x_t} \log p(x_t)$):** This is handled by our standard, pre-trained unconditional DDPM model $\hat{\epsilon}_\theta(x_t, t)$.
2.  **Classifier Gradient ($\nabla_{x_t} \log p(y | x_t)$):** This is the gradient of a separate classifier model, trained to predict the class label $y$ from a noisy image $x_t$. The gradient points in the direction of image space that increases the likelihood of the image being classified as class $y$.

### 5.2 Modifying the Reverse Step
During sampling, we can scale the classifier gradient with a parameter $s$ called the **guidance scale**:

$$\nabla_{x_t} \log p(x_t | y) \approx \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p(y | x_t)$$

Translating this back to our noise-prediction formulation using our score-to-noise identity, the modified noise predictor becomes:

$$\tilde{\epsilon}_\theta(x_t, t) = \hat{\epsilon}_\theta(x_t, t) - s \cdot \sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p(y | x_t)$$

Consequently, during the reverse denoising process, the mean step $\tilde{\mu}$ is shifted:

$$\tilde{\mu}(x_t, t) = \mu_\theta(x_t, t) + s \cdot \Sigma_q(t) \nabla_{x_t} \log p(y | x_t)$$

This allows us to take an unconditional diffusion model and "guide" its generation towards a target class $y$ simply by adding the gradients of an external classifier at each sampling step!

---

## Summary of Key Takeaways for Quick Recall

*   **DDPM ELBO Simplification:** By substituting the forward-step reparameterization of $x_0$ into the analytical Gaussian mean $\mu_q(x_t, x_0)$, the ELBO loss simplifies down to a simple mean-squared error (MSE) loss predicting the added noise $\epsilon_t$.
*   **Practical Training:** For every image in a batch, we sample a random time step $t$ and train the network on the corresponding noise. This makes training highly parallel and computationally efficient.
*   **Time Embedding:** We project the scalar $t$ to a high-dimensional vector using Sinusoidal Embeddings (geometric progressions of sine/cosine) to prevent the temporal signal from being ignored by the U-Net.
*   **Tweedie's Formula:** This connects noise prediction to score-based modeling, showing that the predicted noise is mathematically equivalent to the score of the data distribution: $\nabla_{x_t} \log p(x_t) \propto -\epsilon_t$.
*   **Classifier Guidance:** Conditional generation is achieved by decomposing the conditional score into the sum of the unconditional score and the gradient of a classifier: $\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$.
# Deep Generative Models: From DDPM to Denoising Diffusion Implicit Models (DDIM)

In modern deep generative modeling, diffusion models have emerged as a highly effective class of generative frameworks. This lecture covers the transition from Denoising Diffusion Probabilistic Models (DDPMs) to Denoising Diffusion Implicit Models (DDIMs), explaining the theoretical motivations, mathematical formulations, and practical implications of this shift.

---

## 1. Context: Latent Diffusion & Stable Diffusion

Before diving into DDIM, it is helpful to place these models in the context of advanced frameworks like Stable Diffusion. 

To make diffusion models computationally feasible for high-resolution images, we avoid training the diffusion process directly in the high-dimensional pixel space. Instead, we train a **Denoising Diffusion Probabilistic Model (DDPM)** within the lower-dimensional latent space of a pre-trained Autoencoder, specifically a **Vector Quantized Variational Autoencoder (VQ-VAE)**.

```
       [ Encoder ]             [ Diffusion Process ]             [ Decoder ]
Pixel Space ---> Latent Space -----------------------> Latent Space ---> Pixel Space
  (Image x)         (z_0)      (DDPM Forward/Reverse)     (z_0 predicted)  (Reconstructed x)
```

The generation pipeline works as follows:
1. **Sampling:** We sample a new latent representation $z_T$ from a prior distribution (typically standard Gaussian).
2. **Denoising:** We pass this sample through the reverse diffusion chain to obtain a denoised latent vector $z_0$.
3. **Decoding:** This denoised latent vector $z_0$ is passed through the Decoder of the VQ-VAE to render the final image in pixel space.

---

### Student Doubt 1: Quantized Latent Space in Stable Diffusion
During the lecture, a student raised a key question regarding the nature of the latent space used in this generation process.

> **Student Question:** *"Sir, I have a question regarding Stable Diffusion. Is the latent space used in the generation right after the Encoder, or is it after mapping to the dictionary of embeddings?"*

**Instructor's Explanation:** 
It is the **quantized latent space** (i.e., after mapping to the discrete dictionary of codebook embeddings). 

> **Follow-up Student Question:** *"But how do we sample from that quantized latent space since it represents a discrete set of quantized vectors?"*

**Instructor's Explanation:** 
Even though the codebook vectors are discrete indices, we treat the encodings (the continuous representations before index lookup or the continuous codebook embeddings themselves) as the latent space. We build the continuous generative model (the diffusion model) directly on top of these continuous latent embeddings corresponding to the entire dataset. Once the diffusion model generates a valid latent vector, it can be decoded back to pixel space.

---

## 2. The Limitations of DDPMs

While DDPMs produce high-quality samples, they suffer from two major limitations:

### Limitation A: Slow Sampling (Inference Bottleneck)
DDPMs are **Markovian** (specifically, first-order Markov chains). The forward step $q(x_t | x_{t-1})$ and the parameterized reverse step $p_\theta(x_{t-1} | x_t)$ assume that the state at $t-1$ depends strictly on the state at $t$. 

Because of this Markovian assumption, the generation process must step sequentially through every single step $t = T, T-1, \dots, 1$. If $T$ is large (typically $T \approx 1000$ to ensure the Gaussian transition perturbation remains small), the sampler must evaluate the neural network ($\epsilon_\theta$) $1000$ times to generate a single image. This makes real-time sampling extremely slow.

### Limitation B: Inability to Perform Deterministic Posterior Inference (Inversion)
Posterior inference—often called **inversion**—refers to finding the unique latent noise vector $x_T$ that corresponds to a given real starting image $x_0$. 

In DDPM, the forward process is stochastic (we add random noise at each step). If you run the forward process on the same image $x_0$ multiple times, you will end up with different noise representations $x_T$. Similarly, because the reverse process is stochastic, denoising a latent vector $x_T$ does not guarantee that you will recover the exact same image $x_0$. This lack of a deterministic mapping makes tasks like image editing, style transfer, or latent space interpolation challenging.

---

## 3. Introducing Denoising Diffusion Implicit Models (DDIMs)

To address these limitations, **Denoising Diffusion Implicit Models (DDIMs)** were introduced. DDIMs generalize the diffusion framework to a class of **non-Markovian** forward processes. 

Key advantages of DDIMs include:
1. **Faster Sampling:** They allow skipping steps during inference, reducing the number of evaluations from $T \approx 1000$ to $S \approx 50$ steps.
2. **Deterministic Inversion:** They enable a fully deterministic mapping between the image space and the latent space.
3. **No Retraining Required:** A trained DDPM model can be used directly as a DDIM during inference. **You do not need to train a new model to get the benefits of DDIM.**

---

## 4. The Core Theoretical Insight of DDIM

The fundamental mathematical insight behind DDIM is that the **DDPM loss objective (the ELBO) depends only on the marginal distributions $q(x_t | x_0)$ and not directly on the joint transition path $q(x_{1:T} | x_0)$**.

Recall that the simplified training objective for DDPM is:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

Here, the noisy sample $x_t$ is constructed directly from $x_0$ using the marginal formula:
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

This means that any forward process that shares the same marginals $q(x_t | x_0)$ will yield the exact same objective function. Consequently, we can define a completely different, **non-Markovian** forward process that shares these marginals, allowing us to use our pre-trained DDPM network with a new, faster, and potentially deterministic inference trajectory.

---

## 5. Defining the Non-Markovian Forward Process

Let us define a family of non-Markovian forward distributions parameterized by a vector of standard deviations $\sigma \in \mathbb{R}^T_{\geq 0}$:

$$q_\sigma(x_{1:T} | x_0) = q_\sigma(x_T | x_0) \prod_{t=2}^T q_\sigma(x_{t-1} | x_t, x_0)$$

Where the terminal marginal is the standard Gaussian perturbation:
$$q_\sigma(x_T | x_0) = \mathcal{N}\left(x_T; \sqrt{\bar{\alpha}_T} x_0, (1 - \bar{\alpha}_T)\mathbf{I}\right)$$

And the transition distribution is conditioned on **both** $x_t$ and $x_0$ (which breaks the Markov property):
$$q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \mathbf{m}_t(x_t, x_0), \sigma_t^2 \mathbf{I}\right)$$

To ensure that the marginals $q_\sigma(x_t|x_0)$ match those of DDPM, the mean $\mathbf{m}_t(x_t, x_0)$ must be defined as:
$$\mathbf{m}_t(x_t, x_0) = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \left( \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}} \right)$$

### Understanding the Terms in the Transition Mean
The mean of the transition $q_\sigma(x_{t-1} | x_t, x_0)$ consists of two primary components:
1. **The $x_0$ direction ($\sqrt{\bar{\alpha}_{t-1}} x_0$):** Points directly toward the reconstructed, clean image.
2. **The noise direction ($\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}$):** Points in the direction of the noise at the current step $t$.

The parameter $\sigma_t^2$ controls how much stochastic noise is explicitly added back at step $t-1$.

---

## 6. Defining the Denoising (Reverse) Process

In the reverse process, we want to sample $p_\theta(x_{t-1} | x_t)$. Since we do not have access to the true clean image $x_0$ during generation, we use our trained neural network $\epsilon_\theta(x_t)$ to predict it.

Using the forward relation, we can write our model's estimate of $x_0$ at step $t$ as:
$$f_\theta(x_t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}}$$

We then define our parameterized reverse transition by replacing the true $x_0$ in $q_\sigma(x_{t-1} | x_t, x_0)$ with this estimate $f_\theta(x_t)$:
$$p_\theta(x_{t-1} | x_t) = q_\sigma(x_{t-1} | x_t, f_\theta(x_t))$$

Substituting $f_\theta(x_t)$ into our transition formula yields:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} f_\theta(x_t) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t), \sigma_t^2 \mathbf{I}\right)$$

During generation, we sample $x_{t-1}$ from this distribution:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t) + \sigma_t \epsilon_t$$
where $\epsilon_t \sim \mathcal{N}(0, \mathbf{I})$.

---

## 7. The Magic of $\sigma_t = 0$ (Deterministic Trajectories)

The choice of the variance parameter $\sigma_t$ defines the behavior of our model during inference:

### Case 1: Recovering DDPM
If we set $\sigma_t$ to:
$$\sigma_t = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$
the transition matches the standard DDPM reverse transition. This shows that DDPM is a stochastic sub-case of the broader DDIM family.

### Case 2: DDIM ($\sigma_t = 0$)
If we set $\sigma_t = 0$ for all $t$, the random noise term $\sigma_t \epsilon_t$ disappears completely from the generation equation. The transition becomes fully deterministic:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t)$$

Under this setting, the forward and reverse paths become a deterministic mapping.

---

### Student Doubt 2: Why are DDPMs Non-Invertible Compared to DDIMs?
At this point in the lecture, a student asked a question about the fundamental cause of DDPM's non-invertibility.

> **Student Question:** *"So the reason why DDPMs are non-invertible, is it because the Markovian processes are non-reversible?"*

**Instructor's Explanation:** 
No, it is not because of the Markovian property itself. It is because the DDPM forward process is **stochastic** (it adds random noise at each step). 

In DDIM with $\sigma_t = 0$, the forward process is entirely deterministic. If you start with a real image $x_0$, there is a single, deterministic trajectory that maps it to a latent noise vector $x_T$. Because this mapping is deterministic, running the reverse process starting from $x_T$ will return you to the exact same starting image $x_0$. 

This deterministic property is what makes DDIM invertible, enabling applications such as high-fidelity image editing and latent space interpolation.

---

## 8. Summary Comparison: DDPM vs. DDIM

| Feature | DDPM | DDIM ($\sigma_t = 0$) |
| :--- | :--- | :--- |
| **Markovian Assumption** | Yes (First-order Markov chain) | No (Conditioned on both $x_t$ and $x_0$) |
| **Sampling Process** | Stochastic (Random noise added at each step) | Deterministic (No noise added during reverse steps) |
| **Sampling Speed** | Slow (Requires evaluating all $T \approx 1000$ steps) | Fast (Can skip steps, generating samples in $S \approx 50$ steps) |
| **Invertibility** | No (Stochastic path cannot be precisely reversed) | Yes (Deterministic forward and reverse paths) |
| **Training Requirement** | Standard DDPM training | Uses the exact same trained DDPM model |