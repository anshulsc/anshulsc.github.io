# Lecture Notes: Proximal Policy Optimization (PPO) from First Principles LLMS


---

## 1. Foundations of Reinforcement Learning (RL)

Reinforcement Learning is a framework where an agent learns to make decisions by interacting with an environment to maximize some cumulative reward.

### Core Terminology
*   **Agent**: The decision-maker (e.g., a physical robot like a Roomba, or a digital system like a Large Language Model).
*   **Environment**: Everything outside the agent with which it interacts (e.g., a living room, or human users and software tools).
*   **Goal**: The objective the agent aims to achieve (e.g., sweeping the entire floor surface, or correctly solving a mathematical proof).
*   **Episode**: A complete sequence of interactions from an initial state to a terminal state (e.g., a full room-cleaning session, or generating a complete text response until an end-of-sequence token is reached).
*   **Steps ($t$)**: Discrete time increments within an episode. At each step $t$:
    1.  The agent observes the current state $s_t$.
    2.  The agent selects an action $a_t$.
    3.  The environment transitions to a new state $s_{t+1}$ and provides a scalar reward $r_t$.
*   **Trajectory ($\tau$)**: The chronological sequence of states, actions, and rewards experienced over an entire episode:
    $$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_{T-1}, a_{T-1}, r_{T-1}, s_T)$$

---

## 2. Reinforcement Learning for Large Language Models (LLMs)

When adapting RL to LLMs, the traditional vocabulary is mapped onto text generation processes:

| RL Concept | Physical Agent (e.g., Roomba) | Generative LLM (e.g., Llama) |
| :--- | :--- | :--- |
| **Agent** | Robotic vacuum cleaner | The Large Language Model |
| **Environment** | The layout of the room, obstacles | Human prompts, tools (e.g., Python interpreters), databases |
| **State ($s_t$)** | Current spatial coordinates $(x, y)$ | The prompt concatenated with all tokens generated up to step $t$ |
| **Action ($a_t$)** | Physical move (e.g., "move 1 foot right") | The next token predicted from the vocabulary |
| **Reward ($r_t$)** | Percentage of surface cleaned | Evaluation of text properties (e.g., correctness, style, safety) |

### Generative Transition Dynamics
For an LLM, state transitions are deterministic additions of the chosen action (token) to the existing context window:
$$s_{t+1} = s_t \mathbin{\Vert} a_t$$
where $\Vert$ denotes string concatenation. The generation loop terminates when the agent emits a special terminal token (such as `"<end>"` or `"<|eot_id|>"`).

### Sparse vs. Dense Rewards
In tasks like mathematical reasoning (e.g., the GSM8K dataset), evaluating intermediate tokens is often impractical. Instead, the environment provides a **sparse reward** at the final step $T$:
$$r_t = 0 \quad \text{for } t < T$$
$$r_T = \begin{cases} 1 & \text{if the final extracted answer matches the ground truth} \\ 0 & \text{otherwise} \end{cases}$$
Alternatively, a **dense reward** setup can be engineered using an auxiliary reward model (e.g., in RLHF) to score the output at smaller intervals (such as at the end of every sentence).

---

## 3. The Optimization Challenge & Policy Gradients

Let the LLM represent a stochastic **policy** $\pi_\theta(a|s)$, which defines a probability distribution over the vocabulary of actions $a$ given the current token history $s$, parameterized by model weights $\theta$.

Our objective is to find the optimal parameters $\theta^*$ that maximize the expected cumulative reward over all possible trajectories:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$
where $R(\tau) = \sum_{t=0}^T r_t$ is the total return of trajectory $\tau$.

### The Credit Assignment Problem
Using a single scalar reward $r_T$ at the end of an episode to update billions of parameters $\theta$ is computationally difficult. We must determine which specific tokens (actions $a_t$) in a sequence of hundreds or thousands of steps were responsible for the final positive or negative outcome.

### Supervised Fine-Tuning (SFT) vs. Reinforcement Learning (RL)
*   **Supervised Fine-Tuning (SFT)**: Given a human reference response, the target distribution is a one-hot vector where the correct next token has a probability of $1$. The model minimizes cross-entropy loss to match this target:
    $$L^{SFT}(\theta) = -\sum_{t} \log \pi_\theta(a_t^* | s_t)$$
*   **Reinforcement Learning (RL)**: Without a reference response, we must generate candidate outputs, observe the resulting rewards, and adjust token probabilities up or down.

### Derivation of the Basic Policy Gradient
To update the policy parameters using gradient ascent on expected return, we compute the gradient $\nabla_\theta J(\theta)$.

Using the probability of a trajectory $\tau$ under policy parameters $\theta$:
$$P(\tau|\theta) = P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$$

The expected return is:
$$J(\theta) = \int P(\tau|\theta) R(\tau) d\tau$$

Taking the gradient with respect to $\theta$:
$$\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau|\theta) R(\tau) d\tau$$

Using the **log-derivative trick** (also known as the REINFORCE algorithm):
$$\nabla_\theta \log P(\tau|\theta) = \frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)} \implies \nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)$$

Substituting this back into the gradient equation:
$$\nabla_\theta J(\theta) = \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) R(\tau) \right]$$

Now, taking the logarithm of the trajectory probability:
$$\log P(\tau|\theta) = \log P(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1}|s_t, a_t)$$

Because only the policy term $\pi_\theta(a_t|s_t)$ depends on $\theta$, the gradient simplifies to:
$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Thus, the gradient of our expected return is:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$$

To apply this on a per-token level, we scale each action's gradient by an estimated measure of its relative quality, called the **Advantage** $A_t$:
$$\nabla_\theta J(\theta) \approx \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right]$$

This corresponds to the gradient of the standard **Policy Gradient Loss**:
$$L^{PG}(\theta) = \sum_{t=0}^T \log \pi_\theta(a_t|s_t) A_t$$

---

## 4. The Actor-Critic Framework and Value Function

The advantage $A_t$ answers the question: *How much better than average was the action $a_t$ taken at state $s_t$?*
$$A_t = Q(s_t, a_t) - V(s_t)$$
where:
*   $Q(s_t, a_t)$ is the expected return of taking action $a_t$ in state $s_t$.
*   $V(s_t)$ is the expected return of being in state $s_t$, representing the average outcome over all possible actions.

### The Value Function
The true value of a state $s_t$ under policy $\pi$ is defined as:
$$V(s_t) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{k=t}^T \gamma^{k-t} r_k \;\middle|\; s_t \right]$$
where $\gamma \in [0, 1]$ is a **discount factor** that downweights the importance of rewards received further in the future.

Calculating this expected value directly is computationally intractable for language models due to the exponential branching factor of the vocabulary:
$$\text{Branching Factor} = |V|^{T-t}$$
where $|V|$ is the vocabulary size (often $>100,000$).

### Critic Network
To make this computation tractable, we use a separate model called the **Critic**, represented as a value network $V_\phi(s)$ with parameters $\phi$. The critic learns to approximate the expected future return of any state $s_t$.

```
                 ┌───────────────┐
                 │  State (s_t)  │
                 └───────┬───────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
   ┌──────────────┐            ┌──────────────┐
   │    Actor     │            │    Critic    │
   │  π_θ(a|s)    │            │    V_φ(s)    │
   └──────┬───────┘            └──────┬───────┘
          │                           │
          ▼                           ▼
     Action (a_t)              State Value Estimate
```

The Critic's parameters $\phi$ are optimized via gradient descent on the Mean Squared Error (MSE) loss against the actual observed discounted returns $G_t$:
$$L^{VF}(\phi) = \sum_{t} \left( V_\phi(s_t) - G_t \right)^2$$
where:
$$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$$

---

## 5. Advantage Estimation and GAE

To compute the advantage $A_t = Q(s_t, a_t) - V_\phi(s_t)$, we must determine $Q(s_t, a_t)$. Three primary estimation techniques exist, presenting different trade-offs between bias and variance:

```
─────────────────────────────────────────────────────────────────────────
Bias-Variance Spectrum of Advantage Estimation:

  Monte Carlo (MC)                                Temporal Difference (TD)
  [Low Bias / High Variance]                     [High Bias / Low Variance]
  ◄──────────────────────────────────────────────────────────────────────►
                                   GAE
                            [Tunable Balance]
─────────────────────────────────────────────────────────────────────────
```

### 1. Monte Carlo (MC) Estimation
We wait for the entire episode to complete and use the actual observed return $G_t$ as the empirical target for $Q(s_t, a_t)$:
$$Q(s_t, a_t) \approx G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T$$
$$\hat{A}_t^{MC} = G_t - V_\phi(s_t)$$

*   **Bias**: Extremely low. It uses real environment transitions without relying on model approximations.
*   **Variance**: High. Random environment factors or noise across multiple steps accumulate over the course of the trajectory.
*   **Constraint**: Cannot compute intermediate advantages until the full episode terminates.

### 2. Bootstrapping / Temporal Difference (TD) Estimation
We estimate $Q(s_t, a_t)$ by taking the immediate reward $r_t$ and adding the discounted value of the next state as estimated by our critic $V_\phi(s_{t+1})$:
$$Q(s_t, a_t) \approx r_t + \gamma V_\phi(s_{t+1})$$
This yields the **TD residual** (or one-step advantage) $\delta_t$:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

*   **Bias**: High. If the critic network $V_\phi$ is inaccurate, the bias propagates into the policy update.
*   **Variance**: Low. It only relies on a single transition step and reward sample.

### 3. Generalized Advantage Estimation (GAE)
Generalized Advantage Estimation (Schulman et al.) bridges these two extremes by introducing a hyperparameter $\lambda \in [0, 1]$ to take an exponentially weighted average of multi-step TD residuals.

Let the $k$-step advantage estimators be:
$$\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 V_\phi(s_{t+2}) - V_\phi(s_t)$$
$$\hat{A}_t^{(k)} = \sum_{i=0}^{k-1} \gamma^i \delta_{t+i} = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V_\phi(s_{t+k}) - V_\phi(s_t)$$

The GAE estimator combines these estimates:
$$\hat{A}_t^{GAE} = (1-\lambda) \left( \hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \dots \right)$$
$$\hat{A}_t^{GAE} = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}$$

#### Hyperparameter Analysis
*   When $\lambda = 0$:
    $$\hat{A}_t^{GAE} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
    This matches the high-bias, low-variance TD(0) estimator.
*   When $\lambda = 1$:
    $$\hat{A}_t^{GAE} = \sum_{k=0}^{T-t} \gamma^k \delta_{t+k} = \sum_{k=0}^{T-t} \gamma^k r_{t+k} - V_\phi(s_t)$$
    This matches the unbiased, high-variance Monte Carlo estimator.

---

## 6. On-Policy Optimization and the Multi-Epoch Challenge

The baseline Actor-Critic training algorithm operates in an **on-policy** loop:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Initialize policy weights θ_0 and critic weights φ_0     │
│ 2. Loop for iteration n = 0, 1, 2, ...                      │
│    a. Collect trajectories D_n using current policy π_θ_n   │
│    b. Compute rewards and discounted returns G_t            │
│    c. Compute advantages A_t using current critic V_φ_n     │
│    d. Calculate loss gradients and update policy to θ_n+1   │
│    e. Update critic parameters to φ_n+1                     │
└─────────────────────────────────────────────────────────────┘
```

### The Multi-Epoch Challenge
On-policy models discard collected trajectories $\mathcal{D}_n$ immediately after a single gradient step. In deep learning, we typically reuse data over multiple epochs to improve sample efficiency.

However, if we perform multiple gradient updates using the same trajectories $\mathcal{D}_n$, the policy parameters change from $\theta_{old}$ to some new parameters $\theta_{new}$. Because the trajectories in $\mathcal{D}_n$ were sampled under $\pi_{\theta_{old}}$, using them directly to compute expected values under $\pi_{\theta_{new}}$ causes a distribution mismatch that can destabilize training.

---

## 7. Importance Sampling and the Surrogate Loss

We can resolve this distribution mismatch using **Importance Sampling**, a mathematical identity that allows us to estimate expectations under a target distribution $p$ using samples drawn from a proposal distribution $q$.

$$\mathbb{E}_{x \sim p}[g(x)] = \int g(x) p(x) dx = \int g(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q} \left[ g(x) \frac{p(x)}{q(x)} \right]$$

### Applying Importance Sampling to Policies
We define the probability ratio of an action under the current policy versus the old policy as:
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

*   If $r_t(\theta) > 1$: The action is more likely under the updated policy than the old policy.
*   If $r_t(\theta) < 1$: The action is less likely under the updated policy.

By applying this ratio to our objective, we can reuse trajectories collected under $\pi_{\theta_{old}}$ to estimate gradients for the updated parameters $\theta$:
$$L^{CPI}(\theta) = \sum_{t} r_t(\theta) A_t$$
where $L^{CPI}$ represents the **Conservative Policy Iteration** objective (Kakade & Langford).

Because this loss is an approximation of the true policy gradient objective, it is referred to as a **surrogate loss**.

---

## 8. Clipping in PPO (The Trust Region)

While importance sampling mathematically corrects the expectation estimate, it assumes that the ratio $r_t(\theta)$ is close to $1$. If the updated policy $\pi_\theta$ diverges too far from $\pi_{\theta_{old}}$, the variance of the importance sampling estimator grows, leading to unstable training.

To prevent excessively large policy updates, we must enforce a constraint. While TRPO (Trust Region Policy Optimization) enforces this with a hard Kullback-Leibler (KL) divergence constraint using second-order derivatives:
$$\text{maximize } \mathbb{E} \left[ r_t(\theta) A_t \right] \quad \text{subject to } \mathbb{E} \left[ D_{KL}(\pi_{\theta_{old}} \mathbin{\Vert} \pi_\theta) \right] \le \delta_c$$
calculating this on-the-fly is computationally expensive for large neural networks.

### The Clipped Surrogate Objective
PPO (Schulman et al., 2017) replaces the second-order optimization constraint of TRPO with a simpler **clipping** mechanism. The policy ratio is restricted to a small interval around 1, typically defined by a hyperparameter $\epsilon \in (0, 1)$ (commonly $\epsilon = 0.1$ or $0.2$):
$$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$$

The **PPO Clipped Objective** is defined as:
$$L^{CLIP}(\theta) = \sum_{t} \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)$$

Taking the minimum of the unclipped and clipped objectives ensures that we only update the policy when it is supported by the advantage function.

```
Advantage > 0 (Positive Advantage)            Advantage < 0 (Negative Advantage)
   Loss                                          Loss
    │         / Unclipped objective               │   ___________
    │        /                                    │  /           \ Clipped & Unclipped
    │  _____/____ Clipped objective               │ /             \ are identical
    │ /     :                                     │/               \
    │/      :                                     │:               :
────┼───────:─────────────► Ratio                 ┼────────────────┼────────► Ratio
    │ 1-ε   1   1+ε                               │ 1-ε   1       1+ε
```

### Case Analysis

#### Case 1: Positive Advantage ($A_t > 0$)
The selected action performed better than average.
*   **When $r_t(\theta) \le 1 + \epsilon$**: The gradient updates increase $r_t(\theta)$ to make this favorable action more likely.
*   **When $r_t(\theta) > 1 + \epsilon$**: The ratio exceeds the upper bound. The clipping function limits the value to $(1+\epsilon) A_t$, zeroing out the gradient. This prevents the policy from over-correcting and making the action excessively dominant.

#### Case 2: Negative Advantage ($A_t < 0$)
The selected action performed worse than average.
*   **When $r_t(\theta) \ge 1 - \epsilon$**: The gradient updates decrease $r_t(\theta)$ to make this unfavorable action less likely.
*   **When $r_t(\theta) < 1 - \epsilon$**: The ratio falls below the lower bound. Because $A_t$ is negative, multiplying by a smaller value would normally increase the loss. Taking the minimum of the unclipped and clipped terms ensures the gradient does not drive the policy to make extremely large changes.

---

## 9. The Complete PPO Training Cycle

The standard implementation of Proximal Policy Optimization combines these components into a joint objective optimized at each step:

$$L^{PPO}(\theta, \phi) = \hat{\mathbb{E}}_t \left[ L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 S[\pi_\theta](s_t) \right]$$

where:
*   $c_1, c_2$ are scaling hyperparameters.
*   $L^{CLIP}(\theta)$ is the clipped policy surrogate loss.
*   $L^{VF}(\phi)$ is the mean squared error value function loss: $\left(V_\phi(s_t) - G_t\right)^2$.
*   $S[\pi_\theta](s_t)$ is an optional entropy bonus that encourages exploration by penalizing overly deterministic policies. In large language model training, this term is often set to $0$ ($c_2=0$) because pre-trained models already possess high output entropy.