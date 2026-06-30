# Lecture Notes: Group Relative Policy Optimization (GRPO) & Reinforcement Learning for LLMs

---

## 1. Introduction to GRPO
**Group Relative Policy Optimization (GRPO)** is a reinforcement learning (RL) algorithm that gained prominence as a core technology behind models like **DeepSeek-R1** and **Qwen**. 

### Key Characteristics:
* **Cost Efficiency:** GRPO allows the training of reasoning models at a fraction of the computational cost typically associated with traditional reinforcement learning workflows (such as those using PPO).
* **Algorithmic Heritage:** GRPO is not an entirely new paradigm but rather an elegant optimization of existing policy gradient methods—specifically, OpenAI's **Proximal Policy Optimization (PPO)**.
* **Removal of the Critic:** The defining innovation of GRPO is the elimination of the separate "Critic" (Value) network, replacing it with a group-relative baseline calculation.

---

## 2. The LLM Training Pipeline

To understand where GRPO fits, we must examine the standard Large Language Model (LLM) training pipeline:

```
[Random Initialization] 
       │
       ▼ (Next-token prediction on web-scale data)
  [Pre-training] ──► yields [Base Model]
       │
       ▼ (Instruction-response pairs)
[Supervised Fine-Tuning (SFT)] ──► yields [Instruction-Tuned Model]
       │
       ▼ (Alignment phase via RL / Implicit signals)
[Preference & Reasoning Fine-Tuning] ──► yields [Aligned / Reasoning Model]
```

### The Three Main Alignment/Fine-Tuning Paradigms:
1. **Supervised Fine-Tuning (SFT):** Training the model on curated prompt-response pairs. Once high-quality human-annotated data is exhausted, SFT hits a performance ceiling.
2. **Preference Fine-Tuning (e.g., RLHF, DPO):**
   * **RLHF (Reinforcement Learning with Human Feedback):** Uses human preferences to train a *Reward Model*, which then guides the policy optimization via PPO.
   * **DPO (Direct Preference Optimization):** A mathematically reformulated alternative that optimizes the policy directly from preference pairs without training a separate reward model.
3. **Reasoning Fine-Tuning / Reinforcement Learning with Verifiable Rewards (RLVR):**
   * Applied to domains with objective, deterministic ground truths (e.g., mathematics, coding, formal logic).
   * Instead of relying on subjective human preferences, the system evaluates correctness programmatically (e.g., executing code or checking a mathematical result).
   * **PPO** and **GRPO** serve as the underlying optimization engines for this stage.

---

## 3. Foundations of Reinforcement Learning (RL)

### Classical RL (Physical World)
In classical RL (e.g., a robotic vacuum cleaner like a Roomba navigating a room):
* **Agent:** The decision-maker (the robot).
* **Environment:** The world the agent interacts with (the living room).
* **State ($s_t$):** The agent's current situation at time $t$ (coordinates in the room).
* **Action ($a_t$):** The decision made by the agent (move forward, turn right).
* **Reward ($r_t$):** Scalar feedback from the environment indicating the quality of the action.
* **Trajectory ($\tau$):** A sequence of states and actions: $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$.

### RL applied to Large Language Models (LLMs)
When translating reinforcement learning to generative language models:
* **Agent:** The LLM (parameterized by weights $\theta$, acting as the policy $\pi_\theta$).
* **Environment:** Everything outside the model (the prompt, user context, compilers, APIs).
* **State ($s_t$):** The text sequence generated so far (the prompt plus all tokens generated up to step $t$).
* **Action ($a_t$):** The generation of the *next single token* from the vocabulary.
* **Reward ($R$ or $r_t$):** 
  * *Outcome-based:* A single reward evaluated at the end of the full generation ($r_{T-1} = R$, and all intermediate $r_t = 0$).
  * *Process-based:* Step-by-step rewards given at the end of intermediate reasoning steps.

### Autoregressive Token Generation as a Trajectory
Consider a math problem where the target answer is `72`.
* **$s_0$ (Initial State):** The system prompt and user query.
* **$a_0$ (First Action):** The first generated token (e.g., `"To"`).
* **$s_1$ (New State):** Prompt + `"To"`.
* **$a_1$ (Second Action):** The second token (e.g., `"find"`).
* This autoregressive loop continues until the model outputs the end-of-sequence token (`<end>`), completing the trajectory $\tau$.

---

## 4. Policy Gradient Derivation from First Principles

The goal of policy-based reinforcement learning is to find the model parameters $\theta$ that maximize the expected trajectory reward under the policy $\pi_\theta$.

Let the objective function be:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

Where $R(\tau)$ is the total reward of trajectory $\tau$, and the probability of a trajectory occurring under policy $\pi_\theta$ is $P(\tau; \theta)$. We can expand the expectation as a sum over all possible trajectories:
$$J(\theta) = \sum_{\tau} P(\tau; \theta) R(\tau)$$

To maximize $J(\theta)$ using gradient ascent, we compute the gradient with respect to $\theta$:
$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) R(\tau) = \sum_{\tau} \nabla_\theta P(\tau; \theta) R(\tau)$$

Because we cannot sum over the infinite space of all possible trajectories, we must convert this expression back into an expectation so it can be estimated via sampling. We use the identity from calculus (the **log-derivative trick**):
$$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)} \implies \nabla_\theta f(\theta) = f(\theta) \nabla_\theta \log f(\theta)$$

Substituting $P(\tau; \theta)$ for $f(\theta)$:
$$\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$$

Now, substitute this back into our gradient equation:
$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log P(\tau; \theta) R(\tau)]$$

### Deconstructing Trajectory Probability $P(\tau; \theta)$
The probability of generating trajectory $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ is:
$$P(\tau; \theta) = P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) P(s_{t+1} | s_t, a_t)$$

Where $P(s_0)$ is the initial state distribution and $P(s_{t+1} | s_t, a_t)$ is the environmental state transition probability. Taking the natural logarithm:
$$\log P(\tau; \theta) = \log P(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} | s_t, a_t)$$

Now, we take the gradient with respect to $\theta$. Because $P(s_0)$ and the transition dynamics $P(s_{t+1} | s_t, a_t)$ do not depend on the LLM's parameters $\theta$, their gradients are zero:
$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t)$$

Substituting this back into our expectation yields the **Policy Gradient Theorem** formula:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right]$$

### Parameter Update Rule
At each step of gradient ascent, the parameters $\theta$ are updated using:
$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R$$
* $\alpha$: Learning rate.
* $R$: Trajectory-level reward (scalar).

---

## 5. Variance Reduction, Baselines, and the Critic

### The Variance Problem
In its raw form (equation above), if all rewards are positive (e.g., $R \in [100, 101, 102]$), the gradient update will increase the probability of *every* action taken during the trajectory, regardless of whether individual actions were exceptionally good or mediocre. This causes high variance and extremely slow, unstable convergence.

### Subtracting a Baseline
To mitigate this, we subtract a state-dependent baseline $b(s_t)$ from the reward. The baseline represents the expected return. Subtracting it ensures that we only reinforce actions that perform *better* than average, while penalizing those that perform *worse* than average.

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \underbrace{(G_t - b(s_t))}_{\text{Advantage } A_t}$$

Where:
* $G_t = \sum_{k=0}^{T-1-t} \gamma^k r_{t+k}$ is the discounted return at step $t$.
* $b(s_t)$ is the baseline.

### Actor-Critic Framework (e.g., PPO)
In Actor-Critic algorithms like PPO:
* **Actor ($\pi_\theta$):** The policy network (the LLM itself) that selects tokens.
* **Critic ($V_\phi$):** A separate neural network initialized to estimate the expected return from any given state: $V_\phi(s_t) \approx \mathbb{E}[G_t | s_t]$.
* **Advantage ($A_t$):** Calculated as $A_t = G_t - V_\phi(s_t)$.

#### Drawbacks of the Actor-Critic Setup for LLMs:
1. **GPU Memory Overhead:** If the Actor is a 70B parameter model, the Critic must often be of similar scale to accurately assess states. This doubles the GPU memory requirement during training.
2. **Computational Footprint:** Training and backpropagating through two massive models simultaneously slows down iteration speeds.
3. **Hyperparameter Instability:** Balancing the learning dynamics between the actor and the critic adds significant optimization complexity.

---

## 6. How GRPO Works: Group Relative Policy Optimization

GRPO completely bypasses the need for a separate Critic network by calculating the baseline directly from statistical properties of a **group** of parallel generations.

```
                  ┌──► Output 1 ──► Reward R₁ ┐
                  ├──► Output 2 ──► Reward R₂ │
Prompt (State s₀) ┼──► Output 3 ──► Reward R₃ ┼──► Compute Mean (μ) & Std (σ) ──► Normalize Advantages (A_i)
                  ├──► Output 4 ──► Reward R₄ │
                  └──► Output G ──► Reward R_G┘
```

### The GRPO Advantage Formulation
Instead of evaluating the absolute quality of a single output sequence using a Critic, GRPO generates $G$ different responses $\{\tau_1, \tau_2, \dots, \tau_G\}$ for the *same* prompt.

1. Generate $G$ independent trajectories from the current policy $\pi_\theta$.
2. Obtain a reward for each trajectory: $\{R_1, R_2, \dots, R_G\}$.
3. Calculate the mean ($\mu$) and standard deviation ($\sigma$) of the rewards across the group:
   $$\text{mean}(r_g) = \frac{1}{G} \sum_{j=1}^G R_j$$
   $$\text{std}(r_g) = \sqrt{\frac{1}{G} \sum_{j=1}^G (R_j - \text{mean}(r_g))^2}$$
4. Compute the normalized **relative advantage** $A_i$ for each response in the group:
   $$A_i = \frac{R_i - \text{mean}(r_g)}{\text{std}(r_g)}$$

This relative advantage replaces the baseline-subtracted return $(G_t - V_\phi(s_t))$. 

### Computational Benefits:
* **Critic Elimination:** Eliminating $V_\phi$ saves up to 50% of the GPU memory and reduces training time.
* **No Extra Weights:** The baseline is calculated dynamically from sampling outputs, requiring no additional learnable parameters.

---

## 7. Process vs. Outcome Supervision in GRPO

| Feature | Outcome Supervision (GRPO + OS) | Process Supervision (GRPO + PS) |
| :--- | :--- | :--- |
| **Reward Delivery** | Single evaluation at the very end of the output sequence. | Multiple evaluations at individual intermediate steps. |
| **Example (Math)** | Correct answer = $1.0$, Incorrect = $0.0$. | Step 1 correct = $0.2$, Step 2 correct = $0.3$, etc. |
| **Advantage Calculation** | Constant advantage across all tokens in a trajectory: $A_{i,t} = A_i$. | Token-level advantage based on the step-level reward $r_j$ it belongs to: $A_{i,t} = \frac{r_{j \ge t} - \text{mean}(r_g)}{\text{std}(r_g)}$. |
| **Complexity** | Extremely simple; relies on straightforward verifiers (regex, code execution). | Complex; requires step-level verifiers or auxiliary reward models. |

### Empirical Insights (DeepSeek-R1 Paper)
Comparative benchmarks show that while process supervision (GRPO + PS) offers some stability gains on certain mathematical datasets (e.g., GSM8K), the performance delta over outcome supervision (GRPO + OS) is often marginal. Given the significant added complexity of step-level verifiers, the prevailing research consensus is to favor outcome-based reward mechanisms where possible ("keep it simple").

---

## 8. Mathematical Formulations: PPO Loss vs. GRPO Loss

To protect against policy collapse due to high-variance updates, we must constrain how much the policy parameters $\theta$ can change in a single training step.

### Step 1: The Probability Ratio
Define the probability ratio $r_t(\theta)$ between the action probabilities under the new policy $\pi_\theta$ and the old policy $\pi_{\theta_{old}}$:
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

### Step 2: The PPO Clipping Objective
PPO constraints the step update by clipping this ratio if it wanders outside of an interval defined by hyperparameter $\epsilon$ (typically $0.1$ or $0.2$):

$$\mathcal{L}^{PPO}(\theta) = \sum_{t=0}^{T-1} \min \left( r_t(\theta) A_t, \,\, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)$$

* **If $A_t > 0$ (Action is advantageous):** We want to increase $r_t(\theta)$. The objective increases linearly with $r_t$, but the `clip` function caps this growth at $1+\epsilon$. The `min` operator ensures we do not get reward credit for moving $r_t$ beyond this threshold.
* **If $A_t < 0$ (Action is disadvantageous):** We want to decrease $r_t(\theta)$. The `min` operator ensures that if the ratio becomes very small, we still penalize the policy appropriately.

---

### Step 3: The GRPO Objective Formulation
GRPO takes the PPO clipping objective, averages it over the sampled group $G$, and appends a **Kullback-Leibler (KL) divergence** regularization penalty.

$$\mathcal{L}^{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \sum_{t=0}^{T-1} \left[ \min \left( r_{i,t}(\theta) A_i, \,\, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A_i \right) \right] - \beta D_{KL}(\pi_\theta \,\, || \,\, \pi_{ref})$$

#### Key Components of the GRPO Loss:
1. **Group Averaging ($\frac{1}{G} \sum_{i=1}^G$):** Normalizes the objective across the parallel sampled trajectories.
2. **Probability Ratio for Group Output $i$ at token $t$ ($r_{i,t}(\theta)$):**
   $$r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} | s_{i,t})}{\pi_{\theta_{old}}(a_{i,t} | s_{i,t})}$$
3. **Relative Group Advantage ($A_i$):** Computed purely from the relative rewards within the sampled cohort.
4. **KL Regularization Term ($\beta D_{KL}(\pi_\theta \, || \, \pi_{ref})$):**
   * Measures the divergence between the active policy $\pi_\theta$ and a fixed reference policy $\pi_{ref}$ (typically the model's state immediately after Supervised Fine-Tuning, before RL began).
   * **Why it is critical in GRPO:** Because GRPO discards the Critic network, the optimization process is more vulnerable to variance. The explicit KL penalty prevents the active policy from drifting too far from the SFT model, acting as an anchor to maintain language coherence and formatting alignment.

---

## 9. Structural Comparison: PPO vs. GRPO

During a single training step, the execution loop differences are as follows:

| Process Component | Proximal Policy Optimization (PPO) | Group Relative Policy Optimization (GRPO) |
| :--- | :--- | :--- |
| **Memory Allocation** | Requires Actor ($\pi_\theta$) AND Critic ($V_\phi$) networks. | Requires ONLY Actor ($\pi_\theta$) network. |
| **Output Generation** | Single trajectory $\tau$ is generated. | A group of $G$ trajectories are generated in parallel. |
| **Baseline Source** | Estimated value function $V_\phi(s_t)$. | Mathematical average of the group rewards: $\text{mean}(r_g)$. |
| **Advantage Source** | Temporal difference error: $G_t - V_\phi(s_t)$. | Group-normalized reward: $\frac{R_i - \text{mean}(r_g)}{\text{std}(r_g)}$. |
| **Regularization** | Primarily relies on PPO clipping. | PPO clipping + explicit reference SFT policy KL penalty. |