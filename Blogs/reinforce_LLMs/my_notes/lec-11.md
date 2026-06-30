# Lecture Notes: Policy Gradient Methods and the Policy Gradient Theorem

---

## 1. Introduction and Contextual Relevance
Policy gradient methods represent a foundational pillar of modern reinforcement learning (RL). While value-based methods (such as Q-learning) have historically dominated the landscape of discrete control tasks, direct policy optimization has emerged as the premier paradigm for complex, high-dimensional, and continuous spaces. 

In the context of modern artificial intelligence—particularly **Reasoning Large Language Models (LLMs)**—policy gradient methods are of central importance. Algorithms such as RLHF (Reinforcement Learning from Human Feedback), PPO (Proximal Policy Optimization), and GRPO (Group Relative Policy Optimization) are deployed to train state-of-the-art reasoning engines (e.g., OpenAI's o1/o3 and DeepSeek-R1). These models formulate text generation as a sequential decision-making process where the "policy" ($\pi$) outputs tokens, and reinforcement learning is used to optimize the model's reasoning trajectories.

---

## 2. Why Policy Gradient Methods?

### 2.1 Limitations of Value-Based and Tabular Methods
In value-based reinforcement learning, the objective is to estimate an action-value function, $q(s,a)$, which represents the expected return of taking action $a$ in state $s$ and following a given policy thereafter. The optimal policy is implicitly derived by selecting the greedy action:

$$\pi(s) = \arg\max_{a} q(s,a)$$

To understand why this approach can become impractical, consider two distinct paradigms of Q-estimation:

#### Tabular Representation
For small environments, we can represent $q(s,a)$ as a lookup table. 
* **The Process:** For a given state $s$ with actions $\{a_1, a_2, a_3, a_4\}$, we compute or retrieve their respective expected returns $\{q_1, q_2, q_3, q_4\}$. We store these values in a table and choose the action corresponding to the highest $q$-value.
* **The Bottleneck:** As the state-action space grows, tabular representations suffer from the **curse of dimensionality**. For example, the game of Chess features approximately $10^{46}$ valid states. Storing such a table is physically impossible due to memory constraints.

```
       [ State s ]
      /    |    \    \
    a_1   a_2   a_3   a_4
    |     |     |     |
   q_1   q_2   q_3   q_4  ==> Store in Lookup Table ==> Select max(q_i)
```

#### Value Function Approximation
To bypass tabular limits, we approximate the Q-function using a parameterized function (e.g., a neural network) $f(s, a; \mathbf{w}) \approx q_{\pi}(s,a)$. While this solves the memory bottleneck, it introduces a secondary issue: **we are optimizing a proxy (the value function) to find a policy, rather than optimizing the policy itself.**

```
   Inputs: (s, a)  ===>  [ Deep Neural Network (w) ]  ===>  Output: Predicted Q(s,a)
```

### 2.2 The Case for Direct Policy Optimization
Policy gradient methods propose a more direct approach: **parametrize and optimize the policy directly without intermediate value function lookup/maximization steps.** 

Instead of asking, *"What is the expected value of every action in this state?"* we parameterize a policy $\pi(a \mid s, \mathbf{\theta})$ that directly outputs a probability distribution over actions. We then update the parameters $\mathbf{\theta}$ using gradient ascent to maximize expected cumulative reward.

---

## 3. Formulating Parameterized Policies

### 3.1 Mathematical Notation
Let $\pi(a \mid s, \mathbf{\theta})$ represent a parameterized, stochastic policy:

$$\pi(a \mid s, \mathbf{\theta}) = \mathbb{P}(A_t = a \mid S_t = s, \mathbf{\theta}_t = \mathbf{\theta})$$

* $s \in \mathcal{S}$ is the current state.
* $a \in \mathcal{A}$ is the action.
* $\mathbf{\theta} \in \mathbb{R}^{d'}$ is the parameter vector.

If the parameter space is $2$-dimensional, $\mathbf{\theta} = \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix}$. If it is $3$-dimensional, $\mathbf{\theta} = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \theta_3 \end{bmatrix}$, and so on. By tuning $\mathbf{\theta}$, we smoothly warp the action probability distribution across different states.

---

## 4. Policy Parameterization via Softmax

To ensure that the policy outputs a valid probability distribution, the parameterized representation must satisfy two fundamental axioms of probability:
1. **Non-negativity:** $\pi(a \mid s, \mathbf{\theta}) \ge 0 \quad \forall a \in \mathcal{A}$
2. **Normalization:** $\sum_{a \in \mathcal{A}} \pi(a \mid s, \mathbf{\theta}) = 1$

A common and robust method to achieve this for discrete action spaces is **Softmax Parameterization**.

### 4.1 Numerical Preference Functions
We first define a real-valued **numerical preference function** $h(s, a, \mathbf{\theta}) \in \mathbb{R}$. The preference function represents the unnormalized "affinity" of taking action $a$ in state $s$ under parameters $\mathbf{\theta}$. The function $h$ can be modeled as a linear combination of features, a polynomial, or a deep neural network.

#### Intuitive Example
Suppose an agent is in state $s$ and can choose from five actions $\{a_1, a_2, a_3, a_4, a_5\}$. Let us assume we have some prior numerical preferences for these actions:

| Action ($a$) | Preference score $h(s,a,\mathbf{\theta})$ | Qualitative Rank |
| :--- | :---: | :--- |
| $a_1$ | $3$ | Moderate |
| $a_2$ | $2$ | Low |
| $a_3$ | $4$ | High |
| $a_4$ | $5$ | Highest (Preferred) |
| $a_5$ | $1$ | Lowest |

We can visualize these raw scores using a continuous approximation curve:

```
Preference h(s,a,θ)
  ^
5 |             x (a_4)
4 |       x (a_3) 
3 | x (a_1)
2 |       x (a_2)
1 |                   x (a_5)
0 +--------------------------> Action (a)
```

Because these raw scores do not lie in the $[0,1]$ interval and do not sum to $1$, we cannot use them directly as a policy.

---

### 4.2 The Softmax Transformation
To convert these arbitrary real-valued preferences into a valid probability distribution, we apply the softmax function:

$$\pi(a \mid s, \mathbf{\theta}) = \frac{e^{h(s, a, \mathbf{\theta})}}{\sum_{b \in \mathcal{A}} e^{h(s, b, \mathbf{\theta})}}$$

#### Step-by-Step Numerical Walkthrough
Using our five-action preference table, let us calculate the probability $\pi(a_1 \mid s, \mathbf{\theta})$:

1. **Exponentiate individual preferences:**
   * $e^{h(s, a_1, \mathbf{\theta})} = e^3 \approx 20.086$
   * $e^{h(s, a_2, \mathbf{\theta})} = e^2 \approx 7.389$
   * $e^{h(s, a_3, \mathbf{\theta})} = e^4 \approx 54.598$
   * $e^{h(s, a_4, \mathbf{\theta})} = e^5 \approx 148.413$
   * $e^{h(s, a_5, \mathbf{\theta})} = e^1 \approx 2.718$

2. **Sum the exponentiated terms (The Partition Function):**
   $$\sum_{b \in \mathcal{A}} e^{h(s, b, \mathbf{\theta})} = e^3 + e^2 + e^4 + e^5 + e^1$$
   $$\sum_{b \in \mathcal{A}} e^{h(s, b, \mathbf{\theta})} \approx 20.086 + 7.389 + 54.598 + 148.413 + 2.718 = 233.204$$

3. **Normalize to find the probability of $a_1$:**
   $$\pi(a_1 \mid s, \mathbf{\theta}) = \frac{e^3}{233.204} \approx \frac{20.086}{233.204} \approx 0.086 \quad (8.6\%)$$

Repeating this calculation for all actions yields the final parameterized probability distribution:

```
Probability π(a|s,θ)
  ^
  |                  ■ (63.6%)
  |
  |            ■ (23.4%)
  |
  |      ■ (8.6%)
  |            ■ (3.2%)
  |                  ■ (1.2%)
0 +--------------------------> Actions
        a_1   a_2   a_3   a_4   a_5
```

This distribution preserves the relative ranking of the original preference values while satisfying all probability axioms.

---

## 5. The Policy Gradient Theorem

To optimize the policy parameter $\mathbf{\theta}$ directly, we need a scalar performance measure, $J(\mathbf{\theta})$, that evaluates the quality of the policy $\pi_{\mathbf{\theta}}$. Once defined, we can perform **gradient ascent**:

$$\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha \nabla_{\theta} J(\mathbf{\theta}_t)$$

where $\alpha > 0$ is the step size (learning rate) and $\nabla_{\theta} J(\mathbf{\theta})$ is the gradient of the performance measure with respect to the policy parameters.

```
                  Optimal Parameters (Point B)
                           / \
                          /   \  <--- Walk up the gradient (Ascent)
                         /     \
   Start Parameters ----*
      (Point A)
```

---

### 5.1 Defining the Performance Measure $J(\mathbf{\theta})$
In episodic tasks, a natural choice for the performance measure is the expected start value of the episode. Let $s_0$ be the designated, non-random start state of every episode. We define:

$$J(\mathbf{\theta}) \doteq v_{\pi_{\mathbf{\theta}}}(s_0)$$

where $v_{\pi_{\mathbf{\theta}}}(s_0)$ is the true state-value function of $s_0$ under the policy parameterized by $\mathbf{\theta}$.

---

### 5.2 The Fundamental Challenge of Policy Gradients
The performance $J(\mathbf{\theta})$ depends on two distinct factors, both of which are affected by the policy parameters $\mathbf{\theta}$:
1. **The Action Selection Probabilities:** How likely is the agent to take specific actions? This is directly determined by the policy: $\pi(a \mid s, \mathbf{\theta})$.
2. **The State Distribution:** Which states does the agent visit, and how frequently? This is represented by the stationary state distribution $\mu(s)$. 

Because the agent’s actions alter its path through the environment, changing $\mathbf{\theta}$ implicitly changes the distribution of future states $\mu(s)$. Mathematically, if we expand $J(\mathbf{\theta})$:

$$J(\mathbf{\theta}) = \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} \pi(a \mid s, \mathbf{\theta}) q_{\pi_{\mathbf{\theta}}}(s, a)$$

Taking the gradient with respect to $\mathbf{\theta}$ using the product rule yields:

$$\nabla_{\theta} J(\mathbf{\theta}) = \sum_{s \in \mathcal{S}} \left[ \nabla_{\theta} \mu(s) \sum_{a \in \mathcal{A}} \pi(a \mid s, \mathbf{\theta}) q_{\pi_{\mathbf{\theta}}}(s, a) + \mu(s) \nabla_{\theta} \left( \sum_{a \in \mathcal{A}} \pi(a \mid s, \mathbf{\theta}) q_{\pi_{\mathbf{\theta}}}(s, a) \right) \right]$$

The term **$\nabla_{\theta} \mu(s)$** is highly problematic. It represents the gradient of the state visitation frequency with respect to the policy parameters. This term depends on the unknown and complex transition dynamics of the environment (the state-transition probability matrix $P(s' \mid s, a)$). Without a perfect model of the world, calculating $\nabla_{\theta} \mu(s)$ directly is impossible.

---

### 5.3 The Policy Gradient Theorem Formulation
The **Policy Gradient Theorem** provides an analytical solution to this problem. It proves that we do not need to calculate the gradient of the state distribution $\nabla_{\theta} \mu(s)$. Under expectation, the terms involving the gradient of the state distribution cancel out.

The theorem states that:

$$\nabla_{\theta} J(\mathbf{\theta}) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} q_{\pi}(s, a) \nabla_{\theta} \pi(a \mid s, \mathbf{\theta})$$

#### Key Components of the Policy Gradient Formula

$$\nabla_{\theta} J(\mathbf{\theta}) \propto \sum_{s \in \mathcal{S}} \underbrace{\mu(s)}_{\substack{\text{State} \\ \text{Distribution}}} \sum_{a \in \mathcal{A}} \underbrace{q_{\pi}(s, a)}_{\substack{\text{Action-Value} \\ \text{Function}}} \underbrace{\nabla_{\theta} \pi(a \mid s, \mathbf{\theta})}_{\substack{\text{Gradient of} \\ \text{the Policy}}}$$

* **$\mu(s)$**: The stationary state distribution under the current policy $\pi_{\mathbf{\theta}}$. It weights the states by how often the agent visits them.
* **$q_{\pi}(s,a)$**: The expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter. This term acts as a scalar weight, scaling the gradient step to favor actions that yield higher expected rewards.
* **$\nabla_{\theta} \pi(a \mid s, \mathbf{\theta})$**: The direction in parameter space that increases the probability of choosing action $a$ in state $s$.

---

### 5.4 Intuitive Analytical Analogy
An intuitive way to understand why this simplification occurs is to consider a standard derivative from calculus:

$$\frac{d}{dx} \left[ f(u(x)) \right]$$

In general, this derivative requires the chain rule and involves computing the derivative of the inner function, $\frac{du}{dx}$. 

The Policy Gradient Theorem proves that for the reinforcement learning objective, the expected gradient of the performance measure behaves as if the inner derivative ($\nabla_{\theta} \mu(s)$) does not exist:

$$\nabla_{\theta} J(\mathbf{\theta}) \approx \mathbb{E}_{s \sim \mu, a \sim \pi} \left[ q_{\pi}(s, a) \nabla_{\theta} \ln \pi(a \mid s, \mathbf{\theta}) \right]$$

This allows us to obtain an unbiased estimate of the gradient purely through experience, without requiring any knowledge of the transition dynamics of the environment.

---

## 6. Summary of Key Concept Differences

| Attribute | Tabular Value Methods | Value Function Approximation | Policy Gradient Methods |
| :--- | :--- | :--- | :--- |
| **Target Representation** | Q-table ($|S| \times |A|$) | Parameterized Value Network $Q(s, a; \mathbf{w})$ | Parameterized Policy Network $\pi(a \mid s; \mathbf{\theta})$ |
| **Optimization Objective** | Minimize Bellman error locally | Minimize mean squared value error | Maximize expected return directly: $J(\mathbf{\theta})$ |
| **Action Selection** | Deterministic / Greedy ($\arg\max$) | Deterministic / Greedy ($\arg\max$) | Stochastic (directly outputs probabilities) |
| **Dimensional Limits** | Fails in large/continuous spaces | Scales well, but relies on value proxy | Scales effectively to continuous/complex spaces |

---

## 7. Looking Ahead: Practical Policy Gradient Algorithms
The analytical formulation provided by the Policy Gradient Theorem forms the basis for several practical, model-free reinforcement learning algorithms:

1. **REINFORCE (Monte Carlo Policy Gradient):** Bypasses the calculation of the true $q_{\pi}(s,a)$ by replacing it with the actual observed return $G_t$ from sample trajectories.
2. **REINFORCE with Baseline:** Subtracts a state-dependent baseline $b(s)$ (typically an estimate of the state-value function $v(s)$) from the return $G_t$ to reduce the variance of the gradient estimator without introducing bias.
3. **Actor-Critic Methods:** Uses two parameterized systems: an **Actor** (the parameterized policy $\pi_{\mathbf{\theta}}$) and a **Critic** (a parameterized value function $v_{\mathbf{w}}$ used to bootstrap and evaluate the quality of the selected actions).