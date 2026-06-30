# Lecture Notes: Monte Carlo Methods in Reinforcement Learning

---

## 1. Context and Course Positioning

This lecture is part of the reinforcement learning (RL) phase of the course. The primary objective of this phase is to explore methods that evaluate policies (estimate their value functions) and subsequently use those evaluations to discover optimal policies. 

In previous discussions, **Dynamic Programming (DP)** was presented as the foundational class of methods for this task. This lecture introduces **Monte Carlo (MC) Methods**, highlighting how they overcome a fundamental limitation of Dynamic Programming—namely, the requirement of a complete model of the environment.

---

## 2. Review of Dynamic Programming & Its Limitations

To understand the motivation behind Monte Carlo methods, we must first review the mechanics and limitations of Dynamic Programming.

### 2.1 The Concept of Bootstrapping
In Dynamic Programming, we estimate the value of a state by using the estimated values of its successor states. This process is known as **bootstrapping**. 

Mathematically, the Bellman expectation equation for the state-value function $V^\pi(s)$ under a policy $\pi$ is written as:

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} \sum_{r \in R} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

In an iterative policy evaluation setting, the update rule at iteration $k+1$ is:

$$V_{k+1}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} \sum_{r \in R} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

Here, the calculation of $V_{k+1}(s)$ relies on the estimated values of the next states, $V_k(s')$. Because we use one estimate to update another, we are "bootstrapping."

### 2.2 Generalized Policy Iteration (GPI)
Dynamic Programming algorithms typically follow the framework of Generalized Policy Iteration (GPI), which consists of two interacting processes:
1. **Policy Evaluation:** Computing the value function $V^\pi$ for the current policy $\pi$.
2. **Policy Improvement:** Making the policy greedy with respect to the current value function:

$$\pi'(s) = \arg\max_{a \in A} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

This evaluation-improvement loop continues iteratively until both the policy and the value function converge to their optimal counterparts ($\pi^*$ and $V^*$).

```
       Evaluation: V_k ---> V^π
      /                         \
  Policy π_k               Improved Policy π_{k+1}
      \                         /
       Improvement: π_{k+1} = greedy(V_k)
```

### 2.3 The Fundamental Limitation of DP
The major limitation of Dynamic Programming is its dependence on a **complete model of the environment**. Specifically, to calculate the expected value in both the evaluation and improvement steps, the agent must know the transition probabilities and reward dynamics of the environment, represented by:

$$P(s', r | s, a) \doteq \mathbb{P}(S_{t} = s', R_{t} = r \mid S_{t-1} = s, A_{t-1} = a)$$

In most practical, real-world scenarios, this model is unavailable:
* **Chess:** Calculating the exact transition probability of moving from one board state to another based on an opponent's unknown strategy is computationally and practically intractable.
* **Robotics / Mars Rover:** A physical rover deployed on Mars cannot possess a prior, mathematically exact model of the terrain's friction, slope variances, or structural hazards. 

Thus, we require methods that can learn directly from **experience** without requiring prior knowledge of the environment's transition and reward dynamics.

---

## 3. Introduction to Monte Carlo (MC) Methods

Monte Carlo methods are **model-free learning algorithms**. Instead of computing expectations using a mathematical model of the environment, they estimate value functions by averaging the returns experienced during interactions with the environment.

### 3.1 Core Characteristics of MC
* **Model-Free:** No prior knowledge of the transition probability $P(s', r | s, a)$ is required.
* **Learns from Experience:** The agent interacts with the environment, gathers samples of state-action-reward sequences, and learns from these experienced trajectories.
* **No Bootstrapping:** Unlike DP, Monte Carlo methods do *not* update value estimates based on other value estimates. Instead, they update estimates using the actual, empirical returns ($G_t$) observed at the end of each episode.
* **Episode-Based:** MC methods are restricted to **episodic tasks**. Value updates only occur after an episode terminates, as the complete return $G_t$ must be known before an update can be computed.

---

## 4. Monte Carlo Prediction (Policy Evaluation)

The objective of Monte Carlo prediction is to estimate the state-value function $V^\pi(s)$ for a given policy $\pi$.

### 4.1 The Empirical Return
The value of a state is the expected return (expected discounted sum of future rewards) starting from that state:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$$

where the return $G_t$ is defined as:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T$$

Here, $\gamma \in [0, 1]$ is the discount factor, and $T$ is the terminal step of the episode.

### 4.2 Estimation via Sample Averages
Because we do not have a model to compute the expectation $\mathbb{E}_\pi$, we estimate it using the sample average of the returns observed after visiting state $s$. If state $s$ is visited $N(s)$ times across multiple episodes, and the return following the $i$-th visit is $G_i(s)$, the Monte Carlo estimate is:

$$V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)$$

By the **Law of Large Numbers**, as the number of observed visits $N(s) \to \infty$, the sample average converges to the true expected value:

$$\lim_{N(s) \to \infty} \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s) = V^\pi(s)$$

### 4.3 First-Visit vs. Every-Visit MC
There are two primary approaches to counting state visits within an episode:
1. **First-Visit MC:** Only the first time a state $s$ is visited within a single episode is counted. The return $G_t$ following that first visit is appended to the list of returns for $s$, and $V(s)$ is updated by averaging these first-visit returns across all episodes.
2. **Every-Visit MC:** Every time a state $s$ is visited within an episode, the subsequent return from that point is counted and averaged.

### 4.4 Algorithmic Implementation of MC Prediction
Below is the pseudo-algorithm for First-Visit Monte Carlo prediction:

```python
Initialize:
    π = policy to be evaluated
    V = arbitrary state-value function (e.g., all zeros)
    Returns(s) = empty list for all s ∈ S

Loop forever (for each episode):
    Generate an episode following π: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T
    G_t = 0
    Loop for each step of the episode, t = T-1, T-2, ..., 0:
        G_t = R_{t+1} + γ * G_t
        Unless S_t appears in S_0, S_1, ..., S_{t-1}:
            Append G_t to Returns(S_t)
            V(S_t) = average(Returns(S_t))
```

---

## 5. From State Values ($V$) to Action Values ($Q$)

In Dynamic Programming, we can perform policy improvement using the state-value function $V(s)$ because the model $P(s', r | s, a)$ allows us to look one step ahead:

$$\pi'(s) = \arg\max_{a \in A} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V(s') \right]$$

However, in a **model-free** setting, we do not know $P(s', r | s, a)$. Therefore, even if we have evaluated $V(s)$ for all states, we cannot determine which action $a$ is best because we cannot compute the transition dynamics to the next states.

To resolve this, model-free control algorithms estimate the **action-value function** $Q^\pi(s, a)$ instead of $V^\pi(s)$. 

### 5.1 The Action-Value Function $Q(s, a)$
$Q^\pi(s, a)$ represents the expected return starting from state $s$, taking action $a$, and thereafter following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]$$

With access to $Q(s, a)$, policy improvement can be performed directly without a model of the environment. The greedy action choice is simply:

$$\pi'(s) = \arg\max_{a \in A} Q(s, a)$$

---

## 6. Monte Carlo Control & Exploration

Monte Carlo Control follows the same pattern as GPI, alternating between policy evaluation (estimating $Q(s, a)$) and policy improvement.

```
       Evaluation: Q_k ---> Q^π
      /                         \
  Policy π_k               Improved Policy π_{k+1}
      \                         /
       Improvement: π_{k+1} = argmax_a Q_k(s, a)
```

### 6.1 The Exploration Problem
If we use a deterministic greedy policy to interact with the environment, we run into a major issue: **we may never visit certain state-action pairs**.

If an action $a$ at state $s$ is never selected, we cannot collect returns for the pair $(s, a)$. Consequently, we can never update its $Q$-value estimate $Q(s, a)$, and we might miss out on discovering that action $a$ is actually superior to our current choice. This is the **Exploration vs. Exploitation** dilemma.

### 6.2 $\epsilon$-Greedy Policies
To ensure that the agent continues to explore all actions, we can use an **$\epsilon$-greedy policy**. Under an $\epsilon$-greedy policy, the agent selects the greedy action most of the time, but with a small probability $\epsilon$, it selects an action at random.

For a state $s$ with a set of possible actions $A(s)$, the probability of selecting action $a$ is defined as:

$$\pi(a|s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|A(s)|} & \text{if } a = a^* \text{ (the greedy action)} \\ 
\frac{\epsilon}{|A(s)|} & \text{if } a \neq a^* \text{ (non-greedy actions)} 
\end{cases}$$

Where $|A(s)|$ is the number of possible actions in state $s$, and $a^* = \arg\max_{a \in A} Q(s, a)$.

#### Verification of Probability Summation
To confirm this forms a valid probability distribution, we sum the probabilities over all actions:

$$\sum_{a \in A(s)} \pi(a|s) = \left( 1 - \epsilon + \frac{\epsilon}{|A(s)|} \right) + (|A(s)| - 1) \cdot \frac{\epsilon}{|A(s)|}$$

$$\sum_{a \in A(s)} \pi(a|s) = 1 - \epsilon + \frac{\epsilon}{|A(s)|} + \epsilon - \frac{\epsilon}{|A(s)|} = 1$$

This policy guarantees that every action has a non-zero probability of selection ($\ge \frac{\epsilon}{|A(s)|}$), ensuring continuous exploration of the entire state-action space over time.

---

## 7. On-Policy vs. Off-Policy Learning

When evaluating and improving policies using experienced trajectories, we can categorize algorithms based on whether the policy used to collect data is the same as the policy being optimized:

### 7.1 On-Policy Methods
In **On-Policy** methods, the policy we are evaluating and optimizing is the *same* policy used to interact with the environment and generate the episodes.
* **Target Policy:** $\pi$ (the policy we want to evaluate/improve, which is soft/exploratory, such as $\epsilon$-greedy).
* **Behavior Policy:** $\pi$ (the policy used to generate experience).
* **Limitation:** The final learned policy must remain exploratory (soft) so that data collection continues to cover all actions. Thus, we cannot learn a purely optimal deterministic policy.

### 7.2 Off-Policy Methods
In **Off-Policy** methods, we use two separate policies:
1. **Behavior Policy ($\mu$):** The policy used to generate the experienced trajectories (typically highly exploratory, e.g., uniform random or $\epsilon$-greedy with a large $\epsilon$).
2. **Target Policy ($\pi$):** The policy we are evaluating and optimizing (typically a deterministic greedy policy, $\pi(a|s) = 1$ for the optimal action, $0$ otherwise).

---

## 8. Off-Policy Monte Carlo & Importance Sampling

Because the episodes are generated using the behavior policy $\mu$, the experienced returns $G_t$ reflect the distribution of trajectories under $\mu$, not our target policy $\pi$. To evaluate $\pi$ using data from $\mu$, we must scale the returns using **Importance Sampling**.

### 8.1 Mathematical Derivation of Importance Sampling
Suppose we want to estimate the expectation of a function $f(x)$ where $x$ is sampled from a target distribution $P$. However, we only have samples of $x$ drawn from a proposal/behavior distribution $Q$.

$$\mathbb{E}_{x \sim P} [f(x)] = \sum_{x} f(x) P(x) = \sum_{x} f(x) P(x) \frac{Q(x)}{Q(x)}$$

$$\mathbb{E}_{x \sim P} [f(x)] = \sum_{x} \left( f(x) \frac{P(x)}{Q(x)} \right) Q(x) = \mathbb{E}_{x \sim Q} \left[ f(x) \frac{P(x)}{Q(x)} \right]$$

The term $\frac{P(x)}{Q(x)}$ is called the **Importance Sampling Ratio**. It scales the samples drawn from $Q$ so that their weighted average equals the expectation under $P$.

### 8.2 Trajectory Probability Ratio
In reinforcement learning, the probability of a state-action trajectory $S_t, A_t, S_{t+1}, A_{t+1}, \dots, S_T$ under a policy $\pi$ is:

$$\mathbb{P}(A_t, S_{t+1}, A_{t+1}, \dots, S_T \mid S_t, \pi) = \prod_{k=t}^{T-1} \pi(A_k | S_k) P(S_{k+1} | S_k, A_k)$$

Note that the environment transition probabilities $P(S_{k+1} | S_k, A_k)$ are identical regardless of the policy. 

When we calculate the ratio of the trajectory probability under the target policy $\pi$ to that under the behavior policy $\mu$, the environment dynamics terms cancel out:

$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k | S_k) P(S_{k+1} | S_k, A_k)}{\prod_{k=t}^{T-1} \mu(A_k | S_k) P(S_{k+1} | S_k, A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{\mu(A_k | S_k)}$$

This is the **Importance Sampling Ratio** for a trajectory. We can use it to scale the experienced returns:

$$G_t^{\text{adjusted}} = \rho_{t:T-1} G_t$$

These adjusted returns are then averaged to estimate $V^\pi(s)$ or $Q^\pi(s, a)$. This approach allows us to learn about a deterministic target policy $\pi$ while behaving according to a completely different exploratory policy $\mu$.

### 8.3 Importance Sampling Numerical Example
Consider a single-step transition where we observe:
* **Target Policy Probability:** $\pi(a_1|s_1) = 0.775$
* **Behavior Policy Probability:** $\mu(a_1|s_1) = 0.587$
* **Observed Return:** $G_t = 1.0$

The importance sampling ratio for this step is:

$$\rho_t = \frac{\pi(a_1|s_1)}{\mu(a_1|s_1)} = \frac{0.775}{0.587} \approx 1.319$$

The adjusted return under the target policy is calculated as:

$$G_t^{\text{adjusted}} = \rho_t \cdot G_t = 1.319 \times 1.0 = 1.319$$

**Intuition:** Because the target policy is more likely to choose action $a_1$ than the behavior policy ($0.775 > 0.587$), the return obtained from this transition is scaled up to reflect its higher expected frequency under the target policy.

---

## 9. On-Policy First-Visit MC Control: Gridworld Simulation

The interactive demonstration in the lecture illustrates On-Policy First-Visit Monte Carlo Control on a $4 \times 4$ Gridworld.

```
+---+---+---+---+
| S |   |   |   |   S = Start (0,0)
+---+---+---+---+   H = Hazard / Obstacle (Red X)
|   | H |   |   |   G = Goal (Green Circle)
+---+---+---+---+
|   |   |   |   |
+---+---+---+---+
|   |   |   | G |
+---+---+---+---+
```

### 9.1 Environment Parameters
* **Grid Size:** $4 \times 4$ (16 discrete states, index from $(0,0)$ to $(3,3)$).
* **Actions:** $\text{Up, Down, Left, Right}$ ($|A| = 4$).
* **Rewards:** 
  * Reaching the Goal cell $G$ yields a positive reward (e.g., $+1$).
  * Entering a Hazard cell $H$ (marked with a red **X**) yields a negative reward (e.g., $-1$).
  * Every other step incurs a small step penalty (e.g., $-0.1$) to encourage finding the shortest path.
* **Exploration Parameter:** $\epsilon = 0.2$.

### 9.2 Step-by-Step Policy Probability Assignment
With $\epsilon = 0.2$ and $|A| = 4$, the probabilities for action selection are computed as follows:

1. **Greedy Action probability:**

$$1 - \epsilon + \frac{\epsilon}{|A|} = 1 - 0.2 + \frac{0.2}{4} = 0.8 + 0.05 = 0.85 \text{ (or } 85\%\text{)}$$

2. **Non-Greedy Actions probability:**

$$\frac{\epsilon}{|A|} = \frac{0.2}{4} = 0.05 \text{ (or } 5\%\text{)}$$

This distribution ensures that the agent prioritizes the greedy path while dedicating a combined $15\%$ probability to exploring alternative moves.

### 9.3 Mechanics of Value & Policy Updates in the Simulation
1. **Episode Generation:** An episode is generated from the start state to the terminal goal state using the current $\epsilon$-greedy policy.
2. **First-Visit Return Calculation:** For each state-action pair $(s, a)$ visited in the episode:
   * The empirical return $G_t$ is computed from the first occurrence of $(s, a)$ to the end of the episode.
   * $G_t$ is appended to the historical list of returns for $(s, a)$.
3. **Action-Value Update:** The action-value estimate is updated by averaging the list of returns:
   
$$Q(s, a) = \text{average}(\text{Returns}(s, a))$$

4. **Policy Improvement:** The greedy action is updated:
   
$$a^* = \arg\max_{a} Q(s, a)$$

The probability distribution $\pi(a|s)$ is then recalculated using the updated $a^*$. As the simulation runs for more episodes, the estimated action values converge, and the policy stabilizes toward the shortest path to the goal that avoids the hazards.

---

## 10. Summary and Comparison: DP vs. MC

| Feature | Dynamic Programming (DP) | Monte Carlo (MC) Methods |
| :--- | :--- | :--- |
| **Model Requirements** | Requires a complete model of the environment ($P(s', r \mid s, a)$). | Model-free (requires only raw experience). |
| **Bootstrapping** | Yes (updates estimates based on other estimates). | No (updates estimates based on actual empirical returns). |
| **Task Constraints** | Can be applied to both continuous and episodic tasks. | Strictly limited to episodic tasks. |
| **Update Frequency** | Updates values step-by-step during an iteration. | Updates values only at the end of an episode. |
| **Source of Value** | Solves systems of Bellman expectation equations. | Averages experienced sample returns. |

### Looking Forward
* **Dynamic Programming** benefits from bootstrapping, which allows it to update estimates step-by-step without waiting for an episode to end, but it is limited by its need for an environment model.
* **Monte Carlo** is model-free and learns from real-world experience, but it cannot bootstrap and must wait until the end of an episode to make updates.
* **The Next Step:** In the next lecture, we will explore **Temporal Difference (TD) Learning**. TD learning combines the best of both worlds: it is model-free (like Monte Carlo) but updates its estimates step-by-step using bootstrapping (like Dynamic Programming).