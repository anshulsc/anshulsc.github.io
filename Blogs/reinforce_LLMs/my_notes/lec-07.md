# Comprehensive Lecture Notes: Temporal Difference (TD) Methods

---

## 1. Introduction and Overview
Temporal Difference (TD) learning is a central and novel approach to reinforcement learning (RL). It represents a blend of two foundational RL methodologies: **Monte Carlo (MC) methods** and **Dynamic Programming (DP)**. 

To contextualize TD methods, it is helpful to contrast them with DP and MC based on two key dimensions: **model dependency** and **bootstrapping**.

| Dimension | Dynamic Programming (DP) | Monte Carlo (MC) | Temporal Difference (TD) |
| :--- | :--- | :--- | :--- |
| **Model Dependency** | Model-based (requires transition probabilities $P(s' \mid s, a)$ and reward models). | Model-free (learns directly from raw environmental experience). | Model-free (learns directly from raw environmental experience). |
| **Bootstrapping** | Yes (updates estimates based on other learned estimates). | No (updates estimates based entirely on complete, actual returns $G_t$). | Yes (updates estimates based on other learned estimates). |
| **Update Frequency** | After each step (or complete sweep of state space). | Only at the end of complete episodes. | After each individual time step (online updates). |

---

## 2. The Prediction Problem: Policy Evaluation
The goal of the prediction problem is to estimate the state-value function $v_\pi(s)$ for a given policy $\pi$. This function represents the expected return starting from state $s$ and following policy $\pi$ thereafter:
$$v_\pi(s) \doteq \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$$

---

## 3. Mathematical Derivation of Update Rules

### 3.1 Recursive Average and the Monte Carlo Update
To understand how updates are constructed, we begin with the incremental update of a running average. 

Suppose we have observed a sequence of returns $G_0, G_1, G_2, \dots, G_{k-1}$ from $k$ independent episodes starting from a specific state $s$. The estimated value $V_k(s)$ after $k$ experiences is the sample average:
$$V_k(s) = \frac{G_0 + G_1 + G_2 + \dots + G_{k-1}}{k}$$

If we observe a new return $G_k$ from the $(k+1)$-th episode, the updated estimate $V_{k+1}(s)$ is:
$$V_{k+1}(s) = \frac{G_0 + G_1 + G_2 + \dots + G_{k-1} + G_k}{k+1}$$

To express $V_{k+1}(s)$ in terms of the previous estimate $V_k(s)$, we can manipulate the numerator:
$$(k+1) \cdot V_{k+1}(s) = \left( G_0 + G1 + G_2 + \dots + G_{k-1} \right) + G_k$$

Using the fact that $\sum_{i=0}^{k-1} G_i = k \cdot V_k(s)$:
$$(k+1) \cdot V_{k+1}(s) = k \cdot V_k(s) + G_k$$

Divide both sides by $k+1$:
$$V_{k+1}(s) = \frac{k}{k+1} V_k(s) + \frac{1}{k+1} G_k$$

Rewrite the fraction $\frac{k}{k+1}$ as $1 - \frac{1}{k+1}$:
$$V_{k+1}(s) = \left(1 - \frac{1}{k+1}\right) V_k(s) + \frac{1}{k+1} G_k$$
$$V_{k+1}(s) = V_k(s) + \frac{1}{k+1} \left( G_k - V_k(s) \right)$$

By substituting the step-size parameter $\frac{1}{k+1}$ with a constant step-size parameter $\alpha \in (0, 1]$, we arrive at the general **Monte Carlo update rule**:
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$

### 3.2 Key Components of the Update Rule
*   **$V(S_t)$**: The current estimate of the state's value (the old estimate).
*   **$G_t$**: The actual target return received after visiting state $S_t$. In Monte Carlo, this is the total discounted reward accumulated from time $t$ until the end of the episode:
    $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+t+3} + \dots$$
*   **$\alpha$**: The step-size parameter, which determines how much weight is given to the new experience versus the historical estimate.
*   **$[G_t - V(S_t)]$**: The estimation error (or TD error). It measures the difference between the actual return observed ($G_t$) and our prior estimate ($V(S_t)$).

> [!IMPORTANT]
> Because Monte Carlo methods rely on the actual return $G_t$, they must wait until the episode terminates before performing any updates. This is a significant limitation for long or non-terminating (continuing) tasks.

---

### 3.3 The Temporal Difference (TD(0)) Update
Temporal Difference (TD) methods resolve the latency issue of Monte Carlo by replacing the actual return $G_t$ with an estimate.

By definition, the expected return from state $S_t$ can be expanded recursively:
$$G_t = R_{t+1} + \gamma G_{t+1}$$

Using the current value function estimate $V(S_{t+1})$ as a proxy for the future return from step $t+1$ onwards, we approximate the target as:
$$G_t \approx R_{t+1} + \gamma V(S_{t+1})$$

Substituting this approximate target into the update equation yields the **$\text{TD}(0)$ (one-step TD) update rule**:
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

#### Analysis of the TD Target
*   **TD Target**: $R_{t+1} + \gamma V(S_{t+1})$
*   **TD Error ($\delta_t$)**: $\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

Unlike MC, which must wait for $G_t$ to be fully realized at the end of the episode, TD(0) requires waiting only until the next time step $t+1$ to observe $R_{t+1}$ and $S_{t+1}$. It then uses the current estimate $V(S_{t+1})$ to update $V(S_t)$. This process of updating an estimate based on another estimate is called **bootstrapping**.

---

## 4. Conceptual Intuition: The Commuting Example

Consider the daily scenario of driving home from work. The goal is to estimate the expected travel time remaining from each intermediate location.

### 4.1 Environment Formulation
*   **States ($S$)**: Major junctions along the commute route (Junctions 1 through 8, where Junction 1 is the Office and Junction 8/9 is Home).
*   **Actions ($A$)**: Fixed (we are evaluating travel times along a single, predetermined route).
*   **Rewards ($R$)**: The elapsed time (negative reward or positive cost) taken to travel between consecutive junctions.
*   **Discount Factor ($\gamma$)**: $1$ (undiscounted episodic task).

```
[Office/S1] ---> (S2) ---> (S3) ---> (S4) ---> (S5) ---> (S6) ---> (S7) ---> (S8) ---> [Home/S9]
```

### 4.2 Monte Carlo vs. TD Updates in Practice

Suppose you are at **Junction 3** and have a current estimate that it will take 30 minutes to get home ($V(S_3) = 30$). 

During your drive, you encounter a severe traffic jam between Junction 3 and Junction 4, which alone takes 25 minutes.

#### Under Monte Carlo:
To update your estimate for Junction 3, you cannot act on this delay immediately. You must:
1.  Complete the entire journey home through junctions 4, 5, 6, 7, and 8.
2.  Record the total time actually taken from Junction 3 to Home (e.g., $G_3 = 55$ minutes).
3.  Perform the update at the end of the trip:
    $$V(S_3) \leftarrow V(S_3) + \alpha [55 - V(S_3)]$$

#### Under Temporal Difference:
As soon as you cross the traffic jam and arrive at **Junction 4**, you observe that it took 25 minutes ($R_{t+1} = 25$) and you retrieve your current estimate of the remaining time from Junction 4 (e.g., $V(S_4) = 15$ minutes). 
Without waiting to reach home, you can immediately update $V(S_3)$:
$$\text{TD Target} = R_{t+1} + V(S_{t+1}) = 25 + 15 = 40 \text{ minutes}$$
$$V(S_3) \leftarrow V(S_3) + \alpha [40 - V(S_3)]$$

**Intuitive Benefit**: The TD agent learns online. If the environment changes or unexpected delays occur, the value estimates are updated step-by-step, allowing the agent to utilize updated values for decision-making within the same episode.

---

## 5. Detailed Algorithm: Tabular $\text{TD}(0)$ for Estimating $v_\pi$

Below is the complete algorithmic procedure for tabular $\text{TD}(0)$ policy evaluation:

```
Input: The policy π to be evaluated
Parameter: Step size α ∈ (0, 1], discount factor γ ∈ [0, 1]
Initialize V(s) arbitrarily (e.g., V(s) = 0) for all s ∈ S+, where V(terminal) = 0

Loop for each episode:
    Initialize S
    Loop for each step of episode (until S is terminal):
        A ← action given by π for S
        Take action A, observe reward R and next state S'
        V(S) ← V(S) + α * [R + γ * V(S') - V(S)]
        S ← S'
```

---

## 6. Empirical Evaluation: The Random Walk

The properties of TD(0) can be analyzed using a standard 1D Markov Decision Process known as the **Random Walk**.

### 6.1 Problem Specification
An agent starts in the center state **C** of a chain of five states: **A, B, C, D, and E**.

```
       0          0          0          0          0          1
[Left] <--- (A) <--- (B) <--- (C) ---> (D) ---> (E) ---> [Right]
Terminal                                                Terminal
```

*   **State Space**: $\mathcal{S} = \{A, B, C, D, E\}$, with two terminal states $\mathcal{S}^+ = \{\text{Left}, A, B, C, D, E, \text{Right}\}$.
*   **Policy ($\pi$)**: Equal probability of moving left or right from any state:
    $$\pi(\text{left} \mid s) = \pi(\text{right} \mid s) = 0.5 \quad \forall s \in \mathcal{S}$$
*   **Transitions**: Move to the adjacent state on the left or right.
*   **Rewards**:
    *   $R = 0$ for all transitions, except when transitioning to the right terminal state:
        $$R_{t+1} = 1 \quad \text{if } S_{t+1} = \text{Right}$$
*   **Discount Factor ($\gamma$)**: $1$ (undiscounted).

---

### 6.2 Analytical Derivation of True Values
Since this is a simple Markov chain, the true state values $v_\pi(s)$ can be computed analytically using the Bellman expectation equations:
$$v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma v_\pi(s') \right]$$

Setting $\gamma = 1$:
1.  $V(A) = 0.5(0) + 0.5 V(B) \implies V(A) = 0.5 V(B)$
2.  $V(B) = 0.5 V(A) + 0.5 V(C)$
3.  $V(C) = 0.5 V(B) + 0.5 V(D)$
4.  $V(D) = 0.5 V(C) + 0.5 V(E)$
5.  $V(E) = 0.5 V(D) + 0.5(1) \implies V(E) = 0.5 V(D) + 0.5$

Solving this linear system of 5 equations:
Substitute (1) into (2):
$$V(B) = 0.25 V(B) + 0.5 V(C) \implies 0.75 V(B) = 0.5 V(C) \implies V(B) = \frac{2}{3} V(C)$$
$$V(A) = \frac{1}{3} V(C)$$

Substitute (5) into (4):
$$V(D) = 0.5 V(C) + 0.25 V(D) + 0.25 \implies 0.75 V(D) = 0.5 V(C) + 0.25 \implies V(D) = \frac{2}{3} V(C) + \frac{1}{3}$$
$$V(E) = \frac{1}{3} V(C) + \frac{2}{3}$$

Substitute $V(B)$ and $V(D)$ into the equation for $V(C)$:
$$V(C) = 0.5 V(B) + 0.5 V(D)$$
$$V(C) = 0.5 \left( \frac{2}{3} V(C) \right) + 0.5 \left( \frac{1}{3} V(C) + \frac{1}{3} + \frac{1}{2} \right)$$

Solving these equations yields the analytical true values:
$$V(A) = \frac{1}{6}, \quad V(B) = \frac{2}{6}, \dots, \quad V(E) = \frac{5}{6}$$

Or generally:
$$V(s) = \frac{\text{index of state}}{6}$$
where state index for $A=1, B=2, C=3, D=4, E=5$.

---

### 6.3 Empirical Performance Comparison

Running the experiment with step-size $\alpha = 0.1$ for both TD(0) and Monte Carlo yields the following insights:

1.  **Value Function Accuracy**:
    *   After 100 episodes, the value estimates generated by **$\text{TD}(0)$** match the true analytical values almost perfectly.
    *   **Monte Carlo (MC)** estimates show higher variance and deviate more from the true linear progression, even after 100 episodes.

2.  **Learning Curves (RMS Error Over Episodes)**:
    *   **Root Mean Square (RMS) Error** is defined as:
        $$\text{RMS Error} \doteq \sqrt{\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( V(s) - v_\pi(s) \right)^2}$$
    *   The empirical data indicates that $\text{TD}(0)$ methods converge faster to lower RMS error rates than MC methods across a wide range of step sizes ($\alpha$).
    *   MC methods suffer from high variance because they rely on actual returns ($G_t$), which are affected by many random transitions along the full trajectory. TD methods reduce this variance by bootstrapping (using the current estimate of the next state's value), leading to more stable and faster learning.