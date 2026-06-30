# Lecture 10: Temporal Difference Methods — Control Algorithms (SARSA & Q-Learning)

---

## 1. Introduction and Recap of TD Prediction

Temporal Difference (TD) learning methods combine the key advantages of Dynamic Programming (DP) and Monte Carlo (MC) methods:
* **Model-Free:** Like Monte Carlo methods, TD methods can learn directly from raw experience without requiring a transition probability model of the environment.
* **Bootstrapping:** Like Dynamic Programming, TD methods update their estimates based on other, previously learned estimates, without waiting for the final outcome of an episode.

### 1.1 The Prediction Problem (Policy Evaluation)
The prediction problem involves estimating the state-value function $v_\pi(s)$ for a given policy $\pi$. 

In the previous lecture, the **TD(0)** algorithm was introduced. The update rule for TD(0) is:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

Where:
* $S_t$ is the state visited at time step $t$.
* $R_{t+1}$ is the immediate reward received at step $t+1$.
* $S_{t+1}$ is the next state visited at step $t+1$.
* $\alpha \in (0, 1]$ is the step-size parameter (learning rate).
* $\gamma \in [0, 1]$ is the discount factor.
* $R_{t+1} + \gamma V(S_{t+1})$ is known as the **TD Target**.
* $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD Error**.

### 1.2 The Concept of Bootstrapping
When the value function is updated using TD(0), the target uses the current estimate of the next state's value $V(S_{t+1})$ instead of the actual return $G_t$ of the entire episode. 

$$\begin{aligned}
\text{Monte Carlo Target (No Bootstrapping):} \quad & G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T \\
\text{TD Target (Bootstrapping):} \quad & R_{t+1} + \gamma V(S_{t+1})
\end{aligned}$$

Because TD methods bootstrap, updates are performed incrementally at every single time step. We do not need to wait until the terminal state $S_T$ is reached to update our estimates, making TD algorithms highly suited for continuous (non-episodic) environments.

---

## 2. Transitioning from Prediction to Control

To solve the control problem—where the objective is to determine the optimal policy $\pi^*$—we must modify our prediction framework.

### 2.1 The Need for Action-Value Functions $Q(s, a)$
In model-based settings (like Dynamic Programming), knowing the state-value function $V(s)$ along with the transition dynamics $P(s' \mid s, a)$ is sufficient to derive a policy:

$$\pi(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]$$

However, in model-free settings, the transition probabilities $P(s' \mid s, a)$ are unknown. Even if we estimate a highly accurate state-value function $V(s)$, we cannot determine which action $a$ is optimal because we do not know which action leads to which next states. 

To overcome this limitation, model-free control algorithms estimate the **action-value function** $Q(s, a)$ instead of the state-value function $V(s)$. The action-value function represents the expected return when starting in state $s$, taking action $a$, and subsequently following policy $\pi$. 

Once $Q(s, a)$ is learned, a greedy policy can be determined directly without requiring any model of the environment:

$$\pi(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$

---

## 3. On-Policy TD Control: SARSA

**SARSA** is an on-policy TD control algorithm. The name "SARSA" represents the sequence of variables that are processed during a single transition update: State, Action, Reward, State, Action.

$$\left( S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1} \right)$$

### 3.1 The SARSA Update Derivation
To derive the SARSA update rule, we apply the TD prediction formula directly to action-values. Instead of estimating $V(S_t)$ using $V(S_{t+1})$, we estimate $Q(S_t, A_t)$ using $Q(S_{t+1}, A_{t+1})$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

* **TD Target for SARSA:** $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$
* **TD Error for SARSA:** $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$

This update occurs at every step of an episode. The next action $A_{t+1}$ must be chosen using the same policy $\pi$ that generated $A_t$ in state $S_t$. This is why SARSA is classified as an **on-policy** algorithm: the policy being evaluated and improved is the exact same policy that is used to select the actions.

### 3.2 The SARSA Control Loop
To guarantee that the algorithm explores all state-action pairs while still converging toward the optimal policy, we utilize an exploratory policy, typically the **$\epsilon$-greedy** policy.

```
Initialize Q(s, a) arbitrarily for all s in S, a in A (and Q(terminal, .) = 0)
For each episode:
    Initialize S
    Choose A from S using a policy derived from Q (e.g., epsilon-greedy)
    For each step of the episode (until S is terminal):
        Take action A, observe reward R and next state S'
        Choose A' from S' using a policy derived from Q (e.g., epsilon-greedy)
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        S <- S'
        A <- A'
```

---

## 4. The Exploration-Exploitation Dilemma and $\epsilon$-Greedy Policies

If an agent acts purely greedily based on its current action-value estimates, it may fail to find the optimal path. 

### 4.1 The Need for Exploration
Consider a state $s$ with three possible actions, $\{a_1, a_2, a_3\}$. Suppose our current estimates are:
* $Q(s, a_1) = 32$
* $Q(s, a_2) = 28$
* $Q(s, a_3) = 0$ (unexplored)

If the agent always acts greedily, it will choose $a_1$. However, the true (but unknown) optimal value for $a_3$ might be $45$. Because $Q(s, a_3)$ starts at $0$, a purely greedy agent will never choose $a_3$, completely missing the optimal action. 

In Dynamic Programming, this is not an issue because we have a mathematical model of the environment that provides the full transition landscape. In model-free learning, we must actively balance **exploitation** (choosing the action with the highest estimated value to maximize reward) and **exploration** (trying unfamiliar actions to gain more information).

### 4.2 The $\epsilon$-Greedy Strategy
The $\epsilon$-greedy policy is a simple and effective method to balance exploration and exploitation. Under this policy, the agent behaves as follows:
* With probability $1 - \epsilon$, the agent **exploits** its current knowledge by selecting the greedy action:
  $$A_t = \arg\max_{a \in \mathcal{A}} Q(S_t, a)$$
* With probability $\epsilon$, the agent **explores** by selecting an action uniformly at random from the set of all available actions $\mathcal{A}(S_t)$ (including the greedy action).

The probability of choosing any arbitrary action $a$ under an $\epsilon$-greedy policy with $|\mathcal{A}|$ total actions is:

$$\pi(a \mid s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{if } a \neq \arg\max_{a'} Q(s, a') 
\end{cases}$$

---

## 5. Off-Policy TD Control: Q-Learning

Q-Learning, developed by Chris Watkins in 1989, is an **off-policy** temporal difference control algorithm. 

### 5.1 Intuitive Difference: On-Policy vs. Off-Policy
To understand the conceptual distinction, consider the analogy of learning to ride a bicycle:
* **On-Policy Learning (SARSA):** You learn to ride the bicycle by actually riding it yourself. You make mistakes, experience falls, and directly update your strategy based on your own trials and errors. Your target policy is identical to your behavioral policy.
* **Off-Policy Learning (Q-Learning):** You ride the bicycle (behavior policy, which might include exploration/mistakes), but you also watch a professional cyclist (target policy, which represents optimal behavior). When you evaluate your actions, you do not ask *"How well did my actual next exploratory step work?"* Instead, you ask: *"What would have happened if I had executed the absolute best possible action at that point, like the expert would have?"*

In off-policy learning, we have two distinct policies:
1. **Behavior Policy ($\mu$):** The policy used to generate behavior and select actions (often exploratory, such as $\epsilon$-greedy).
2. **Target Policy ($\pi$):** The policy that is being evaluated and improved (typically the optimal, purely greedy policy).

### 5.2 The Q-Learning Update Formula
In Q-learning, the update target is calculated using the maximum possible action-value of the next state $S_{t+1}$, rather than the action-value of the actual action $A_{t+1}$ chosen by the behavior policy.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

### 5.3 Comparative Analysis of Updates

| Component | SARSA (On-Policy) | Q-Learning (Off-Policy) |
| :--- | :--- | :--- |
| **Next Action Selection** | Chosen from $S_{t+1}$ using the behavior policy (e.g., $\epsilon$-greedy). | Evaluated using the greedy choice $\max_a Q(S_{t+1}, a)$. |
| **TD Target** | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ | $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ |
| **Philosophy** | Evaluates the actual actions taken, including exploration errors. | Evaluates optimal behavior, assuming no future exploration errors are made. |

---

## 6. Practical Case Study: Cliff Walking

The differences between SARSA and Q-Learning are demonstrated in the classic benchmark problem: **Cliff Walking**.

```
   -------------------------------------------------
  |   |   |   |   |   |   |   |   |   |   |   |   |
   -------------------------------------------------
  |   |   |   |   |   |   |   |   |   |   |   |   |
   -------------------------------------------------
  |   |   |   |   |   |   |   |   |   |   |   |   |
   -------------------------------------------------
  | S | x | x | x | x | x | x | x | x | x | x | G |  <-- Bottom Row
   -------------------------------------------------
    S: Start, G: Goal, x: The Cliff
```

### 6.1 Environment Rules
* **Grid:** $4 \times 12$ gridworld.
* **Start State ($S$):** Bottom-left corner.
* **Goal State ($G$):** Bottom-right corner.
* **The Cliff ($x$):** The cells on the bottom row between $S$ and $G$.
* **Actions:** Up, Down, Left, Right.
* **Reward Structure:**
  * $R = -1$ for each step that does not land on the cliff.
  * $R = -100$ if the agent steps into the cliff. The agent is also immediately reset to the start state $S$.

### 6.2 Path Choices: Safe Path vs. Optimal Path
There are two primary pathways to reach the goal:
1. **The Optimal Path (The Edge):** Walking directly along the row immediately above the cliff. This path is the shortest distance (13 steps), yielding a total episodic reward of $-13$.
2. **The Safe Path:** Walking up, traveling along the top row of the grid, and then walking down to the goal. This path is longer (17 steps), yielding a total episodic reward of $-17$.

### 6.3 Behavior of SARSA vs. Q-Learning

```
  Safe Path (SARSA):     =======================>
                         ^                      |
                       [ S ] [ Cliff  Cliff ... ] [ G ]

  Optimal Path (Q):    =========================>
                       [ S ] [ Cliff  Cliff ... ] [ G ]
```

#### Why does SARSA choose the Safe Path?
SARSA is on-policy and evaluates its current $\epsilon$-greedy behavior policy, which includes exploration errors. If the agent walks along the optimal edge path, there is always a probability $\epsilon$ that it will randomly select a "Down" action and fall into the cliff, incurring a massive penalty of $-100$. 

Because SARSA factors these exploration errors into its expected values, the edge path is evaluated as having a low expected return. As a result, SARSA chooses the longer, safer path, where an unexpected exploratory step will not result in a fall off the cliff.

#### Why does Q-Learning choose the Optimal Path?
Q-Learning is off-policy. It behaves explorationally (using $\epsilon$-greedy to select actions) but evaluates its target policy assuming it will act purely greedily ($\max_a Q(S_{t+1}, a)$) in the future. It learns the values of the optimal policy, which never makes exploratory errors. 

Because Q-learning assumes it will always make the perfect, greedy choice along the edge, it evaluates the optimal path as superior to the safe path. During training, Q-learning actually falls off the cliff more frequently due to exploration, but it successfully converges to the optimal edge path.

---

## 7. Looking Forward: Tabular Methods vs. Function Approximation

All methods covered up to this point (Dynamic Programming, Monte Carlo, and Tabular Temporal Difference) are classified as **tabular methods**.

### 7.1 Limitations of Tabular Methods
In tabular reinforcement learning, we maintain a table with a distinct entry for every state $V(s)$ or state-action pair $Q(s, a)$. While mathematically convenient, this approach does not scale to complex problems:
* **Memory Constraints:** Games like Chess have approximately $10^{120}$ states. Go has approximately $10^{170}$ states. It is physically impossible to store tables of this size in memory.
* **Continuous Spaces:** Real-world applications (such as robotics, autonomous driving, and physical control) operate in continuous state and action spaces (e.g., exact coordinates, velocities, joint angles). Continuous spaces contain an infinite number of states, making tabular representations impossible.
* **Lack of Generalization:** Tabular methods cannot generalize. If the agent encounters a state that is slightly different from any state it has visited before, it has no estimate for its value and must learn it entirely from scratch.

### 7.2 Value Function Approximation
To solve these scale limitations, we must transition from tabular methods to **Function Approximation**. Instead of storing values in a table, we represent the value function as a parameterized function:

$$\hat{v}(s, \mathbf{w}) \approx v_\pi(s) \quad \text{or} \quad \hat{q}(s, a, \mathbf{w}) \approx q_\pi(s, a)$$

Where $\mathbf{w}$ is a weight vector of much lower dimensionality than the state space. This function can be:
* A linear combination of state features.
* A deep neural network (acting as a **Universal Function Approximator**).

```
   Tabular representation:
   State S1  --->  [ Value = 100 ]
   State S2  --->  [ Value = 200 ]
   
   Function approximation:
   State Features (S)  --->  [ Neural Network (weights w) ]  --->  Estimated Value V(S, w)
```

By adjusting the weights $\mathbf{w}$ using gradient-based optimization, a single update generalizes to other similar states. This forms the foundation for **Deep Reinforcement Learning (DRL)**, enabling algorithms to solve high-dimensional, continuous control problems.