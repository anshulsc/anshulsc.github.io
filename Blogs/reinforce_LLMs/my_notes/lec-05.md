# Lecture Notes: Dynamic Programming in Reinforcement Learning

---

## 1. Introduction to Dynamic Programming (DP)

In Reinforcement Learning (RL), **Dynamic Programming (DP)** refers to a collection of algorithms used to compute optimal policies given a perfect model of the environment represented as a **Markov Decision Process (MDP)**. 

### Key Characteristics of Dynamic Programming:
* **Model-Based:** DP assumes that the transition probabilities and reward dynamics of the environment are fully known.
* **Core Idea:** The core utility of DP in RL is to use value functions to structure and organize the search for good policies.
* **Primary Task:** Once we can compute the value functions satisfying the Bellman expectation or optimality equations, we can easily obtain optimal policies.

---

## 2. Policy Evaluation (The Prediction Problem)

**Policy Evaluation** is the process of computing the state-value function $v_\pi(s)$ for an arbitrary policy $\pi$. This is also referred to as the **prediction problem**.

### 2.1 The Bellman Expectation Equation

For any state $s \in \mathcal{S}$, the value of $s$ under a policy $\pi$ is defined as the expected return starting from $s$ and following $\pi$ thereafter:

$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v_\pi(s') \right]$$

#### Component-by-Component Explanation:
* $v_\pi(s)$: The expected long-term return starting from state $s$ under policy $\pi$.
* $\pi(a|s)$: The probability of taking action $a$ in state $s$ under policy $\pi$.
* $p(s', r | s, a)$: The transition probability of reaching state $s'$ with reward $r$, given that action $a$ was taken in state $s$.
* $r + \gamma v_\pi(s')$: The immediate reward $r$ plus the discounted value ($\gamma$ is the discount factor, $0 \le \gamma < 1$) of the next state $s'$.

---

### 2.2 System of Linear Equations

If the state space $\mathcal{S}$ is finite (containing $n$ states), the Bellman expectation equation represents a system of $n$ simultaneous linear equations in $n$ unknowns (the values $v_\pi(s)$ for each $s \in \mathcal{S}$). 

In principle, this system can be solved analytically using matrix inversion:

$$\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi$$

However, direct analytical computation is computationally expensive for large state spaces because matrix inversion has a complexity of $\mathcal{O}(n^3)$. 

---

### 2.3 Iterative Policy Evaluation

To overcome the computational limits of direct analytical methods, we use an iterative approach. We construct a sequence of value approximations $v_0, v_1, v_2, \dots$, where each approximation maps $\mathcal{S}$ to $\mathbb{R}$.

1. **Initialization:** Start with an initial approximation $v_0(s) = 0$ (or any arbitrary value) for all states $s \in \mathcal{S}$.
2. **Update Rule (Successive Approximation):** Use the Bellman expectation equation as an update rule to calculate the next approximation $v_{k+1}$ from the current approximation $v_k$:

$$v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v_k(s') \right]$$

Under the discount factor $\gamma < 1$ or guaranteed termination, the sequence $\{v_k\}$ is guaranteed to converge to the true value function $v_\pi$ as $k \to \infty$.

#### Intuitive Example (5-State System)
Consider an MDP with 5 states: $s_1, s_2, s_3, s_4, s_5$.

Using the backup diagrams shown in the lecture:
* **State $s_1$:** From $s_1$, the agent can take action $a_1$ (leading to $s_2$) or action $a_2$ (leading to $s_3$). Thus, the value update for $s_1$ depends recursively on the values of $s_2$ and $s_3$:
  $$v_{k+1}(s_1) = f_1\left(v_k(s_2), v_k(s_3)\right)$$
* **State $s_2$:** From $s_2$, the transition leads back to $s_1$ or to $s_4$:
  $$v_{k+1}(s_2) = f_2\left(v_k(s_1), v_k(s_4)\right)$$

By recursively applying these updates across all states in each iteration (a process known as a **sweep**), the state values gradually stabilize to their true values.

---

## 3. Policy Improvement

Once we have evaluated a policy $\pi$ and computed its value function $v_\pi$, we want to determine if we can make the policy better. 

### 3.1 Evaluating an Alternative Action

For a given state $s$, we want to know if it is better to select an action $a \neq \pi(s)$. We can compute the expected return of taking action $a$ in state $s$, and thereafter following the existing policy $\pi$:

$$q_\pi(s, a) = \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v_\pi(s') \right]$$

* If $q_\pi(s, a) > v_\pi(s)$, it means selecting action $a$ once and then following $\pi$ yields a higher expected return than choosing $\pi(s)$ immediately.
* If this inequality holds, we can improve our policy by updating it to choose action $a$ in state $s$.

---

### 3.2 The Policy Improvement Theorem

The **Policy Improvement Theorem** states that if we update the policy greedily by choosing the action that maximizes the action-value function $q_\pi(s, a)$ at each state, the new policy $\pi'$ is guaranteed to be as good as or better than the old policy $\pi$:

$$v_{\pi'}(s) \ge v_\pi(s) \quad \forall s \in \mathcal{S}$$

The updated greedy policy $\pi'$ is defined as:

$$\pi'(s) = \operatorname{argmax}_{a \in \mathcal{A}} q_\pi(s, a)$$

$$\pi'(s) = \operatorname{argmax}_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v_\pi(s') \right]$$

---

## 4. Policy Iteration

By alternating between **Policy Evaluation** (computing the value function of a policy) and **Policy Improvement** (generating a better policy based on the computed value function), we can find the optimal policy. This process is called **Policy Iteration**.

### The Policy Iteration Loop:

$$\pi_0 \xrightarrow{\text{Evaluation}} v_{\pi_0} \xrightarrow{\text{Improvement}} \pi_1 \xrightarrow{\text{Evaluation}} v_{\pi_1} \dots \xrightarrow{\text{Improvement}} \pi_* \xrightarrow{\text{Evaluation}} v_*$$

```
   +---------------------------------------+
   |                                       |
   |           Policy Evaluation           |  <-- Compute v_pi from current pi
   |  (Solve Bellman Expectation Eq.)      |      using iterative sweeps.
   |                                       |
   +-------------------+-------------------+
                       |
                       | v_pi
                       v
   +-------------------+-------------------+
   |                                       |
   |          Policy Improvement           |  <-- Create greedy policy pi'
   |     (Greedy policy update)            |      using argmax of q_pi(s,a).
   |                                       |
   +-------------------+-------------------+
                       |
                       | New policy pi'
                       +---- If pi' == pi, stop (Optimal Policy found!)
                             Otherwise, set pi = pi' and repeat loop.
```

Because a finite MDP has only a finite number of policies, this process is guaranteed to converge to an optimal policy $\pi_*$ and optimal value function $v_*$ in a finite number of iterations.

---

## 5. Practical Example: Car Rental Problem

To demonstrate how policy iteration works, the lecture uses a car rental scenario (adapted from the classic "Jack's Car Rental" problem).

### 5.1 Problem Setup
* **State:** The number of cars at each of two locations at the end of the day.
  * Location 1: Max 20 cars.
  * Location 2: Max 20 cars.
  * Total state space size: $21 \times 21 = 441$ states.
* **Actions:** Moving up to 5 cars between the two locations overnight.
  * Positive action: Move cars from Location 1 to Location 2.
  * Negative action: Move cars from Location 2 to Location 1.
* **Reward:** 
  * $+10$ for each car rented to a customer.
  * $-2$ for each car moved overnight between locations.
* **Transition Dynamics:** Customer requests and car returns at each location follow Poisson distributions:
  * Location 1: Requests ($\lambda_{\text{rent}} = 3$), Returns ($\lambda_{\text{return}} = 3$)
  * Location 2: Requests ($\lambda_{\text{rent}} = 4$), Returns ($\lambda_{\text{return}} = 2$)

### 5.2 Poisson Distribution Recap
The probability of $x$ events occurring in a fixed interval of time, given an expected rate $\lambda$, is:

$$P(x) = \frac{\lambda^x e^{-\lambda}}{x!}$$

### 5.3 Implementation and Results
The Python script shown in the Google Colab environment implements policy iteration to solve this problem:

1. **Initialization:** The state-value array is initialized to zeros, and the starting policy is to move 0 cars.
2. **Policy Evaluation:** The code calculates the expected returns for renting and moving cars, taking into account the probability distributions of customer requests and returns.
3. **Policy Improvement:** The code updates the policy greedily based on the expected returns of moving cars.
4. **Convergence:** The policy converges when the greedy action choices stop changing from one iteration to the next.

#### Optimal Policy Visualization:
The resulting plot shows the optimal decision policy based on the state of the two locations:

* **High car count at Location 1, low at Location 2:** The optimal policy moves cars from Location 1 to Location 2 (represented by positive values / yellow-green regions).
* **Low car count at Location 1, high at Location 2:** The optimal policy moves cars from Location 2 to Location 1 (represented by negative values / dark blue regions).
* **Balanced states:** The optimal policy is to move 0 cars (represented by the medium-green region).

```
   Cars at Location 1
     ^
  20 | [ +5 ][ +5 ][ +4 ][ +3 ][ +2 ][  0 ][  0 ][  0 ]  <-- Yellow/Green: Move cars 
     | [ +5 ][ +4 ][ +3 ][ +2 ][ +1 ][  0 ][  0 ][ -1 ]      from Loc 1 to Loc 2.
     | [ +4 ][ +3 ][ +2 ][ +1 ][  0 ][  0 ][ -1 ][ -2 ]
     | [  0 ][  0 ][  0 ][  0 ][  0 ][  0 ][ -3 ][ -4 ]  <-- Medium Green: Move 0 cars.
     | [  0 ][  0 ][  0 ][  0 ][  0 ][ -2 ][ -4 ][ -5 ]
   0 | [  0 ][  0 ][  0 ][ -1 ][ -2 ][ -3 ][ -5 ][ -5 ]  <-- Dark Blue: Move cars 
     +------------------------------------------------>      from Loc 2 to Loc 1.
     0                                               20
                                    Cars at Location 2
```

---

## 6. Value Iteration

One drawback of Policy Iteration is that each of its iterations involves policy evaluation, which itself requires multiple sweeps through the state space to converge.

### 6.1 The Update Rule
**Value Iteration** simplifies this process by combining policy evaluation and policy improvement into a single update. It updates the state values by maximizing over all actions in a single sweep, without waiting for the policy to fully converge:

$$v_{k+1}(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v_k(s') \right]$$

This update is derived directly from the **Bellman Optimality Equation**:

$$v_*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v_*(s') \right]$$

---

## 7. Drawbacks of Dynamic Programming Algorithms

While mathematically sound, DP has two main limitations in practical applications:

### 1. Assumption of a Perfect Model
DP requires complete knowledge of the environment's transition probabilities $p(s', r | s, a)$ and reward dynamics. For many real-world problems (such as controlling a rover on Mars or navigating complex physical terrain), an accurate mathematical model is difficult to obtain. Instead, agents must learn from experience.

### 2. Computational Complexity (The Curse of Dimensionality)
DP algorithms require sweeping through the entire state space at each iteration. If the state space is very large (e.g., backgammon has over $10^{20}$ states), performing these full sweeps becomes computationally impractical. Subsequent methods, such as Monte Carlo and Temporal Difference learning, address these limitations by learning from sampled experience.