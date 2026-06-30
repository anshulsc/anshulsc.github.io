# Lecture Notes: Approximate Solution Methods & Value Function Approximation

---

## 1. Introduction and Recap of Reinforcement Learning Methods

In the study of reinforcement learning (RL) within the context of the course *Reasoning LLMs from Scratch*, the primary objective is to estimate value functions to discover an optimal policy for an agent. Over previous sessions, this has been approached using three classical frameworks, often referred to as the "Trio of Methods":

1. **Dynamic Programming (DP):** Computes exact value functions by solving the Bellman equations. However, it requires a complete and accurate model of the environment, specifically the transition probabilities $P(s' \mid s, a)$ and reward dynamics. In most real-world applications, this transition model is unknown.
2. **Monte Carlo (MC) Methods:** A model-free approach that learns directly from episodes of experience. It does not require transition probabilities but must wait until the end of an episode to compute the actual return $G_t$ before updating the value of any visited state.
3. **Temporal Difference (TD) Learning:** Combining elements of both DP and MC, TD methods are model-free (learning from experience like MC) and bootstrap their estimates (updating values based on other estimates, similar to DP). This allows updates to be performed at every time step without waiting for an episode to terminate.

---

## 2. The Limitations of Tabular Methods

Up to this point, value functions have been represented using **Tabular Methods**. In a tabular representation, the value function is maintained as a lookup table:

```
            State (s)     │    Value V(s)
         ─────────────────┼──────────────────
               s_1        │      V(s_1)
               s_2        │      V(s_2)
               ...        │       ...
               s_N        │      V(s_n)
```

Each state has an independent entry, and updating the estimate of one state has no direct mathematical influence on the estimate of any other state. 

### The Office-to-Home Commute Example
Consider a daily commute from an office to home, passing through 7 distinct junctions. The states are the junctions $J_1, J_2, \dots, J_7$, and the value of each state represents the estimated remaining travel time to reach home.

```
 [Office] ──► J1 ──► J2 ──► J3 ──► J4 ──► J5 ──► J6 ──► J7 ──► [Home]
```

* **Monte Carlo Update:** Suppose you experience an unusually heavy traffic delay at Junction 3 ($J_3$). Under a Monte Carlo approach, you cannot update your estimate for $J_3$ immediately. You must complete the entire commute, arrive home, calculate the total commute duration, and only then update the value entry for $J_3$. This delay in updating is highly inefficient.
* **Temporal Difference Update:** Under a TD approach, as soon as you transition from $J_3$ to $J_4$ and experience the delay, you can immediately update your estimate for $J_3$ using the step's elapsed time and your pre-existing estimate for $J_4$.

### The Scale Problem (Curse of Dimensionality)
While tabular methods work well for simple environments with small state spaces, they fail to scale to complex, real-world problems.

* **Example: The Game of Chess**
  The state space for chess consists of all legal board configurations. This is estimated to be approximately:
  $$\lvert \mathcal{S} \rvert \approx 10^{46} \text{ states}$$
  To store a tabular value function for chess, a computer would require a table with $10^{46}$ entries. This is physically impossible due to memory limitations. Furthermore, because tabular methods require visiting each state to learn its value, an agent would need to play an astronomical number of games to populate the table.

---

## 3. Transitioning to Function Approximation & Generalization

To handle large or continuous state spaces, we must transition from tabular lookup tables to **Function Approximation**. The core idea is to represent the value function not as a table, but as a parameterized functional mapping:

$$\hat{v}(s, w) \approx v_\pi(s)$$

where $w \in \mathbb{R}^d$ is a vector of adjustible weights (parameters) that is much smaller than the number of states ($d \ll \lvert \mathcal{S} \rvert$). 

### The Concept of Generalization
Function approximation enables **generalization**: when an update is made to the weight vector $w$ based on experience at a single state $s$, the estimated values of many other states change as well. 

```
   Tabular representation:
   [ Update State s_i ]  ──►  Only Entry V(s_i) changes.

   Function Approximation:
   [ Update Weights w ]   ──►  Entire function surface shift; 
                               unseen states s_j also get updated.
```

If an agent has visited only 25 out of 50 possible states, a well-designed function approximator can generalize from those 25 visited states to produce reasonable value estimates for the remaining 25 unvisited states.

### Supervised Learning Analogies

#### 1. Linear Regression (Continuous Estimation)
Consider predicting a student's exam score based on study hours. Suppose we collect three data points:
* 2 hours $\rightarrow$ 20%
* 4 hours $\rightarrow$ 40%
* 8 hours $\rightarrow$ 60%

Instead of keeping a lookup table for every possible study duration, we fit a continuous function (e.g., a curve or line of best fit) passing near these points.

```
  Score (%)
     ▲
  60 │                  ● (8 hrs, 60%)
     │               /
  40 │            ● (4 hrs, 40%)
     │         ./
  20 │      ● (2 hrs, 20%)
     │     /
     └──────────────────────────────► Hours
            2     4  5     8
```

If we want to estimate the score for $5$ hours of study (a value not present in our training data), we evaluate our fitted curve at $x = 5$, obtaining an approximation of $\sim 45\%$. This is function approximation via regression.

#### 2. Image Classification (Feature Generalization)
In an MRI-based brain tumor classification task, a supervised learning model is trained on thousands of labeled images categorized as *Benign* or *Malignant*. When presented with a completely new MRI scan from a new patient, the model does not look for an exact image match in a database. Instead, it extracts generalized features learned during training to make a prediction.

---

## 4. Key Differences: Reinforcement Learning vs. Supervised Learning

Although both frameworks use function approximation, applying supervised learning techniques directly to Reinforcement Learning introduces two distinct challenges:

1. **Delayed and Unknown Targets:** In supervised learning, the target labels $y$ (e.g., the true diagnostic class or the true exam score) are known and fixed in the dataset beforehand. In RL, we do not know the true value function $v_\pi(s)$ ahead of time. The targets must be estimated dynamically from experience, which often depends on future rewards that are delayed.
2. **Non-Stationarity:** The data distribution in RL is non-stationary. As the agent interacts with the environment and updates its policy, the states it visits and the rewards it receives change. Thus, the target value function is constantly shifting during training (e.g., similar to a self-driving car adapting to changing traffic conditions over time).

---

## 5. Mathematical Formulation of Value Function Approximation

We define our approximate state-value function as:

$$\hat{v}(s, w)$$

where:
* $s \in \mathcal{S}$ is the state.
* $w = [w_1, w_2, \dots, w_d]^T \in \mathbb{R}^d$ is the parameter weight vector.
* $\hat{v}$ can be a linear function of features, a decision tree, or a multi-layer deep neural network.

### The Mean Squared Value Error ($\overline{VE}$)
To quantify how well our parameterized function $\hat{v}(s, w)$ approximates the true value function $v_\pi(s)$ across the state space, we define an objective loss function called the **Mean Squared Value Error (MSVE)**, denoted as $\overline{VE}(w)$:

$$\overline{VE}(w) = \sum_{s \in \mathcal{S}} \mu(s) \left[ v_\pi(s) - \hat{v}(s, w) \right]^2$$

Where:
* $v_\pi(s)$ is the true state-value under policy $\pi$.
* $\hat{v}(s, w)$ is the estimated state-value parameterized by weights $w$.
* $\mu(s) \ge 0$ is the **state distribution**, representing the relative importance of each state. Typically, $\sum_{s \in \mathcal{S}} \mu(s) = 1$.

### Intuition of the State Distribution $\mu(s)$
We cannot approximate all states with equal accuracy due to the limited capacity of our parameter vector $w$ ($d \ll \lvert \mathcal{S} \rvert$). The state distribution $\mu(s)$ acts as a weighting factor:
* If state $s$ is visited frequently by the agent, $\mu(s)$ is high. The loss function penalizes errors at state $s$ heavily, forcing the function approximator to prioritize accuracy for this state.
* If state $s$ is rarely or never visited, $\mu(s)$ is low. We accept larger approximation errors at $s$ to preserve model capacity for more frequent states.

```
       State Distribution μ(s)
           ▲
           │          _.._
           │        .'    `.     ◄── State 0 is visited most frequently
           │       /        \
           │      |          |
           │    _.'          `._
         ──┴───┴──────┼──────┴───┴──► State s
              -1      0      1
```

---

## 6. Step-by-Step Mathematical Derivation of Gradient Descent Updates

Our goal is to find a weight vector $w$ that minimizes the objective function:

$$\overline{VE}(w) = \sum_{s \in \mathcal{S}} \mu(s) \left[ v_\pi(s) - \hat{v}(s, w) \right]^2$$

Since the true state-value $v_\pi(s)$ is unknown, we must perform updates based on individual experienced states. Let $S_t$ be the state observed at time step $t$, and let $U_t$ be a sample or estimate of its true value (acting as our target). We define a localized squared error loss function for a single step:

$$J(w) = \frac{1}{2} \left[ U_t - \hat{v}(S_t, w) \right]^2$$

*(The factor of $\frac{1}{2}$ is introduced to simplify the arithmetic during differentiation).*

### General Gradient Descent Update Rule
To update our weights $w$ in the direction that minimizes $J(w)$, we compute the gradient with respect to $w$ and adjust the weights in the opposite direction:

$$w_{t+1} = w_t - \alpha \nabla_w J(w_t)$$

where $\alpha > 0$ is the step-size parameter (learning rate).

### Step-by-Step Derivation of the Gradient
We apply the chain rule of calculus to compute $\nabla_w J(w)$:

1. **Write down the function:**
   $$J(w) = \frac{1}{2} f(w)^2 \quad \text{where} \quad f(w) = U_t - \hat{v}(S_t, w)$$

2. **Differentiate with respect to the weight vector $w$:**
   $$\nabla_w J(w) = \frac{1}{2} \cdot 2 \cdot f(w) \cdot \nabla_w f(w)$$
   $$\nabla_w J(w) = \left[ U_t - \hat{v}(S_t, w) \right] \cdot \nabla_w \left[ U_t - \hat{v}(S_t, w) \right]$$

3. **Distribute the gradient operator:**
   $$\nabla_w J(w) = \left[ U_t - \hat{v}(S_t, w) \right] \cdot \left[ \nabla_w U_t - \nabla_w \hat{v}(S_t, w) \right]$$

Depending on the nature of our target $U_t$, we evaluate $\nabla_w U_t$ using one of two methods:

---

### Case A: True Gradient Update (Monte Carlo Target)
In Monte Carlo learning, the target is the actual return $G_t$ observed from time step $t$ until the end of the episode:

$$U_t = G_t$$

Because $G_t$ is a sample of actual rewards obtained from the environment, it is completely independent of the current parameter weights $w$. Thus, its gradient with respect to $w$ is zero:

$$\nabla_w U_t = \nabla_w G_t = 0$$

Substituting this back into our gradient equation:

$$\nabla_w J(w) = \left[ G_t - \hat{v}(S_t, w) \right] \cdot \left[ 0 - \nabla_w \hat{v}(S_t, w) \right]$$
$$\nabla_w J(w) = - \left[ G_t - \hat{v}(S_t, w) \right] \nabla_w \hat{v}(S_t, w)$$

Substituting this gradient into our weight update rule:

$$w_{t+1} = w_t - \alpha \nabla_w J(w_t)$$
$$w_{t+1} = w_t - \alpha \left( - \left[ G_t - \hat{v}(S_t, w_t) \right] \nabla_w \hat{v}(S_t, w_t) \right)$$
$$w_{t+1} = w_t + \alpha \left[ G_t - \hat{v}(S_t, w_t) \right] \nabla_w \hat{v}(S_t, w_t)$$

This is the exact **gradient-descent value function update rule** for Monte Carlo targets.

---

### Case B: Semi-Gradient Update (Temporal Difference Target)
In one-step Temporal Difference learning (TD(0)), our target is the bootstrapped estimate:

$$U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t)$$

Unlike the Monte Carlo return, this target is an estimate that *does* depend on the weight parameter vector $w$. A mathematically rigorous gradient descent would require us to compute the derivative of the target itself:

$$\nabla_w U_t = \nabla_w \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, w) \right] = \gamma \nabla_w \hat{v}(S_{t+1}, w)$$

Computing this gradient is computationally demanding and can lead to instability under certain conditions. 

To simplify the process, we use **Semi-Gradient Methods**. In a semi-gradient update, we treat the target estimate $U_t$ as a constant with respect to $w$ during differentiation, ignoring its dependence on the weights. Therefore, we set:

$$\nabla_w U_t \approx 0$$

Using this assumption, the update formula remains identical in structure to the gradient update but uses the TD target:

$$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t) \right] \nabla_w \hat{v}(S_t, w_t)$$

#### Intuitive Comparison of Update Terms
* **Monte Carlo Target ($G_t$):** An unbiased but high-variance estimate of $v_\pi(S_t)$. Since it does not bootstrap, we can perform true gradient descent.
* **TD Target ($R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t)$):** A biased but low-variance estimate. Because it bootstraps, we rely on semi-gradient updates.

---

## 7. Visualizing Weight Updates and Error Surfaces

To gain intuition on how gradient descent works, we can visualize the MSVE loss landscape over a two-dimensional weight space consisting of $w_1$ and $w_2$.

### 2D Loss Surface with Weight Optimization Path

```
  w_2
   ▲
2.0│  [Green Zone: High Loss]
   │    \   
1.5│     ● (Start: w_0)
   │      \  
1.0│       \───►  [Path of Gradient Descent Steps]
   │            \
0.5│             `───► ● (w_min) ───► [Purple Zone: Minimal Loss]
   │
0.0└──────────────────────────────────► w_1
  -2.0  -1.5  -1.0  -0.5   0.0   0.5   1.0
```

* **Color Gradient:** The color transitions from light green (representing high error/loss) to dark purple/blue (representing low error/loss).
* **Optimization Path:** Starting at an arbitrary initial weight configuration $w_0 = [w_1, w_2]^T$ (marked with a green dot), the gradient descent algorithm calculates the direction of steepest descent at each step.
* **Convergence:** The weights are updated iteratively, tracing a path perpendicular to the error contours, eventually converging toward $w_{\min}$ (marked with a red star), where the approximation error is minimized.