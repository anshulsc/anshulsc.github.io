# Lecture Notes: Approximate Solution Methods & Deep Reinforcement Learning

---

## 1. Introduction & Motivation: Moving Beyond Tabular Methods

In reinforcement learning, **tabular methods** represent the value function ($V(s)$ or $Q(s, a)$) using a lookup table, where every state or state-action pair has a distinct entry. 

### 1.1 The Curse of Dimensionality
Tabular representation becomes impractical when dealing with large state spaces due to:
* **Memory constraints:** Storing $10^{46}$ states (such as in the game of Chess) or infinite states (in continuous control) is physically impossible.
* **Data scarcity:** An agent cannot visit every state often enough to compute accurate lookup estimates.

### 1.2 Generalization
To scale reinforcement learning to complex real-world environments, we must use **Function Approximation**. Instead of storing individual values, we represent the value function using a parameterized functional form:

$$\hat{v}(s, w) \approx v_{\pi}(s)$$

where $w \in \mathbb{R}^d$ is a weight vector of much lower dimensionality than the state space ($d \ll |S|$). Function approximation allows the agent to generalize: learning from a small subset of visited states generalizes to predict the values of unvisited, similar states.

---

## 2. Reinforcement Learning vs. Supervised Learning

Function approximation in reinforcement learning resembles supervised regression, but there are distinct differences:

| Metric | Supervised Learning | Reinforcement Learning |
| :--- | :--- | :--- |
| **Target Availability** | Known beforehand (labeled data) | Unknown; must be estimated online via experience |
| **Data Stationarity** | Stationary distribution | Non-stationary distribution (the target changes as the policy and weight vector $w$ update) |

---

## 3. Mathematical Foundations of Parameter Updates

To train the parameter vector $w$, we define a loss function that measures the discrepancy between our estimate $\hat{v}(s, w)$ and the true value $v_{\pi}(s)$.

### 3.1 Loss Function
We minimize the Mean Squared Value Error (MSVE) over the state distribution $\mu(s)$:

$$J(w) = \sum_{s \in S} \mu(s) \left[ v_{\pi}(s) - \hat{v}(s, w) \right]^2$$

To update the parameters iteratively, we calculate the gradient of the loss for a single observed state $s_t$:

$$J_t(w) = \frac{1}{2} \left[ v_{\pi}(s_t) - \hat{v}(s_t, w) \right]^2$$

### 3.2 Gradient Descent Derivation
We update the weights in the direction of the steepest descent:

$$w_{t+1} = w_t - \alpha \nabla_w J_t(w_t)$$

where $\alpha$ is the step-size parameter (learning rate). 

Taking the gradient of $J_t(w)$ with respect to $w$:

$$\nabla_w J_t(w) = \nabla_w \left( \frac{1}{2} \left[ v_{\pi}(s_t) - \hat{v}(s_t, w) \right]^2 \right)$$

Applying the chain rule:

$$\nabla_w J_t(w) = \left[ v_{\pi}(s_t) - \hat{v}(s_t, w) \right] \cdot \nabla_w \left( v_{\pi}(s_t) - \hat{v}(s_t, w) \right)$$

Since $v_{\pi}(s_t)$ is independent of our weight parameters $w$:

$$\nabla_w \left( v_{\pi}(s_t) - \hat{v}(s_t, w) \right) = 0 - \nabla_w \hat{v}(s_t, w)$$

Substituting this back:

$$\nabla_w J_t(w) = - \left[ v_{\pi}(s_t) - \hat{v}(s_t, w) \right] \nabla_w \hat{v}(s_t, w)$$

Substituting this gradient into our update rule yields:

$$w_{t+1} = w_t + \alpha \left[ v_{\pi}(s_t) - \hat{v}(s_t, w_t) \right] \nabla_w \hat{v}(s_t, w_t)$$

---

## 4. Estimating the Target: True Gradient vs. Semi-Gradient

Because the true value $v_{\pi}(s_t)$ is unknown, we substitute it with an approximation target $U_t$.

$$\mathbb{E}[U_t \mid S_t = s] \approx v_{\pi}(s)$$

### 4.1 Monte Carlo (MC) Target
In Monte Carlo methods, the target $U_t$ is the actual return $G_t$:

$$U_t = G_t$$

Since the return $G_t$ is observed at the end of the episode, it does not depend on the current weight parameter $w$. This yields a true gradient descent update:

$$w_{t+1} = w_t + \alpha \left[ G_t - \hat{v}(s_t, w_t) \right] \nabla_w \hat{v}(s_t, w_t)$$

### 4.2 Temporal Difference (TD) Target
In Temporal Difference learning, we bootstrap our estimate using the one-step target:

$$U_t = R_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)$$

Here, the target $U_t$ depends on $w_t$. A true gradient update would require taking the derivative of the target itself:

$$\nabla_w \left( \frac{1}{2} \left[ R_{t+1} + \gamma \hat{v}(s_{t+1}, w) - \hat{v}(s_t, w) \right]^2 \right) = \left[ R_{t+1} + \gamma \hat{v}(s_{t+1}, w) - \hat{v}(s_t, w) \right] \left( \gamma \nabla_w \hat{v}(s_{t+1}, w) - \nabla_w \hat{v}(s_t, w) \right)$$

### 4.3 Semi-Gradient Methods
Computing the full gradient is complex and can lead to instability when bootstrapping. In practice, we ignore the target's dependence on $w_t$ and only take the gradient with respect to the estimated value of the current state $\hat{v}(s_t, w_t)$. This is called a **semi-gradient method**:

$$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t) \right] \nabla_w \hat{v}(s_t, w_t)$$

---

## 5. Value Function Representations

There are two primary categories of function approximation:

```
                  ┌───────────────────────────────┐
                  │ Function Approximation (v̂)    │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────┐                               ┌─────────────────┐
│ Linear Methods  │                               │  Non-Linear     │
│  v̂ = wᵀ x(s)    │                               │  (Neural Nets)  │
└─────────────────┘                               └─────────────────┘
```

### 5.1 Linear Methods
In linear function approximation, the value function is represented as the inner product of the weight vector $w$ and a feature vector $x(s)$:

$$\hat{v}(s, w) = w^T x(s) = \sum_{i=1}^d w_i x_i(s)$$

#### Step-by-Step Derivation of the Linear Gradient
For a two-dimensional case:

$$w = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}, \quad x(s) = \begin{bmatrix} x_1(s) \\ x_2(s) \end{bmatrix}$$

$$\hat{v}(s, w) = w_1 x_1(s) + w_2 x_2(s)$$

We calculate the partial derivatives with respect to each parameter:

$$\frac{\partial \hat{v}(s, w)}{\partial w_1} = x_1(s)$$

$$\frac{\partial \hat{v}(s, w)}{\partial w_2} = x_2(s)$$

Thus, the gradient vector is:

$$\nabla_w \hat{v}(s, w) = \begin{bmatrix} \frac{\partial \hat{v}}{\partial w_1} \\ \frac{\partial \hat{v}}{\partial w_2} \end{bmatrix} = \begin{bmatrix} x_1(s) \\ x_2(s) \end{bmatrix} = x(s)$$

In linear approximation, the gradient of the value function is simply the feature vector itself:

$$\nabla_w \hat{v}(s, w) = x(s)$$

#### Linear Parameter Update Rules
* **Monte Carlo Linear Update:**

  $$w_{t+1} = w_t + \alpha \left[ G_t - \hat{v}(s_t, w_t) \right] x(s_t)$$

* **Temporal Difference (TD) Linear Update:**

  $$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t) \right] x(s_t)$$

*Note on features:* Choosing appropriate representations for $x(s)$ (such as coarse coding, tile coding, or radial basis functions) is critical for linear models to capture spatial structures or non-linear patterns.

---

### 5.2 Non-Linear Methods: Neural Networks & Deep RL
Linear approximations are limited to functions that are linear in the features. For highly complex, high-dimensional tasks, we use non-linear estimators like Artificial Neural Networks.

```
Input (States, s)       Hidden Layers (Representations)       Output (v̂(s, w))
      ○ ───────────────►        ○ ───────────────►                 ○
      ○ ───────────────►        ○ ───────────────►                 
      ○ ───────────────►        ○ ───────────────►                 
                     Connections contain weights (w)
```

In Deep RL:
* **Inputs:** Raw observations or states $s$ (e.g., board positions, pixel inputs).
* **Weights ($w$):** Network parameters (synaptic weights) optimized via backpropagation.
* **Output:** The approximate value function $\hat{v}(s, w)$ or state-action value function $\hat{q}(s, a, w)$.

This integration of neural network function approximation with reinforcement learning updates is the basis of **Deep Reinforcement Learning (Deep RL)**, as demonstrated by systems like AlphaGo.

---

## 6. Control with Function Approximation: Finding Optimal Policies

To transition from predicting value functions to finding optimal policies, we approximate the state-action value function $\hat{q}(s, a, w) \approx q_*(s, a)$.

### 6.1 Action Selection via the $\epsilon$-Greedy Policy
At each step, the agent balances exploration and exploitation:
* With probability $1 - \epsilon$, select the greedy action:

  $$a^* = \arg\max_a \hat{q}(s, a, w)$$

* With probability $\epsilon$, select a random action from all possible actions.

#### Numeric Example
Let there be four actions $\{a_1, a_2, a_3, a_4\}$ with current estimates:

$$\hat{q}(s, a_1, w) = 20, \quad \hat{q}(s, a_2, w) = 30, \quad \hat{q}(s, a_3, w) = 10, \quad \hat{q}(s, a_4, w) = 15$$

For an exploration rate $\epsilon = 0.05$:
* **95% of the time (Exploitation):** The agent selects the highest-value action: $a_2$.
* **5% of the time (Exploration):** The agent chooses randomly among $\{a_1, a_2, a_3, a_4\}$ with equal probability ($1.25\%$ per action).

---

## 7. Semi-Gradient SARSA Algorithm

By combining Temporal Difference target updates, action-value function approximation, and $\epsilon$-greedy action selection, we arrive at the **Semi-Gradient SARSA** algorithm.

### 7.1 State-Action Transition Cycle
The update uses the transition cycle of SARSA:

$$S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} \dots$$

```
   State S_t (Action A_t chosen)
                │
                ▼ (Environment Transition)
   Reward R_{t+1} & Next State S_{t+1}
                │
                ▼ (Choose Next Action A_{t+1} via ϵ-greedy)
   Update weights (w) based on the target error
```

### 7.2 Weight Update Formula

$$w_{t+1} = w_t + \alpha \left[ R_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)$$

---

### 7.3 Complete Pseudocode

```python
def semi_gradient_sarsa(env, alpha, gamma, epsilon, num_episodes):
    # Initialize weight vector w arbitrarily (or to zeros)
    w = initialize_weights()
    
    for episode in range(num_episodes):
        # Initialize state
        s = env.reset()
        
        # Choose action a from s using policy derived from q̂(s, ·, w) (e.g., ϵ-greedy)
        a = epsilon_greedy_action(s, w, epsilon)
        
        while not done:
            # Take action a, observe reward r and next state s_prime
            s_prime, r, done = env.step(a)
            
            if done:
                # Terminal update: no future state-action value
                target = r
                w += alpha * (target - q_hat(s, a, w)) * gradient_q_hat(s, a, w)
                break
            
            # Choose action a_prime from s_prime using policy derived from q̂(s_prime, ·, w)
            a_prime = epsilon_greedy_action(s_prime, w, epsilon)
            
            # Compute temporal difference target
            target = r + gamma * q_hat(s_prime, a_prime, w)
            
            # Update weights using semi-gradient TD step
            w += alpha * (target - q_hat(s, a, w)) * gradient_q_hat(s, a, w)
            
            # Transition variables
            s = s_prime
            a = a_prime
            
    return w
```