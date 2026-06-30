# Lecture Notes: Policy Gradient Methods

---

## 1. Introduction & Recap of the Policy Gradient Theorem

In policy-based reinforcement learning, the objective is to optimize a policy $\pi(a|s, \theta)$ parameterized by $\theta \in \mathbb{R}^d$ to maximize a performance measure $J(\theta)$ (such as the expected return). To do this, we can use gradient ascent to update the policy parameters:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)$$

The **Policy Gradient Theorem** provides an analytical expression for the gradient of the performance measure with respect to the policy parameter $\theta$ without requiring the derivative of the state distribution. The theorem states:

$$\nabla_\theta J(\theta) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} q_\pi(s, a) \nabla_\theta \pi(a|s, \theta)$$

Where:
*   $\mu(s)$ is the **state distribution** under the policy $\pi$ (the stationary distribution or the discounted visitor frequency of state $s$).
*   $q_\pi(s, a)$ is the true **action-value function** of the state-action pair $(s, a)$ under policy $\pi$.
*   $\nabla_\theta \pi(a|s, \theta)$ is the **gradient of the policy** with respect to its parameters $\theta$.

### Key Insight:
A remarkable feature of the Policy Gradient Theorem is that $\nabla_\theta J(\theta)$ does not involve the gradient of the state distribution ($\nabla_\theta \mu(s)$), which would otherwise be highly complex and impractical to calculate in model-free scenarios.

---

## 2. Deriving the REINFORCE Algorithm

The Policy Gradient Theorem involves an explicit sum over all states ($\sum_{s \in \mathcal{S}}$) and all actions ($\sum_{a \in \mathcal{A}}$). In practice, we cannot sum over the entire state space because the environment dynamics may be unknown, and the state space is often high-dimensional or continuous. We therefore transition from an analytical sum to an **expected value** so we can approximate the gradient using samples (Monte Carlo simulation).

### Mathematical Concept 1: Expected Value
For a discrete random variable $X$ with a probability mass function $P(x)$, the expected value of a function $f(X)$ is defined as:

$$\mathbb{E}[f(X)] = \sum_{x \in \mathcal{X}} P(x) f(x)$$

#### Die Roll Example:
If you roll a fair six-sided die, each face $x \in \{1, 2, 3, 4, 5, 6\}$ has a probability $P(x) = \frac{1}{6}$. The expected value $\mathbb{E}[X]$ is calculated as:

$$\mathbb{E}[X] = \sum_{x=1}^{6} P(x) \cdot x = \frac{1}{6}(1) + \frac{1}{6}(2) + \frac{1}{6}(3) + \frac{1}{6}(4) + \frac{1}{6}(5) + \frac{1}{6}(6)$$

$$\mathbb{E}[X] = \frac{1}{6}(21) = 3.5$$

---

### Step 1: Converting the State Summation to an Expectation

We can rewrite the state summation in the Policy Gradient Theorem as an expectation over states visited under the policy $\pi$. Since $\mu(s)$ is the probability distribution of states under policy $\pi$:

$$\sum_{s \in \mathcal{S}} \mu(s) \left[ \sum_{a \in \mathcal{A}} q_\pi(s, a) \nabla_\theta \pi(a|s, \theta) \right] = \mathbb{E}_\pi \left[ \sum_{a \in \mathcal{A}} q_\pi(S_t, a) \nabla_\theta \pi(a|S_t, \theta) \right]$$

Thus, the performance gradient can be written as:

$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi \left[ \sum_{a \in \mathcal{A}} q_\pi(S_t, a) \nabla_\theta \pi(a|S_t, \theta) \right]$$

---

### Step 2: Converting the Action Summation to an Expectation

To eliminate the explicit sum over actions ($\sum_{a \in \mathcal{A}}$), we employ a mathematical identity. We multiply and divide the terms inside the summation by the policy probability $\pi(a|S_t, \theta)$:

$$\sum_{a \in \mathcal{A}} q_\pi(S_t, a) \nabla_\theta \pi(a|S_t, \theta) = \sum_{a \in \mathcal{A}} \pi(a|S_t, \theta) \left( q_\pi(S_t, a) \frac{\nabla_\theta \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)} \right)$$

This expression is now in the form $\sum_{a} P(a) f(a)$, which represents an expectation over the actions selected by policy $\pi$:

$$\sum_{a \in \mathcal{A}} \pi(a|S_t, \theta) \left( q_\pi(S_t, a) \frac{\nabla_\theta \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)} \right) = \mathbb{E}_\pi \left[ q_\pi(S_t, A_t) \frac{\nabla_\theta \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right]$$

*Note: In alternative literature, the fractional term $\frac{\nabla_\theta \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}$ is frequently written as $\nabla_\theta \ln \pi(A_t|S_t, \theta)$ because of the derivative rule $\nabla \ln f(x) = \frac{\nabla f(x)}{f(x)}$.*

---

### Step 3: Combining Expectations and Sampling the Return

By combining the expectations over states and actions, we obtain:

$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi \left[ q_\pi(S_t, A_t) \frac{\nabla_\theta \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right]$$

The true action-value function $q_\pi(S_t, A_t)$ represents the expected cumulative return from state $S_t$ taking action $A_t$:

$$q_\pi(S_t, A_t) = \mathbb{E}_\pi [G_t | S_t, A_t]$$

Where $G_t$ is the actual cumulative discounted return obtained from time step $t$ onward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$$

Using the law of total expectation, we can substitute the true expectation $q_\pi(S_t, A_t)$ with the empirical sample return $G_t$:

$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi \left[ G_t \frac{\nabla_\theta \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right]$$

### The REINFORCE Parameter Update Rule
Dropping the expectation operator and performing stochastic gradient ascent using individual trajectory samples yields the parameter update rule for the **REINFORCE** algorithm:

$$\theta_{t+1} = \theta_t + \alpha G_t \frac{\nabla_\theta \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}$$

---

## 3. Physical & Geometric Intuition of REINFORCE

To understand how the REINFORCE algorithm adjusts policy parameters, we can analyze the update vector geometrically:

$$\Delta \theta = \alpha G_t \frac{\nabla_\theta \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}$$

```
                      \  New Policy Parameters (N)
                       \  
                        ^  Update Vector: α * G_t * (∇π / π)
                         \ 
                          \
   ------------------------o Old Policy Parameters (O)
```

1.  **Direction ($\nabla_\theta \pi(A_t|S_t, \theta)$)**:
    The gradient of the policy probability with respect to $\theta$ points in the direction in parameter space that *most rapidly increases* the probability of choosing action $A_t$ in state $S_t$.
2.  **Return Weighting ($G_t$)**:
    *   If the selected action results in a high positive return ($G_t \gg 0$), the parameters are shifted significantly in this direction, making the action much more likely to occur again in that state.
    *   If the return is near zero or negative, the update step is negligible or moves in the opposite direction.
3.  **The Normalization Term ($\frac{1}{\pi(A_t|S_t, \theta)}$)**:
    Without this division, actions that are selected frequently under the current policy would receive massive updates simply due to their occurrence rate, even if they yield mediocre returns. Dividing by $\pi(A_t|S_t, \theta)$ ensures that rare, highly beneficial actions receive a proportionally larger boost, while common actions are adjusted more conservatively.

---

## 4. REINFORCE with Baseline

While REINFORCE is unbiased, it is a Monte Carlo method that suffers from **high variance**. Because the return $G_t$ can vary significantly between episodes, the gradient updates can be noisy, slowing down convergence.

To address this issue, we can subtract an action-independent **baseline** $b(s)$ from the action-value estimate:

$$\nabla_\theta J(\theta) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} (q_\pi(s, a) - b(s)) \nabla_\theta \pi(a|s, \theta)$$

### Proof that subtracting an action-independent baseline does not introduce bias:
To ensure the baseline does not alter the expected gradient, the expected value of the baseline term must equal zero:

$$\sum_{a \in \mathcal{A}} b(s) \nabla_\theta \pi(a|s, \theta) = b(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s, \theta) = b(s) \nabla_\theta \sum_{a \in \mathcal{A}} \pi(a|s, \theta)$$

Since $\sum_{a \in \mathcal{A}} \pi(a|s, \theta) = 1$ (the policy is a valid probability distribution that sum to 1 over all actions):

$$b(s) \nabla_\theta (1) = 0$$

Because the expected value of the subtracted baseline is zero, it introduces no bias but can significantly reduce variance by centering the return values.

### Choosing the State-Value Function as the Baseline
A natural choice for the baseline $b(s)$ is the **state-value function** $\hat{v}(s, w)$, parameterized by weights $w$:

$$b(s) = \hat{v}(s, w)$$

This yields the update rule for **REINFORCE with Baseline**:

$$\theta_{t+1} = \theta_t + \alpha (G_t - \hat{v}(S_t, w)) \frac{\nabla_\theta \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}$$

---

## 5. Actor-Critic Methods

A lingering limitation of both REINFORCE and REINFORCE with Baseline is that they are **offline** methods: they require complete episodes to compute the cumulative return $G_t$. This means we cannot update the policy during continuing (infinite-horizon) tasks or long episodes.

### Transitioning to Temporal Difference (TD) Learning
To enable online, step-by-step updates, we replace the Monte Carlo return $G_t$ with a bootstrapped **Temporal Difference (TD) target**:

$$G_t \approx R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$$

Substituting this bootstrap target into the baseline-subtracted gradient update gives:

$$\theta_{t+1} = \theta_t + \alpha \left( R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w) \right) \frac{\nabla_\theta \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}$$

The term inside the brackets is the **TD Error** (often denoted as $\delta_t$):

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$$

This formulation is the foundation of **Actor-Critic Methods**:

```
                       +-------------------+
                       |    Environment    |
                       +-------------------+
                         | State      ^ Reward
                         v            |
                       +-------------------+
                       |      Critic       |-----> Updates Value weights (w)
                       |     (v(s, w))     |       using TD error (δ)
                       +-------------------+
                         | TD Error (δ)
                         v
                       +-------------------+
                       |       Actor       |-----> Updates Policy parameters (θ)
                       |     (π(a|s, θ))   |       using Actor-Critic rule
                       +-------------------+
```

### Component Roles:
1.  **The Actor (Policy $\pi$)**:
    *   Responsible for selecting actions.
    *   It is updated in the direction of the policy gradient, scaled by the critic's feedback ($\delta_t$).
2.  **The Critic (Value Function $\hat{v}$)**:
    *   Responsible for evaluating the actions selected by the actor.
    *   It calculates the TD error $\delta_t$ to determine whether the outcome of the actor's choice was better or worse than expected.
    *   The critic is updated concurrently using standard temporal-difference methods (e.g., TD(0)).

---

## 6. Summary Comparison of Methods

| Method | Return/Evaluation Term | Variance | Update Type |
| :--- | :--- | :--- | :--- |
| **REINFORCE** | $G_t$ (Full Return) | High | Offline (Episode end) |
| **REINFORCE with Baseline** | $G_t - \hat{v}(S_t, w)$ | Medium | Offline (Episode end) |
| **Actor-Critic** | $R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$ | Low | Online (Step-by-step) |