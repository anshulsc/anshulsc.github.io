# Lecture Notes: Generalized Advantage Estimation (GAE)

---

## 1. Recap of Policy Gradient Methods

In reinforcement learning, the objective is to optimize a policy $\pi$ to maximize the expected cumulative reward. Policy gradient methods optimize the policy parameters directly.

### Step 1: Parametrizing the Policy
An unparametrized policy is denoted as $\pi(a | s)$, which represents the probability of taking action $a$ given state $s$. To optimize this policy using gradient-based optimization, we parameterize it using a parameter vector $\theta \in \mathbb{R}^d$ (such as the weights of a neural network):

$$\pi(a | s) \to \pi_\theta(a | s) = P(A_t = a \mid S_t = s; \theta)$$

### Step 2: Defining the Performance Measure $J(\theta)$
We define a performance measure $J(\theta)$ that quantifies the quality of the parameterized policy. A common performance measure is the expected total return starting from the initial state of the episode, $s_0$:

$$J(\theta) = v_{\pi_\theta}(s_0)$$

where $v_{\pi_\theta}(s_0)$ is the state-value function under policy $\pi_\theta$ at the initial state $s_0$:

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \;\middle|\; S_0 = s_0 \right]$$

### Step 3: Gradient Ascent
To find the optimal policy parameters $\theta$ that maximize $J(\theta)$, we perform gradient ascent. We compute the gradient of the performance measure with respect to the parameters, $\nabla_\theta J(\theta)$, and update the parameters in that direction:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta)$$

where $\alpha > 0$ is the learning rate.

### Step 4 & 5: The Policy Gradient Theorem
When we modify the policy parameters $\theta$, two things change:
1. The action selection probabilities $\pi_\theta(a|s)$ change directly.
2. The state visitation distribution (how often the agent visits each state $s$), denoted as $\mu(s)$, changes as a consequence of the new policy behavior.

Because the state distribution depends on the environment's transition dynamics (which are typically unknown), computing the gradient of the state distribution with respect to $\theta$ is challenging. The **Policy Gradient Theorem** provides a crucial simplification: it shows that the gradient of $J(\theta)$ does not depend on the derivative of the state distribution.

The theorem states that:

$$\nabla_\theta J(\theta) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} q_\pi(s, a) \nabla_\theta \pi_\theta(a|s)$$

where:
*   $\mu(s)$ is the stationary distribution of states under policy $\pi_\theta$.
*   $q_\pi(s, a)$ is the action-value function under policy $\pi_\theta$.

---

## 2. Derivation of the Expectation Form

To make the policy gradient theorem practical for model-free reinforcement learning, we must express the gradient as an expectation that can be approximated using sample trajectories.

### Step-by-Step Derivation

1.  **Start with the Policy Gradient expression:**
    $$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} q_\pi(s, a) \nabla_\theta \pi_\theta(a|s)$$

2.  **Multiply and divide by $\pi_\theta(a|s)$ inside the inner summation:**
    $$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} \pi_\theta(a|s) q_\pi(s, a) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

3.  **Apply the log-derivative trick:**
    From calculus, the derivative of the natural logarithm of a function $f(x)$ is:
    $$\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}$$
    Applying this to the policy $\pi_\theta(a|s)$, we get:
    $$\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \nabla_\theta \log \pi_\theta(a|s)$$

4.  **Substitute this back into the gradient equation:**
    $$\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} \pi_\theta(a|s) q_\pi(s, a) \nabla_\theta \log \pi_\theta(a|s)$$

5.  **Convert the summation to an expectation:**
    Because $\mu(s)$ represents the probability of visiting state $s$ and $\pi_\theta(a|s)$ is the probability of taking action $a$ in state $s$, the double sum represents an expectation over the state-action pairs encountered while following policy $\pi_\theta$:
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ q_\pi(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

---

## 3. Three Methods for Representing the Policy Gradient

Depending on how we approximate or substitute the action-value term $q_\pi(s_t, a_t)$, we obtain different policy gradient formulations.

### Method 1: True Action-Value
This is the direct representation:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ q_\pi(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$
*   **Characteristics:** It requires an accurate estimate of $q_\pi(s_t, a_t)$ for all actions, which usually necessitates a learned critic function.

### Method 2: REINFORCE (Monte Carlo Return)
Since $q_\pi(s_t, a_t) = \mathbb{E}_{\pi_\theta} [G_t \mid S_t = s_t, A_t = a_t]$, we can replace the true action-value with the empirical sample return $G_t$ (where $G_t = \sum_{l=0}^{\infty} \gamma^l R_{t+l+1}$):
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$
*   **Characteristics:** This is the **REINFORCE** algorithm. It is unbiased but suffers from **high variance** because the return $G_t$ depends on many random transitions and rewards over the course of an episode.

### Method 3: REINFORCE with Baseline
To reduce variance without introducing bias, we can subtract a state-dependent baseline $b(s_t)$ from the return $G_t$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

#### Proof of Zero Bias for Baseline Subtraction (First Principles)
We must show that the expected value of the baseline term is exactly zero:

$$\mathbb{E}_{a_t \sim \pi_\theta} \left[ b(s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \;\middle|\; S_t = s_t \right] = \sum_{a_t \in \mathcal{A}} \pi_\theta(a_t \mid s_t) b(s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Substitute the log-derivative identity back:

$$= \sum_{a_t \in \mathcal{A}} \pi_\theta(a_t \mid s_t) b(s_t) \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}$$
$$= b(s_t) \sum_{a_t \in \mathcal{A}} \nabla_\theta \pi_\theta(a_t \mid s_t)$$
$$= b(s_t) \nabla_\theta \sum_{a_t \in \mathcal{A}} \pi_\theta(a_t \mid s_t)$$

Since $\pi_\theta(a_t | s_t)$ is a valid probability distribution, the sum of probabilities over all possible actions must equal 1:

$$= b(s_t) \nabla_\theta (1) = b(s_t) \cdot 0 = 0$$

Because the expected value of this baseline term is zero, subtracting $b(s_t)$ does not introduce bias into our gradient estimate. In practice, the state-value function $v_\pi(s_t)$ is commonly used as the baseline:

$$b(s_t) \approx v_\pi(s_t)$$

---

## 4. The Advantage Function

The baseline subtraction method leads directly to the concept of the **Advantage Function** $A_\pi(s, a)$, which is defined as:

$$A_\pi(s_t, a_t) = q_\pi(s_t, a_t) - v_\pi(s_t)$$

### Intuition
*   The state-value function $v_\pi(s_t)$ represents the expected return if we follow the policy $\pi$ starting from state $s_t$ (representing the default or average performance at that state).
*   The action-value function $q_\pi(s_t, a_t)$ represents the expected return of taking a specific action $a_t$ and then following the policy $\pi$.
*   The advantage $A_\pi(s_t, a_t)$ measures whether a specific action $a_t$ is better or worse than the policy's average action:
    *   **$A_\pi(s_t, a_t) > 0$:** The action is better than average. The gradient update will increase the probability of selecting this action.
    *   **$A_\pi(s_t, a_t) < 0$:** The action is worse than average. The gradient update will decrease the probability of selecting this action.

Using the advantage function, the policy gradient is written as:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ A_\pi(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

---

## 5. Estimating the Advantage Function

Since we do not have access to the true functions $q_\pi$ and $v_\pi$, we must estimate the advantage function. We use $V(s)$ as our learned estimate of $v_\pi(s)$.

### 1-Step TD Advantage Estimate (Temporal Difference)
Using the Bellman equation, we can approximate the action-value function as:
$$q_\pi(s_t, a_t) \approx \mathbb{E} \left[ R_t + \gamma V(S_{t+1}) \;\middle|\; S_t = s_t, A_t = a_t \right]$$

Substituting this approximation into the advantage equation gives us the 1-step advantage estimate, which is identical to the 1-step TD error $\delta_t$:

$$\hat{A}_t^{(1)} = R_t + \gamma V(S_{t+1}) - V(S_t) = \delta_t$$

### 2-Step TD Advantage Estimate
Looking two steps into the future, we have:
$$\hat{A}_t^{(2)} = R_t + \gamma R_{t+1} + \gamma^2 V(S_{t+2}) - V(S_t)$$

We can express this in terms of TD errors:
$$\delta_t + \gamma \delta_{t+1} = \left( R_t + \gamma V(S_{t+1}) - V(S_t) \right) + \gamma \left( R_{t+1} + \gamma V(S_{t+2}) - V(S_{t+1}) \right)$$
$$= R_t + \gamma R_{t+1} + \gamma^2 V(S_{t+2}) - V(S_t) = \hat{A}_t^{(2)}$$

Thus:
$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1}$$

### $n$-Step TD Advantage Estimate
Extending this recursively, the $n$-step advantage estimate is:

$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l R_{t+l} + \gamma^n V(S_{t+n}) - V(S_t) = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$

### Infinite-Step (Monte Carlo) Advantage Estimate
If we look ahead to the end of the episode (letting $n \to \infty$), we get the Monte Carlo return $G_t$:

$$\hat{A}_t^{(\infty)} = G_t - V(S_t) = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$$

---

## 6. Generalized Advantage Estimation (GAE)

Instead of choosing a specific $n$-step lookahead, **Generalized Advantage Estimation** (GAE) takes an exponentially weighted average of all $n$-step advantage estimators. This is parameterized by $\lambda \in [0, 1]$.

### Derivation of the GAE Formula

We define the GAE estimator as the weighted average of the $n$-step advantage estimators:

$$\hat{A}_t^{GAE(\gamma, \lambda)} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}$$

Expanding this expression using the TD-error representation $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$:

$$\hat{A}_t^{GAE(\gamma, \lambda)} = (1-\lambda) \left[ \hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \dots \right]$$
$$\hat{A}_t^{GAE(\gamma, \lambda)} = (1-\lambda) \left[ \delta_t + \lambda (\delta_t + \gamma \delta_{t+1}) + \lambda^2 (\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}) + \dots \right]$$

Now, we collect the coefficients for each individual TD error term $\delta_{t+l}$:

*   **For $\delta_t$:**
    $$(1-\lambda) \left( 1 + \lambda + \lambda^2 + \lambda^3 + \dots \right) \delta_t = (1-\lambda) \left( \frac{1}{1-\lambda} \right) \delta_t = \delta_t$$

*   **For $\delta_{t+1}$:**
    $$(1-\lambda) \left( \gamma \lambda + \gamma \lambda^2 + \gamma \lambda^3 + \dots \right) \delta_{t+1} = (1-\lambda) \gamma \lambda \left( 1 + \lambda + \lambda^2 + \dots \right) \delta_{t+1}$$
    $$= (1-\lambda) \gamma \lambda \left( \frac{1}{1-\lambda} \right) \delta_{t+1} = (\gamma \lambda) \delta_{t+1}$$

*   **For $\delta_{t+2}$:**
    $$(1-\lambda) \left( \gamma^2 \lambda^2 + \gamma^2 \lambda^3 + \dots \right) \delta_{t+2} = (1-\lambda) \gamma^2 \lambda^2 \left( \frac{1}{1-\lambda} \right) \delta_{t+2} = (\gamma \lambda)^2 \delta_{t+2}$$

By induction, the coefficient for the general term $\delta_{t+l}$ is $(\gamma \lambda)^l$. The GAE estimator simplifies to:

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

### Analysis of the Parameter $\lambda$

The parameter $\lambda$ controls the trade-off between bias and variance:

```
  TD(0)                                                           Monte Carlo
  1-step TD                                                       Infinite-step
  |---------------------------------------------------------------------------|
  λ = 0                                                           λ = 1
  Bias: High                                                      Bias: Low
  Variance: Low                                                   Variance: High
```

#### Case 1: $\lambda = 0$
When we set $\lambda = 0$, the GAE formula simplifies to:
$$\hat{A}_t^{GAE(\gamma, 0)} = \delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$$
*   **Bias:** High (relies heavily on the accuracy of $V$).
*   **Variance:** Low (only depends on the immediate transition).

#### Case 2: $\lambda = 1$
When we set $\lambda = 1$, the GAE formula simplifies to:
$$\hat{A}_t^{GAE(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = G_t - V(S_t)$$
*   **Bias:** Low (unbiased estimate of the advantage).
*   **Variance:** High (accumulates variance over the entire trajectory).

### Summary Table

| $\lambda$ Value | Variance | Bias | Method Type |
| :--- | :--- | :--- | :--- |
| $\lambda = 0$ | Low | High | 1-step TD |
| $\lambda = 1$ | High | Low | Monte Carlo |
| $\lambda \in (0, 1)$ | Intermediate | Intermediate | GAE (Balanced) |

In modern reinforcement learning algorithms like PPO (Proximal Policy Optimization), $\lambda$ is typically set to an intermediate value (e.g., $0.95 \le \lambda \le 0.98$) to balance bias and variance, leading to stable updates.