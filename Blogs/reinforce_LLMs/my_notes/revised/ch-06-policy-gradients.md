# Chapter 6: Policy Gradients — Optimizing the Policy Directly

*Chapter 5 ended with function approximation: approximate $Q(s,a)$ with a neural network, then derive the policy via $\arg\max$. But this indirect approach has a fundamental limitation — it requires a maximization step over actions, which breaks for continuous action spaces and can't represent stochastic policies naturally. The move to direct policy optimization is the single most important transition in this entire set of notes. Everything from here — REINFORCE, Actor-Critic, GAE, TRPO, PPO — flows from one idea: parameterize the policy $\pi_\theta$ and use gradient ascent to maximize expected return.*

---

## 6.1 Why Direct Policy Optimization?

### Limitations of Value-Based Methods

| Attribute | Tabular Value Methods | Value Function Approximation | Policy Gradient Methods |
| :--- | :--- | :--- | :--- |
| **Target** | Q-table ($|\mathcal{S}| \times |\mathcal{A}|$) | Neural network $Q(s, a; \mathbf{w})$ | Neural network $\pi(a \mid s; \theta)$ |
| **Optimization** | Minimize Bellman error locally | Minimize mean squared value error | Maximize expected return directly: $J(\theta)$ |
| **Action Selection** | Deterministic / Greedy ($\arg\max$) | Deterministic / Greedy ($\arg\max$) | Stochastic (outputs probabilities) |
| **Scaling** | Fails in large/continuous spaces | Scales but relies on value proxy | Scales to continuous/complex spaces |

The key problem: value-based methods optimize a **proxy** (the value function) to find a policy. Policy gradients optimize the policy itself.

### Parameterized Policies

$$\pi(a \mid s, \theta) = \mathbb{P}(A_t = a \mid S_t = s, \theta_t = \theta)$$

* $\theta \in \mathbb{R}^{d'}$: parameter vector (neural network weights).
* By tuning $\theta$, we smoothly warp the action probability distribution.

### Softmax Parameterization (Discrete Actions)

Define a preference function $h(s, a, \theta) \in \mathbb{R}$ (can be a neural network), then convert to probabilities:

$$\pi(a \mid s, \theta) = \frac{e^{h(s, a, \theta)}}{\sum_{b \in \mathcal{A}} e^{h(s, b, \theta)}}$$

#### Worked Example
5 actions with preferences $\{3, 2, 4, 5, 1\}$:

| Action | $h(s,a,\theta)$ | $e^h$ | $\pi(a \mid s,\theta)$ |
|---|---|---|---|
| $a_1$ | 3 | 20.09 | 8.6% |
| $a_2$ | 2 | 7.39 | 3.2% |
| $a_3$ | 4 | 54.60 | 23.4% |
| $a_4$ | 5 | 148.41 | **63.6%** |
| $a_5$ | 1 | 2.72 | 1.2% |
| **Sum** | | **233.20** | **100%** |

The softmax preserves the ranking while satisfying probability axioms (non-negative, sums to 1).

---

## 6.2 The Policy Gradient Theorem

### The Performance Measure

For episodic tasks, define:
$$J(\theta) \doteq v_{\pi_\theta}(s_0)$$

The expected return starting from the initial state under policy $\pi_\theta$. We optimize via gradient ascent:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)$$

### The Fundamental Challenge

$J(\theta)$ depends on two things, **both** affected by $\theta$:
1. **Action probabilities**: $\pi(a|s,\theta)$ — directly controlled by $\theta$.
2. **State distribution**: $\mu(s)$ — which states the agent visits. Changes implicitly when $\theta$ changes the policy's behavior.

Expanding $J(\theta)$:
$$J(\theta) = \sum_{s} \mu(s) \sum_{a} \pi(a|s,\theta) q_\pi(s,a)$$

Taking the gradient via product rule:
$$\nabla_\theta J = \sum_s \left[ \nabla_\theta \mu(s) \sum_a \pi(a|s,\theta) q_\pi(s,a) + \mu(s) \nabla_\theta \left(\sum_a \pi(a|s,\theta) q_\pi(s,a)\right) \right]$$

The term $\nabla_\theta \mu(s)$ is **intractable** — it depends on the unknown environment dynamics $P(s'|s,a)$.

### The Theorem

The **Policy Gradient Theorem** proves that under expectation, we don't need $\nabla_\theta \mu(s)$:

$$\nabla_\theta J(\theta) \propto \sum_{s} \underbrace{\mu(s)}_{\text{state dist.}} \sum_{a} \underbrace{q_\pi(s,a)}_{\text{action-value}} \underbrace{\nabla_\theta \pi(a|s,\theta)}_{\text{policy gradient}}$$

This is remarkable: we can compute the gradient of the objective without knowing how the state distribution changes with $\theta$.

---

## 6.3 REINFORCE: From Theorem to Algorithm

### Step 1: State Sum → Expectation

Since $\mu(s)$ is a probability distribution over states:

$$\sum_s \mu(s) [\cdots] = \mathbb{E}_\pi [\cdots]$$

$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi \left[\sum_a q_\pi(S_t, a) \nabla_\theta \pi(a|S_t, \theta)\right]$$

### Step 2: Action Sum → Expectation (Log-Derivative Trick)

Multiply and divide by $\pi(a|S_t,\theta)$:

$$\sum_a q_\pi(S_t,a) \nabla_\theta \pi(a|S_t,\theta) = \sum_a \pi(a|S_t,\theta) \cdot q_\pi(S_t,a) \cdot \frac{\nabla_\theta \pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}$$

The ratio is the log-derivative: $\frac{\nabla f}{f} = \nabla \ln f$

$$= \mathbb{E}_{a \sim \pi} \left[ q_\pi(S_t, A_t) \nabla_\theta \ln \pi(A_t|S_t,\theta) \right]$$

### Step 3: Replace $q_\pi$ with Sample Return $G_t$

Since $q_\pi(S_t, A_t) = \mathbb{E}[G_t | S_t, A_t]$, we can substitute the actual observed return:

$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi \left[ G_t \nabla_\theta \ln \pi(A_t|S_t,\theta) \right]$$

### The REINFORCE Update

$$\theta_{t+1} = \theta_t + \alpha G_t \frac{\nabla_\theta \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}$$

### Geometric Intuition

$$\Delta\theta = \alpha \cdot \underbrace{G_t}_{\text{scale}} \cdot \underbrace{\frac{\nabla_\theta \pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}}_{\text{direction}}$$

```
                      \  New θ
                       \
                        ^  Update: α · G_t · (∇π / π)
                         \
                          \
   -----------------------o Old θ
```

* **Direction**: $\nabla_\theta \pi$ points toward increased probability of action $A_t$.
* **Scale** ($G_t$): High return → big step. Low/negative return → small/opposite step.
* **Normalization** ($1/\pi$): Rare but good actions get proportionally larger boosts. Without it, frequently taken actions would dominate the update regardless of quality.

---

## 6.4 Baselines & the Advantage Function

### The Variance Problem
REINFORCE is unbiased but has **high variance** — $G_t$ varies wildly between episodes, making convergence slow and noisy.

### Subtracting a Baseline
Subtract any action-independent baseline $b(s)$ from the return:

$$\nabla_\theta J(\theta) \propto \mathbb{E}_\pi \left[ (q_\pi(s,a) - b(s)) \nabla_\theta \pi(a|s,\theta) \right]$$

### Proof: Zero Bias

$$\sum_a b(s) \nabla_\theta \pi(a|s,\theta) = b(s) \nabla_\theta \underbrace{\sum_a \pi(a|s,\theta)}_{= 1} = b(s) \cdot \nabla_\theta(1) = 0$$

Since $\pi$ is a valid probability distribution, its sum is always 1, whose gradient is 0. So subtracting $b(s)$ adds zero expected contribution → no bias, but can massively reduce variance.

### The Advantage Function

The natural baseline: $b(s) = v_\pi(s)$.

$$A_\pi(s,a) = q_\pi(s,a) - v_\pi(s)$$

* $A > 0$: Action is **better than average** → increase its probability.
* $A < 0$: Action is **worse than average** → decrease its probability.
* $A = 0$: Action matches the average.

### REINFORCE with Baseline

$$\theta_{t+1} = \theta_t + \alpha (G_t - \hat{v}(S_t, \mathbf{w})) \frac{\nabla_\theta \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}$$

---

## 6.5 Actor-Critic: Online Policy Gradients

### The Remaining Problem
REINFORCE (even with baseline) is a Monte Carlo method — needs complete episodes to compute $G_t$. Can't update mid-episode.

### Solution: Replace $G_t$ with TD Target

$$G_t \approx R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$$

Substituting into the baseline-subtracted update:

$$\theta_{t+1} = \theta_t + \alpha \underbrace{\left(R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})\right)}_{\delta_t \text{ (TD error)}} \frac{\nabla_\theta \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}$$

### Architecture

```
                       +-------------------+
                       |    Environment    |
                       +-------------------+
                         | State      ^ Reward
                         v            |
                       +-------------------+
                       |      Critic       |-----> Updates value weights (w)
                       |     (v̂(s, w))     |       using TD error (δ)
                       +-------------------+
                         | TD Error (δ)
                         v
                       +-------------------+
                       |       Actor       |-----> Updates policy parameters (θ)
                       |     (π(a|s, θ))   |       using δ · ∇ln π
                       +-------------------+
```

### Comparison

| Method | Return Estimate | Variance | Update Type |
| :--- | :--- | :--- | :--- |
| **REINFORCE** | $G_t$ (full return) | High | Offline (episode end) |
| **REINFORCE + Baseline** | $G_t - \hat{v}(S_t, \mathbf{w})$ | Medium | Offline (episode end) |
| **Actor-Critic** | $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$ | Low | Online (step-by-step) |

---

## 6.6 Generalized Advantage Estimation (GAE)

### The Bias-Variance Tradeoff in Advantage Estimation

The 1-step TD advantage $\hat{A}_t^{(1)} = \delta_t$ has low variance but high bias (depends on accuracy of $\hat{v}$). The MC advantage $G_t - V(S_t)$ has low bias but high variance. Can we interpolate?

### n-Step Advantage Estimates

$$\hat{A}_t^{(1)} = \delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$$
$$\hat{A}_t^{(2)} = \delta_t + \gamma\delta_{t+1} = R_t + \gamma R_{t+1} + \gamma^2 V(S_{t+2}) - V(S_t)$$
$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$
$$\hat{A}_t^{(\infty)} = G_t - V(S_t)$$

### GAE: Exponentially Weighted Average

Instead of picking a single $n$, take a weighted average of **all** $n$-step estimates:

$$\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}$$

### Derivation

Expand and collect coefficients of each $\delta_{t+l}$:

**For $\delta_t$:**
$$(1-\lambda)(1 + \lambda + \lambda^2 + \cdots)\delta_t = (1-\lambda) \cdot \frac{1}{1-\lambda} \cdot \delta_t = \delta_t$$

**For $\delta_{t+1}$:**
$$(1-\lambda)(\gamma\lambda + \gamma\lambda^2 + \cdots)\delta_{t+1} = (1-\lambda)\gamma\lambda \cdot \frac{1}{1-\lambda} \cdot \delta_{t+1} = (\gamma\lambda)\delta_{t+1}$$

**For $\delta_{t+l}$ (by induction):** coefficient = $(\gamma\lambda)^l$

$$\boxed{\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}}$$

### The $\lambda$ Parameter

```
  TD(0)                                                           Monte Carlo
  1-step TD                                                       Full return
  |---------------------------------------------------------------------------|
  λ = 0                                                           λ = 1
  Bias: High                                                      Bias: Low
  Variance: Low                                                   Variance: High
```

* **$\lambda = 0$**: $\hat{A}_t = \delta_t$ — pure TD, high bias, low variance.
* **$\lambda = 1$**: $\hat{A}_t = G_t - V(S_t)$ — pure MC, low bias, high variance.
* **Practice**: $\lambda \in [0.95, 0.98]$ balances both. Used in PPO, GRPO.

---

## 6.7 TRPO: Trust Regions for Safe Updates

### The Problem with Vanilla Policy Gradients
Standard gradient ascent can take destructively large steps — the policy changes so much in one update that it enters a bad region of parameter space and never recovers.

### Policy Performance Difference Identity

For any two policies $\pi'$ and $\pi$:

$$\eta(\pi') = \eta(\pi) + \mathbb{E}_{\tau \sim \pi'} \left[ \sum_{t=0}^{\infty} \gamma^t A_\pi(s_t, a_t) \right]$$

#### Proof (Telescoping Sum)

Expand the discounted sum of advantages:
$$\mathbb{E}_{\tau \sim \pi'} \left[\sum_t \gamma^t A_\pi(s_t,a_t)\right] = \mathbb{E}_{\tau \sim \pi'} \left[\sum_t \gamma^t (R(s_t) + \gamma v_\pi(s_{t+1}) - v_\pi(s_t))\right]$$

Split into three sums:
$$= \mathbb{E}_{\tau \sim \pi'} \left[\sum_t \gamma^t R(s_t)\right] + \underbrace{\sum_t \gamma^{t+1} v_\pi(s_{t+1}) - \sum_t \gamma^t v_\pi(s_t)}_{\text{telescoping: } = -v_\pi(s_0)}$$

$$= \eta(\pi') - \eta(\pi)$$

Rearranging: $\eta(\pi') = \eta(\pi) + \mathbb{E}_{\tau \sim \pi'}[\sum_t \gamma^t A_\pi(s_t,a_t)]$  ∎

### The Surrogate Objective

The identity requires sampling from $\pi'$ (the unknown new policy). Approximate by replacing the state distribution $\rho_{\pi'}$ with $\rho_\pi$ (the known old policy):

$$L_\pi(\pi') = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \pi'(a|s) A_\pi(s,a)$$

**Key property**: At $\pi' = \pi$, the surrogate matches the true objective in **both value and gradient**.

### Monotonic Improvement Bound

How far can $\pi'$ deviate from $\pi$ before the approximation breaks?

$$\eta(\pi') \ge L_\pi(\pi') - C \cdot D_{KL}^{\max}(\pi, \pi')$$

where $C = \frac{4\epsilon\gamma}{(1-\gamma)^2}$ and $\epsilon = \max_{s,a} |A_\pi(s,a)|$.

### From Penalty to Constraint

The penalty constant $C$ is enormous (since $\gamma \approx 0.99$), restricting steps to microscopic sizes. TRPO reformulates as a **constrained** problem:

$$\max_{\pi'} L_\pi(\pi') \quad \text{s.t.} \quad D_{KL}^{\max}(\pi, \pi') \le \delta$$

### Solving via Taylor Approximations

Let $x = \theta - \theta_{old}$, $g = \nabla_\theta L|_{\theta_{old}}$ (= vanilla policy gradient), and $\mathbf{A}$ = Fisher Information Matrix.

* **Objective** (1st-order Taylor): $f(\theta) \approx g^T x$
* **Constraint** (2nd-order Taylor): $D_{KL} \approx \frac{1}{2} x^T \mathbf{A} x$

The constrained problem becomes: $\max_x g^T x$ s.t. $\frac{1}{2} x^T \mathbf{A} x \le \delta$

### Lagrangian Solution

$$\mathcal{L}(x, \lambda) = g^T x - \lambda(\frac{1}{2} x^T \mathbf{A} x - \delta)$$

Set $\nabla_x \mathcal{L} = 0$: $g - \lambda \mathbf{A} x = 0 \implies x^* = \frac{1}{\lambda} \mathbf{A}^{-1} g$

Substitute into the constraint boundary:
$$\frac{1}{2\lambda^2} g^T \mathbf{A}^{-1} g = \delta \implies \lambda = \sqrt{\frac{g^T \mathbf{A}^{-1} g}{2\delta}}$$

### Final TRPO Step

$$\theta_{k+1} = \theta_k + \sqrt{\frac{2\delta}{g^T \mathbf{A}^{-1} g}} \cdot \mathbf{A}^{-1} g$$

With backtracking line search: $\theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2\delta}{g^T \mathbf{A}^{-1} g}} \cdot \mathbf{A}^{-1} g$

### The Computational Bottleneck
$\mathbf{A}$ is $N \times N$ where $N$ = number of parameters. For modern neural networks ($N$ = millions to billions):
* Storing $\mathbf{A}$: $O(N^2)$ memory — prohibitive.
* Computing $\mathbf{A}^{-1} g$: $O(N^3)$ — impractical.
* **Workaround**: Conjugate Gradient to approximate $\mathbf{A}^{-1} g$ without storing $\mathbf{A}$. Still complex.

This bottleneck directly motivated PPO.

---

## 6.8 PPO: Clipping Instead of Constraints

PPO achieves the stability of TRPO using only **first-order** derivatives and a simple clipping mechanism.

### The Probability Ratio

$$r_t(\theta) = \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{old}}(A_t|S_t)}$$

* $r_t = 1$: New policy same as old.
* $r_t > 1$: Action became more likely.
* $r_t < 1$: Action became less likely.

### The Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

### Case 1: Positive Advantage ($\hat{A}_t > 0$)

Good action → we want to increase its probability ($r_t > 1$). But cap at $1+\epsilon$:

```
   L^CLIP (A_t > 0)
     ^
     |                / Unclipped
     |               /
(1+ε)|  ┌───────────▀   ← Clipped: no more gradient
     |  │
     | /
     |/
─────┼──────┬───────┬──── r_t
     │      1      1+ε
```

Once $r_t > 1+\epsilon$: gradient = 0. Prevents the policy from becoming over-confident on this action.

### Case 2: Negative Advantage ($\hat{A}_t < 0$)

Bad action → we want to decrease its probability ($r_t < 1$). Cap at $1-\epsilon$:

```
   L^CLIP (A_t < 0)
     ^
     |      1-ε     1
─────┼──────┼───────┼──── r_t
     |     /│
     |    / │
(1-ε)|  ┌▀  │           ← Clipped: stop penalizing further
     |  │   │
     │  │   \
     ▼  │    \  ← Unclipped: full penalty if r_t > 1 (making bad action MORE likely)
```

Key asymmetry: if $r_t > 1+\epsilon$ for a bad action (policy accidentally made it more likely), the full unclipped penalty applies — no mercy.

### Complete PPO Objective

$$L^{PPO}(\theta) = \mathbb{E}_t \left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](S_t) \right]$$

| Term | Purpose |
|---|---|
| $L^{CLIP}$ | Clipped policy gradient loss |
| $L^{VF} = (V_\theta(S_t) - V_t^{target})^2$ | Value function accuracy |
| $S[\pi_\theta] = -\sum_a \pi_\theta(a|S_t) \ln \pi_\theta(a|S_t)$ | Entropy bonus — encourages exploration |

### PPO vs VPG vs TRPO

| Metric | VPG | TRPO | PPO |
| :--- | :--- | :--- | :--- |
| **Optimization** | First-Order | Second-Order | **First-Order** |
| **Computation** | Low | High (FIM inversion) | **Low** |
| **Stability** | Low (collapse risk) | High (KL constraint) | **High (clipping)** |
| **Implementation** | Easy | Hard | **Easy** |

---

## Key Takeaways

1. **Policy Gradient Theorem**: We can compute the gradient of expected return without knowing the state distribution's gradient.
2. **REINFORCE**: Replace $q_\pi$ with sample return $G_t$ → model-free policy gradient.
3. **Baselines**: Subtracting $v_\pi(s)$ introduces zero bias but massively reduces variance.
4. **Actor-Critic**: Replace $G_t$ with TD target → online updates.
5. **GAE**: Interpolate between TD and MC via $\lambda$ → tunable bias-variance.
6. **TRPO**: Trust region constraint guarantees monotonic improvement, but requires second-order optimization.
7. **PPO**: Clips the probability ratio → achieves TRPO's stability with first-order simplicity.

---

*PPO gives us a stable, scalable policy optimizer. Now: how do we actually apply this to language models — where the "action" is a token, the "state" is a prompt, and the "reward" comes from a human or a verifier? → Chapter 7.*
