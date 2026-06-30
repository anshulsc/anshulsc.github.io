# Chapter 4: Dynamic Programming — When You Know the World

*Chapter 3 gave us the Bellman equations that define optimal behavior. Now: how to compute it? Dynamic Programming (DP) assumes we have the full model — the transition dynamics $p(s',r|s,a)$ — and uses the Bellman equations as iterative update rules. DP is impractical for large problems (chess has $10^{46}$ states), but the ideas here — policy evaluation, policy improvement, value iteration — are the conceptual backbone of every algorithm that follows.*

---

## 1. What Is Dynamic Programming?

A collection of algorithms that compute optimal policies **given a perfect model** of the environment as an MDP.

* **Requires**: Complete knowledge of $p(s', r \mid s, a)$ for all $s, s', a, r$.
* **Method**: Turn the Bellman equations into iterative update rules.
* **Limitation**: Impractical for large state spaces (curse of dimensionality).

Two key operations:
1. **Policy Evaluation**: Given a policy $\pi$, compute its value function $v_\pi(s)$.
2. **Policy Improvement**: Use the value function to derive a better policy.

---

## 2. Policy Evaluation (Iterative)

**Goal**: Compute $v_\pi(s)$ for a given policy $\pi$.

Recall the Bellman equation for $v_\pi$ (→ §3.10):
$$v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_\pi(s')]$$

This is a system of $|\mathcal{S}|$ linear equations with $|\mathcal{S}|$ unknowns. We can solve it iteratively:

$$v_{k+1}(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k(s')] \quad \forall s \in \mathcal{S}$$

Start with arbitrary $v_0$ → apply update → $v_1 \to v_2 \to \dots \to v_\pi$.

### Convergence
As $k \to \infty$, $v_k \to v_\pi$ is guaranteed. We stop when changes are negligible:
$$\Delta = \max_s |v_{k+1}(s) - v_k(s)| < \theta$$

### Pseudocode

```
ITERATIVE POLICY EVALUATION:
  Input: policy π to evaluate, threshold θ
  Initialize V(s) = 0 for all s ∈ S
  
  Loop:
    Δ ← 0
    For each state s ∈ S:
      v ← V(s)                                          # Store old value
      V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ V(s')]   # Bellman update
      Δ ← max(Δ, |v - V(s)|)                            # Track max change
    Until Δ < θ
  
  Return V ≈ v_π
```

### Worked Example: 5-State Linear Walk

```
State Space:
   [s_1] --- [s_2] --- [s_3] --- [s_4] --- [s_5]
     ^                                        ^
   Terminal                               Terminal
  r = 0                                   r = +1
```

Equiprobable random policy: $\pi(\text{left}|s) = \pi(\text{right}|s) = 0.5$, discount $\gamma = 1$.

| Iteration | $V(s_1)$ | $V(s_2)$ | $V(s_3)$ | $V(s_4)$ | $V(s_5)$ |
|---|---|---|---|---|---|
| $k=0$ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| $k=1$ | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 |
| $k=\infty$ | 0.167 | 0.333 | 0.5 | 0.667 | 0.833 |

The steady-state values form a linear gradient — states closer to the $+1$ terminal have higher value.

---

## 3. Policy Improvement

Given $v_\pi(s)$, can we find a **better** policy $\pi'$?

### The Policy Improvement Theorem

For all $s \in \mathcal{S}$:

$$q_\pi(s, \pi'(s)) \ge v_\pi(s) \implies v_{\pi'}(s) \ge v_\pi(s)$$

If taking the greedy action $\pi'(s) = \arg\max_a q_\pi(s,a)$ is at least as good as the current policy's value in every state, then the new policy $\pi'$ is at least as good overall.

### Greedy Policy Improvement

$$\pi'(s) = \arg\max_{a} q_\pi(s, a) = \arg\max_{a} \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_\pi(s')]$$

---

## 4. Policy Iteration

Combine evaluation and improvement into a loop:

```
π_0 --[Eval]--> V_π0 --[Improve]--> π_1 --[Eval]--> V_π1 --[Improve]--> π_2 ---> ... ---> π*

E = Policy Evaluation
I = Policy Improvement

π_0 →E→ v_π0 →I→ π_1 →E→ v_π1 →I→ π_2 →E→ ... →I→ π* →E→ v*
```

### Pseudocode

```
POLICY ITERATION:
  1. Initialize V(s) and π(s) arbitrarily for all s ∈ S

  2. Policy Evaluation:
     Loop:
       Δ ← 0
       For each s ∈ S:
         v ← V(s)
         V(s) ← Σ_{s',r} p(s',r|s,π(s)) [r + γ V(s')]
         Δ ← max(Δ, |v - V(s)|)
     Until Δ < θ

  3. Policy Improvement:
     policy_stable ← true
     For each s ∈ S:
       old_action ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
       If old_action ≠ π(s): policy_stable ← false
     
     If policy_stable: return V ≈ v*, π ≈ π*
     Else: go to Step 2
```

### Convergence
Finite MDPs have a finite number of deterministic policies → policy iteration must converge (no cycles — each step is strictly better or equal). Typically converges in very few iterations (often < 10 even for large MDPs).

---

## 5. Case Study: Jack's Car Rental

### Problem Setup
* Jack manages **two locations** of a car rental company.
* Each day, customers arrive to rent and return cars. Revenue = $10 per rental.
* Overnight, Jack can transfer up to **5 cars** between locations (cost: $2 per car moved).
* Rental and return requests follow **Poisson distributions**:

| Location | Expected Rentals ($\lambda$) | Expected Returns ($\lambda$) |
|---|---|---|
| Location 1 | 3 | 3 |
| Location 2 | 4 | 2 |

**Poisson Probability**: $P(n) = \frac{\lambda^n e^{-\lambda}}{n!}$

### State, Action, Reward
* **State**: $(n_1, n_2)$ — number of cars at each location at end of day. Max 20 per location → $21 \times 21 = 441$ states.
* **Action**: Net cars moved overnight from Location 1 to Location 2: $a \in \{-5, -4, \dots, 0, \dots, 4, 5\}$.
* **Reward**: $+10$ per rental fulfilled $- 2 \cdot |a|$ (moving cost).

### Policy Iteration Results

```
   Policy Iteration Sequence:
   
   π_1 (Move 0)     π_2 (Move some)    π_3 (Refined)     π_4 ≈ π*
   ┌──────────┐     ┌──────────┐       ┌──────────┐      ┌──────────┐
   │  All 0s  │     │  1 0 -1  │       │  2 1 0   │      │  3 2 1   │
   │  0  0  0 │ --> │  2 1  0  │  -->  │  3 2 1   │ -->  │  4 3 2   │
   │  0  0  0 │     │  3 2  1  │       │  4 3 2   │      │  5 4 3   │
   └──────────┘     └──────────┘       └──────────┘      └──────────┘
```

The optimal policy moves more cars to Location 2 (higher demand, lower natural return rate).

---

## 6. Value Iteration

Policy iteration requires full policy evaluation (inner loop to convergence) at each step. **Value iteration** cuts this short: do just **one** Bellman backup per state, combining evaluation and improvement:

$$v_{k+1}(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k(s')]$$

### Key Insight
Instead of iterating evaluation to convergence before improving, we take the max immediately. This is equivalent to turning the Bellman optimality equation into an update rule.

### Pseudocode

```
VALUE ITERATION:
  Initialize V(s) = 0 for all s ∈ S
  
  Loop:
    Δ ← 0
    For each s ∈ S:
      v ← V(s)
      V(s) ← max_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
      Δ ← max(Δ, |v - V(s)|)
  Until Δ < θ
  
  Extract policy:
    π(s) = argmax_a Σ_{s',r} p(s',r|s,a) [r + γ V(s')]
  
  Return V ≈ v*, π ≈ π*
```

### Policy Iteration vs. Value Iteration

| Aspect | Policy Iteration | Value Iteration |
|---|---|---|
| Evaluation | Full convergence (inner loop) | One-step backup |
| Improvement | Separate step | Embedded in the backup |
| Speed per iteration | Slow (many evaluation sweeps) | Fast (single sweep) |
| Iterations to converge | Fewer outer iterations | More iterations overall |
| Practical winner | Small-medium MDPs | Large MDPs |

---

## 7. Limitations of Dynamic Programming

DP is the gold standard for computing optimal policies — if the model is known. But:

1. **Model Dependency**: Requires full knowledge of $p(s', r \mid s, a)$. In most real problems (robotics, games, LLM alignment), the environment dynamics are unknown.

2. **Curse of Dimensionality**: Computation scales with $|\mathcal{S}| \times |\mathcal{A}|$. Chess: $\sim 10^{46}$ states. Go: $\sim 10^{170}$. Storing a value table is physically impossible.

3. **No Learning**: DP computes solutions offline using the model. It doesn't learn from experience.

---

## Key Takeaways

1. **Policy Evaluation** iteratively solves the Bellman equation for a given policy.
2. **Policy Improvement** computes a greedy policy from the value function.
3. **Policy Iteration** alternates between these two → guaranteed convergence to $\pi_*$.
4. **Value Iteration** combines both into a single max-backup → faster per iteration.
5. All DP methods need the full model. Without it, we need model-free methods.

---

*DP needs a perfect model and can't handle enormous state spaces. Chess has $10^{46}$ states. We can't even store the table. So what do we do when we don't have the model? We learn directly from experience — through Monte Carlo sampling and Temporal Difference bootstrapping → Chapter 5.*
