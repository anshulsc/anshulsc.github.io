# Chapter 3: MDPs, Value Functions & Bellman Equations

*Bandits (Chapter 2) isolated the exploration-exploitation tradeoff in a single-state setting. Now we bring in the full picture: multiple states, state transitions, and the need to plan ahead. The Markov Decision Process (MDP) is the formal mathematical framework that captures all of this. By the end of this chapter, we'll have the Bellman equations — the recursive relationships that define what "optimal" means, and that every algorithm in Chapters 4–7 tries to solve or approximate.*

---

## 1. The Agent-Environment Interface

The interaction between agent and environment is a step-by-step decision-making loop:

```
             Action (At)
        +----------------->
        |                |
   +----+---+        +---+----+
   |        |        |        |
   | Agent  |        | Environ|
   |        |        | -ment  |
   +----+---+        +---+----+
        ^                |
        |  Reward (Rt+1) |
        +----------------+
           State (St+1)
```

### The Discrete Time-Step Cycle
At discrete time steps $t = 0, 1, 2, 3, \dots$:
1. **State Observation** ($S_t$): Agent receives state $S_t \in \mathcal{S}$.
2. **Action Selection** ($A_t$): Based on $S_t$, agent selects $A_t \in \mathcal{A}(s)$.
3. **Transition and Reward** ($S_{t+1}, R_{t+1}$): Environment transitions to $S_{t+1} \in \mathcal{S}$ and sends reward $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$.

### Notation
* **Capital Letters** ($S_t, A_t, R_t$): Random variables at time $t$.
* **Lowercase Letters** ($s, a, r$): Specific values/realizations (e.g., $s \in \mathcal{S}$).

---

## 2. The Agent's Policy ($\pi$)

The policy formalizes the agent's decision-making strategy — a mapping from states to action probabilities:

$$\pi_t(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$$

* If stationary (doesn't change over time): simply $\pi(a \mid s)$.
* Can be deterministic (maps state → single action) or stochastic (maps state → probability distribution over actions).

---

## 3. Goals and Rewards

### The Reward Hypothesis
> *All of what we mean by goals and purposes can be well-formulated as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).*

### Formulating Goals Correctly
Rewards must encode **what** to achieve, not **how** to achieve it.

* **Bad (Chess)**: Reward $+1$ for capturing opponent's queen → agent may sacrifice the game to capture a queen.
* **Good (Chess)**: Reward $+1$ for winning, $-1$ for losing, $0$ for draw → forces alignment with the actual objective.

---

## 4. Returns ($G_t$)

The agent's objective: maximize **cumulative** reward, not immediate reward. This cumulative measure = the **Return**.

### Episodic Tasks
Interaction breaks into self-contained episodes ending at terminal state at time $T$:
$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T$$

### Continuing Tasks (Discounting)
If the task runs forever ($T = \infty$), the return could diverge. Fix: introduce a **discount rate** $\gamma \in [0, 1)$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

#### Intuitive Analogy
₹100 today is worth more than ₹100 five years from now (inflation). Similarly:
* Immediate rewards: full face value ($1 \cdot R_{t+1}$)
* Future rewards: scaled down exponentially ($\gamma^k R_{t+k+1}$)

#### The Role of $\gamma$
* **$\gamma = 0$ (Myopic)**: Only cares about immediate reward $R_{t+1}$. Ignores future.
* **$\gamma \to 1$ (Far-Sighted)**: Values future rewards almost as much as immediate ones → encourages long-term planning.

---

## 5. The Markov Property

A state signal $S_t$ has the **Markov Property** if the current state contains all information needed to predict the future — the history of how the agent arrived is irrelevant.

### Mathematical Formulation
$$\mathbb{P}(R_{t+1} = r, S_{t+1} = s' \mid S_t = s, A_t = a) = \mathbb{P}(R_{t+1} = r, S_{t+1} = s' \mid S_t, A_t, R_t, S_{t-1}, \dots, S_0, A_0)$$

The current state and action are **sufficient** to describe transition dynamics.

### Examples

| Example | State Representation | Markovian? | Why |
| :--- | :--- | :--- | :--- |
| **Chess** | Current board configuration | **Yes** | Don't need move history to find the optimal next move. |
| **Physics** (Cannonball) | Position $p$ + velocity $v$ | **Yes** | Current $(p, v)$ determines trajectory. Previous path irrelevant. |
| **Conversations** | Last spoken sentence | **No** | Need entire conversation context for a coherent response. |
| **Draw Poker** | Cards in your hand | **No** | Need history of bets, discards, bluffing patterns. |

---

## 6. Finite Markov Decision Processes

An RL task with the Markov property = **Markov Decision Process (MDP)**. If state, action, and reward spaces are finite = **Finite MDP**.

The dynamics are governed by:
$$p(s', r \mid s, a) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a)$$

From this single function, we derive:

### State-Transition Probabilities
$$p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$$

### Expected Rewards for State-Action Pairs
$$r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)$$

### Expected Rewards for State-Action-Next-State Triples
$$r(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r \mid s, a)}{p(s' \mid s, a)}$$

---

## 7. Case Studies

### Case Study 1: Bioreactor Control
```
       [Nutrients/Inputs]
               |
               v
       +---------------+
       |  BIOREACTOR   | ---> [Sensors: Temp, pH, Pressure]
       |  (Chemical    |
       |   Production) | ---> [Heaters & Stirring Motors]
       +---------------+
               |
               v
       [Chemical Product]
```
* **State**: Sensor readings (temperature, pressure, pH)
* **Action**: Control inputs (target temperature, stirring rate)
* **Reward**: Rate of chemical production

### Case Study 2: Pick-and-Place Robot Arm
* **State**: Joint angles ($\theta_1, \theta_2$) and angular velocities ($\omega_1, \omega_2$)
* **Action**: Voltages applied to motors ($V_1, V_2$)
* **Reward**: $+1$ per successful pick-and-place, $-0.02$ for jerky movements

### Case Study 3: Recycling Robot (Finite MDP)
An autonomous can-collecting robot with battery states {High, Low}:

```
                   +--------+  Search (prob α)  +--------+
                   |        | ------------------> |        |
                   |  HIGH  |                     |  LOW   |
                   |        | <------------------ |        |
                   +--------+   Recharge (prob 1) +--------+
                     |    ^                         |    ^
        Wait (prob 1)|    | Wait (prob 1)           |    | Search (prob 1-β)
                     v    |                         v    | (depleted)
                   +--------+                     +--------+
                   |  HIGH  |                     | RESCUE | (R = -3)
                   +--------+                     +--------+
```

#### Full Transition Table

| $s$ | $a$ | $s'$ | $p(s' \mid s, a)$ | $r(s, a, s')$ |
| :--- | :--- | :--- | :--- | :--- |
| high | search | high | $\alpha$ | $r_{\text{search}}$ |
| high | search | low | $1 - \alpha$ | $r_{\text{search}}$ |
| high | wait | high | $1$ | $r_{\text{wait}}$ |
| low | search | low | $\beta$ | $r_{\text{search}}$ |
| low | search | high (Rescue) | $1 - \beta$ | $-3$ |
| low | wait | low | $1$ | $r_{\text{wait}}$ |
| low | recharge | high | $1$ | $0$ |

### Case Study 4: Cart-Pole
```
                  || <-- Pole (Angle θ)
                  ||
              +---|---+
              |  [O]  | <-- Cart (Position p)
              +-------+
               O     O  <-- Wheels
         ----------------------- <-- Track
```

**As Episodic Task**: Episode ends when $|\theta| > 15°$ or $|p| > 2.4$m. Reward $+1$ per step balanced. Return = total steps of success.

**As Continuing Task**: $R = -1$ on failure, $0$ otherwise. Discounted return penalizes future failures.

---

## 8. The State-Value Function $v_\pi(s)$

Now we get to the heart of RL: **value functions**. Unlike immediate reward (short-term), value estimates the long-term cumulative reward. An agent may accept a negative immediate reward if it leads to a high-value state.

The **state-value function** under policy $\pi$:

$$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]$$

* $v_\pi(s)$: Expected return starting from state $s$, following policy $\pi$ thereafter.
* $\mathbb{E}_\pi[\cdot]$: Expectation under policy $\pi$.
* $\gamma$: Discount rate — reward $k$ steps in the future is worth $\gamma^{k-1}$ times its immediate value.

---

## 9. The Action-Value Function $q_\pi(s, a)$

$$q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a \right]$$

### State-Value vs. Action-Value
* **$v_\pi(s)$**: How good is it to **be** in state $s$?
* **$q_\pi(s, a)$**: How good is it to **take action** $a$ from state $s$?

The action-value function lets us compare actions directly — crucial for model-free control (→ §5.1).

---

## 10. The Bellman Equation for $v_\pi$

Value functions satisfy a recursive relationship. The value of the current state can be expressed in terms of successor states' values.

### Derivation

Start with the definition:
$$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

Expand $G_t$ into immediate reward + discounted future:
$$v_\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

Sum over all possible actions, next states, and rewards:
$$v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] \right]$$

Since $\mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] = v_\pi(s')$:

$$\boxed{v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]}$$

### Component Breakdown
* $\pi(a \mid s)$: Policy — probability of taking action $a$ in state $s$.
* $p(s', r \mid s, a)$: Transition dynamics — probability of landing in $s'$ with reward $r$.
* $r$: Immediate reward.
* $\gamma v_\pi(s')$: Discounted value of successor state.

### Backup Diagram

```
        ( s )           <- Current State s
        / | \
       /  |  \          <- Policy chooses action 'a' with prob π(a|s)
     (a) (a) (a)        <- Action choices
     / \   \   / \
    /   \   \ /   \     <- Environment transitions with prob p(s',r|s,a)
  (s') (s') (s') (s')   <- Next States s'
```

### Worked Example (3 actions, 6 successor states)

```
                     ( s )
                    /  |  \
            _______/   |   \_______
           /           |           \
        [ a1 ]      [ a2 ]      [ a3 ]
         /  \        /  \        /  \
        /    \      /    \      /    \
      (s1)  (s2)  (s3)  (s4)  (s5)  (s6)
      [r1]  [r2]  [r3]  [r4]  [r5]  [r6]
```

$$v_\pi(s) = \underbrace{\pi(a_1 \mid s) \big( p(s_1, r_1 \mid s, a_1)[r_1 + \gamma v_\pi(s_1)] + p(s_2, r_2 \mid s, a_1)[r_2 + \gamma v_\pi(s_2)] \big)}_{\text{Term 1}} + \text{Term 2} + \text{Term 3}$$

Each term sums over successor states reachable by that action, weighted by transition probabilities.

---

## 11. Optimal Value Functions

The goal: find a policy that maximizes expected return. A policy $\pi$ is better than $\pi'$ if:
$$\pi \ge \pi' \iff v_\pi(s) \ge v_{\pi'}(s) \quad \forall s \in \mathcal{S}$$

There always exists at least one **optimal policy** $\pi_*$.

### Optimal State-Value Function
$$v_*(s) = \max_{\pi} v_\pi(s) \quad \forall s \in \mathcal{S}$$

### Optimal Action-Value Function
$$q_*(s, a) = \max_{\pi} q_\pi(s, a) \quad \forall s \in \mathcal{S}, a \in \mathcal{A}(s)$$

The relationship:
$$v_*(s) = \max_{a \in \mathcal{A}(s)} q_*(s, a)$$

### The Bellman Optimality Equation

$$\boxed{v_*(s) = \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]}$$

Key difference from the standard Bellman equation: instead of averaging over actions (weighted by $\pi(a|s)$), we **maximize** — assuming the agent will greedily choose the best action.

---

## 12. Worked Example: Recycling Robot Bellman Optimality Equations

Using the recycling robot from §7 (Case Study 3), we write explicit Bellman optimality equations.

### For state $s = \text{High}$:
Agent chooses Search or Wait — take the max:

$$v_*(\text{High}) = \max \begin{cases}
\alpha \left[ R_{\text{search}} + \gamma v_*(\text{High}) \right] + (1-\alpha) \left[ R_{\text{search}} + \gamma v_*(\text{Low}) \right], \\
R_{\text{wait}} + \gamma v_*(\text{High})
\end{cases}$$

### For state $s = \text{Low}$:
Agent chooses Search, Wait, or Recharge — take the max:

$$v_*(\text{Low}) = \max \begin{cases}
\beta \left[ R_{\text{search}} + \gamma v_*(\text{Low}) \right] + (1-\beta) \left[ -3 + \gamma v_*(\text{High}) \right], \\
R_{\text{wait}} + \gamma v_*(\text{Low}), \\
\gamma v_*(\text{High})
\end{cases}$$

### Convergent Values (via value iteration, → Chapter 4)
* $v_*(\text{High}) \approx 25.42$
* $v_*(\text{Low}) \approx 22.88$

```
  Value
   ^
30 |
25 |---------------------------- v*(High) ≈ 25.42
20 |-------------------------- v*(Low) ≈ 22.88
15 |
10 |   /
 5 |  /
 0 +----------------------------> Iterations
   0   20  40  60  80  100  120
```

Substituting these values back reveals which action yields the max in each state → the **optimal policy** $\pi_*$.

---

## Key Takeaways

1. **MDPs** formalize the sequential decision-making problem: states, actions, transitions, rewards.
2. **The Markov property** says the current state is all you need — history is irrelevant.
3. **Value functions** ($v_\pi$, $q_\pi$) quantify long-term expected return.
4. **The Bellman equation** expresses value recursively: current value = immediate reward + discounted future value.
5. **The Bellman optimality equation** defines what "optimal" means: choose the action that maximizes expected value.

---

*We now know what "optimal" looks like mathematically (Bellman optimality equation). But how do we actually compute it? If we know the full model $p(s',r|s,a)$, we can use Dynamic Programming → Chapter 4. If we don't have a model, we learn from experience → Chapter 5.*
