# Lecture Notes: Value Functions in Reinforcement Learning

---

## 1. Introduction to Value Functions and MDPs

In reinforcement learning (RL), we model the interaction between an agent and its environment as a **Markov Decision Process (MDP)**. 

### Key Concepts Recap:
* **Markov Property:** A state signal possesses the Markov property if it retains all relevant information from the past. This means the future transition depends only on the current state and action, not on the sequence of states that preceded it:
  $$p(s', r \mid s, a) = \Pr\{S_{t} = s', R_{t} = r \mid S_{t-1} = s, A_{t-1} = a\}$$
* **Markov Decision Process (MDP):** A reinforcement learning task that satisfies the Markov property is called an MDP. If the state and action spaces are finite, it is a finite MDP.
* **Significance in Large Language Models (LLMs):** Understanding value functions is crucial for modern alignment techniques (such as RLHF/RLAIF used in models like DeepSeek), where reinforcement learning is applied to optimize LLM outputs based on reward signals.

---

## 2. Defining "Value" in Reinforcement Learning

The concept of "value" is one of the four fundamental elements of any reinforcement learning system (alongside policy, reward signal, and optionally, a model of the environment).

### What is Value?
* Unlike an immediate **reward** (which is a short-term signal indicating what is good in the immediate moment), the **value** of a state is the total expected cumulative reward an agent can accumulate in the long run, starting from that state.
* An agent may choose a transition that yields a low or even negative immediate reward if it leads to a state with a much higher long-term value.

### Intuitive Analogy: Player Performance in a Tournament
Consider a cricket team selecting a player for a 14-match tournament:
* In the first 2 matches, the player performs poorly (representing low immediate rewards).
* However, the team management chooses not to drop the player because they believe the player's overall **value** is high. 
* They expect that over the course of the 14 matches, the player's cumulative performance (long-term return) will be highly beneficial. 
* This demonstrates how long-term value can override temporary, low immediate rewards.

---

## 3. The State-Value Function, $v_\pi(s)$

The **state-value function** of a state $s$ under a policy $\pi$, denoted as $v_\pi(s)$, is the expected return starting from state $s$ and following policy $\pi$ thereafter.

### Mathematical Formulation:
$$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

For a discounted task, the return $G_t$ is defined as:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Substituting $G_t$ into the state-value equation yields:
$$v_\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]$$

### Definition of Terms:
* **$v_\pi(s)$**: State-value function for state $s$ under policy $\pi$.
* **$\mathbb{E}_\pi[\cdot]$**: Expected value of a random variable given that the agent follows policy $\pi$.
* **$G_t$**: Expected return (cumulative discounted reward) at time step $t$.
* **$S_t = s$**: The starting state condition at time step $t$.
* **$\gamma$**: Discount rate ($0 \le \gamma \le 1$), which determines the present value of future rewards. A reward received $k$ steps in the future is worth only $\gamma^{k-1}$ times its immediate value.

---

## 4. The Action-Value Function, $q_\pi(s, a)$

The **action-value function** of taking action $a$ in state $s$ under a policy $\pi$, denoted as $q_\pi(s, a)$, is the expected return starting from state $s$, taking action $a$, and thereafter following policy $\pi$.

### Mathematical Formulation:
$$q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

Substituting the discounted return $G_t$ yields:
$$q_\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a \right]$$

### State-Value vs. Action-Value:
* **State-Value ($v_\pi(s)$)** evaluates the quality of being in a state $s$ under policy $\pi$.
* **Action-Value ($q_\pi(s, a)$)** evaluates the quality of performing a specific action $a$ from state $s$, allowing the agent to compare the outcomes of different choices before committing to a policy.

---

## 5. The Bellman Equation for $v_\pi(s)$

A fundamental property of value functions in reinforcement learning is that they satisfy recursive relationships. For any policy $\pi$ and state $s$, the Bellman equation expresses the value of the current state in terms of the values of its possible successor states.

### Derivation:
Starting with the definition of $v_\pi(s)$:
$$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

Expand the return $G_t$ into its immediate reward $R_{t+1}$ and the discounted future return $\gamma G_{t+1}$:
$$v_\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

To compute this expectation, we sum over all possible actions $a$, next states $s'$, and rewards $r$, weighted by their probabilities of occurrence under policy $\pi$ and transition dynamics $p$:
$$v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} \sum_{r} p(s', r \mid s, a) \left[ r + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] \right]$$

Since $\mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] = v_\pi(s')$, we obtain the **Bellman Equation for $v_\pi$**:
$$v_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]$$

### Component Breakdown:
* **$\pi(a \mid s)$**: The policy; the probability of taking action $a$ while in state $s$.
* **$p(s', r \mid s, a)$**: The transition probability; the joint probability of transitioning to state $s'$ with reward $r$, given current state $s$ and action $a$.
* **$r$**: The immediate reward received upon transitioning.
* **$\gamma v_\pi(s')$**: The discounted value of the successor state $s'$.

---

## 6. Understanding Backup Diagrams

**Backup diagrams** are visual representations of the recursive update relationships defined by the Bellman equations. They trace relationships "back" from successor states to a parent state.

### State-Value Backup Diagram:

```
        ( s )           <- Current State s
        / | \
       /  |  \          <- Policy chooses action 'a' with prob π(a|s)
     (a) (a) (a)        <- Action choices
     / \   \   / \
    /   \   \ /   \     <- Environment transitions to next state s' with prob p(s',r|s,a)
  (s') (s') (s') (s')   <- Next States s'
```

### Concrete Step-by-Step State-Value Calculation:

Consider an agent in state $s$ with three possible actions $\{a_1, a_2, a_3\}$. Taking these actions leads to next states $\{s_1, s_2, s_3, s_4, s_5, s_6\}$ and immediate rewards $\{r_1, r_2, r_3, r_4, r_5, r_6\}$ as shown in the diagram:

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

We can write the calculation of $v_\pi(s)$ as the sum of three terms, one for each action branch:

$$v_\pi(s) = \text{Term 1} + \text{Term 2} + \text{Term 3}$$

* **Term 1 (for action $a_1$):**
  $$\text{Term 1} = \pi(a_1 \mid s) \left( p(s_1, r_1 \mid s, a_1)[r_1 + \gamma v_\pi(s_1)] + p(s_2, r_2 \mid s, a_1)[r_2 + \gamma v_\pi(s_2)] \right)$$
* **Term 2 (for action $a_2$):**
  $$\text{Term 2} = \pi(a_2 \mid s) \left( p(s_3, r_3 \mid s, a_2)[r_3 + \gamma v_\pi(s_3)] + p(s_4, r_4 \mid s, a_2)[r_4 + \gamma v_\pi(s_4)] \right)$$
* **Term 3 (for action $a_3$):**
  $$\text{Term 3} = \pi(a_3 \mid s) \left( p(s_5, r_5 \mid s, a_3)[r_5 + \gamma v_\pi(s_5)] + p(s_6, r_6 \mid s, a_3)[r_6 + \gamma v_\pi(s_6)] \right)$$

---

## 7. Optimal Value Functions and the Bellman Optimality Equation

The objective of an RL agent is to find a policy that maximizes the expected return over the long term. 

### Policy Ordering:
A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ ($\pi \ge \pi'$) if and only if its expected return is greater than or equal to that of $\pi'$ for all states:
$$\pi \ge \pi' \iff v_\pi(s) \ge v_{\pi'}(s) \quad \forall s \in \mathcal{S}$$

There is always at least one policy that is better than or equal to all other policies. This is the **optimal policy**, denoted as $\pi_*$.

### Optimal State-Value Function ($v_*$):
The optimal state-value function is the maximum value function over all policies:
$$v_*(s) = \max_{\pi} v_\pi(s) \quad \forall s \in \mathcal{S}$$

### Optimal Action-Value Function ($q_*$):
The optimal action-value function is the maximum action-value function over all policies:
$$q_*(s, a) = \max_{\pi} q_\pi(s, a) \quad \forall s \in \mathcal{S}, a \in \mathcal{A}(s)$$

Because $q_*(s, a)$ represents the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy, we can express $v_*(s)$ in terms of $q_*(s, a)$ as:
$$v_*(s) = \max_{a \in \mathcal{A}(s)} q_*(s, a)$$

### The Bellman Optimality Equation for $v_*(s)$:
This equation formalizes the intuition that the value of a state under an optimal policy must equal the expected return from the best action for that state:

$$v_*(s) = \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]$$

Unlike the standard Bellman equation, the Bellman Optimality Equation does not depend on the policy's action probabilities ($\pi(a|s)$). Instead, it assumes the agent will greedily choose the action that maximizes the expected value.

---

## 8. Case Study: The Recycling Robot

To illustrate the Bellman Optimality Equation, we analyze the **Recycling Robot** environment.

```
       [Search]  <-- probability alpha (battery stays high)
      /          <-- probability 1-alpha (battery goes low)
    (High)
      \
       [Wait]    <-- probability 1 (battery stays high)

       [Search]  <-- probability beta (battery stays low)
      /          <-- probability 1-beta (robot runs out of power -> charging state)
    (Low)
      \
       [Wait]    <-- probability 1 (battery stays low)
      /
       [Recharge] <-- probability 1 (battery becomes high)
```

### Environment Parameters:
* **States ($\mathcal{S}$):** $\{\text{High}, \text{Low}\}$ (battery charge level).
* **Actions ($\mathcal{A}$):**
  * In state $\text{High}$: $\{\text{Search}, \text{Wait}\}$
  * In state $\text{Low}$: $\{\text{Search}, \text{Wait}, \text{Recharge}\}$
* **Transition and Reward Specifications:**
  * **Search from High:** stays High with probability $\alpha$ (reward $R_{\text{search}}$); becomes Low with probability $1-\alpha$ (reward $R_{\text{search}}$).
  * **Wait from High:** stays High with probability $1$ (reward $R_{\text{wait}}$).
  * **Search from Low:** stays Low with probability $\beta$ (reward $R_{\text{search}}$); depletes completely with probability $1-\beta$ (requires rescue, transitioning to High with reward $-3$).
  * **Wait from Low:** stays Low with probability $1$ (reward $R_{\text{wait}}$).
  * **Recharge from Low:** transitions to High with probability $1$ (reward $0$).

### Transition Dynamics Table:

| Current State ($s$) | Action ($a$) | Next State ($s'$) | Transition Probability $p(s' \mid s, a)$ | Reward $r(s, a, s')$ |
| :-------------------:| :------------:| :-----------------:| :----------------------------------------:| :--------------------:|
| High                | Search       | High              | $\alpha$                                 | $R_{\text{search}}$  |
| High                | Search       | Low               | $1-\alpha$                               | $R_{\text{search}}$  |
| Low                 | Search       | Low               | $\beta$                                  | $R_{\text{search}}$  |
| Low                 | Search       | High              | $1-\beta$                                | $-3$                 |
| High                | Wait         | High              | $1$                                      | $R_{\text{wait}}$    |
| Low                 | Wait         | Low               | $1$                                      | $R_{\text{wait}}$    |
| Low                 | Recharge     | High              | $1$                                      | $0$                  |

---

### Constructing Bellman Optimality Equations for the Recycling Robot

We write out the explicit Bellman optimality equations for each state.

#### 1. For state $s = \text{High}$:
The agent can choose either $a = \text{Search}$ or $a = \text{Wait}$. We take the maximum of these two options:

$$v_*(\text{High}) = \max \begin{cases} 
\text{Term 1} & \text{(Search)} \\
\text{Term 2} & \text{(Wait)} 
\end{cases}$$

* **Term 1 (Search from High):**
  $$\text{Term 1} = \alpha \left[ R_{\text{search}} + \gamma v_*(\text{High}) \right] + (1-\alpha) \left[ R_{\text{search}} + \gamma v_*(\text{Low}) \right]$$
* **Term 2 (Wait from High):**
  $$\text{Term 2} = 1 \cdot \left[ R_{\text{wait}} + \gamma v_*(\text{High}) \right]$$

Thus:
$$v_*(\text{High}) = \max \begin{cases}
\alpha \left[ R_{\text{search}} + \gamma v_*(\text{High}) \right] + (1-\alpha) \left[ R_{\text{search}} + \gamma v_*(\text{Low}) \right], \\
R_{\text{wait}} + \gamma v_*(\text{High})
\end{cases}$$

#### 2. For state $s = \text{Low}$:
The agent can choose $a = \text{Search}$, $a = \text{Wait}$, or $a = \text{Recharge}$. We take the maximum of these three options:

$$v_*(\text{Low}) = \max \begin{cases} 
\text{Term 1} & \text{(Search)} \\
\text{Term 2} & \text{(Wait)} \\
\text{Term 3} & \text{(Recharge)}
\end{cases}$$

* **Term 1 (Search from Low):**
  $$\text{Term 1} = \beta \left[ R_{\text{search}} + \gamma v_*(\text{Low}) \right] + (1-\beta) \left[ -3 + \gamma v_*(\text{High}) \right]$$
* **Term 2 (Wait from Low):**
  $$\text{Term 2} = 1 \cdot \left[ R_{\text{wait}} + \gamma v_*(\text{Low}) \right]$$
* **Term 3 (Recharge from Low):**
  $$\text{Term 3} = 1 \cdot \left[ 0 + \gamma v_*(\text{High}) \right]$$

Thus:
$$v_*(\text{Low}) = \max \begin{cases}
\beta \left[ R_{\text{search}} + \gamma v_*(\text{Low}) \right] + (1-\beta) \left[ -3 + \gamma v_*(\text{High}) \right], \\
R_{\text{wait}} + \gamma v_*(\text{Low}), \\
\gamma v_*(\text{High})
\end{cases}$$

---

### Google Colab Implementation Results

Running a value iteration program with specific environment parameters yields the convergent optimal state-values:

* **Optimal value at High ($v_*(\text{High})$):** $\approx 25.42$
* **Optimal value at Low ($v_*(\text{Low})$):** $\approx 22.88$

#### Value Function Convergence Plot:
Below is a conceptual representation of how the state values converge over iterations during the dynamic programming solution:

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

By substituting these convergent values back into the expectation terms of the Bellman Optimality Equation, we can identify which action yielded the maximum value in each state, thereby revealing the **optimal policy** ($\pi_*$).