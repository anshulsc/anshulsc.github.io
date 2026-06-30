# Lecture Notes: Finite Markov Decision Processes (MDPs)

---

## 1. Introduction to Finite Markov Decision Processes (MDPs)
Finite Markov Decision Processes (MDPs) serve as the formal mathematical framework for defining reinforcement learning (RL) problems. Unlike simpler reward-selection frameworks such as multi-armed bandits, MDPs incorporate the concept of **associative search**—where the agent must not only choose the best action for a given situation but also evaluate how its actions influence future states and future rewards. 

Understanding MDPs requires a formal understanding of how an autonomous agent interacts with its surrounding environment through state transitions, actions, rewards, and long-term goals.

---

## 2. The Agent-Environment Interface
The interaction between an agent and its environment is a continuous, step-by-step decision-making process. This relationship is formalized through the **Agent-Environment Interface**.

```
             Action (At)
        +---------------->
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
The interaction occurs at discrete time steps, $t = 0, 1, 2, 3, \dots$:
1. **State Observation ($S_t$):** At time step $t$, the agent receives some representation of the environment's state, $S_t \in \mathcal{S}$, where $\mathcal{S}$ is the set of all possible states.
2. **Action Selection ($A_t$):** Based on the state $S_t$, the agent selects an action $A_t \in \mathcal{A}(s)$, where $\mathcal{A}(s)$ denotes the set of actions available in state $s$.
3. **Transition and Reward ($S_{t+1}, R_{t+1}$):** One time step later, as a consequence of the agent's action, the environment transitions to a new state $S_{t+1} \in \mathcal{S}$ and passes a numerical reward $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ back to the agent.

### Notational Nomenclature
* **Capital Letters ($S_t, A_t, R_t$):** Represent random variables associated with time step $t$.
* **Lowercase Letters ($s, a, r$):** Represent specific, concrete values or realizations within their respective sets (e.g., a state $s \in \mathcal{S}$, an action $a \in \mathcal{A}(s)$, and a reward $r \in \mathcal{R}$).

---

## 3. The Agent's Policy ($\pi$)
The agent's decision-making strategy is formalized as a **policy**. A policy is a mapping from the state space to a probability distribution over the action space.

$$\pi_t(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$$

* $\pi_t(a \mid s)$ represents the probability that the agent selects action $A_t = a$ when the current state is $S_t = s$.
* If the policy is stationary (does not change over time), it is simply written as $\pi(a \mid s)$.
* Policies can be deterministic (mapping a state directly to a single action) or stochastic (assigning probabilities to multiple possible actions).

---

## 4. Goals and Rewards
A defining feature of reinforcement learning is its reliance on a scalar reward signal to formalize the concept of a goal. This is encapsulated by the **Reward Hypothesis**:

> *All of what we mean by goals and purposes can be well-formulated as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).*

### Formulating Goals Correctly
To ensure an agent learns the desired behavior, rewards must be structured around *what* the agent should achieve, rather than *how* it should achieve it.

* **The Danger of Sub-Goals:** Designers are often tempted to reward intermediate milestones (sub-goals) to guide the agent. However, this can introduce unintended behaviors. 
  * *Example (Chess):* If you reward the agent $+1$ for capturing an opponent's queen (a sub-goal), it may develop a policy that prioritizes capturing the queen even if doing so results in losing the match. 
  * *Correct Formulation:* Reward the agent $+1$ only for winning the game, $-1$ for losing, and $0$ for a draw. This forces the agent to align all actions with the ultimate objective of winning.

---

## 5. Returns and expected Return ($G_t$)
The agent's objective is to select actions that maximize its long-term cumulative reward. This cumulative measure is formalized as the **Return ($G_t$)**.

### Option A: Episodic Tasks
In episodic tasks, the agent-environment interaction naturally breaks down into self-contained sequences called **episodes** (e.g., a game of chess, navigating a maze). Each episode terminates in a special state called the **terminal state** at a final time step $T$. 

For episodic tasks, the return is calculated as:

$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T$$

### Option B: Continuing Tasks and the Need for Discounting
In continuing tasks, the agent-environment interaction does not have a natural end and can go on indefinitely ($T = \infty$, e.g., an ongoing planetary rover mission, continuous industrial process control). 

If we use the episodic return formula where $T = \infty$, the return $G_t$ could diverge to infinity:

$$G_t = \sum_{k=0}^{\infty} R_{t+k+1} \to \infty$$

To prevent infinite returns and prioritize immediate rewards over distant future rewards, we introduce a **discount rate ($\gamma$)**, where $\gamma \in [0, 1)$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

#### Intuitive Analogy for Discounting
Consider the value of money over time: $100$ Rupees today is typically worth more to you than $100$ Rupees promised five years from now, partly due to inflation. Similarly, in RL:
* **Immediate rewards** are valued at face value ($1 \cdot R_{t+1}$).
* **Future rewards** are scaled down exponentially ($\gamma^k R_{t+k+1}$).

#### The Role of the Discount Factor ($\gamma$)
* **Myopic (Short-Sighted) Agent ($\gamma = 0$):** The agent only cares about maximizing the immediate reward $R_{t+1}$. It ignores the future consequences of its current actions.
* **Far-Sighted Agent ($\gamma \to 1$):** The agent values future rewards almost as much as immediate rewards, encouraging it to plan long-term strategies.

---

## 6. The Markov Property
A state signal $S_t$ is said to possess the **Markov Property** if it contains all relevant information from the past required to predict the future state and reward. In a Markov state, the history of how the agent arrived at the current state is irrelevant for predicting what will happen next.

### Mathematical Formulation
For any general process, the probability of the next state and reward depends on the entire history of states, actions, and rewards:

$$\mathbb{P}(R_{t+1} = r, S_{t+1} = s' \mid S_t, A_t, R_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0)$$

A state signal $S_t$ has the **Markov Property** if and only if:

$$\mathbb{P}(R_{t+1} = r, S_{t+1} = s' \mid S_t = s, A_t = a) = \mathbb{P}(R_{t+1} = r, S_{t+1} = s' \mid S_t, A_t, R_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0)$$

This means the current state $S_t$ and selected action $A_t$ are sufficient to describe the transition dynamics to $S_{t+1}$ and $R_{t+1}$.

### Examples of Markovian vs. Non-Markovian States

| Example | State Representation | Markovian Status | Explanation |
| :--- | :--- | :--- | :--- |
| **Chess** | The current board configuration of chess pieces. | **Markovian** | You do not need to know the sequence of moves that led to this board layout to determine the optimal next move. All necessary information is visible on the board. |
| **Physics** (Cannonball) | Current position ($p$) and velocity ($v$). | **Markovian** | Under classical mechanics, knowing the current position and velocity is sufficient to predict the cannonball's trajectory. You do not need its previous path. |
| **Conversations** | The last spoken sentence. | **Non-Markovian** | To formulate a coherent response, you usually need the context of the entire past conversation, not just the last sentence. |
| **Draw Poker** | Just the cards in your hand. | **Non-Markovian** | To play optimally, you must consider the history of other players' bets, cards discarded, and bluffing patterns over previous rounds. |

---

## 7. Mathematical Formulations of MDPs
An RL task that satisfies the Markov property is classified as a **Markov Decision Process (MDP)**. If the state, action, and reward spaces are finite, the process is a **Finite MDP**.

The dynamics of a Finite MDP are governed by a joint probability distribution:

$$p(s', r \mid s, a) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a)$$

From this single function $p(s', r \mid s, a)$, we can derive other useful quantities:

### 1. State-Transition Probabilities
The probability of transitioning to state $s'$ from state $s$ by taking action $a$:

$$p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)$$

### 2. Expected Rewards for State-Action Pairs
The expected immediate reward received after taking action $a$ in state $s$:

$$r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a)$$

### 3. Expected Rewards for State-Action-Next-State Triples
The expected reward given that the agent was in state $s$, took action $a$, and transitioned to state $s'$:

$$r(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r \mid s, a)}{p(s' \mid s, a)}$$

---

## 8. Practical Case Studies

### Case Study 1: The Bioreactor Control Problem
A bioreactor is a vessel used to grow organisms (like yeast or bacteria) to produce useful chemical products.

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

* **Objective:** Maximize the rate at which useful chemicals are produced.
* **State Space (Vector):** Sensor readings including temperature, pressure, pH levels, and thermocouple measurements.
* **Action Space (Vector):** Control inputs such as target temperature (activating a heating element) and target stirring rate (activating motor speeds).
* **Reward (Scalar):** The moment-to-moment measurement of the rate of chemical production.

---

### Case Study 2: Pick-and-Place Robot Arm
An industrial robotic arm tasked with picking up blocks of different colors and placing them smoothly on a conveyor belt.

* **Objective:** Pick and place blocks successfully while maintaining smooth joint movement.
* **State Space (Vector):** Joint angles ($\theta_1, \theta_2$) and angular velocities ($\omega_1, \omega_2$).
* **Action Space (Vector):** Control voltages applied to the electric motors at each joint ($V_1, V_2$).
* **Reward (Scalar):**
  * $+1$ for each object successfully picked and placed.
  * $-0.02$ (a small negative penalty) for jerky or unsteady movements to encourage smooth trajectories.

---

### Case Study 3: The Recycling Robot (Finite MDP)
An autonomous robot is designed to collect empty soda cans in an office space. It has a battery that can be in two levels: **High** or **Low**.

```
                   +--------+  Search (prob alpha)  +--------+
                   |        | --------------------> |        |
                   |  HIGH  |                       |  LOW   |
                   |        | <-------------------- |        |
                   +--------+   Recharge (prob 1)   +--------+
                     |    ^                           |    ^
        Wait (prob 1)|    | Wait (prob 1)             |    | Search (prob 1-beta)
                     v    |                           v    | (depleted battery)
                   +--------+                       +--------+
                   |  HIGH  |                       | RESCUE | (Reward = -3)
                   +--------+                       +--------+
```

#### State Space
$$\mathcal{S} = \{\text{high}, \text{low}\}$$

#### Action Space
* $\mathcal{A}(\text{high}) = \{\text{search}, \text{wait}\}$
* $\mathcal{A}(\text{low}) = \{\text{search}, \text{wait}, \text{recharge}\}$

#### Parameters
* **$\alpha$:** Probability that the battery remains *high* after a period of searching.
* **$1 - \alpha$:** Probability that searching drains the battery from *high* to *low*.
* **$\beta$:** Probability that the battery remains *low* after a period of searching.
* **$1 - \beta$:** Probability that searching drains the battery completely, causing it to shut down and require manual rescue.
* **$r_{\text{search}}$:** Expected number of cans found while actively searching.
* **$r_{\text{wait}}$:** Expected number of cans found while waiting (people bringing cans to the robot).

#### Complete State Transition and Reward Table

| Current State ($s$) | Action ($a$) | Next State ($s'$) | Transition Probability $p(s' \mid s, a)$ | Expected Reward $r(s, a, s')$ |
| :--- | :--- | :--- | :--- | :--- |
| **high** | search | high | $\alpha$ | $r_{\text{search}}$ |
| **high** | search | low | $1 - \alpha$ | $r_{\text{search}}$ |
| **high** | wait | high | $1$ | $r_{\text{wait}}$ |
| **high** | wait | low | $0$ | $r_{\text{wait}}$ |
| **low** | search | high | $0$ | $r_{\text{search}}$ |
| **low** | search | low | $\beta$ | $r_{\text{search}}$ |
| **low** | search | high (Rescue) | $1 - \beta$ | $-3$ |
| **low** | wait | high | $0$ | $r_{\text{wait}}$ |
| **low** | wait | low | $1$ | $r_{\text{wait}}$ |
| **low** | recharge | high | $1$ | $0$ |
| **low** | recharge | low | $0$ | $0$ |

---

## 9. The Pole-Balancing Task (Cart-Pole)
A classic control benchmark problem in RL involves balancing an unjointed pole on top of a moving cart. The cart moves along a track of finite length.

```
                  || <-- Pole (Angle theta)
                  ||
              +---|---+
              |  [O]  | <-- Cart (Position p)
              +-------+
               O     O  <-- Wheels
         ----------------------- <-- Track
```

We can formulate the cart-pole task in two different ways depending on our design requirements.

### Option 1: Formulating as an Episodic Task
* **Termination Conditions:** The episode ends (terminates) when either:
  1. The pole falls past a critical angle (e.g., $|\theta| > 15^\circ$).
  2. The cart runs off the edge of the track (e.g., $|p| > 2.4\text{ meters}$).
* **Reward:** 
  * $+1$ for every time step the pole remains upright and the cart is on the track.
  * $0$ at the step of failure (terminal state).
* **Return Maximization:** 
  $$G_t = R_{t+1} + R_{t+2} + \dots + R_T = 1 + 1 + \dots + 1 = \text{Total time steps of success}$$
  Maximizing this return encourages the agent to keep the pole balanced for as long as possible.

### Option 2: Formulating as a Continuing Task (with Discounting)
* **Interaction Horizon:** The task is framed as ongoing with no terminal state ($T = \infty$).
* **Reward:**
  * $-1$ if the pole falls over or if the cart goes off-track (failure/fall event).
  * $0$ otherwise.
* **Return Maximization:**
  $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
  Since any failure incurs a reward of $-1$, maximizing this discounted sum of negative rewards encourages the agent to avoid failure events for as long as possible.