# Lecture Notes: Multi-Arm Bandits
**Course:** Reasoning LLMs from Scratch  
**Lecture 5:** Multi-Arm Bandits

---

## 1. Introduction and Context

In the study of Reinforcement Learning (RL), we examine how agents learn to make decisions by interacting with an environment to maximize some notion of cumulative reward. 

### Recap of Prior Concepts
To understand multi-arm bandits, it is helpful to recall the core components of reinforcement learning established in previous lectures:
*   **Supervised vs. Unsupervised vs. Reinforcement Learning:** Unlike supervised learning (which learns from labeled training data) or unsupervised learning (which finds hidden patterns in unlabeled data), reinforcement learning relies on a trial-and-error paradigm where the agent learns from the consequences of its actions.
*   **Key Elements of RL:**
    *   **Policy ($\pi$):** The agent's decision-making strategy (mapping from states to actions).
    *   **Reward Signal ($R$):** The immediate feedback from the environment defining the goal.
    *   **Value Function ($V$ or $Q$):** The long-term expected return from a state or action.
    *   **Model of the Environment:** The agent's representation of how the world behaves (transitions and rewards).

While complex RL problems (such as chess or navigation) involve sequential decision-making across multiple states, the **Multi-Arm Bandit** framework serves as a simplified, single-state formulation that isolates one of RL's most fundamental challenges: the **exploration-exploitation trade-off**.

---

## 2. Defining the Bandit Problem

### 2.1 The One-Armed Bandit
The term "one-armed bandit" is a colloquialism for a traditional slot machine found in casinos. 

```
   [ Slot Machine ]
   +-------------+
   |   (7) (7) (7|   <-- Display Symbols
   +-------------+
   |             |   _
   |             |  (_)  <-- The "One Arm" (Lever)
   |             |   |
   |             |---|
   +-------------+
```

*   **Mechanism:** A player inserts money, pulls the single lever ("one arm"), and the machine displays a set of symbols.
    *   If the symbols align, the player receives a financial payout (reward).
    *   If they do not align, the player loses their stake.
*   **Etymology:** It is called a "bandit" because the statistical probability of payout is designed such that, over time, the machine systematically deprives the player of money, akin to a thief.

### 2.2 The Multi-Arm Bandit (k-Armed Bandit)
If a one-armed bandit has a single lever, a **multi-arm bandit** (or $k$-armed bandit) represents a setup with $k$ distinct levers (actions).

```
          [ Multi-Arm Bandit Machine ]
          +--------------------------+
          |         (7) (7) (7)      |
          +--------------------------+
            |       |          |       |
           _|_     _|_        _|_     _|_
          (_)     (_)        (_)     (_)   <-- "k" Levers (Actions)
           1       2         ...      k
```

*   **The Scenario:** You stand before a machine with $k$ different levers. Each lever has its own stationary probability distribution of rewards.
*   **The Core Problem:** You do not know the underlying reward distributions of the levers beforehand.
*   **The Goal:** You want to maximize your total expected reward over a series of trials (time steps $t = 1, 2, 3, \dots, T$).

---

## 3. Action-Value Methods

To maximize rewards, we must estimate the value of each action. **Action-Value Methods** are algorithms that estimate the values of actions and use these estimates to make decisions.

### 3.1 Mathematical Definitions
Let $a$ denote a specific action (pulling a specific lever).

*   **True Value ($q_*(a)$):** The actual expected reward of action $a$. This is an objective, hidden property of the environment.
    $$q_*(a) = \mathbb{E}[R_t \mid A_t = a]$$
    *Where $A_t$ is the action chosen at time $t$, and $R_t$ is the resulting reward.*

*   **Estimated Value ($Q_t(a)$):** The agent’s estimate of the value of action $a$ at time step $t$. We want $Q_t(a)$ to be as close to $q_*(a)$ as possible.

### 3.2 The Sample-Average Method
One straightforward way to estimate the value of an action is by averaging the rewards actually received when that action was selected in the past:

$$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}(A_i = a)}{\sum_{i=1}^{t-1} \mathbb{I}(A_i = a)}$$

Where:
*   $\mathbb{I}(\text{predicate})$ is an indicator function that equals $1$ if the predicate is true, and $0$ otherwise.
*   The denominator represents the number of times action $a$ has been chosen prior to time $t$, denoted as $N_t(a)$.

If $N_t(a) = 0$, we define $Q_t(a)$ to be a default initial value (e.g., $0$).

By the **Law of Large Numbers**, as an action is selected many times ($N_t(a) \to \infty$), its estimated value converges to its true value:
$$\lim_{N_t(a) \to \infty} Q_t(a) = q_*(a)$$

---

## 4. The Exploration vs. Exploitation Dilemma

Imagine a 4-armed bandit problem where, at time step $t$, your estimated values are:

| Action ($a$) | Estimated Value ($Q_t(a)$) |
|---|---|
| Action 1 | $Q_t(1) = 1.5$ |
| Action 2 | $Q_t(2) = 3.0$ |
| Action 3 | $Q_t(3) = 2.5$ |
| Action 4 | $Q_t(4) = 2.0$ |

Based on this table, you have two potential strategies for your next move:

### 4.1 Exploitation (Greedy Choice)
*   **Definition:** Selecting the action with the highest estimated value to maximize immediate reward.
*   **In this scenario:** You choose **Action 2** ($Q_t(2) = 3.0$).
*   **Characteristics:**
    *   Exploits current knowledge.
    *   Provides short-term optimization.
    *   Does not gather new information about potentially superior actions (e.g., what if Action 3 actually has a true value $q_*(3) = 5.0$, but a poor initial sample made its estimate $2.5$?).

### 4.2 Exploration (Non-Greedy Choice)
*   **Definition:** Selecting a suboptimal estimated action to gather more information about its true value.
*   **In this scenario:** You choose **Action 1, 3, or 4**.
*   **Characteristics:**
    *   Sacrifices immediate reward.
    *   Improves the accuracy of value estimates.
    *   Enables the discovery of better options, potentially leading to much higher long-term cumulative rewards.

---

## 5. Balancing Exploration and Exploitation: The $\epsilon$-Greedy Algorithm

The simplest strategy to balance exploration and exploitation is the **$\epsilon$-greedy** (epsilon-greedy) algorithm.

```
                  [ Choose Action ]
                         |
                +--------+--------+
                |                 |
       With Prob. 1 - epsilon     With Prob. epsilon
                |                 |
         [ EXPLOITATION ]   [ EXPLORATION ]
         Choose Greedy      Choose Random
         Action             Action (from all)
```

### 5.1 The Rule
*   With a small probability $\epsilon$ (where $0 < \epsilon \ll 1$), select an action **at random** from all available actions with equal probability.
*   With probability $1 - \epsilon$, select the **greedy action** (the action with the maximum estimated value: $A_t = \operatorname{argmax}_a Q_t(a)$).

### 5.2 Behavioral Impact of Epsilon ($\epsilon$)
*   **$\epsilon = 0$ (Purely Greedy):** The agent always exploits. It is highly susceptible to getting stuck in local optima.
*   **$\epsilon = 0.1$:** Out of every 100 steps, approximately 90 steps are exploitative, and 10 steps are exploratory. This agent learns quickly but will always waste $10\%$ of its steps exploring, even after values are well-estimated.
*   **$\epsilon = 0.01$:** The agent explores less frequently. It takes longer to identify the optimal action, but once found, it exploits it $99\%$ of the time, resulting in higher asymptotic performance than the $\epsilon = 0.1$ variant in the long run.

---

## 6. Case Study: 10-Armed Testbed Simulation

To empirically analyze these methods, we simulate a standard **10-Armed Bandit** testbed.

### 6.1 Simulation Design
*   **Number of Arms ($k$):** $10$.
*   **True Values ($q_*(a)$):** For each of the 10 actions, the true value $q_*(a)$ is selected from a standard normal distribution (Gaussian with mean $\mu = 0$ and variance $\sigma^2 = 1$).
*   **Reward Generation:** When action $A_t$ is selected, the reward $R_t$ returned by the environment is the true value $q_*(A_t)$ plus a normally distributed noise term with mean $0$ and variance $1$:
    $$R_t \sim \mathcal{N}(q_*(A_t), 1)$$

```
Probability Density
       ^
       |             _..---.._             <- Distribution of Rewards for Arm 'a'
       |          .-'         '-.             Mean = q_*(a)
       |        .'               '.           Variance = 1
       |       /                   \
       |      |          |          |
       |      |          |          |
       +------|----------|----------|--------> Reward Value
                         ^
                      q_*(a)
```

Because of the added noise, the agent cannot identify the best arm from a single pull; it must observe multiple samples to compute a reliable average.

### 6.2 Python Implementation (Google Colab Overview)
Below is a structured Python script representing the core simulation loop described in the lecture:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation Configuration
n_bandits = 10         # Number of arms (k)
n_steps = 1000         # Number of plays per run
n_tasks = 2000         # Number of independent runs to average over
epsilons = [0, 0.1, 0.01]  # Epsilon values to compare

# Results storage: steps x epsilons
average_rewards = np.zeros((n_steps, len(epsilons)))

# Simulation Loop
for i, epsilon in enumerate(epsilons):
    print(f"Running simulation for epsilon = {epsilon}...")
    task_rewards = np.zeros((n_tasks, n_steps))
    
    for task in range(n_tasks):
        # 1. Initialize true values q_*(a)
        q_true = np.random.normal(0, 1, n_bandits)
        
        # 2. Initialize estimates Q(a) and selection counts N(a)
        q_est = np.zeros(n_bandits)
        action_counts = np.zeros(n_bandits)
        
        for step in range(n_steps):
            # 3. Action Selection (epsilon-greedy)
            if np.random.rand() < epsilon:
                # Explore: choose a random action
                action = np.random.choice(n_bandits)
            else:
                # Exploit: choose greedy action (break ties randomly)
                max_val = np.max(q_est)
                best_actions = np.where(q_est == max_val)[0]
                action = np.random.choice(best_actions)
            
            # 4. Generate Reward (True Value + Noise)
            reward = np.random.normal(q_true[action], 1)
            
            # 5. Update estimates using Sample-Average
            action_counts[action] += 1
            q_est[action] += (reward - q_est[action]) / action_counts[action]
            
            # Record reward
            task_rewards[task, step] = reward
            
    # Compute average reward across all tasks for each step
    average_rewards[:, i] = np.mean(task_rewards, axis=0)

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(average_rewards[:, 0], label="$\epsilon$ = 0 (Greedy)", color='red')
plt.plot(average_rewards[:, 1], label="$\epsilon$ = 0.1", color='blue')
plt.plot(average_rewards[:, 2], label="$\epsilon$ = 0.01", color='green')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("$\epsilon$-Greedy Performance on 10-Armed Bandit")
plt.legend()
plt.grid(True)
plt.show()
```

### 6.3 Empirical Performance Analysis

When plotting the average reward obtained over $1000$ steps across $2000$ independent runs, we observe the following performance curves:

```
Average Reward
  ^
  |                                        ==================  epsilon = 0.1 (Explores early, finds best fast)
  |                                     ---------------------  epsilon = 0.01 (Slower start, catches up later)
  |                  _..----------------
  |               .-'
  |             .'  .........................................  epsilon = 0 (Greedy - gets stuck in local optima)
  |           ./
  |         ./
  |       .'
  |     .'
  +----+-----------------------------------------------------> Steps
  0                                                          1000
```

*   **The Greedy Algorithm ($\epsilon = 0$):**
    *   **Performance:** Lowest performance in the long term.
    *   **Reasoning:** It improves slightly at the very beginning by exploiting its first positive discoveries, but it quickly plateaus because it gets stuck choosing suboptimal actions, never discovering the true best-performing arm.
*   **The $\epsilon = 0.1$ Algorithm:**
    *   **Performance:** Rapid initial improvement.
    *   **Reasoning:** The high rate of exploration allows the agent to quickly find the optimal action. Its reward rate rises fast and stabilizes near its maximum potential (limited only by the $10\%$ exploration penalty).
*   **The $\epsilon = 0.01$ Algorithm:**
    *   **Performance:** Slower initial improvement, but steady upward trajectory.
    *   **Reasoning:** Because it explores less, it takes more steps to collect enough information to identify the best arm. However, once identified, it exploits it $99\%$ of the time, eventually outperforming the $\epsilon = 0.1$ algorithm in very long trials.

---

## 7. Bandits vs. Full Reinforcement Learning

The multi-arm bandit problem is often called a **nonassociative** RL task because it does not associate different actions with different states. It is a critical stepping stone to full RL, but it is important to distinguish between the two:

| Characteristic | Multi-Arm Bandit Problems | Full Reinforcement Learning Problems |
|---|---|---|
| **State Space** | **Single State:** The environment remains in the same state/situation regardless of the action taken. | **Multiple States:** The environment changes state dynamically based on the actions selected by the agent. |
| **Planning** | **No Planning Required:** Actions only affect immediate rewards. There is no temporal credit assignment over long horizons. | **Planning Required:** Actions have long-term consequences. The agent must balance immediate rewards against future state values (e.g., chess moves). |
| **Action Dependency** | **Independent Actions:** The feedback for an action is independent of previous choices. | **Sequential Dependencies:** The reward and state transition of action $a_t$ depend on prior choices $a_{t-1}, a_{t-2}, \dots$ |

---

## 8. Summary of Key Takeaways

1.  **Value Estimation:** We estimate action values using averages of observed rewards. Over time, these estimations ($Q_t(a)$) converge to the true hidden values ($q_*(a)$).
2.  **Exploitation vs. Exploration:** 
    *   *Exploitation* maximizes immediate rewards using current knowledge but risks missing better options.
    *   *Exploration* sacrifices short-term gains to collect data that can yield higher long-term cumulative returns.
3.  **$\epsilon$-Greedy Strategy:** A simple and effective heuristic to balance this trade-off by taking greedy actions most of the time ($1-\epsilon$) and exploring randomly with a small probability ($\epsilon$).