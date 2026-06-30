# Chapter 2: Bandits & The Exploration Problem

*In Chapter 1 we saw the tic-tac-toe agent choose between exploitation (greedy move) and exploration (random move). That tension deserves its own study. The multi-arm bandit strips away states, transitions, and long-term planning — leaving only the purest form of the exploration-exploitation dilemma. Understanding it here makes everything from ε-greedy SARSA (→ §5.3) to PPO's entropy bonus (→ §6.8) click.*

---

## 1. Defining the Bandit Problem

### The One-Armed Bandit
The term "one-armed bandit" = a slot machine. Pull the lever, maybe get a payout, probably lose money. Called a "bandit" because the statistics are designed to rob you over time.

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

### The Multi-Arm Bandit ($k$-Armed Bandit)
Now imagine $k$ different levers, each with its own unknown reward distribution.

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

* **The Setup**: You stand before $k$ levers. Each has a stationary probability distribution of rewards.
* **The Core Problem**: You don't know the underlying distributions.
* **The Goal**: Maximize total expected reward over $T$ time steps.

---

## 2. Action-Value Methods

### Mathematical Definitions
Let $a$ denote pulling a specific lever.

* **True Value** ($q_*(a)$): The actual expected reward — an objective, hidden property:
  $$q_*(a) = \mathbb{E}[R_t \mid A_t = a]$$

* **Estimated Value** ($Q_t(a)$): Our estimate at time step $t$. We want $Q_t(a) \approx q_*(a)$.

### The Sample-Average Method
Average the rewards actually received when action $a$ was selected:

$$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}(A_i = a)}{\sum_{i=1}^{t-1} \mathbb{I}(A_i = a)}$$

Where $\mathbb{I}(\text{predicate})$ = 1 if true, 0 otherwise. The denominator = $N_t(a)$ (times action $a$ was chosen before time $t$).

If $N_t(a) = 0$, define $Q_t(a)$ = some default (e.g., $0$).

By the **Law of Large Numbers**: as $N_t(a) \to \infty$, the estimate converges to truth:
$$\lim_{N_t(a) \to \infty} Q_t(a) = q_*(a)$$

---

## 3. The Exploration vs. Exploitation Dilemma

Imagine a 4-armed bandit where your current estimates are:

| Action ($a$) | Estimated Value ($Q_t(a)$) |
|---|---|
| Action 1 | $Q_t(1) = 1.5$ |
| Action 2 | $Q_t(2) = 3.0$ |
| Action 3 | $Q_t(3) = 2.5$ |
| Action 4 | $Q_t(4) = 2.0$ |

### Exploitation (Greedy Choice)
* Pick **Action 2** ($Q_t(2) = 3.0$).
* Maximizes immediate reward using current knowledge.
* Risk: what if Action 3 actually has true value $q_*(3) = 5.0$, but a poor initial sample made its estimate $2.5$?

### Exploration (Non-Greedy Choice)
* Pick Action 1, 3, or 4.
* Sacrifices immediate reward to gather information.
* Enables discovery of better options → potentially higher long-term cumulative rewards.

This is the same tension from tic-tac-toe (→ §1.6), but isolated to its purest form.

---

## 4. The ε-Greedy Algorithm

The simplest strategy to balance exploration and exploitation:

```
                  [ Choose Action ]
                         |
                +--------+--------+
                |                 |
       With Prob. 1 - ε          With Prob. ε
                |                 |
         [ EXPLOITATION ]   [ EXPLORATION ]
         Choose Greedy      Choose Random
         Action             Action (from all)
```

### The Rule
* With probability $\epsilon$ (small, e.g. $0.01$ or $0.1$): select an action **at random** from all $k$ actions with equal probability.
* With probability $1 - \epsilon$: select the **greedy action** $A_t = \operatorname{argmax}_a Q_t(a)$.

### Impact of ε
* **$\epsilon = 0$ (Purely Greedy)**: Always exploits. Susceptible to local optima.
* **$\epsilon = 0.1$**: ~90 greedy steps per 100. Learns quickly but wastes 10% of steps exploring even after values are well-estimated.
* **$\epsilon = 0.01$**: Slower to find the best action, but once found, exploits it 99% of the time → higher long-run performance.

This exact same ε-greedy strategy reappears in Monte Carlo control (→ §5.1), SARSA (→ §5.3), and semi-gradient SARSA (→ §5.4).

---

## 5. Case Study: 10-Armed Testbed

### Simulation Design
* **Arms ($k$)**: $10$
* **True Values**: $q_*(a) \sim \mathcal{N}(0, 1)$ for each arm
* **Reward Generation**: $R_t \sim \mathcal{N}(q_*(A_t), 1)$ — true value plus noise

```
Probability Density
       ^
       |             _..---.._             <- Distribution of Rewards for Arm 'a'
       |          .-'         '-.             Mean = q_*(a)
       |        .'               '.           Variance = 1
       |       /                   \
       |      |          |          |
       +------|----------|----------|--------> Reward Value
                         ^
                      q_*(a)
```

Because of noise, the agent can't identify the best arm from a single pull — it must observe multiple samples.

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Configuration
n_bandits = 10         # Number of arms (k)
n_steps = 1000         # Number of plays per run
n_tasks = 2000         # Independent runs to average over
epsilons = [0, 0.1, 0.01]

# Results storage
average_rewards = np.zeros((n_steps, len(epsilons)))

for i, epsilon in enumerate(epsilons):
    print(f"Running simulation for epsilon = {epsilon}...")
    task_rewards = np.zeros((n_tasks, n_steps))
    
    for task in range(n_tasks):
        # 1. Initialize true values q_*(a)
        q_true = np.random.normal(0, 1, n_bandits)
        
        # 2. Initialize estimates Q(a) and counts N(a)
        q_est = np.zeros(n_bandits)
        action_counts = np.zeros(n_bandits)
        
        for step in range(n_steps):
            # 3. ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(n_bandits)       # explore
            else:
                max_val = np.max(q_est)
                best_actions = np.where(q_est == max_val)[0]
                action = np.random.choice(best_actions)     # exploit (break ties randomly)
            
            # 4. Generate reward
            reward = np.random.normal(q_true[action], 1)
            
            # 5. Update estimate using sample-average
            action_counts[action] += 1
            q_est[action] += (reward - q_est[action]) / action_counts[action]
            
            task_rewards[task, step] = reward
            
    average_rewards[:, i] = np.mean(task_rewards, axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(average_rewards[:, 0], label="ε = 0 (Greedy)", color='red')
plt.plot(average_rewards[:, 1], label="ε = 0.1", color='blue')
plt.plot(average_rewards[:, 2], label="ε = 0.01", color='green')
plt.xlabel("Steps"); plt.ylabel("Average Reward")
plt.title("ε-Greedy Performance on 10-Armed Bandit")
plt.legend(); plt.grid(True); plt.show()
```

### Performance Analysis

```
Average Reward
  ^
  |                                        ==================  ε = 0.1 (fast exploration)
  |                                     ---------------------  ε = 0.01 (slow but higher ceiling)
  |                  _..----------------
  |               .-'
  |             .'  .........................................  ε = 0 (greedy, gets stuck)
  |           ./
  |         ./
  |       .'
  |     .'
  +----+-----------------------------------------------------> Steps
  0                                                          1000
```

* **Greedy ($\epsilon = 0$)**: Plateaus early — gets stuck choosing suboptimal arms, never discovers the best one.
* **$\epsilon = 0.1$**: Rapid improvement. High exploration rate finds optimal arm quickly, but 10% exploration penalty limits asymptotic performance.
* **$\epsilon = 0.01$**: Slow start but steady climb. Once it finds the best arm, it exploits 99% of the time → eventually outperforms $\epsilon = 0.1$ in long trials.

---

## 6. Bandits vs. Full Reinforcement Learning

| Characteristic | Multi-Arm Bandit Problems | Full Reinforcement Learning |
|---|---|---|
| **State Space** | **Single State**: same situation regardless of action. | **Multiple States**: environment changes dynamically based on actions. |
| **Planning** | **No Planning**: actions only affect immediate rewards. | **Planning Required**: actions have long-term consequences (e.g., chess moves). |
| **Action Dependency** | **Independent**: feedback for an action is independent of previous choices. | **Sequential**: reward and next state depend on prior choices $a_{t-1}, a_{t-2}, \dots$ |

---

## 7. Key Takeaways

1. **Value Estimation**: We estimate action values using averages of observed rewards. Over time, $Q_t(a) \to q_*(a)$.
2. **Exploitation vs. Exploration**: Exploitation maximizes immediate reward but risks missing better options. Exploration sacrifices short-term gains to discover higher long-term returns.
3. **ε-Greedy Strategy**: Simple and effective — greedy most of the time ($1-\epsilon$), random with probability $\epsilon$.

---

*Bandits have one state — the agent sits in the same situation forever. Real problems have many states, and actions affect what state you land in next. To handle that, we need a formal mathematical framework: Markov Decision Processes → Chapter 3.*
