# Chapter 1: The RL Problem — What Are We Even Doing?

*This is where it all starts. The course is about reasoning LLMs, and this chapter lays out why reinforcement learning matters: it's the framework that turns a non-reasoning model into a reasoning one. Everything that follows — bandits, Bellman equations, policy gradients, PPO, GRPO — builds on the ideas introduced here.*

---

## 1. Three Paradigms of Machine Learning

| Feature | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
| :--- | :--- | :--- | :--- |
| **Data Type** | Labeled data (input-output pairs) | Unlabeled data | No predefined labels; interactive data |
| **Primary Source** | Human-curated datasets | Raw, unstructured datasets | Active interaction with an environment |
| **Objective** | Generalize/extrapolate to unseen data | Find hidden structures/patterns within data | Learn how to act to maximize a reward signal |
| **Example** | Classifying brain MRIs as *malignant* or *benign*. | Clustering *Iris* flower species; Netflix/YouTube recommendations. | Game-playing agents (Chess, Tic-Tac-Toe); robotics. |

The key distinction: in RL, the agent learns from the **consequences of its decisions** rather than being explicitly told what the correct action is — trial and error, not labeled examples.

---

## 2. Core Characteristics

* **Learning by Trial and Error**: No teacher provides correct answers. The agent acts, observes what happens, and adjusts.
* **Biological Relevance**: RL is the closest CS paradigm to how humans and animals naturally learn. A gazelle calf doesn't have labeled training data for "how to run" — it adapts motor control through physical feedback within 30 minutes.
* **Historical Context (1960s–1980s)**: Early AI researchers classified RL-related approaches as **weak methods**. The dominant belief was that performance depended on volume of human-provided training data. RL shifted this by emphasizing learning from environmental interaction.

---

## 3. Real-Life Examples

These examples ground the abstraction. Notice how each one has the same three-part structure: an agent, an environment, and a goal.

### I. A Master Chess Player
* Two concurrent cognitive processes:
  1. **Planning**: Anticipating moves, counter-moves, calculating lines several steps ahead.
  2. **Intuitive Judgments**: Assessing the "desirability" of a board state based on past experience.

### II. A Gazelle Calf
* Shortly after birth, a newborn gazelle calf struggles to stand (trial and error). Within 30 minutes, it can run at 36 km/h.
* No database of labeled "how-to-run" examples — it adapts motor control parameters almost instantly through physical feedback.

### III. A Mobile Robot (e.g., Roomba)
* **Decision Questions**: Should the robot navigate to a new room to collect more trash, or return to its docking station to recharge?
* **Trade-off**: Balancing immediate rewards (cleaning) with long-term survival (maintaining battery power).

### IV. Phil Preparing Breakfast
* Sequence of sub-actions: walking to cupboard → opening it → selecting cereal → reaching → grasping → retrieving → grabbing plate → finding spoon.
* Small physical actions serve immediate sub-goals, which serve the ultimate goal of obtaining nourishment.

---

## 4. The Unified Structure of RL Problems

Every RL problem decomposes into three entities:

```
[Active Decision-Making Agent] <--- (Interaction) ---> [Environment] ---> [Goal]
```

1. **Agent**: The entity taking actions (the chess player, the gazelle, the robot, Phil).
2. **Environment**: Everything external to the agent's decision-making core.
   * *Note on boundaries*: The boundary isn't always physical. A robot's battery level or Phil's memories can mathematically be treated as part of the environment — they dictate the state from which decisions are made.
3. **Goal**: The objective (winning chess, escaping predators, maximizing collected trash, gaining nourishment).

---

## 5. The Four Elements of an RL System

1. **Policy** ($\pi$):
   * *Informal*: The agent's way of behaving at any given time.
   * *Formal*: A mapping from perceived states of the environment to actions to be taken in those states.

2. **Reward Signal** ($R$):
   * The immediate goal. At each time step, the environment sends a single numerical value (reward) to the agent.
   * The agent's sole objective: maximize cumulative reward over the long run.
   * *Analogy*: Pleasure/pain in biological systems — touching a boiling pot gives a sharp negative reward, teaching the system not to repeat the action.

3. **Value Function** ($V$):
   * *Reward vs. Value*: Reward = what is good *immediately*. Value = what is good in the *long run*.
   * The value of a state = total estimated reward the agent can expect to accumulate over the future, starting from that state.
   * *Tournament analogy*: If Player X performs poorly in the first 2 matches of a 14-game tournament, immediate reward is low. But if the player is highly skilled, their long-term value remains high.
   * → This distinction between reward and value is **central** to everything that follows. The Bellman equation (→ §3.5) formalizes exactly this relationship.

4. **Model of the Environment** *(Optional)*:
   * Mimics the behavior of the environment, allowing predictions about how the environment responds.
   * **Model-Based RL**: Uses a model to plan actions by predicting future states.
   * **Model-Free RL**: Learns purely through trial and error without predicting the next state. Most of this course focuses on model-free methods.

---

## 6. Case Study: Tic-Tac-Toe

A concrete example showing all four elements working together. We design a model-free RL agent to play Tic-Tac-Toe as 'X' against an opponent playing 'O'.

### Key Definitions
* **State**: Any unique board configuration of X's and O's on the 3×3 grid.
* **Policy**: The rule dictating which move to make for any given board state.

### The Value Function Table
A lookup table: every possible board state → estimated probability of winning from that state.

* **Winning States** (three X's in a row): Value = $1.0$
* **Losing States** (three O's in a row): Value = $0.0$
* **Draw States**: Value = $0.5$
* **Initial/Intermediate States**: Initialized to $0.5$ (neutral)

```
       State Column                 Value Column
+-------------------------+        +--------------+
|     X  |  X  |  X       |        |              |
|   -----+-----+-----     | -----> |     1.0      |
|     O  |  O  |          |        | (Win State)  |
+-------------------------+        +--------------+
|     O  |  O  |  O       |        |              |
|   -----+-----+-----     | -----> |     0.0      |
|     X  |  X  |          |        | (Loss State) |
+-------------------------+        +--------------+
|     X  |     |  O       |        |              |
|   -----+-----+-----     | -----> |     0.5      |
|        |     |          |        | (Default Init|
+-------------------------+        +--------------+
```

### Action Selection: Exploitation vs. Exploration
1. **Exploitation (Greedy Move)**: Examine all legal moves, look up resulting states, select the one with the highest value (e.g., $0.6$ over $0.4$).
2. **Exploration (Exploratory Move)**: Occasionally choose at random — allows the agent to evaluate board states it might otherwise overlook.

This tension between exploitation and exploration is the **central problem** of Chapter 2 (→ §2.4).

### Updating Value Estimates ("Backing Up")
When the agent transitions from state $S$ to successor state $S'$ via a greedy move:

$$V(S) \leftarrow V(S) + \alpha \left[ V(S') - V(S) \right]$$

Where:
* $V(S)$: current value of earlier state
* $V(S')$: value of successor state
* $\alpha$: step-size parameter (learning rate), a small positive fraction

#### Worked Example
* Current: $V(S) = 0.8$, Successor: $V(S') = 0.9$, Step-size: $\alpha = 0.01$

$$V(S) \leftarrow 0.8 + 0.01 \times (0.9 - 0.8) = 0.8 + 0.001 = 0.801$$

The value of the earlier state shifts slightly toward the value of the later state. This is called **backing up**. By repeatedly playing games, values converge to true winning probabilities.

*Note:* This update rule is actually a special case of **Temporal Difference learning** (→ §5.2), which we'll derive formally later.

---

## 7. Key Takeaways

1. **Interaction-Driven Learning**: In RL, learning occurs through active trial-and-error interaction with the environment.
2. **Goal-Directed Behavior**: Actions are guided by a clear goal, formalized by a reward signal.
3. **Long-Term Perspective**: The value function helps evaluate long-term consequences rather than just seeking immediate gratification.
4. **Model-Free Simplicity**: Effective learning can occur without an explicit model of the environment (as the tic-tac-toe example shows).

---

*But tic-tac-toe essentially had one situation: "look at the board, pick a move." What happens when we need to choose between multiple levers without knowing which pays best, and there's no board to look at — just raw action-reward feedback? That's the multi-arm bandit problem → Chapter 2.*
