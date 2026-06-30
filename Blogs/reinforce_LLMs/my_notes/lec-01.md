Here are the detailed lecture notes from the video **Lecture 4: Introduction to Reinforcement Learning**.

---

# Lecture Notes: Introduction to Reinforcement Learning (Lecture 4)

## 1. Context and Overview
This lecture is the fourth in a series focusing on reasoning-based Large Language Models (LLMs). It transitions the discussion from inference-time techniques to model training via Reinforcement Learning.

### Recap of Methods for Inducing Reasoning in LLMs:
1. **Inference-Time Compute Scaling** *(covered in previous lectures)*:
   * Processes are applied during the inference stage without modifying the pre-trained weights of the model.
   * **Prompting**: Techniques such as *Chain of Thought (CoT)* and *Zero-Shot Chain of Thought* nudge larger models to perform better on reasoning tasks (e.g., adding "Let's think step-by-step").
   * **Verifiers**: The LLM generates multiple candidate answers, and a verification layer (utilizing outcome-based or process-based reward models) selects the best output. This is typically implemented via:
     * Majority voting
     * Best-of-N sampling
     * Beam search
2. **Pure Reinforcement Learning (RL)** *(the focus of today's lecture)*:
   * Unlike inference-time scaling, pure RL directly updates/tweaks the parameters of the model to turn a non-reasoning model into a reasoning model.

### Primary Reference Material:
* **Textbook**: *Reinforcement Learning: An Introduction* (Second Edition) by Richard S. Sutton and Andrew G. Barto.

---

## 2. Paradigms of Machine Learning: A Comparison
To understand Reinforcement Learning, it is helpful to compare it to the other two primary machine learning paradigms.

| Feature | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
| :--- | :--- | :--- | :--- |
| **Data Type** | Labeled data (input-output pairs) | Unlabeled data | No predefined labels; interactive data |
| **Primary Source** | Human-curated datasets | Raw, unstructured datasets | Active interaction with an environment |
| **Objective** | Generalize/extrapolate to unseen data | Find hidden structures/patterns within data | Learn how to act to maximize a reward signal |
| **Example** | Classifying brain MRIs as *malignant* or *benign*. | Clustering *Iris* flower species; Netflix/YouTube recommendations. | Game-playing agents (Chess, Tic-Tac-Toe); robotics. |

---

## 3. Core Characteristics of Reinforcement Learning
* **Learning by Trial and Error**: The agent learns from the consequences of its decisions rather than being explicitly told what the correct action is.
* **Biological Relevance**: RL is the closest computer science paradigm to how humans and animals naturally learn.
* **Historical Context (1960s–1980s)**:
  * Early AI researchers often classified RL-related approaches ("search" or "learning" methods) as **weak methods**.
  * The dominant belief was that performance depended primarily on the volume of human-provided training data. RL shifted this by emphasizing learning directly from environmental interaction.

---

## 4. Real-Life Examples Illustrating RL Concepts

### I. A Master Chess Player
* The player makes a move based on two concurrent cognitive processes:
  1. **Planning**: Anticipating moves, counter-moves, and calculating lines several steps ahead.
  2. **Intuitive Judgments**: Assessing the immediate "desirability" or value of a board state based on past experience.

### II. A Gazelle Calf
* **Rapid Adaptation**: Shortly after birth, a newborn gazelle calf struggles to stand (trial and error). Within 30 minutes, it can run at speeds of up to 36 km/h.
* **Observation**: It does not have a database of labeled "how-to-run" examples; it adapts its motor control parameters almost instantly through physical feedback.

### III. A Mobile Robot (e.g., Roomba)
* **Decision Questions**: Should the robot navigate to a new room to collect more trash, or return to its docking station to recharge?
* **Parameters**: Current battery charge, history of how long it took to find the charger in previous runs.
* **Trade-off**: Balancing immediate rewards (cleaning) with long-term survival (maintaining battery power).

### IV. Phil Preparing Breakfast
* **Sequence of Sub-actions**: Walking to the cupboard, opening it, selecting a cereal box, reaching, grasping, retrieving, grabbing a plate, and finding a spoon.
* **Goal Hierarchy**: Small physical actions (such as grabbing a spoon) serve immediate sub-goals, which in turn serve the ultimate goal of obtaining nourishment.

---

## 5. The Unified Structure of RL Problems

Every reinforcement learning problem can be structured around three main entities:

```
[Active Decision-Making Agent] <--- (Interaction) ---> [Environment] ---> [Goal]
```

1. **Active Decision-Making Agent**: The entity taking actions (e.g., the chess player, the gazelle, the robot, or Phil).
2. **Environment**: Everything external to the agent's decision-making core. 
   * *Note on Boundaries*: The boundary between the agent and the environment is not always physical. For example, a robot’s internal battery level or Phil’s internal memories and preferences can mathematically be treated as part of the environment, as they dictate the state from which decisions are made.
3. **Goal**: The objective the agent is trying to achieve (e.g., winning the chess game, running fast to escape predators, maximizing collected trash, or gaining nourishment).

---

## 6. The Four Main Elements of an RL System

An RL system consists of four foundational components:

1. **Policy**:
   * *Informal*: The agent's way of behaving at any given time.
   * *Formal*: A mapping from perceived states of the environment to the actions to be taken in those states.
2. **Reward Signal**:
   * Defines the immediate goal of the problem. At each time step, the environment sends a single numerical value (the reward) to the agent.
   * The agent's sole objective is to maximize the cumulative reward over the long run.
   * *Analogy*: Equivalent to pleasure or pain in biological systems (e.g., touching a boiling pot gives a sharp negative reward, teaching the system not to repeat the action).
3. **Value Function**:
   * *Reward vs. Value*: While the reward signal indicates what is good *immediately*, the value function specifies what is good in the *long run*.
   * The value of a state is the total estimated reward an agent can expect to accumulate over the future, starting from that state.
   * *Sports Tournament Analogy*: If Player X performs poorly in the first two matches of a 14-game tournament, the immediate *reward* is low. However, if the player is highly skilled and likely to win the remaining 12 matches, their long-term *value* remains high.
4. **Model of the Environment** *(Optional)*:
   * A component that mimics the behavior of the environment, allowing the agent to make predictions about how the environment will respond to actions.
   * **Model-Based RL**: Uses a model to plan actions by predicting future states.
   * **Model-Free RL**: The agent learns purely through trial and error without predicting the next state mathematically (e.g., learning to play a game solely by playing it).

---

## 7. Practical Case Study: Tic-Tac-Toe

To illustrate how these elements function together, consider designing a model-free RL agent to play Tic-Tac-Toe (playing as 'X' against an opponent playing as 'O').

### Key Definitions:
* **State**: Any unique board configuration of X's and O's on the 3x3 grid.
* **Policy**: The rule that dictates which move to make for any given board state.

### The Value Function Table:
We maintain a lookup table containing every possible board state and its corresponding "value" (the estimated probability of winning from that state).

* **Winning States** (three X's in a row): Value = $1.0$ (certain win)
* **Losing States** (three O's in a row): Value = $0.0$ (certain loss)
* **Draw States**: Value = $0.5$
* **Initial/Intermediate States**: Initialized to $0.5$ (neutral expectation)

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
When it is the agent's turn to play:
1. **Exploitation (Greedy Move)**:
   * The agent examines all possible legal moves, looks up their resulting states in the Value Table, and selects the move leading to the state with the highest value (e.g., choosing a state valued at $0.6$ over $0.4$).
2. **Exploration (Exploratory Move)**:
   * The agent occasionally chooses a move at random rather than the greedy move. This allows it to experience and evaluate board states it might otherwise overlook.

### Updating Value Estimates ("Backing Up")
As the game is played, the values of the visited states must be updated so they reflect the true probability of winning.

When the agent transitions from an earlier state ($S$) to a successor state ($S'$) via a greedy move, the value of the earlier state is updated using the **temporal-difference update rule**:

$$V(S) \leftarrow V(S) + \alpha \left[ V(S') - V(S) \right]$$

Where:
* $V(S)$ is the current value of the earlier state.
* $V(S')$ is the value of the successor state.
* $\alpha$ is a small positive fraction representing the **step-size parameter** (learning rate).

#### Mathematical Example:
Assume:
* Current state value: $V(S) = 0.8$
* Successor state value: $V(S') = 0.9$
* Step-size parameter: $\alpha = 0.01$

The updated value for state $S$ is calculated as:
$$V(S) \leftarrow 0.8 + 0.01 \times (0.9 - 0.8)$$
$$V(S) \leftarrow 0.8 + 0.01 \times (0.1)$$
$$V(S) \leftarrow 0.8 + 0.001 = 0.801$$

This method of shifting the value of an earlier state slightly toward the value of the later state is called **backing up**. By repeatedly playing games and applying this rule, the values in the lookup table will eventually converge to the true probabilities of winning from each state.

---

## 8. Summary of Key Takeaways
1. **Interaction-Driven Learning**: In RL, learning occurs through active trial-and-error interaction with the environment.
2. **Goal-Directed Behavior**: Actions are guided by a clear goal, formalized by a reward signal.
3. **Long-Term Perspective**: The value function is central to RL, helping the agent evaluate the long-term consequences of its decisions (planning/delayed rewards) rather than just seeking immediate gratification.
4. **Model-Free Simplicity**: Effective learning can occur without an explicit mathematical model of the environment (as shown in the Tic-Tac-Toe example).

---

## 9. Homework Problem
* **Question**: *How is reinforcement learning different from evolutionary methods, such as genetic algorithms?*