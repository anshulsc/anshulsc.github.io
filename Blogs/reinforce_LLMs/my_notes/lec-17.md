# Lecture Notes: Integrating Reinforcement Learning with Large Language Models
**Course:** Reasoning LLMs from Scratch  
**Subject:** Framing Language Generation as an Agent-Environment RL Interface

---

## 1. Introduction and Foundations Recap

Before adapting Reinforcement Learning (RL) to Large Language Models (LLMs), it is helpful to outline the progression of classical RL concepts that led to modern optimization algorithms like PPO (Proximal Policy Optimization).

```
   [Tabular RL] ---> [Function Approximation] ---> [Policy Gradient Methods]
                                                          |
                                           [TRPO] <--- [REINFORCE / Actor-Critic]
                                             |
                                           [PPO]
```

### 1.1 Tabular Methods to Function Approximation
*   **Tabular RL:** Useful in environments with small, discrete state and action spaces (e.g., small grid worlds). Exact state-action values ($Q(s, a)$) are stored in a lookup table.
*   **Value Function Approximation:** As state-action spaces grow exponentially, storing values in tables becomes intractable. We transition to using function approximators (such as deep neural networks) to estimate value functions: $V(s) \approx \hat{V}(s; \mathbf{w})$ or $Q(s, a) \approx \hat{Q}(s, a; \mathbf{w})$.

### 1.2 Policy Gradient Methods
Rather than indirectly deriving a policy from value estimates (which can lead to oscillatory behavior during training), policy gradient methods directly parameterize the policy $\pi_\theta(a|s)$ and optimize the parameters $\theta$ to maximize expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

*   **REINFORCE:** A Monte Carlo policy gradient algorithm that uses sample trajectories to update policy parameters:
    $$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) G_t$$
    where $G_t$ is the return from time step $t$.
*   **Actor-Critic Methods:** To reduce the high variance associated with Monte Carlo returns ($G_t$), Actor-Critic architectures introduce a baseline—typically the state-value function $V(s)$—transforming the update to use the advantage function $A(s, a) = Q(s, a) - V(s)$.

### 1.3 Trust Region and Proximal Policy Optimization
*   **TRPO (Trust Region Policy Optimization):** Standard policy gradient steps can occasionally cause destructively large policy updates, making subsequent training unstable. TRPO addresses this by enforcing a Kullback-Leibler (KL) divergence constraint on the policy update:
    $$\mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}} \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \parallel \pi_\theta(\cdot|s)) \right] \le \delta$$
    Calculating the inverse of the Fisher Information Matrix required for this constraint is computationally expensive.
*   **PPO (Proximal Policy Optimization):** To simplify training while retaining the stability of trust regions, PPO uses a clipped surrogate objective function:
    $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
    where the probability ratio is $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$. This approach avoids complex second-order derivative calculations and has become a standard algorithm for alignment tasks.

---

## 2. Framing Language Modeling as an RL Problem

Historically, modeling text generation as an interactive agent-environment system was not an obvious design pattern. Standard autoregressive language models are trained via supervised pre-training on massive corpora using a cross-entropy loss. However, framing text generation as an RL problem allows the model to optimize for non-differentiable objectives (such as truthfulness, helpfulness, and safety).

### 2.1 The Classical Agent-Environment Interface
In standard RL, an agent interacts with an environment over discrete time steps:

```
               Action (a_t)
         +--------------------> [ Environment ]
         |                         |
     [ Agent ]                     | State (s_{t+1}), Reward (r_{t+1})
         ^                         v
         +-------------------------+
```

1.  At time $t$, the agent observes the state $s_t \in \mathcal{S}$.
2.  The agent selects an action $a_t \in \mathcal{A}$ based on its policy $\pi(a|s)$.
3.  The environment transitions to a new state $s_{t+1}$ and yields a scalar reward $r_t \in \mathbb{R}$.

---

## 3. Mapping the RL Interface to LLMs

To apply RL to language models, we establish a mapping between the components of language generation and the standard RL MDP (Markov Decision Process):

| RL Component | LLM Translation | Description / Meaning |
| :--- | :--- | :--- |
| **Agent** | The LLM ($\pi_\theta$) | The neural network parameterizing the policy. |
| **Environment** | The context buffer/User interaction | Keeps track of the generated tokens and passes the complete sequence to a evaluator/reward system. |
| **State ($s_t$)** | Prompt + Generated tokens so far | The sequence context available to the model at generation step $t$. |
| **Action ($a_t$)** | The next predicted token ($y_t$) | Selecting a single token/word from the vocabulary space $\mathcal{V}$. |
| **Reward ($r_t$)** | Correctness/Alignment score | Feedback evaluating the quality of the generated sequence. |
| **Policy ($\pi_\theta$)** | Token-generation probabilities | The probability distribution over the vocabulary given the current state. |

---

## 4. Step-by-Step Generative Trace (The "Roger" Example)

To observe this mapping in action, consider the following prompt and the step-by-step token generation process.

*   **Prompt ($x$):** *"Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"*
*   **Correct Output:** *"Roger has 11 tennis balls."*

Let us trace how the state, action, and environment transition at each discrete step $t$:

### Step 1: Initial Prompt

```
[State s_0] = [Prompt]
     |
     v
 [Agent] ---> Predicts next word ---> [Action a_0] = "Roger"
                                           |
                                           v
                                   [State s_1] = [Prompt, "Roger"]
```

### Step 2: Incorporating First Generation

```
[State s_1] = [Prompt, "Roger"]
     |
     v
 [Agent] ---> Predicts next word ---> [Action a_1] = "has"
                                           |
                                           v
                                   [State s_2] = [Prompt, "Roger", "has"]
```

### Intermediate Generation Steps:
*   **$s_2 = \{ \text{Prompt}, \text{"Roger"}, \text{"has"} \}$** $\rightarrow$ Agent takes action **$a_2 = \text{"eleven"}$** $\rightarrow$ **$s_3 = \{s_2, \text{"eleven"}\}$**
*   **$s_3 = \{ \text{Prompt}, \dots, \text{"eleven"} \}$** $\rightarrow$ Agent takes action **$a_3 = \text{"tennis"}$** $\rightarrow$ **$s_4 = \{s_3, \text{"tennis"}\}$**
*   **$s_4 = \{ \text{Prompt}, \dots, \text{"tennis"} \}$** $\rightarrow$ Agent takes action **$a_4 = \text{"balls"}$** $\rightarrow$ **$s_5 = \{s_4, \text{"balls"}\}$**

At each step $t$, the action $a_t$ chosen by the model is appended to the current state $s_t$ to form the next state $s_{t+1}$:

$$s_{t+1} = s_t \cup \{a_t\}$$

---

## 5. Mathematical Formalization of Policy and Generation Probability

Let us derive the probability of a complete generated text sequence from first principles.

### 5.1 Formal Definitions
Let:
*   $\mathcal{V}$ be the vocabulary (the set of all possible tokens).
*   $x = \{x_1, x_2, \dots, x_{T_x}\}$ be the input prompt, where $x_i \in \mathcal{V}$.
*   $y = \{y_1, y_2, \dots, y_{T_y}\}$ be the generated completion, where $y_i \in \mathcal{V}$.
*   $y_{<t} = \{y_1, y_2, \dots, y_{t-1}\}$ denote the sequence of generated tokens prior to step $t$.

### 5.2 Derivation of the Complete Sequence Probability
By the **Chain Rule of Probability**, the joint probability of a sequence of dependent random variables $A, B, C, \dots$ can be decomposed into a product of conditional probabilities:

$$P(A, B, C) = P(A) \cdot P(B | A) \cdot P(C | A, B)$$

Applying this rule directly to the generation of a completion sequence $y$ conditioned on the prompt $x$:

$$P(y_1, y_2, \dots, y_{T_y} | x) = P(y_1 | x) \cdot P(y_2 | x, y_1) \cdot P(y_3 | x, y_1, y_2) \cdots P(y_{T_y} | x, y_1, \dots, y_{T_y-1})$$

We can write this product compactly using product notation:

$$P(y | x) = \prod_{t=1}^{T_y} P(y_t | x, y_1, y_2, \dots, y_{t-1})$$

Using our shorthand notation for prior tokens $y_{<t}$:

$$P(y | x) = \prod_{t=1}^{T_y} P(y_t | x, y_{<t})$$

Since the LLM parameterizes this conditional distribution using weights $\theta$, we replace the probability $P$ with our policy notation $\pi_\theta$:

$$\pi_\theta(y | x) = \prod_{t=1}^{T_y} \pi_\theta(y_t | x, y_{<t})$$

### 5.3 Intuition of $\pi_\theta$ as a Probability Distribution
For any given state $s_t = \{x, y_{<t}\}$, the policy $\pi_\theta(\cdot | s_t)$ outputs a vector of probabilities over the entire vocabulary space $\mathcal{V}$:

$$\pi_\theta(\cdot | s_t) \in [0, 1]^{|\mathcal{V}|} \quad \text{such that} \quad \sum_{w \in \mathcal{V}} \pi_\theta(w | s_t) = 1$$

During generation, the agent chooses an action $a_t = y_t$ by sampling from this distribution or using decoding strategies like greedy search:

```
State (s_t): [ "I", "am", "going" ]
                  |
                  v
                [LLM]
                  |
                  v  (Softmax Distribution over Vocabulary)
          +---------------+------+
          | Word          | Prob |
          +---------------+------+
          | "to"          | 0.45 |  <-- Selected Action (a_t)
          | "home"        | 0.20 |
          | "there"       | 0.15 |
          | ...           | ...  |
          +---------------+------+
```

---

## 6. The Reward Mechanics: Sparse and Delayed Feedback

A primary characteristic of applying RL to LLMs is the nature of the reward signal.

### 6.1 Intermediate vs. Terminal Rewards
For intermediate generation steps ($t < T_y$), there is typically no immediate signal indicating if a single token is correct or incorrect in isolation. The feedback is **delayed** and **sparse**.

$$r_t = 0 \quad \text{for all} \quad t < T_y$$

Once the model generates a terminal token (e.g., an `<EOS>` token) at step $T_y$, the complete generated completion $y$ is evaluated by the environment:

$$r_{T_y} = \text{Evaluator}(x, y)$$

```
 Step 1: "Roger"      --->  r_0 = 0
 Step 2: "has"        --->  r_1 = 0
 Step 3: "eleven"     --->  r_2 = 0
 Step 4: "tennis"     --->  r_3 = 0
 Step 5: "balls."     --->  r_4 = 1.0 (Terminal Evaluation)
```

This reward structure is mathematically similar to games like chess, where intermediate moves receive a reward of $0$, and a non-zero reward (win, loss, or draw) is determined only at the end of the match.

---

## 7. The Reward Modeling Challenge in LLMs

While math problems have objective rewards ($1$ for correct, $0$ for incorrect), many text generation tasks (such as summarization, creative writing, or dialogue) are subjective.

```
       [Objective Reward Tasks]                     [Subjective Reward Tasks]
    (e.g., Mathematics, Coding)                   (e.g., Creative Writing, Essays)
                 |                                               |
                 v                                               v
        Deterministic rules/                             Requires human input/
        exact answer checks                              learned Reward Models
```

### 7.1 Human Preference Collection
Because manual human evaluation for every training step is slow and expensive, AI labs train an auxiliary model called a **Reward Model (RM)** to act as a proxy for human judgment. To train this model, developers collect pairwise comparison data:

1.  A prompt $x$ is sent to the LLM.
2.  The LLM generates two distinct candidates: $y^{(1)}$ and $y^{(2)}$.
3.  A human annotator reviews both candidates and indicates their preference:
    $$y^{(1)} \succ y^{(2)} \quad (\text{meaning } y^{(1)} \text{ is preferred over } y^{(2)})$$

```
Prompt: "Explain photosynthesis."
  |
  +---> Response 1: "Plants use sunlight to make food." (Preferred)
  |
  +---> Response 2: "Photosynthesis is a chemical reaction involving green things."
```

### 7.2 Training the Reward Model (RM)
Using thousands of these pairwise preferences, a reward model $R_\phi(x, y)$ is trained to output a scalar score. The objective function encourages the RM to assign higher scores to human-preferred responses:

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( R_\phi(x, y_w) - R_\phi(x, y_l) \right) \right]$$

where:
*   $y_w$ is the winning (preferred) response.
*   $y_l$ is the losing (less preferred) response.
*   $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function, which maps the score difference into a probability.

Once trained, this learned reward function $R_\phi(x, y)$ provides the scalar reward signal $r_{T}$ to update the policy $\pi_\theta$ using RL algorithms like PPO or GRPO.