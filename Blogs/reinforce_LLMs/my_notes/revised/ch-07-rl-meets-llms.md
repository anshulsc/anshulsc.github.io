# Chapter 7: RL Meets LLMs — PPO, GRPO & Reasoning

*Everything from Chapters 1–6 was classical RL: agents in gridworlds, mountain cars, robotic vacuums. Now we map the entire framework onto language models — where the "agent" is an LLM, the "action" is generating a token, and the "reward" comes from a human judge or a math verifier. This chapter covers RLHF, PPO for LLMs, reward modeling, and GRPO — the algorithm behind DeepSeek-R1 that eliminates the critic network entirely.*

---

## 7.1 Language Generation as an MDP

To apply RL to text generation, we cast token-by-token generation into the standard Agent-Environment interface (→ §3.1):

| RL Component | LLM Translation | Description |
| :--- | :--- | :--- |
| **Agent** | The LLM ($\pi_\theta$) | Neural network parameterizing the policy |
| **Environment** | Context buffer / User | Tracks generated tokens, passes to evaluator |
| **State ($s_t$)** | Prompt + tokens generated so far | $s_t = (x, y_1, \dots, y_{t-1})$ |
| **Action ($a_t$)** | Next predicted token ($y_t$) | Selected from vocabulary $\mathcal{V}$ |
| **Reward ($r_t$)** | Correctness / alignment score | Evaluates quality of generated sequence |
| **Policy ($\pi_\theta$)** | Token-generation probabilities | Distribution over vocabulary given current context |

### State Transitions
Deterministic concatenation — the next state is just the current state with the new token appended:
$$s_{t+1} = s_t \cup \{y_t\} = (x, y_1, \dots, y_t)$$

### Worked Trace: "Roger" Example

**Prompt ($x$)**: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

```
Step 0: s_0 = [Prompt]
        Agent → a_0 = "Roger"         → s_1 = [Prompt, "Roger"]

Step 1: s_1 = [Prompt, "Roger"]
        Agent → a_1 = "has"           → s_2 = [Prompt, "Roger", "has"]

Step 2: s_2 = [Prompt, "Roger", "has"]
        Agent → a_2 = "11"            → s_3 = [Prompt, "Roger", "has", "11"]

Step 3: s_3 → a_3 = "tennis"          → s_4
Step 4: s_4 → a_4 = "balls."          → s_5 (terminal)
```

### Complete Sequence Probability

By the chain rule of probability:
$$\pi_\theta(y | x) = \prod_{t=1}^{T_y} \pi_\theta(y_t | x, y_{<t})$$

At each step, the LLM outputs a softmax distribution over the entire vocabulary:

```
State s_t: [ "I", "am", "going" ]
                 |
                 v
               [LLM]
                 |
                 v  (Softmax over Vocabulary)
         +---------------+------+
         | Token         | Prob |
         +---------------+------+
         | "to"          | 0.45 |  ← Selected (sampled or greedy)
         | "home"        | 0.20 |
         | "there"       | 0.15 |
         | ...           | ...  |
         +---------------+------+
```

---

## 7.2 Reward Mechanics

### Sparse & Delayed Feedback
For most RL-in-LLM setups, intermediate tokens get **no reward**. Only the complete output is evaluated:

$$r_t = 0 \quad \text{for } t < T_y$$
$$r_{T_y} = \text{Evaluator}(x, y)$$

```
 "Roger"  → r_0 = 0
 "has"    → r_1 = 0
 "11"     → r_2 = 0
 "tennis" → r_3 = 0
 "balls." → r_4 = 1.0  (correct answer → terminal reward)
```

Similar to chess: intermediate moves get $r=0$, only the final outcome (win/lose/draw) carries a reward.

### Objective vs. Subjective Tasks

| | Objective Reward | Subjective Reward |
|---|---|---|
| **Examples** | Math, coding, formal logic | Creative writing, summarization, dialogue |
| **Evaluation** | Deterministic rules, exact answer check | Human judgment |
| **Reward Signal** | Programmatic verifier | Learned Reward Model |

### Training a Reward Model (RM)

For subjective tasks, we train an auxiliary model $R_\phi(x, y)$ to mimic human preferences:

1. Generate two responses $y^{(1)}, y^{(2)}$ for prompt $x$.
2. Human annotator picks preferred response: $y_w \succ y_l$.
3. Train RM with Bradley-Terry loss:

$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( R_\phi(x, y_w) - R_\phi(x, y_l) \right) \right]$$

where $\sigma(z) = \frac{1}{1+e^{-z}}$ maps the score difference to a probability.

```
Prompt: "Explain photosynthesis."
  |
  +→ Response 1: "Plants use sunlight to make food."        (Preferred ✓)
  |
  +→ Response 2: "Photosynthesis involves green things."    (Rejected ✗)
```

The trained RM provides the scalar reward $r_T = R_\phi(x, y)$ for PPO/GRPO updates.

---

## 7.3 PPO for LLMs

### The 4-Model Setup

```
                            ┌────────────────────────┐
                            ▼                        │
 ┌──────────┐    ┌─────────────┐    ┌─────────────┐  │
 │ Question ├──► │    Actor    ├──► │ Prob. Dist. ├──┘
 │   (x)    │    │ (LLM π_θ)  │    │  (π_θ)      │
 └──────────┘    └──────┬──────┘    └─────────────┘
                        │
                        ▼ Generated completion (y)
                 ┌──────┴──────┐
                 ▼             ▼
          ┌─────────────┐    ┌─────────────┐
          │Reward Model │    │Value Model  │
          │   (Fixed)   │    │  (Critic)   │
          └──────┬──────┘    └──────┬──────┘
                 │ Reward R         │ Baseline V(s)
                 ▼                  ▼
          ┌────────────────────────────────┐
          │      Calculate Advantage       │
          │   A_t = R + γV(s_{t+1}) - V(s) │
          └────────────────┬───────────────┘
                           ▼
          ┌────────────────────────────────┐
          │       PPO Gradient Update      │
          └────────────────────────────────┘
```

Four models simultaneously in memory:
1. **Actor ($\pi_\theta$)**: The LLM being optimized.
2. **Reference ($\pi_{ref}$)**: Frozen copy of the SFT model — used for KL penalty to prevent drift.
3. **Reward Model (RM)**: Fixed, provides scalar evaluation of outputs.
4. **Critic ($V_\phi$)**: Actively trained to estimate expected return from any state.

### Token-Level PPO

The probability ratio at token level:
$$r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\theta_{old}}(y_t | x, y_{<t})}$$

The clipped objective (→ §6.8, not re-derived here):
$$L_{PPO}(\theta) = \mathbb{E} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \min\left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

### The Memory Bottleneck

$$\text{Memory} \propto \text{Size(Actor)} + \text{Size(Critic)} + \text{Size(Reference)} + \text{Size(Reward)}$$

* If the Actor is 70B parameters, the Critic must be comparable in size → **double** the parameter footprint.
* Optimizer states (AdamW: 8 bytes/param) → doubles again.
* Using an entire 70B network just to compute a baseline $V(s)$ is mathematically wasteful.

---

## 7.4 GRPO: Eliminating the Critic

### The Key Insight

Recall the advantage function (→ §6.4):
$$A(s,a) = Q_\pi(s,a) - V_\pi(s)$$

The baseline $V_\pi(s) = \mathbb{E}_{a \sim \pi}[Q_\pi(s,a)]$ is just the **average expected reward** over all possible actions. Instead of training a massive neural network to approximate this average, **estimate it statistically** by sampling multiple outputs for the same prompt.

### Group Sampling

```
                        ┌─────────────┐
                        │ Prompt (x)  │
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           ┌──────────┐  ┌──────────┐  ┌──────────┐
           │Output y_1│  │Output y_2│  │Output y_G│
           └────┬─────┘  └────┬─────┘  └────┬─────┘
                │              │              │
                ▼              ▼              ▼
           ┌──────────┐  ┌──────────┐  ┌──────────┐
           │Reward r_1│  │Reward r_2│  │Reward r_G│
           └────┬─────┘  └────┬─────┘  └────┬─────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
               ┌──────────────────────────────┐
               │   Compute Mean (μ) and       │
               │   Standard Deviation (σ)     │
               └──────────────┬───────────────┘
                              ▼
                    A_i = (r_i - μ) / σ
```

### Algorithm

1. For prompt $x$, sample $G$ completions: $\{y_1, \dots, y_G\} \sim \pi_\theta(\cdot|x)$
2. Score each: $r_i = R(x, y_i)$
3. Compute group statistics:
   $$\mu = \frac{1}{G} \sum_{i=1}^G r_i$$
   $$\sigma = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu)^2 + \epsilon_0}$$
4. Normalize to get relative advantage:
   $$A_i = \frac{r_i - \mu}{\sigma}$$

### Intuition
* $r_i > \mu$ (better than group average) → $A_i > 0$ → make those tokens more likely.
* $r_i < \mu$ (worse than group average) → $A_i < 0$ → make those tokens less likely.

The group mean $\mu$ **is the baseline**. No critic needed. Memory drops from 4 models to 2 (Actor + Reference).

---

## 7.5 Outcome vs. Process Supervision

### Outcome-Supervised RL (ORM)

Reward only at the very end:
```
 "Roger starts with 5..." ──► "He buys 2 more..." ──► "Answer is 11." [R = 1.0]
```
* All tokens in the trajectory share the same advantage $A_i$.
* Simple: just check if the final answer is correct.
* Limitation: If the model gets the right answer through wrong reasoning, it still gets $R=1$. Bad reasoning habits get reinforced.

### Process-Supervised RL (PRM)

Reward at each intermediate reasoning step:
```
 "Roger starts with 5..." ──► "2 × 3 = 6..." ──► "5 + 6 = 11."
       [R_1]                      [R_2]              [R_3]
```

* Cumulative future rewards at step $k$:
  $$G_{k,i} = \sum_{j=k}^{K_i} \gamma^{j-k} R_{i,j}$$
* Step-level advantage via group normalization:
  $$A_{t,i} = \frac{G_{t,i} - \mu_t}{\sigma_t}$$

Reinforces correct reasoning steps individually → more reliable reasoning paths.

### Practical Finding (DeepSeek-R1)
Process supervision provides marginal gains over outcome supervision on most benchmarks, but adds significant complexity (step-level verifiers). Outcome supervision is often sufficient — "keep it simple."

---

## 7.6 The Complete GRPO Loss Function

### 1. Group Surrogate Loss

Average the PPO clipped objective over the sampled group:

$$L_{\text{Surr}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( r_{t,i}(\theta) A_{t,i}, \; \text{clip}(r_{t,i}(\theta), 1-\epsilon, 1+\epsilon) A_{t,i} \right)$$

where $r_{t,i}(\theta) = \frac{\pi_\theta(y_{t,i} | x, y_{<t,i})}{\pi_{\theta_{old}}(y_{t,i} | x, y_{<t,i})}$

### 2. KL Divergence Penalty

Prevent the active policy from drifting too far from the reference (SFT) model:

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \sum_{a \in \mathcal{V}} \pi_\theta(a|s) \log\frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}$$

Token-level estimate:
$$D_{KL} \approx \log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{ref}(y_t|x,y_{<t})}$$

**Why critical in GRPO**: Without a critic to stabilize learning, the optimization is more vulnerable to variance. The KL penalty anchors the policy near the SFT model — preserving language coherence while adapting to rewards.

### 3. The Complete GRPO Objective

$$\boxed{L_{GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left(r_{t,i} A_{t,i}, \; \text{clip}(r_{t,i}, 1-\epsilon, 1+\epsilon) A_{t,i}\right) \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]}$$

---

## 7.7 PPO vs. GRPO: Final Comparison

| Feature | PPO | GRPO |
| :--- | :--- | :--- |
| **Critic / Value Model** | Required (trained neural net) | **None** (statistically estimated) |
| **Models in Memory** | 4 (Actor, Critic, Reference, Reward) | **2** (Actor, Reference) |
| **Memory Footprint** | Extremely High | **Low to Moderate** |
| **Advantage Baseline** | $V_\phi(s)$ (learned) | $\mu$ (group mean of sampled rewards) |
| **Regularization** | Clipping (+ optional KL) | Clipping + explicit KL penalty |
| **Key Use Case** | Classic RLHF (preference alignment) | Step-by-step reasoning (DeepSeek-R1) |

---

## The LLM Training Pipeline (Complete Picture)

```
[Random Initialization]
       │
       ▼ (Next-token prediction on web-scale data)
  [Pre-training] ──► [Base Model]
       │
       ▼ (Instruction-response pairs)
  [Supervised Fine-Tuning (SFT)] ──► [Instruction-Tuned Model]
       │
       ▼ (RL alignment)
  ┌────┴────┐
  ▼         ▼
[RLHF]   [RLVR]
 PPO      GRPO
 DPO      Process Supervision
  │         │
  ▼         ▼
[Aligned] [Reasoning]
 Model     Model
```

---

## Key Takeaways

1. **LLM generation = MDP**: State = prompt + tokens so far, Action = next token, Reward = end-of-sequence evaluation.
2. **PPO for LLMs** requires 4 models in memory — dominated by the Critic's cost.
3. **GRPO eliminates the Critic** by computing the baseline from group statistics ($\mu, \sigma$) of parallel samples.
4. **Outcome supervision** (single final reward) is simpler and often sufficient; **process supervision** (step-level rewards) can improve reasoning reliability.
5. **The KL penalty** against the reference model prevents catastrophic drift while allowing task-specific adaptation.

---

*This is the end of the notes. The arc from Chapter 1 to here: we started with what RL even is (trial and error, reward, value), formalized it as MDPs and Bellman equations, learned to solve it without a model (MC, TD, Q-learning), scaled it with function approximation, optimized policies directly (policy gradients → PPO), and finally applied it to the thing that matters most today — teaching language models to reason.*
