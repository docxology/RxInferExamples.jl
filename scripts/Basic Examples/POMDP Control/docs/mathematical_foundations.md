# Mathematical Foundations

## POMDP Formalism

A **Partially Observable Markov Decision Process (POMDP)** is defined by the tuple:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R, \gamma \rangle$$

| Symbol | Description | This Implementation |
|--------|-------------|---------------------|
| $\mathcal{S}$ | State space | 25 grid positions (5×5) |
| $\mathcal{A}$ | Action space | {up, right, down, left} |
| $\mathcal{O}$ | Observation space | Grid positions (fully observable in this case) |
| $T$ | Transition function $P(s'|s,a)$ | **B matrix** - learned online |
| $O$ | Observation function $P(o|s)$ | **A matrix** - learned online |
| $R$ | Reward function | Implicit via goal prior |
| $\gamma$ | Discount factor | Through planning horizon T |

## Active Inference Framework

Active Inference reformulates control as **inference**. Instead of maximizing reward, the agent minimizes **Free Energy**.

### Generative Model

The generative model specifies the agent's beliefs about how observations are generated:

```
P(o_{1:T}, s_{0:T}, u_{1:T}) = P(s_0) ∏_{t=1}^{T} P(s_t | s_{t-1}, u_t) P(o_t | s_t)
```

In RxInfer notation:
```julia
@model function pomdp_model(...)
    A ~ p_A                                    # Observation model prior
    B ~ p_B                                    # Transition model prior
    previous_state ~ p_previous_state          # State prior
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)
    ...
end
```

### A Matrix: Observation Likelihood

The **A matrix** encodes $P(o|s)$ - the probability of observing $o$ given hidden state $s$:

$$A_{ij} = P(o_i | s_j)$$

- **Shape**: `[n_observations × n_states]` = `[25 × 25]`
- **Prior**: `DirichletCollection(diageye(25) + 0.1)` (weakly diagonal)
- **Interpretation**: Initially assumes observations directly reveal state (identity with noise)

### B Matrix: Transition Dynamics

The **B matrix** encodes $P(s'|s,a)$ - action-conditioned state transitions:

$$B_{ijk} = P(s'_i | s_j, a_k)$$

- **Shape**: `[n_states × n_states × n_actions]` = `[25 × 25 × 4]`
- **Prior**: `DirichletCollection(ones(25, 25, 4))` (uniform - no initial knowledge)
- **Learning**: Updated online as agent experiences transitions

## Expected Free Energy (EFE)

The Expected Free Energy for policy $\pi$ quantifies the "badness" of future trajectories:

$$G(\pi) = \underbrace{\mathbb{E}_{Q}[\log Q(s_\tau|\pi) - \log P(s_\tau)]}_{\text{Epistemic Value (Information Gain)}} + \underbrace{\mathbb{E}_{Q}[-\log P(o_\tau)]}_{\text{Pragmatic Value (Goal-seeking)}}$$

### Components

1. **Epistemic Value**: Preference for reducing uncertainty about states
2. **Pragmatic Value**: Preference for observations consistent with goals

In this implementation, the goal prior $P(s_T) = \text{Categorical at goal position}$ drives pragmatic behavior.

## Belief Updates

### State Estimation

Given observation $o_t$ and action $u_{t-1}$:

$$Q(s_t) \propto P(o_t | s_t) \sum_{s_{t-1}} P(s_t | s_{t-1}, u_{t-1}) Q(s_{t-1})$$

This is implemented via message passing in RxInfer's `DiscreteTransition` nodes.

### Policy Selection

Policies are selected proportionally to negative EFE:

$$P(\pi) \propto \exp(-G(\pi))$$

In the model:
```julia
controls[t] ~ p_control  # vague prior
s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
future_y[t] ~ DiscreteTransition(s[t], m_A)
s[end] ~ p_goal  # Goal prior attracts planning
```

## Matrix Entropy

Matrix entropy measures certainty of learned parameters:

$$H(M) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j} M_{ij} \log M_{ij}$$

- **High entropy**: Uniform (uncertain) - agent hasn't learned yet
- **Low entropy**: Peaked (certain) - agent has learned structure

### References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Da Costa, L. et al. (2020). Active inference on discrete state-spaces
3. Parr, T. et al. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior
