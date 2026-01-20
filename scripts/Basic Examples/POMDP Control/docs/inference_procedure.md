# Inference Procedure

## Overview

The inference procedure implements **Variational Message Passing (VMP)** on a Forney-style Factor Graph (FFG). Each inference call:

1. Constructs the model graph
2. Clamps observed data
3. Runs belief propagation for parameters and states
4. Plans by inferring optimal future controls

## Factor Graph Structure

```
                    ┌─────────┐
                    │   A     │ (DirichletCollection)
                    └────┬────┘
                         │
┌──────────┐    ┌────────▼────────┐    ┌──────────┐
│prev_state├───►│ DiscreteTransition├───►│current_y │ (observed)
└──────────┘    │  (observation)   │    └──────────┘
                └─────────────────┘
                         ▲
                    ┌────┴────┐
                    │current  │
                    │ state   │
                    └────┬────┘
                         │
┌──────────┐    ┌────────▼────────┐
│prev_state├───►│ DiscreteTransition│
│          │    │  (transition)   │
└──────────┘    └────────┬────────┘
                         ▲
                    ┌────┴────┐
                    │    B    │ (DirichletCollection)
                    └─────────┘
```

## Message Passing Schedule

### Iteration Structure

For each of 20 VMP iterations:

1. **Forward messages**: $s_{t-1} \rightarrow s_t$ (prediction)
2. **Backward messages**: $o_t \rightarrow s_t$ (observation update)
3. **Parameter updates**: $s \rightarrow A, B$ (learning)
4. **Policy updates**: $s_{1:T} \rightarrow u_{1:T}$ (planning)

### DiscreteTransition Node

The `DiscreteTransition` node computes:

**Forward (prediction)**:
$$\mu_{s_t}^{\rightarrow} = B_{:,:,u_t} \cdot \mu_{s_{t-1}}$$

**Backward (observation)**:
$$\mu_{s_t}^{\leftarrow} = A^T \cdot \delta_{o_t}$$

**Posterior**:
$$q(s_t) \propto \mu_{s_t}^{\rightarrow} \odot \mu_{s_t}^{\leftarrow}$$

## Planning via Inference

The key insight of Active Inference: **planning = inference over future controls**.

### Unobserved Future

```julia
future_y = UnfactorizedData(fill(missing, T))
```

Future observations are **missing** - the model infers what they "should" be based on:
- Current state belief
- Learned transition dynamics
- Goal prior attraction

### Policy Inference

For each planning step $t$:
```julia
controls[t] ~ p_control  # vague prior = uniform
s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
future_y[t] ~ DiscreteTransition(s[t], m_A)
```

The backward message from `s[end] ~ p_goal` propagates to bias `controls[t]` toward goal-reaching actions.

## Extracting Results

After inference:

```julia
# Current state belief
agent.p_s = last(inference_result.posteriors[:current_state])

# Planned policy
agent.current_policy = last(inference_result.posteriors[:controls])

# Updated parameters
agent.p_A = last(inference_result.posteriors[:A])
agent.p_B = last(inference_result.posteriors[:B])
```

## Action Selection

The policy is a vector of Categorical distributions over actions:

```julia
action_idx = mode(first(agent.current_policy))
```

`mode()` returns the most probable action for the first planning step.

## Online Learning

### Parameter Updates

Each inference call updates A and B posteriors:

```julia
# Pseudo-counts accumulate from observations
p_A_new ∝ p_A_old + evidence_from_observations
p_B_new ∝ p_B_old + evidence_from_transitions
```

### Learning Dynamics

| Experiment | A Entropy | B Entropy | Success Rate |
|------------|-----------|-----------|--------------|
| 1-10       | High      | High      | 0%           |
| 20-30      | Medium    | Medium    | 15-40%       |
| 40-50      | Low       | Low       | 55-64%       |

As entropy decreases, the agent's model becomes more certain → better planning.

## Computational Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Forward pass | O(n² × T) | State space × horizon |
| Backward pass | O(n² × T) | Same |
| Parameter update | O(n² × a) | States × actions |
| Per iteration | O(n² × T × a) | ~25² × 4 × 4 = 10K ops |
| Total (20 iter) | O(200K ops) | Fast for small grids |

## Convergence Criteria

VMP iterations continue until:
- Fixed iteration count (20) reached
- Or KL divergence between successive beliefs < threshold

This implementation uses fixed iterations for simplicity.
