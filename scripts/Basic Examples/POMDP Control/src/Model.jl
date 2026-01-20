# POMDP generative model for inference-as-planning
# Note: RxInfer, Distributions, LinearAlgebra already imported in PODMPControlApp.jl

# POMDP generative model for Windy Grid World control.
#
# Arguments:
# - p_A: Prior on observation likelihood matrix
# - p_B: Prior on transition matrix
# - p_goal: Prior belief about goal state
# - p_control: Prior on control actions
# - previous_control: One-hot encoded previous action
# - p_previous_state: Prior on previous state
# - current_y: Current observation (one-hot encoded position)
# - future_y: Future observations (missing for planning)
# - T: Planning horizon
# - m_A: Mean of observation matrix (for planning)
# - m_B: Mean of transition matrix (for planning)
@model function pomdp_model(
    p_A, p_B, p_goal, p_control,
    previous_control, p_previous_state,
    current_y, future_y, T, m_A, m_B
)
    # Instantiate all model parameters with priors
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    
    # Parameter inference: current state from previous action and observation
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)

    prev_state = current_state
    
    # Inference-as-planning loop
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
    
    # Goal prior: attract final state toward goal
    s[end] ~ p_goal
end

# Factorization constraints for the POMDP model.
@constraints function pomdp_constraints()
    q(previous_state, previous_control, current_state, B) = q(previous_state, previous_control, current_state)q(B)
    q(current_state, current_y, A) = q(current_state, current_y)q(A)
    q(current_state, s, controls, B) = q(current_state, s, controls)q(B)
    q(s, future_y, A) = q(s, future_y)q(A)
end

"""
    make_pomdp_initialization()

Create initialization block for POMDP inference with learned priors.
"""
function make_pomdp_initialization()
    n_states = grid_params.grid_size^2
    n_actions = 4
    
    @initialization begin
        q(A) = DirichletCollection(diageye(n_states) .+ 0.1)
        q(B) = DirichletCollection(ones(n_states, n_states, n_actions))
    end
end

"""
    create_initial_priors()

Create initial prior distributions for A and B matrices.
Returns (p_A, p_B) tuple.
"""
function create_initial_priors()
    n_states = grid_params.grid_size^2
    n_actions = 4
    
    p_A = DirichletCollection(diageye(n_states) .+ 0.1)
    p_B = DirichletCollection(ones(n_states, n_states, n_actions))
    
    return (p_A, p_B)
end

"""
    create_goal_prior(goal_pos)

Create a categorical distribution centered on the goal state.
"""
function create_goal_prior(goal_pos::Tuple{Int,Int})
    goal_idx = grid_location_to_index(goal_pos)
    return Categorical(index_to_one_hot(goal_idx))
end

export pomdp_model, pomdp_constraints, make_pomdp_initialization
export create_initial_priors, create_goal_prior
