# POMDP Agent orchestrating inference and control
# Note: RxInfer, Distributions, RxEnvironments already imported in PODMPControlApp.jl
# See: docs/agent_architecture.md for documentation

"""
    select_action(policy::Categorical) -> Int

Select an action from the policy distribution.

If `experiment_params.stochastic_actions` is true, samples from the policy
distribution with optional temperature scaling. Otherwise, returns the mode.

Temperature effects (when stochastic):
- τ = 1.0: Sample directly from policy probabilities
- τ > 1.0: More uniform (higher exploration)
- τ < 1.0: More peaked (lower exploration, closer to greedy)

See: docs/mathematical_foundations.md#policy-selection
"""
function select_action(policy::Categorical)
    if experiment_params.stochastic_actions
        probs = params(policy)[1]
        τ = experiment_params.exploration_temperature
        
        if τ != 1.0
            # Apply temperature scaling: p_i' = p_i^(1/τ) / Σ p_j^(1/τ)
            log_probs = log.(probs .+ 1e-10)  # Avoid log(0)
            scaled_log_probs = log_probs ./ τ
            scaled_probs = exp.(scaled_log_probs .- maximum(scaled_log_probs))
            scaled_probs = scaled_probs ./ sum(scaled_probs)
            return rand(Categorical(scaled_probs))
        else
            return rand(policy)
        end
    else
        return mode(policy)
    end
end

"""
    PODMPAgent

Orchestrates POMDP inference and control in the WindyGridWorld.
"""
mutable struct PODMPAgent
    # Environment references
    environment::Any
    rx_agent::Any
    observations::Any
    
    # Learned model parameters
    p_A::Any  # Observation model posterior
    p_B::Any  # Transition model posterior
    
    # State beliefs
    p_s::Any  # Current state belief
    
    # Goal specification
    goal::Any
    
    # Policy and action tracking
    current_policy::Any  # Current policy (vector of Categoricals)
    prev_action::Vector{Float64}
    
    # Statistics
    step_count::Int
end

"""
    PODMPAgent(env, agent)

Create a new POMDP agent attached to the given environment.
"""
function PODMPAgent(env, rx_agent)
    # Initialize model priors
    p_A, p_B = create_initial_priors()
    
    # Initial state belief at start position
    start_idx = grid_location_to_index(grid_params.start)
    p_s = Categorical(index_to_one_hot(start_idx))
    
    # Goal prior
    goal = create_goal_prior(grid_params.goal)
    
    # Create observation stream
    observations = keep(Any)
    RxEnvironments.subscribe_to_observations!(rx_agent, observations)
    
    # Send initial observation by resetting environment
    # This ensures the observation stream has the starting position before first step
    reset_env!(env)
    
    # Initialize policy: start with "down" as neutral from starting position
    initial_policy = [Categorical([0.0, 0.0, 1.0, 0.0])]  # Down
    prev_action = action_to_one_hot(ACTION_DOWN)
    
    agent = PODMPAgent(
        env, rx_agent, observations,
        p_A, p_B, p_s, goal,
        initial_policy, prev_action, 0
    )
    
    rxlog("info", "PODMPAgent created with goal=$(grid_params.goal)")
    return agent
end

"""
    reset!(agent)

Reset agent to initial state.
"""
function reset!(agent::PODMPAgent)
    # Reset environment
    reset_env!(agent.environment)
    
    # Reset state belief to start position
    start_idx = grid_location_to_index(grid_params.start)
    agent.p_s = Categorical(index_to_one_hot(start_idx))
    
    # Reset policy and action tracking
    agent.current_policy = [Categorical([0.0, 0.0, 1.0, 0.0])]  # Down
    agent.prev_action = action_to_one_hot(ACTION_DOWN)
    agent.step_count = 0
    
    rxlog("debug", "Agent reset")
end

"""
    step!(agent, remaining_horizon) -> Bool

Execute one inference-control step. Returns true if goal reached.
The order is: action → observe → infer (matching original notebook).
"""
function step!(agent::PODMPAgent, remaining_horizon::Int)
    agent.step_count += 1
    
    # First: Execute ACTION based on current policy (from previous inference or initial)
    # Choose action either stochastically (sample) or deterministically (mode)
    action_idx = select_action(first(agent.current_policy))
    movement = ACTION_MOVEMENTS[action_idx]
    send!(agent.environment, agent.rx_agent, movement)
    agent.prev_action = action_to_one_hot(action_idx)
    
    rxlog("debug", "Step $(agent.step_count): action=$(ACTION_NAMES[action_idx]) -> $movement")
    
    # Second: Get observation AFTER action
    last_obs = RxEnvironments.data(last(agent.observations))
    obs_one_hot = index_to_one_hot(grid_location_to_index(last_obs))
    
    rxlog("debug", "Step $(agent.step_count): obs=$last_obs")
    
    # Check if goal reached
    if last_obs == grid_params.goal
        return true
    end
    
    # Third: Run inference to update beliefs and get next policy
    T = max(remaining_horizon, 1)
    inference_result = infer(
        model = pomdp_model(
            p_A = agent.p_A,
            p_B = agent.p_B,
            T = T,
            p_previous_state = agent.p_s,
            p_goal = agent.goal,
            p_control = vague(Categorical, 4),
            m_A = mean(agent.p_A),
            m_B = mean(agent.p_B)
        ),
        data = (
            previous_control = agent.prev_action,
            current_y = obs_one_hot,
            future_y = UnfactorizedData(fill(missing, T))
        ),
        constraints = pomdp_constraints(),
        initialization = make_pomdp_initialization(),
        iterations = experiment_params.n_iterations
    )
    
    # Update beliefs
    agent.p_s = last(inference_result.posteriors[:current_state])
    agent.current_policy = last(inference_result.posteriors[:controls])
    
    # Update model parameters
    agent.p_A = last(inference_result.posteriors[:A])
    agent.p_B = last(inference_result.posteriors[:B])
    
    return false
end

"""
    run_experiment!(agent, T) -> Bool

Run a complete T-step experiment. Returns true if goal reached.
"""
function run_experiment!(agent::PODMPAgent, T::Int)
    reset!(agent)
    
    for t in 1:T
        goal_reached = step!(agent, T - t)
        if goal_reached
            rxlog("info", "Goal reached at step $t")
            return true
        end
    end
    
    final_pos = RxEnvironments.data(last(agent.observations))
    rxlog("info", "Experiment ended at position $final_pos (goal not reached)")
    return false
end

"""
    get_current_position(agent)

Get the agent's current position in the grid.
"""
function get_current_position(agent::PODMPAgent)
    return RxEnvironments.data(last(agent.observations))
end

export PODMPAgent, reset!, step!, run_experiment!, get_current_position
