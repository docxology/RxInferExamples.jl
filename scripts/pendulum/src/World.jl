function restrict_engine_power(action) 
    return parameters.engine_max_power * tanh(action / parameters.engine_max_power)
end

function state_transition(previous_state, action) 
    # Transition function modeling transition due to gravity, friction and engine control
    (θ, θ̇) = previous_state
    θ̈ = 1 / (parameters.bob_mass * parameters.rod_length ^ 2) * 
        (-parameters.bob_mass * parameters.gravity * parameters.rod_length * sin(θ) - 
            parameters.friction * θ̇ .+ restrict_engine_power(action))
    Δs = (θ̇, θ̈)
    next_state = previous_state .+  Δs .* parameters.worlds_clock_Δt
    return next_state
end

Base.@kwdef mutable struct PendulumWorld
    pendulum_hidden_state :: Tuple{Float64, Float64} = (0.0, 0.0)
    next_registered_action  = 0.0
    noise_free_observations = RecentSubject(Float64)
    noisy_observations      = RecentSubject(Float64)
    ticks                   = Subject(Bool)
    observations_history    = CircularBuffer(30)
    actions_history         = CircularBuffer(30)
end

function tick!(world::PendulumWorld)
    next_hidden_state = state_transition(world.pendulum_hidden_state, world.next_registered_action)
    stochastic_state  = rand(MvNormalMeanPrecision(collect(next_hidden_state), 1e10 * diageye(2)))
            
    noise_free_observation = first(stochastic_state)
    noisy_observation      = rand(NormalMeanVariance(noise_free_observation, parameters.observations_noise))
        
    push!(world.actions_history, restrict_engine_power(world.next_registered_action))
    push!(world.observations_history, noisy_observation)
    
    world.next_registered_action = 0.0
    world.pendulum_hidden_state = (stochastic_state[1], stochastic_state[2])
    
    next!(world.noise_free_observations, noise_free_observation)
    next!(world.noisy_observations, noisy_observation)
    next!(world.ticks, true)
end

function register_next_action(world::PendulumWorld, action)
    world.next_registered_action = action
    return nothing
end

export restrict_engine_power, state_transition, PendulumWorld, tick!, register_next_action
