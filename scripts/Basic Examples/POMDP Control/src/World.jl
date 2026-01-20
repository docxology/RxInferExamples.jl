# WindyGridWorld environment with RxEnvironments integration
# Note: RxEnvironments, Plots already imported in PODMPControlApp.jl

"""
    WindyGridWorld{N}

A grid world environment with wind effects.
- `wind`: Tuple of N integers specifying upward wind force in each column
- `agents`: Vector of agents in the environment
- `goal`: Target (x, y) position
"""
struct WindyGridWorld{N}
    wind::NTuple{N,Int}
    agents::Vector
    goal::Tuple{Int,Int}
end

"""
    WindyGridWorldAgent

Agent with mutable position in the grid world.
"""
mutable struct WindyGridWorldAgent
    position::Tuple{Int,Int}
end

# RxEnvironments interface implementation

RxEnvironments.update!(env::WindyGridWorld, dt) = nothing

function RxEnvironments.receive!(env::WindyGridWorld{N}, agent::WindyGridWorldAgent, action::Tuple{Int,Int}) where {N}
    # Validate that only one axis of movement is non-zero
    if action[1] != 0 && action[2] != 0
        @assert false "Only one of the two actions can be non-zero"
    end
    
    # Compute new position with wind effect
    wind_effect = env.wind[clamp(agent.position[1], 1, N)]
    new_position = (
        agent.position[1] + action[1],
        agent.position[2] + action[2] + wind_effect
    )
    
    # Clamp to grid bounds
    if all(elem -> 0 < elem <= N, new_position)
        agent.position = new_position
    end
end

function RxEnvironments.what_to_send(env::WindyGridWorld, agent::WindyGridWorldAgent)
    return agent.position
end

function RxEnvironments.what_to_send(agent::WindyGridWorldAgent, env::WindyGridWorld)
    return agent.position
end

function RxEnvironments.add_to_state!(env::WindyGridWorld, agent::WindyGridWorldAgent)
    push!(env.agents, agent)
end

"""
    reset_env!(environment)

Reset all agents in the environment to the starting position.
"""
function reset_env!(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    for agent in env.agents
        agent.position = grid_params.start
    end
    for subscriber in RxEnvironments.subscribers(environment)
        send!(subscriber, environment, grid_params.start)
    end
    rxlog("debug", "Environment reset to $(grid_params.start)")
end

"""
    plot_environment(environment)

Create a scatter plot visualization of the environment state.
"""
function plot_environment(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    N = grid_params.grid_size
    
    p1 = scatter(
        [env.goal[1]], [env.goal[2]],
        color=:blue, label="Goal",
        xlims=(0, N + 1), ylims=(0, N + 1),
        xlabel="X", ylabel="Y",
        title="Windy Grid World"
    )
    
    for (i, agent) in enumerate(env.agents)
        scatter!(p1, [agent.position[1]], [agent.position[2]],
            color=:red, label=i == 1 ? "Agent" : nothing)
    end
    
    return p1
end

"""
    create_environment()

Create a new WindyGridWorld environment with an agent.
Returns (environment, agent) tuple.
"""
function create_environment()
    env = RxEnvironment(WindyGridWorld(grid_params.wind, [], grid_params.goal))
    agent = add!(env, WindyGridWorldAgent(grid_params.start))
    rxlog("info", "Created WindyGridWorld environment with goal=$(grid_params.goal)")
    return (env, agent)
end

export WindyGridWorld, WindyGridWorldAgent
export reset_env!, plot_environment, create_environment
