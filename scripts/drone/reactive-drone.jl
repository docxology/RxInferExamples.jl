# This file was automatically generated from /Users/4d/Documents/GitHub/RxInferExamples.jl/interactive/drone/reactive-drone.ipynb
# by NotebookConversion module at 2026-01-19T04:43:01.031
# Do not edit by hand. Edit the notebook instead.
#
# Source notebook: reactive-drone.ipynb

import Pkg; Pkg.activate("."); Pkg.instantiate();

# Used to reset the background task
const taskidcounter = Ref{Any}(1)

using RxInfer, Plots, LinearAlgebra, StableRNGs, StaticArrays, BenchmarkTools, DataStructures
import GLMakie, Makie 

function state_transition(state, action, drone, environment)
	dt, gravity, _ = environment
	_, radius, mass, limit = drone
	
	left_engine, right_engine = action
    position_x, position_y, acceleration_x, acceleration_y, angle = state
    
    left_engine = clamp(left_engine, -limit, limit)
    right_engine = clamp(right_engine, -limit, limit)
    
    change_in_acceleration_x = (left_engine + right_engine)/mass * sin(angle)
    change_in_acceleration_y = (left_engine + right_engine)/mass * cos(angle) - gravity/mass
    change_in_angle = (left_engine - right_engine) * radius

	new_acceleration_x = acceleration_x + dt * change_in_acceleration_x
	new_acceleration_y = acceleration_y + dt * change_in_acceleration_y
	
    return [
		position_x + dt * new_acceleration_x,
		position_y + dt * new_acceleration_y,
        new_acceleration_x,
        new_acceleration_y,
        rem(angle + dt * change_in_angle, 2π)
		# θ + dt * a_θ
    ]
end

mutable struct EnvironmentSettings 
	dt :: Float64
	Fg :: Float64
	noise :: Float64

	function EnvironmentSettings(; dt = 0.05, gravity = 9.8, noise = 1e-7)
		return new(dt, gravity, noise)
	end
end

begin
	Base.iterate(settings::EnvironmentSettings) = iterate((settings.dt, settings.Fg, settings.noise))
	Base.iterate(settings::EnvironmentSettings, state) = iterate((settings.dt, settings.Fg, settings.noise), state)
end

begin
	function plot_drone!(p, state, radius; color = :black)
		x, y, x_a, y_a, θ = state
		dx = radius * cos(θ)
		dy = radius * sin(θ)

		drone_position = [ x ], [ y ]
		drone_engines  = [ x - dx, x + dx ], [ y + dy, y - dy ]
		drone = [ x - dx, x, x + dx ], [ y + dy, y, y - dy ]

		rotation_matrix = [ cos(-θ) -sin(-θ); sin(-θ) cos(-θ) ]
		engine_shape = [ -1 0 1; 1 -1 1 ]
		drone_shape  = [ -2 -2 2 2 ; -1 1 1 -1 ]
		
		engine_shape = rotation_matrix * engine_shape
		drone_shape  = rotation_matrix * drone_shape
		engine_marker = Shape(engine_shape[1, :], engine_shape[2, :])
		drone_marker  = Shape(drone_shape[1, :], drone_shape[2, :])
		
		scatter!(p, drone_position[1], drone_position[2]; color = color, label = false, marker = drone_marker)
		scatter!(p, drone_engines[1], drone_engines[2]; color = color, label = false, marker = engine_marker, ms = 10)
		plot!(p, drone; color = color, label = false)
	
		return p
	end
	
	function plot_goal!(p, state, radius; color = :red)
		x, y, x_a, y_a, θ = state
		scatter!(p, [ x ], [ y ], markersize = 20, color = color, alpha = 0.3)
	end
end

function plot_drone_states(states)
	n = length(states)
	p = plot(getindex.(states, 1), getindex.(states, 2), xlim = (-1, 1), ylim = (-1, 1))

	states5 = states[1:(round(Int, n / 20)):n]

	for s in states5
		p = plot_drone!(p, s, r)
	end

	plot_goal!(p, goal, r; color = :red)

	p
end

function animate_drone_states(environment, drone, states, goals; fps = 30)
	n = length(states)

	radius = drone.settings.radius
	
	@gif for (state, goal) in zip(states, goals)
		p = plot(xlim = (-1, 1), ylim = (-1, 1), aspect_ratio = :equal)
		plot_drone!(p, state, radius)
		plot_goal!(p, goal, radius; color = :red)
	end fps = fps
end

@model function drone_dynamics(goalM, goalV, drone_settings, environment_settings, scurrent, actions)

	N = drone_settings.horizon
	
	s0 ~ MvNormal(mean = scurrent, covariance = 1e-6 * diageye(5))

	prev_s = s0
	
	for i in 1:N
		F[i] ~ MvNormal(mean = actions[i], covariance = diageye(2))
		s[i] ~ MvNormal(
			mean = state_transition(
				prev_s, F[i], drone_settings, environment_settings
			), 
			covariance = environment_settings.noise * diageye(5)
		)
		s[i] ~ MvNormal(mean = goalM[i], covariance = goalV[i])

		prev_s = s[i]
	end
	
end

@meta function drone_dynamics_approximations()
	state_transition() -> Unscented()
end

# The drone has a goal! Ultimately to survive, 
# but for now the goal is simpler (to reach a point in space)
struct DroneGoal
	goal_point :: Vector{Float64}
	goal_mean  :: Vector{Vector{Float64}}
	goal_cov   :: Vector{Matrix{Float64}}

	function DroneGoal(horizon::Int, n_state::Int)
		goal_point = zeros(n_state)
		goal_mean  = [ zeros(n_state) for i in 1:horizon ]
		goal_cov   = [ 0.1 * diageye(n_state) for i in 1:horizon ]
		return new(goal_point, goal_mean, goal_cov)
	end
end

function set_goal!(goal::DroneGoal, point_a, point_b)
	N = length(goal.goal_mean)
	V = 0.1 * diageye(length(point_a))

	copyto!(goal.goal_point, point_b)

	for (i, intermediate_point) in enumerate(range(point_a, point_b, length = N))
		@inbounds copyto!(goal.goal_mean[i], intermediate_point)
		@inbounds copyto!(goal.goal_cov[i], V)
	end

	return goal
end

function slide_goal!(goal::DroneGoal, current_position)
	n = length(first(goal.goal_mean))
	horizon = length(goal.goal_mean)

	V = 1e-12 * diageye(n)
		
	@inbounds copyto!(goal.goal_mean[1], current_position)
	@inbounds copyto!(goal.goal_cov[1], V)

	# inplace circshift
	@inbounds for i in 2:(horizon - 1)
		copyto!(goal.goal_mean[i], goal.goal_mean[i + 1])
		copyto!(goal.goal_cov[i], goal.goal_cov[i + 1])
	end

	@inbounds copyto!(goal.goal_mean[end], goal.goal_point)
	@inbounds copyto!(goal.goal_cov[end], V)
	
end

# Our drone has a limited capacity and does not remember everything!
# Only a small fraction of the previous history
struct DroneMemory
	states_history  :: CircularBuffer{MvNormalMeanCovariance}
	actions_history :: CircularBuffer{MvNormalMeanCovariance}

	function DroneMemory(n_memory::Int)
		states_history  = CircularBuffer{MvNormalMeanCovariance}(n_memory)
		actions_history = CircularBuffer{MvNormalMeanCovariance}(n_memory)
		return new(states_history, actions_history)
	end
end

function save_in_memory!(memory::DroneMemory, state, action)
	push!(memory.states_history, state)
	push!(memory.actions_history, action)
	return nothing
end

mutable struct DroneSettings 
	horizon :: Int
	radius :: Float64
	mass :: Float64
    engine_power_limit :: Float64

	function DroneSettings(; horizon = 50, radius = 0.5, mass = 5, engine_power_limit = 100)
		return new(horizon, radius, mass, engine_power_limit)
	end
end

begin
	Base.iterate(settings::DroneSettings) = iterate((settings.horizon, settings.radius, settings.mass, settings.engine_power_limit))
	Base.iterate(settings::DroneSettings, state) = iterate((settings.horizon, settings.radius, settings.mass, settings.engine_power_limit), state)
end

struct Drone
	rng :: StableRNG
	settings :: DroneSettings

	current_state  :: Vector{Float64}
	actions        :: Vector{Vector{Float64}}

	goal :: DroneGoal
	memory :: DroneMemory

	function Drone(;
		settings = DroneSettings(),
		current_state::Vector{Float64}, 
		n_memory = 200
	)
		rng     = StableRNG(42)
		n_state = length(current_state)
		actions = [ [ 9.8 / 2, 9.8 / 2 ] for i in 1:settings.horizon ]

		goal   = DroneGoal(settings.horizon, n_state)
		memory = DroneMemory(n_memory)

		return new(
			rng, settings, 
			deepcopy(current_state), actions, goal, memory
		)
	end
end

function set_state!(drone::Drone, state)
	copyto!(drone.current_state, state)
end

function set_goal!(drone::Drone, goal)
	return set_goal!(drone.goal, drone.current_state, goal)
end

function set_actions!(drone::Drone, actions)
	for (i, action) in enumerate(actions)
		copyto!(drone.actions, action)
	end
end

function save_in_memory!(drone::Drone, state, action)
	return save_in_memory!(drone.memory, state, action)
end

function slide_goal!(drone::Drone, estimated_position)
	return slide_goal!(drone.goal, estimated_position)
end

function infer!(drone::Drone, environment::EnvironmentSettings)
	current_estimated_position = drone.current_state
	current_desired_actions    = drone.actions

	probabilistic_model = drone_dynamics(
		drone_settings       = drone.settings,
		environment_settings = environment,
		scurrent 			 = current_estimated_position, 
		actions 			 = current_desired_actions,
	)

	goalM = drone.goal.goal_mean
	goalV = drone.goal.goal_cov
	
	result = infer(
		model = probabilistic_model,
		data = (goalM = goalM, goalV = goalV),
		meta = drone_dynamics_approximations()
	)

	next_estimated_set_of_states  = result.posteriors[:s]
	next_estimated_set_of_actions = result.posteriors[:F]

	return next_estimated_set_of_states, next_estimated_set_of_actions
end

function experiment(; 
		K = 200,
		drone_settings = DroneSettings(),
		environment = EnvironmentSettings(),
		current_state, goal, memory = 1000
	)

	local n = length(current_state)
	local rng = StableRNG(123)
	local drone = Drone(
		settings = drone_settings,
		current_state = current_state,
		n_memory = memory
	)

	local real_states = [ current_state ]
	local goals_history = [ goal ]
	local current_goal = copy(goal)

	set_goal!(drone, current_goal)

	moving_goal = 1
	
	for i in 1:K

		next_estimated_set_of_states, next_estimated_set_of_actions = infer!(
			drone, environment
		)

		taken_action = mean(next_estimated_set_of_actions[2])
		current_real_position = last(real_states)

		next_real_state = state_transition(
			current_real_position, taken_action, drone.settings, environment
		)

		measurement_distribution = MvNormalMeanCovariance(
			next_real_state, environment.noise .* diageye(n)
		)
		noisy_state_measurement = rand(rng, measurement_distribution)
		
		set_state!(drone, noisy_state_measurement)
		slide_goal!(drone, noisy_state_measurement)
		
		save_in_memory!(drone, 
			next_estimated_set_of_states[2], 
			next_estimated_set_of_actions[2]
		)

		push!(real_states, next_real_state)

		if norm(next_real_state[1:2] .- current_goal[1:2]) < 0.1
			# Core.println(K)
			angle = π / 2 * moving_goal
			x = 0.5sin(angle)
			y = 0.5cos(angle)
			copyto!(current_goal, [ 
				x + rand(-0.2:0.01:0.2), 
				y + rand(-0.2:0.01:0.2), 0.0, 0.0, 0.0 ]
			)
			set_goal!(drone, current_goal)

			moving_goal += 1
		end

		push!(goals_history, copy(drone.goal.goal_point))
	end
	
	return real_states, goals_history, drone
end

drone_settings = DroneSettings(
	horizon = 50,
	radius = 0.1,
	mass = 5,
    engine_power_limit = 100
)

environment = EnvironmentSettings(
	dt = 0.05,
	gravity = 9.8,
	noise = 1e-8
)

goal = [ -0.8, 0.1, 0.0, 0.0, 0.0 ]

trajectory, goals, drone = experiment(
	K = 400,
	drone_settings = drone_settings,
	environment = environment,
	goal = goal,
	current_state = zeros(5)
);

g = animate_drone_states(environment, drone, trajectory, goals)

begin
	actions_history = collect(drone.memory.actions_history)
	
	plot(getindex.(mean.(actions_history), 1), ribbon = getindex.(std.(actions_history), 1, 1), label = "left")
	plot!(getindex.(mean.(actions_history), 2), ribbon = getindex.(std.(actions_history), 2, 2), label = "right")
end

GLMakie.activate!(focus_on_show = true, title = "Drone with RxInfer😎")

begin 
    
	function gl_makie_plot_drone!(ax, o_drone_settings, o_states, o_prediction = nothing; n_history = 5)

		o_x = map(state -> getindex(state, 1), o_states)
		o_y = map(state -> getindex(state, 2), o_states)
		o_angle = map(state -> getindex(state, 5), o_states)
        o_minus_angle = map(state -> -getindex(state, 5), o_states)
		o_radius = map(s -> s.radius, o_drone_settings)
        
		o_engine_x_y = map(o_x, o_y, o_angle, o_radius) do x, y, angle, radius
			dx = radius * cos(angle)
			dy = radius * sin(angle)
			return ([ x - dx, x + dx ], [ y + dy, y - dy ])
		end

		o_engine_x = map(c -> getindex(c, 1), o_engine_x_y)
		o_engine_y = map(c -> getindex(c, 2), o_engine_x_y)

        o_drone_marker_scaler = map(o_drone_settings) do settings
			return 8 * settings.mass
		end
        
		o_engine_marker_scaler = map(o_drone_settings) do settings
			return 12 * settings.mass
		end
        
        o_engine_rods_scaler = map(o_drone_settings) do settings
			return 0.75 * settings.mass
		end
		
		GLMakie.scatter!(ax, o_x, o_y; color = :black, label = false, marker = :diamond, rotation = o_minus_angle, markersize = o_drone_marker_scaler)
		GLMakie.scatter!(ax, o_engine_x, o_engine_y; color = :black, label = false, marker = 'T', rotation = o_minus_angle, markersize = o_engine_marker_scaler)
		GLMakie.lines!(ax, o_engine_x, o_engine_y; color = :black, label = false, linewidth = o_engine_rods_scaler)
        
        if !isnothing(o_prediction)
            o_prediction_x = map(prediction -> getindex.(prediction, 1), o_prediction)
            o_prediction_y = map(prediction -> getindex.(prediction, 2), o_prediction)
            GLMakie.scatter!(ax, o_prediction_x, o_prediction_y; color = (:red, 0.1), label = false, markersize = 15)
        end
                
        if n_history > 0
            local history      = CircularBuffer{Vector{Float64}}(n_history)
            local save_counter = 1
            local save_every   = 5
                
            for i in 1:n_history
                push!(history, zeros(5))
            end
                    
            local o_history = map(o_states) do state 
                if save_counter % save_every == 0
                    push!(history, state)
                end
                save_counter += 1
                return history
            end
                    
            local o_history_x = map(state -> getindex.(state, 1), o_history)
            local o_history_y = map(state -> getindex.(state, 2), o_history)
            local history_colors = map(a -> (:black, a), range(0.1, 0.6; length = n_history))
                
            GLMakie.scatter!(ax, o_history_x, o_history_y, color = history_colors, marker = :circle, markersize = o_drone_marker_scaler)
        end
	
		return fig
	end
    
    function gl_makie_plot_goal!(ax, o_goal)
		o_goal_x = map(state -> getindex(state, 1), o_goal)
		o_goal_y = map(state -> getindex(state, 2), o_goal)
        
        GLMakie.scatter!(ax, o_goal_x, o_goal_y, color = (:red, 0.5), marker=:xcross, markersize = 100)
    end
        
    function gl_makie_plot_actions!(ax, o_actions; n_history = 50)
        local history = CircularBuffer{MvNormalMeanCovariance}(n_history)
        for i in 1:n_history
            push!(history, vague(MvNormalMeanCovariance, 2))
        end
                
        local o_history = map(o_actions) do action
            push!(history, action)
            return history
        end
        
        xpoints = 1:50
                
        o_left_engine_band = map(o_history) do actions
            means = mean.(actions)
            stds  = std.(actions)
            return (getindex.(means, 1), getindex.(stds, 1, 1))
        end
                
        o_right_engine_band = map(o_history) do actions
            means = mean.(actions)
            stds  = std.(actions)
            return (getindex.(means, 2), getindex.(stds, 2, 2))
        end
                
        o_left_engine    = map(first, o_left_engine_band)
        o_left_engine_lb = map(band -> band[1] .- 3band[2], o_left_engine_band)
        o_left_engine_ub = map(band -> band[1] .+ 3band[2], o_left_engine_band)
                
        o_right_engine    = map(first, o_right_engine_band)
        o_right_engine_lb = map(band -> band[1] .- 3band[2], o_right_engine_band)
        o_right_engine_ub = map(band -> band[1] .+ 3band[2], o_right_engine_band)
                
        GLMakie.lines!(ax, xpoints, o_left_engine, label = "Left engine", color = :red)
        GLMakie.band!(ax, xpoints, o_left_engine_lb, o_left_engine_ub, color = (:red, 0.2))
                
        GLMakie.lines!(ax, xpoints, o_right_engine, label = "Right engine", color = :blue)
        GLMakie.band!(ax, xpoints, o_right_engine_lb, o_right_engine_ub, color = (:dodgerblue, 0.2))
    end
                
    function gl_makie_plot_states_position!(ax, o_states; n_history = 50)
        local history = CircularBuffer{MvNormalMeanCovariance}(n_history)
        for i in 1:n_history
            push!(history, vague(MvNormalMeanCovariance, 5))
        end
                
        local o_history = map(o_states) do state
            push!(history, state)
            return history
        end
        
        xpoints = 1:50
                
        o_pos_x_band = map(o_history) do states
            means = mean.(states)
            stds  = std.(states)
            return (getindex.(means, 1), getindex.(stds, 1, 1))
        end
                
        o_pos_y_band = map(o_history) do states
            means = mean.(states)
            stds  = std.(states)
            return (getindex.(means, 2), getindex.(stds, 2, 2))
        end
                
        o_pos_x    = map(first, o_pos_x_band)
        o_pos_x_lb = map(band -> band[1] .- 150band[2], o_pos_x_band)
        o_pos_x_ub = map(band -> band[1] .+ 150band[2], o_pos_x_band)
                
        o_pos_y    = map(first, o_pos_y_band)
        o_pos_y_lb = map(band -> band[1] .- 150band[2], o_pos_y_band)
        o_pos_y_ub = map(band -> band[1] .+ 150band[2], o_pos_y_band)
                
        GLMakie.lines!(ax, xpoints, o_pos_x, label = "Position x", color = :red)
        GLMakie.band!(ax, xpoints, o_pos_x_lb, o_pos_x_ub, color = (:red, 0.2))
                
        GLMakie.lines!(ax, xpoints, o_pos_y, label = "Position y", color = :blue)
        GLMakie.band!(ax, xpoints, o_pos_y_lb, o_pos_y_ub, color = (:dodgerblue, 0.2))
    end
        
    function gl_makie_register_mouse_events!(ax, o_goal)
        Makie.on(Makie.events(ax).mousebutton) do event
            if event.button == Makie.Mouse.left
                if event.action == Makie.Mouse.press
                    mp = Makie.events(ax.scene).mouseposition[]
                    width_x, width_y = Makie.events(ax.scene).window_area[].widths
                    new_goal_x = clamp(2(mp[1] / width_x - 0.5), -0.75, 0.75)
                    new_goal_y = clamp(2(mp[2] / width_y - 0.5), -0.75, 0.75)
                    o_goal[] = [ new_goal_x, new_goal_y, 0, 0, 0 ]
                    @async begin 
                        Makie.center!(ax.scene)
                    end
                end
            end
            return Makie.Consume(true)
        end
    end
end

begin 
    c_fontsize = 74
    
	fig = GLMakie.Figure(
        fontsize = c_fontsize, 
        resolution = (1200, 800), 
        dpi = 300,
        fonts = (regular = "Produkt", bold = "Produkt",),
        figure_padding = 150,
    )
    
	ax = GLMakie.Axis(
        fig[1:4, 1:6], 
        limits = (-1, 1, -1, 1), 
        aspect = 1,
        title = "",
        yzoomlock = true,
        xzoomlock = true,
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
    )
    
    engine_ax = GLMakie.Axis(
        fig[5, 1:2], 
        limits = (0, 50, -30, 30), 
        title = "Engines thrust",
        yzoomlock = true,
        xzoomlock = true,
        xgridvisible = false,
        xticks = [-1],
        yticks = [-1000]
    )
    
    states_position_ax = GLMakie.Axis(
        fig[5, 3:4], 
        limits = (0, 50, -1, 1), 
        title = "Estimated position",
        yzoomlock = true,
        xzoomlock = true,
        xgridvisible = false,
        xticks = [-1],
        yticks = [-1000]
    )
    
    GLMakie.hidedecorations!(ax)
    # GLMakie.hidedecorations!(engine_ax)
    
    sg = GLMakie.SliderGrid(
        fig[5, 5:6],
        (label = "Mass", range = 2:0.1:15, format = "{:.1f}g", startvalue = 5.0),
        (label = "Radius", range = 0.05:0.01:0.6, format = "{:.1f}cm", startvalue = 0.1),
        (label = "Engine power", range = 1:0.1:50, format = "{:.1f}m/s²", startvalue = 20),
        (label = "Gravity", range = 4.6:0.1:12.4, format = "{:.1f}m/s²", startvalue = 9.8),
        (label = "Noise", range = map(exp10, range(-9, -4, length=20)), format = "{:.2e}", startvalue = exp10(-8)),
        width = 1100,
        tellheight = true,
        # tellheight = true
    )
    
    r_mass = sg.sliders[1].value
    r_radius = sg.sliders[2].value
    r_engine_limit = sg.sliders[3].value
    r_gravity = sg.sliders[4].value
    r_noise = sg.sliders[5].value
    
    drone_settings = DroneSettings(
        horizon = 50,
        radius = r_radius[],
        mass = r_mass[],
        engine_power_limit = r_engine_limit[]
    )
    
    environment = EnvironmentSettings(
        dt = 0.05,
        gravity = r_gravity[],
        noise = r_noise[]
    )
    
    local rng = StableRNG(123)
    local drone = Drone(
        settings = drone_settings,
        current_state = zeros(5),
        n_memory = 1
    )
        
    o_dynamic_goal   = GLMakie.Observable{Bool}(true)
    o_states         = GLMakie.Observable{Vector{Float64}}(zeros(5))
    o_estimated      = GLMakie.Observable{MvNormalMeanCovariance}(vague(MvNormalMeanCovariance, 5))
    o_actions        = GLMakie.Observable{MvNormalMeanCovariance}(vague(MvNormalMeanCovariance, 2))
    o_goal           = GLMakie.Observable{Vector{Float64}}(zeros(5))
	o_drone_settings = GLMakie.Observable{DroneSettings}(drone.settings)
    o_environment    = GLMakie.Observable{EnvironmentSettings}(environment)
    o_prediction     = GLMakie.Observable{Vector{Vector{Float64}}}([ zeros(5) for _ in 1:drone_settings.horizon ])

    Makie.on(r_mass) do new_mass
        drone_settings.mass = new_mass
        o_drone_settings[] = drone_settings
    end

    Makie.on(r_radius) do new_radius
        drone_settings.radius = new_radius
        o_drone_settings[] = drone_settings
    end

    Makie.on(r_engine_limit) do new_limit
        drone_settings.engine_power_limit = new_limit
        o_drone_settings[] = drone_settings
    end

    Makie.on(r_gravity) do new_gravity
        environment.Fg = new_gravity
        o_environment[] = environment
    end

    Makie.on(r_noise) do new_noise
        environment.noise = new_noise
        o_environment[] = environment
    end

    Makie.on(o_goal) do new_goal 
        set_goal!(drone, o_goal[])
    end

	gl_makie_plot_drone!(ax, o_drone_settings, o_states, o_prediction)
    gl_makie_plot_goal!(ax, o_goal)
    # gl_makie_register_mouse_events!(ax, o_goal)
    gl_makie_plot_actions!(engine_ax, o_actions)
    gl_makie_plot_states_position!(states_position_ax, o_estimated)

	display(fig, scalefactor = 1)
    
    Threads.@spawn begin
        taskidcounter[] += 1
        current_task_id_counter = taskidcounter[]
        i = 1
        goal_id = 1

        local n = length(o_states[])
        
        while i < 10000 && (taskidcounter[] == current_task_id_counter)
            
            next_estimated_set_of_states, next_estimated_set_of_actions = infer!(
                drone, environment
            )
        
            o_prediction[] = mean.(next_estimated_set_of_states)

            taken_action = next_estimated_set_of_actions[2]
        
            # clamp according to the engine limit
            taken_action_mean, taken_action_cov = mean_cov(taken_action)
            taken_action_mean = clamp.(taken_action_mean, -drone.settings.engine_power_limit, drone.settings.engine_power_limit)
        
            current_real_position = o_states[]

            next_real_state = state_transition(
                current_real_position, taken_action_mean, drone.settings, environment
            )

            measurement_distribution = MvNormalMeanCovariance(
                next_real_state, environment.noise .* diageye(n)
            )
            noisy_state_measurement = rand(rng, measurement_distribution)

            set_state!(drone, noisy_state_measurement)
            slide_goal!(drone, noisy_state_measurement)

            save_in_memory!(drone, 
                next_estimated_set_of_states[2], 
                next_estimated_set_of_actions[2]
            )

            if norm(next_real_state[1:2] .- o_goal[][1:2]) < 0.1 || (i % 100 == 0)
                angle = π / 2 * goal_id
                x = 0.5sin(angle)
                y = 0.5cos(angle)
                o_goal[] = [ 
                    x + rand(-0.3:0.01:0.3), y + rand(-0.3:0.01:0.3), 0.0, 0.0, 0.0 
                ]
                goal_id += 1
                # @show std(o_estimated[])[1, 1]
            end
            
            
            o_states[] = next_real_state
            o_estimated[] = next_estimated_set_of_states[2]
            o_actions[] = MvNormalMeanCovariance(taken_action_mean, taken_action_cov)
            Makie.center!(ax.scene)
            sleep(0.02)
            # @show o_trajectory[]
            i += 1
        end
    end
end