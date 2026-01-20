function pendulum_bob_position(angle)
    return Point2f(parameters.rod_length * sin(angle), -parameters.rod_length * cos(angle))
end

function launch_dashboard()
    # Width of the controls
    c_width = 350
    c_fontsize = 14

    fig = Figure(fontsize=c_fontsize, size=(1080, 720))

    controls_grid = fig[1:4, 1] = GridLayout()
    pendulum_grid = fig[1:4, 2:3] = GridLayout()
    auxilary_grid = fig[1:4, 4] = GridLayout()

    display(fig, title="Pendulum with RxInfer😎")

    free_energy_buffer = CircularBuffer{Float64}(100)

    r_world_isrunning = Observable(true) # Is simulation running
    r_origin = Observable([Point2f(0, 0), Point2f(0, 0)]) # Position of the origin
    r_rod = Observable([Point2f(0, 0), Point2f(0, 0)]) # Position of the rod
    r_bob = Observable([Point2f(0, 0)])  # Position of the bob
    r_goal = Observable([Point2f(0, 0)]) # Position of the goal
    r_observations = Observable(Point2f[]) # Positions of noibsy observations (history)
    r_actions = Observable([Point2f(0, 0), Point2f(0, 0)]) # Actions (history)
    r_noise_history = Observable(map(_ -> Point2f(0, 0), 1:30)) # Inferred noise (history)
    r_noise_bandl = Observable(map(_ -> Point2f(0, 0), 1:30)) # Inferred noise lower band (history)
    r_noise_bandu = Observable(map(_ -> Point2f(0, 0), 1:30)) # Inferred noise upper band (history)
    r_free_energy = Observable([Point2f(0, 0)])
    r_free_energy_acc = Observable([Point2f(0, 0)])

    ax_actions_history = Axis(auxilary_grid[1, 1], limits=(0, 30, -1.5, 1.5), title="Agents actions")

    lines!(ax_actions_history, r_actions; linewidth=4, color=:blue)

    # Dashboard buttons and sliders
    b_grid = controls_grid[2, 1] = GridLayout()

    b_run = Button(b_grid[1, 1]; label="Activate agent", width=c_width / 2, fontsize=c_fontsize)
    b_areset = Button(b_grid[2, 1]; label="Erase agents's memory", width=c_width / 2, fontsize=c_fontsize)
    b_stop = Button(b_grid[3, 1]; label="Deactivate agent", width=c_width / 2, fontsize=c_fontsize)

    b_corrupt = Button(b_grid[1, 2]; label="Corrupt world's state", width=c_width / 2, fontsize=c_fontsize)
    b_wreset = Button(b_grid[2, 2]; label="Reset world's parameters", width=c_width / 2, fontsize=c_fontsize)


    sg = SliderGrid(
        controls_grid[1, 1],
        (label="Bob's mass", range=0.15:0.01:0.35, format="{:.3f}g", startvalue=parameters.bob_mass,),
        (label="Rod's Length", range=0.15:0.01:0.25, format="{:.3f}cm", startvalue=parameters.rod_length),
        (label="Maximum engine power", range=0.1:0.1:1.5, format="{:.1f}", startvalue=parameters.engine_max_power),
        (label="Pendulum's friction", range=0.1:0.01:0.3, format="{:.3f}", startvalue=parameters.friction),
        (label="World's gravity", range=1.0:0.1:50.0, format="{:.1f}", startvalue=parameters.gravity),
        (label="Observational noise", range=exp10.(-8.0:0.1:-2), format="{:.6f}", startvalue=parameters.observations_noise),
        (label="VMP iterations", range=1:25, startvalue=5),
        (label="Goal", range=0:0.01:2pi, startvalue=pi),
        (label="Goal variance", range=exp10.(-5.0:0.1:-2), startvalue=exp10(-3)),
        width=c_width,
        tellwidth=true,
        tellheight=true
    )

    r_mass = sg.sliders[1].value
    r_length = sg.sliders[2].value
    r_power = sg.sliders[3].value
    r_friction = sg.sliders[4].value
    r_gravity = sg.sliders[5].value
    r_noise = sg.sliders[6].value
    r_iters = sg.sliders[7].value
    r_goalp = sg.sliders[8].value
    r_goalv = sg.sliders[9].value

    ax_limits = (-0.3, 0.3, -0.3, 0.3)
    ax_pendulum = Axis(pendulum_grid[1, 1], limits=ax_limits, title="Pendulum", aspect=DataAspect())

    lines!(ax_pendulum, r_rod; linewidth=5, color=:black)
    scatter!(ax_pendulum, r_origin; strokewidth=2, strokecolor=:black, color=:black, markersize=20)
    scatter!(ax_pendulum, r_bob; strokewidth=2, strokecolor=:black, color=:black, markersize=map(m -> m * 500, r_mass))
    scatter!(ax_pendulum, r_goal; strokewidth=4, strokecolor=:red, color=(:red, 0.2), markersize=120)
    scatter!(ax_pendulum, r_observations; strokecolor=:green, color=(:green, :0.2), markersize=map(m -> m * 75, r_mass))

    ax_inferred_noise_history = Axis(auxilary_grid[2, 1], yscale=log10, limits=(0, 30, 1e-8, 1.0), title="Estimated noise precision")

    lines!(ax_inferred_noise_history, r_noise_history; linewidth=4, color=:blue)
    band!(ax_inferred_noise_history, r_noise_bandl, r_noise_bandu, color=(:blue, 0.2))
    hlines!(ax_inferred_noise_history, map(e -> [e], r_noise), color=:red)

    ax_free_energy_history = Axis(auxilary_grid[3, 1], limits=((0, 100), nothing), xticklabelsvisible=false, yticklabelsvisible=false, title="Bethe Free Energy")
    lines!(ax_free_energy_history, r_free_energy)

    ax_free_energy_acc_history = Axis(auxilary_grid[4, 1], limits=((1, 15), nothing), xticklabelsvisible=true, yticklabelsvisible=false, title="BFE minimization history")
    lines!(ax_free_energy_acc_history, r_free_energy_acc)

    ## Initialize the environment    

    world = PendulumWorld()
    agent = SuperSmartRxInferAgent(3, labeled(Val((:x_t,)), combineLatest(world.noisy_observations)))

    # Redraw the observations as soon as we have a new data point
    s_ticks = subscribe!(world.ticks, (_) -> begin
        r_observations[] = map(angle -> pendulum_bob_position(angle), world.observations_history)
        r_actions[] = map(((index, force),) -> Point2f(index, force), enumerate(world.actions_history))
        r_free_energy[] = map(((index, value),) -> Point2f(index, value), enumerate(free_energy_buffer))

        if !isnothing(agent.rxinfer_engine)
            if length(agent.rxinfer_engine.history[:n]) == 30
                rfem, rfev = mean(free_energy_buffer), clamp(var(free_energy_buffer), 1e-4, Inf)
                if !isnan(rfem) && !isinf(rfem) && !isnan(rfev) && !isinf(rfev)
                    ylims!(ax_free_energy_history, clamp(rfem - 20sqrt(rfev), 1e-8, Inf), rfem + 20sqrt(rfev))
                end
                rfeaccmin, rfeaccmax = minimum(agent.rxinfer_engine.free_energy_history), maximum(agent.rxinfer_engine.free_energy_history)
                rfeaccm, rfeaccv = mean(agent.rxinfer_engine.free_energy_history), clamp(var(agent.rxinfer_engine.free_energy_history), 1e-4, Inf)
                if !isnan(rfeaccm) && !isinf(rfeaccm) && !isnan(rfeaccv) && !isinf(rfeaccv)
                    xlims!(ax_free_energy_acc_history, 1, length(agent.rxinfer_engine.free_energy_history))
                    ylims!(ax_free_energy_acc_history, clamp(rfeaccmin - sqrt(rfeaccv), 1e-8, Inf), rfeaccmax + sqrt(rfeaccv))
                end

                noise_means = map((q_n) -> mean(q_n), agent.rxinfer_engine.history[:n])
                noise_vars = map((q_n) -> var(q_n), agent.rxinfer_engine.history[:n])
                r_free_energy_acc[] = map(((index, value),) -> Point2f(index, value), enumerate(agent.rxinfer_engine.free_energy_history))
                r_noise_history[] = map(((index, mean),) -> Point2f(index, mean), enumerate(noise_means))
                r_noise_bandl[] = map(((index, mean), var) -> Point2f(index, clamp(mean - sqrt(var), 1e-10, Inf)), enumerate(noise_means), noise_vars)
                r_noise_bandu[] = map(((index, mean), var) -> Point2f(index, clamp(mean + sqrt(var), 1e-10, Inf)), enumerate(noise_means), noise_vars)
            end
        end
    end)

    s_redraw = subscribe!(combineLatest(world.noise_free_observations, agent.the_goal_in_radians), ((angle, goal),) -> begin
        origin_position = Point2f(0.0, 0.0)
        bob_position = pendulum_bob_position(angle)
        r_rod[] = [origin_position, bob_position]
        r_bob[] = [bob_position]
        r_goal[] = [pendulum_bob_position(goal)]
    end)

    # Register a new action as soon as we have it
    s_actions = subscribe!(agent.recent_action, (a) -> register_next_action(world, a))
    s_free_energy = subscribe!(agent.free_energy, (v) -> push!(free_energy_buffer, v))

    ## START THE SHOW!!

    # The world runs independently of the agent, but can be force-stopped as well
    @async begin
        try
            while isopen(fig.scene) && r_world_isrunning[]
                tick!(world) # Changed from tick(world) to tick!(world)
                sleep(1 / 60)
            end
        catch err
            println("An error happened inside our beautiful world!")
            showerror(stderr, err, catch_backtrace())
        end
        unsubscribe!(s_actions)
        unsubscribe!(s_free_energy)
        unsubscribe!(s_ticks)
        unsubscribe!(s_redraw)
    end



    # Implement buttons logic
    on(b_run.clicks) do clicks
        try
            reset!(agent)
            start!(agent)
        catch err
            rxlog("error", "An error happened inside our beautiful click!: $err")
            println("An error happened inside our beautiful click!")
            showerror(stderr, err, catch_backtrace())
        end
    end

    on(b_areset.clicks) do clicks
        reset!(agent)
    end

    on(b_corrupt.clicks) do clicks
        world.pendulum_hidden_state = (0.0, 0.0)
    end

    on(b_wreset.clicks) do clicks
        local rparams = PendulumWorldParameters()
        r_length[] = parameters.rod_length = rparams.rod_length
        r_mass[] = parameters.bob_mass = rparams.bob_mass
        r_friction[] = parameters.friction = rparams.friction
        r_gravity[] = parameters.gravity = rparams.gravity
        r_power[] = parameters.engine_max_power = rparams.engine_max_power
        r_noise[] = parameters.observations_noise = rparams.observations_noise
        parameters.worlds_clock_Δt = rparams.worlds_clock_Δt
    end

    on((_) -> stop!(agent), b_stop.clicks)

    # Implement sliders logic

    on((length) -> begin
        global parameters.rod_length = length
    end, r_length)
    on((mass) -> begin
        global parameters.bob_mass = mass
    end, r_mass)
    on((power) -> begin
        global parameters.engine_max_power = power
    end, r_power)
    on((friction) -> begin
        global parameters.friction = friction
    end, r_friction)
    on((gravity) -> begin
        global parameters.gravity = gravity
    end, r_gravity)
    on((noise) -> begin
        global parameters.observations_noise = noise
    end, r_noise)
    on((iters) -> begin
        next!(agent.vmp_iterations, iters)
    end, r_iters)
    on((goal) -> begin
        next!(agent.the_goal_in_radians, goal)
    end, r_goalp)
    on((var) -> begin
        next!(agent.the_goal_variance, var)
    end, r_goalv)

    return fig
end

export launch_dashboard
