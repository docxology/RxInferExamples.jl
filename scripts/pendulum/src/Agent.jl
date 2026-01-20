mutable struct SuperSmartRxInferAgent
    datastream               :: AbstractSubscribable
    rxinfer_engine           :: Union{Nothing, RxInferenceEngine}
    mean_control_priors      :: Vector{Float64}
    var_control_priors       :: Vector{Float64}
    mean_goal_priors         :: Vector{Float64}
    var_goal_priors          :: Vector{Float64}
    mean_current_state_prior :: Vector{Float64}
    cov_current_state_prior  :: Matrix{Float64}
    subscriptions            :: Vector{Teardown}
    execution_time           :: AbstractSubject
    vmp_iterations           :: AbstractSubject
    recent_action            :: AbstractSubject
    free_energy              :: AbstractSubject
    the_goal_in_radians      :: AbstractSubject
    the_goal_variance        :: AbstractSubject

    function SuperSmartRxInferAgent(T::Int, datastream::AbstractSubscribable)
        mean_control_priors = zeros(T)
        var_control_priors  = zeros(T)
        mean_goal_priors    = zeros(T)
        var_goal_priors     = zeros(T)
        mean_current_state_prior = zeros(2)
        cov_current_state_prior  = zeros(2, 2)
        execution_time           = Subject(Float64)
        vmp_iterations           = BehaviorSubject(5)
        recent_action            = RecentSubject(Float64)
        free_energy              = Subject(Float64)
        the_goal_in_radians      = BehaviorSubject(3.14)
        the_goal_variance        = BehaviorSubject(1e-3)
        subscriptions            = []

        agent = new(datastream, nothing,
            mean_control_priors, var_control_priors,
            mean_goal_priors, var_goal_priors,
            mean_current_state_prior, cov_current_state_prior,
            subscriptions, execution_time, vmp_iterations, recent_action,
            free_energy, the_goal_in_radians, the_goal_variance,
        )

        reset!(agent)

        return agent
    end
end

function reset!(agent::SuperSmartRxInferAgent)
    # Use numeric large/small constants, not the functions `huge`/`tiny`
    fill!(agent.mean_control_priors, 0.0)
    fill!(agent.var_control_priors, 1e10)       # previously `huge` (function)
    fill!(agent.mean_goal_priors, 0.0)
    fill!(agent.var_goal_priors, 1e10)          # previously `huge`
    agent.mean_current_state_prior = [ 0.0, 0.0 ]
    agent.cov_current_state_prior = 1e-12 * diageye(2)  # previously `tiny * diageye(2)`

    rxlog("debug", "reset! called: mean_control_priors=$(agent.mean_control_priors), var_control_priors=$(agent.var_control_priors)")

    return nothing
end

function start!(agent::SuperSmartRxInferAgent)
    if !isnothing(agent.rxinfer_engine)
        stop!(agent)
    end

    rxlog("info", "start! called - creating inference engine")

    # Slide helpers: replace `huge` with numeric large value
    shift_mean_control_priors = shift(agent.mean_control_priors, 0.0)
    shift_var_control_priors  = shift(agent.var_control_priors, 1e10)
    shift_mean_goal_priors    = shift(agent.mean_goal_priors, agent.the_goal_in_radians)
    shift_var_goal_priors     = shift(agent.var_goal_priors, agent.the_goal_variance)

    # valid autoupdate function must take a single argument
    pick_current_state_fn = (states) -> pick_current_state(agent, states)

    autoupdates = @autoupdates begin
        m_u = shift_mean_control_priors(q(u))
        v_u = shift_var_control_priors(q(u))
        m_x = shift_mean_goal_priors(q(x))
        v_x = shift_var_goal_priors(q(x))
        m_u_t_min, v_u_t_min = pick_first_action(q(u))
        m_s_t_min, v_s_t_min = pick_current_state_fn(q(s_t_min))
        n_alpha, n_theta = soft_noise_prior(q(n))
    end

    initial_forces = map(agent.mean_control_priors, agent.var_control_priors) do m, v
        return NormalMeanVariance(m, v)
    end

    T = length(agent.mean_control_priors)
    iterations_ref = Ref(0)

    vmp_iterations_subscription = subscribe!(agent.vmp_iterations, (vmp_iters) -> begin
        iterations_ref[] = vmp_iters
        rxlog("debug", "vmp_iterations => $vmp_iters")
        if !(isnothing(agent.rxinfer_engine))
            agent.rxinfer_engine.fe_actor.score = zeros(vmp_iters, 30)
            agent.rxinfer_engine.fe_actor.cframe = 1
            agent.rxinfer_engine.fe_actor.cindex = 0
            agent.rxinfer_engine.fe_actor.valid = falses(30)
            rxlog("debug", "Reinitialized fe_actor scoring arrays")
        end
    end)

    init_func = make_pendulum_initialization(agent)

    engine = infer(
        model = pendulum(
            T=T, 
            P=1e10*diageye(2), 
            C=[1.0, 0.0],
        ),
        meta = pendulum_meta(),
        constraints = pendulum_constraints(),
        datastream = agent.datastream,
        autoupdates = autoupdates,
        initialization = init_func(),
        autostart = false,
        returnvars = (:u, ),
        historyvars = (u = KeepLast(), s_t = KeepLast(), n = KeepLast()),
        keephistory = 30,
        free_energy = true,
        free_energy_diagnostics = nothing,
        iterations = iterations_ref,
        events = Val((:before_auto_update, :on_tick))
    )

    on_tick_events = engine.events |> filter(event -> event isa RxInferenceEvent{:on_tick})

    on_tick_subscription = subscribe!(on_tick_events, (args...) -> begin
        slide_msg_idx = 3
        graph     = RxInfer.getmodel(engine.model)
        returnval = RxInfer.getreturnval(graph)[1]
        variable  = RxInfer.getvariable(RxInfer.getvarref(graph, returnval))
        predictive_message = getrecent(messageout(variable, slide_msg_idx))
        (m_s_t_min, v_s_t_min) = mean_cov(predictive_message)

        agent.mean_current_state_prior = m_s_t_min
        agent.cov_current_state_prior = v_s_t_min
        # rxlog("debug", "on_tick updated agent mean/cov priors")
    end)

    recent_action_subscription = subscribe!(engine.posteriors[:u], (actions) -> begin
        next!(agent.recent_action, mode(first(actions)))
    end)

    free_energy_subscription = subscribe!(engine.free_energy, (value) -> begin
        next!(agent.free_energy, value)
    end)

    # rxlog("debug", "All subscriptions created, adding to agent")
    push!(agent.subscriptions, vmp_iterations_subscription)
    push!(agent.subscriptions, on_tick_subscription)
    push!(agent.subscriptions, recent_action_subscription)
    push!(agent.subscriptions, free_energy_subscription)

    agent.rxinfer_engine = engine

    rxlog("debug", "Starting inference engine")
    RxInfer.start(engine)
    rxlog("info", "Agent started")

    return nothing, engine
end

function stop!(agent::SuperSmartRxInferAgent)
    rxlog("info", "stop! called")
    if !isnothing(agent.rxinfer_engine)
        RxInfer.stop(agent.rxinfer_engine)
    end
    foreach(subscription -> unsubscribe!(subscription), agent.subscriptions)
    agent.rxinfer_engine = nothing
    agent.subscriptions = []
    rxlog("info", "Agent stopped and subscriptions removed")
    return nothing
end

export SuperSmartRxInferAgent, reset!, start!, stop!
