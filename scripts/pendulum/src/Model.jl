@model function pendulum(T, P, C, m_s_t_min, v_s_t_min, m_u_t_min, v_u_t_min, x_t, m_u, v_u, m_x, v_x, n_alpha, n_theta)
    
    n ~ InverseGamma(n_alpha, n_theta)

    s_t_min ~ MvNormal(mean = m_s_t_min, covariance = v_s_t_min) # Prior for previous state
    u_t_min ~ Normal(mean = m_u_t_min, variance = v_u_t_min)   # Prior for previous action
    u_s_min ~ state_transition(s_t_min, u_t_min)          # Deterministic state transition function
    s_t     ~ MvNormal(mean = u_s_min, precision = P) # Transition uncertainty
    x_t     ~ Normal(mean = dot(C, s_t), variance = n)   # Observational function
    
    s_k_min = s_t
    
    for k in 1:T
        u[k]    ~ Normal(mean = m_u[k], variance = v_u[k])
        u_s[k]  ~ state_transition(s_k_min, u[k])
        s[k]    ~ MvNormal(mean = u_s[k], precision = P)
        x[k]    ~ Normal(mean = dot(C, s[k]), variance = n)
        x[k]    ~ Normal(mean = m_x[k], variance = v_x[k]) 
        s_k_min = s[k]
    end
    return s_t, m_s_t_min, v_s_t_min
end

@meta function pendulum_meta()
    state_transition() -> DeltaMeta(method = Linearization())
end

@constraints function pendulum_constraints()
    q(s_t, x, s, u, n) = q(x, s, u, s_t)q(n)
end

function make_pendulum_initialization(agent)
    @initialization function pendulum_initialization()
        q(u) = map(agent.mean_control_priors, agent.var_control_priors) do m, v
            NormalMeanVariance(m, v)
        end
        q(n) = InverseGamma(4.0, 1.0)
        q(s_t_min) = MvNormalMeanCovariance([0.0; 0.0], 1e-12*diageye(2))
    end
    return pendulum_initialization
end

# Helper functions
pick_first_action = (actions) -> begin
    return mean_var(first(actions))
end

pick_current_state = (agent, states) -> begin
    return (agent.mean_current_state_prior, agent.cov_current_state_prior)
end

soft_noise_prior = (noise) -> begin
    μ = mean(noise)
    μ = μ > 0.1 ? 0.1 : μ
    v = 0.1
    α = μ ^ 2 / v + 2
    θ = μ * (α - 1)
    return (α, θ)
end

export pendulum, pendulum_meta, pendulum_constraints, make_pendulum_initialization, pick_first_action, pick_current_state, soft_noise_prior
