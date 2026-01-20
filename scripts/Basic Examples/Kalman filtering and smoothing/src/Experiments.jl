module Experiments

using RxInfer
using LinearAlgebra
using Random
using Distributions
using Statistics

using ..DataGeneration
using ..Models
using ..Visualization
using ..Analysis
using ..Reporting
using ..LoggingUtils

export run_rotating_ssm, run_identification_problem, run_rx_identification, run_smoothing_example

"""
    run_rotating_ssm(config, output_dir, seed, report_path, logger)

Run the Rotating State-Space Model example.
"""
function run_rotating_ssm(config, output_dir, seed_val, report_path, logger)
    log_section(logger, "Rotating Set-Point SSM")
    logger("Starting Rotating Set-Point SSM...")
    
    exp_dir = joinpath(output_dir, "rotating_ssm")
    mkpath(exp_dir)

    conf = get(config, "rotating_ssm", Dict())
    n = get(conf, "n_observations", 300)
    theta_denom = get(conf, "theta_denominator", 35.0)
    Q_val = get(conf, "noise_variance_Q", 25.0)
    prior_var = get(conf, "prior_variance", 100.0)

    rng = MersenneTwister(seed_val)
    
    θ = π / theta_denom
    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    B = diageye(2)
    Q = Q_val * diageye(2)
    P = diageye(2)

    x, y = generate_rotating_ssm_data(rng, A, B, P, Q, n)
    
    # Visualize data
    p_data = plot_rotating_ssm(x, y, nothing)
    path_data = save_plot(p_data, "rotating_ssm_data.png", exp_dir)
    logger("Saved data plot to: $path_data")

    x0 = MvNormalMeanCovariance(zeros(2), prior_var * diageye(2))
    
    result = infer(
        model = rotate_ssm(x0=x0, A=A, B=B, P=P, Q=Q), 
        data = (y = y,),
        options = (limit_stack_depth = 500, ),
        free_energy = true
    )

    xmarginals = result.posteriors[:x]
    logevidence = -result.free_energy
    logger("Log Evidence (Negative Free Energy): $logevidence")
    
    # Analyze
    est_x = mean.(xmarginals)
    real_x_flat = reduce(vcat, x)
    est_x_flat = reduce(vcat, est_x)
    
    stats = get_statistics(real_x_flat, est_x_flat)
    divergent, div_reason = detect_divergence(est_x_flat)
    
    log_info(logger, "Stats: RMSE=$(stats.rmse), MAE=$(stats.mae), VAF=$(stats.vaf)%, MaxErr=$(stats.max_err)")
    if divergent
        log_warn(logger, "Divergence Detected: $div_reason")
    end
    
    # Validation
    val_result = validate_metrics(stats, conf)
    log_validation(logger, val_result)

    # Generate Markdown Table Report
    headers = ["Metric", "Value"]
    rows = [
        ["RMSE", string(round(stats.rmse, digits=4))],
        ["MAE", string(round(stats.mae, digits=4))],
        ["VAF (%)", string(round(stats.vaf, digits=2))],
        ["Max Error", string(round(stats.max_err, digits=4))],
        ["Status", divergent ? "DIVERGENT ($div_reason)" : "STABLE"],
        ["Validation", val_result.passed ? "PASS" : "FAIL"]
    ]
    report_table = generate_markdown_table(headers, rows)
    append_to_report(report_path, "\n### Rotating SSM (Global)\n\n" * report_table * "\n")

    
    # Diagnostics
    p_err = plot_error_analysis(getindex.(x, 1), getindex.(est_x, 1), "Rotating SSM Dim 1")
    path_err = save_plot(p_err, "rotating_ssm_error_dim1.png", exp_dir)
    logger("Saved error plot to: $path_err")

    # Visualize results
    p_result = plot_rotating_ssm(x, y, xmarginals)
    path_res = save_plot(p_result, "rotating_ssm_result.png", exp_dir)
    logger("Saved result plot to: $path_res")
    
    return result
end

"""
    run_identification_problem(config, output_dir, seed, report_path, logger)

Run the System Identification example (Standard Inference).
"""
function run_identification_problem(config, output_dir, seed_val, report_path, logger)
    log_section(logger, "Identification Problem")
    logger("Starting Identification Problem (Standard)...")
    
    exp_dir = joinpath(output_dir, "identification")
    mkpath(exp_dir)

    conf = get(config, "identification", Dict())
    n = get(conf, "n_observations", 250)
    iters = get(conf, "iterations", 50)
    λ = get(conf, "ar_coefficient", 0.98) # Damping factor

    
    # 1. Linear case (f = +)
    logger("Case 1: Linear model y = x + w")
    real_x, real_w, real_y = generate_identification_data(+, n, seed=seed_val)
    
    p_signals = plot_identification_signals(real_x, real_w, real_y, "x + w")
    save_plot(p_signals, "identification_linear_data.png", exp_dir)
    
    m_x_0, τ_x_0 = -20.0, 1.0
    m_w_0, τ_w_0 = 20.0, 1.0
    a_x, b_x = 0.01, 0.01var(real_x)
    a_w, b_w = 0.01, 0.01var(real_w)
    a_y, b_y = 1.0, 1.0
    
    constraints = @constraints begin 
        q(x0, w0, x, w, τ_x, τ_w, τ_y, s) = q(x, x0, w, w0, s)q(τ_w)q(τ_x)q(τ_y)
    end
    
    xinit = map(r -> NormalMeanPrecision(r, τ_x_0), reverse(range(-60, -20, length = n)))
    winit = map(r -> NormalMeanPrecision(r, τ_w_0), range(20, 60, length = n))

    init = @initialization begin
        μ(x) = xinit
        μ(w) = winit
        q(τ_x) = GammaShapeRate(a_x, b_x)
        q(τ_w) = GammaShapeRate(a_w, b_w)
        q(τ_y) = GammaShapeRate(a_y, b_y)
    end

    logger("Running inference for Linear case (λ=$λ)...")
    result = infer(
        model = identification_problem(f=+, m_x_0=m_x_0, τ_x_0=τ_x_0, a_x=a_x, b_x=b_x, m_w_0=m_w_0, τ_w_0=τ_w_0, a_w=a_w, b_w=b_w, a_y=a_y, b_y=b_y, λ=λ),

        data  = (y = real_y,), 
        options = (limit_stack_depth = 500, ), 
        constraints = constraints, 
        initialization = init,
        iterations = iters,
        free_energy = true
    )
    
    est_x = mean.(result.posteriors[:x][end])
    est_w = mean.(result.posteriors[:w][end])
    
    stats_x = get_statistics(real_x, est_x)
    stats_w = get_statistics(real_w, est_w)
    
    log_info(logger, "Linear Ident X: RMSE=$(stats_x.rmse), VAF=$(stats_x.vaf)%")
    log_info(logger, "Linear Ident W: RMSE=$(stats_w.rmse), VAF=$(stats_w.vaf)%")

    # Validation
    val_result_x = validate_metrics(stats_x, conf)
    val_result_w = validate_metrics(stats_w, conf)
    log_validation(logger, val_result_x)

    headers = ["Component", "RMSE", "MAE", "VAF (%)", "Max Error", "Validation"]
    rows = [
        ["X", string(round(stats_x.rmse, digits=4)), string(round(stats_x.mae, digits=4)), string(round(stats_x.vaf, digits=2)), string(round(stats_x.max_err, digits=4)), val_result_x.passed ? "PASS" : "FAIL"],
        ["W", string(round(stats_w.rmse, digits=4)), string(round(stats_w.mae, digits=4)), string(round(stats_w.vaf, digits=2)), string(round(stats_w.max_err, digits=4)), val_result_w.passed ? "PASS" : "FAIL"]
    ]
    append_to_report(report_path, "\n### Linear Identification\n\n" * generate_markdown_table(headers, rows) * "\n")

    
    # Convergence
    fe = result.free_energy
    converged, change = check_convergence(fe)
    logger("Linear Ident. Converged: $converged (Change: $change)")
    append_to_report(report_path, "Linear Ident. Converged: $converged (Change: $change)\n")
    
    p_fe = plot_free_energy_history(fe)
    save_plot(p_fe, "identification_linear_fe.png", exp_dir)
    
    p_err_x = plot_error_analysis(real_x, est_x, "Linear Ident X")
    save_plot(p_err_x, "identification_linear_error_x.png", exp_dir)
    
    p_res = plot_identification_results(real_x, real_w, real_y, result.posteriors[:x][end], result.posteriors[:w][end], result.posteriors[:s][end])
    path_res = save_plot(p_res, "identification_linear_result.png", exp_dir)
    logger("Saved linear result to: $path_res")

    # 2. Nonlinear case (f = min)
    logger("Case 2: Nonlinear model y = min(x, w)")
    min_real_x, min_real_w, min_real_y = generate_identification_data(min, 200, seed=1, x_i_min=0.0, w_i_min=0.0, noise=1.0, real_x_τ=1.0, real_w_τ=1.0)
    
    p_min_signals = plot_identification_signals(min_real_x, min_real_w, min_real_y, "min(x, w)")
    save_plot(p_min_signals, "identification_nonlinear_data.png", exp_dir)
    
    min_meta = @meta begin 
        smooth_min() -> Linearization()
    end
    
    min_m_x_0, min_τ_x_0 = -1.0, 1.0
    min_m_w_0, min_τ_w_0 = 1.0, 1.0
    min_a_x, min_b_x = 1.0, 1.0
    min_a_w, min_b_w = 1.0, 1.0
    min_a_y, min_b_y = 1.0, 1.0
    
    min_init = @initialization begin
       μ(x) = NormalMeanPrecision(min_m_x_0, min_τ_x_0) 
       μ(w) = NormalMeanPrecision(min_m_w_0, min_τ_w_0)
       q(τ_x) = GammaShapeRate(min_a_x, min_b_x) 
       q(τ_w) = GammaShapeRate(min_a_w, min_b_w)
       q(τ_y) = GammaShapeRate(min_a_y, min_b_y)
    end
    
    logger("Running inference for Nonlinear case (λ=$λ)...")
    min_result = infer(
        model = identification_problem(f=smooth_min, m_x_0=min_m_x_0, τ_x_0=min_τ_x_0, a_x=min_a_x, b_x=min_b_x, m_w_0=min_m_w_0, τ_w_0=min_τ_w_0, a_w=min_a_w, b_w=min_b_w, a_y=min_a_y, b_y=min_b_y, λ=λ),

        data  = (y = min_real_y,), 
        options = (limit_stack_depth = 500, ), 
        constraints = constraints, 
        initialization = min_init,
        meta = min_meta,
        iterations = iters,
        free_energy = true
    )

    est_min_x = mean.(min_result.posteriors[:x][end])
    est_min_w = mean.(min_result.posteriors[:w][end])

    stats_min_x = get_statistics(min_real_x, est_min_x)
    stats_min_w = get_statistics(min_real_w, est_min_w)
    
    log_info(logger, "Nonlinear Ident X: RMSE=$(stats_min_x.rmse), VAF=$(stats_min_x.vaf)%")
    log_info(logger, "Nonlinear Ident W: RMSE=$(stats_min_w.rmse), VAF=$(stats_min_w.vaf)%")
    
    # Validation
    val_nl_x = validate_metrics(stats_min_x, conf)
    val_nl_w = validate_metrics(stats_min_w, conf)
    log_validation(logger, val_nl_x)
    
    headers_nl = ["Component", "RMSE", "MAE", "VAF (%)", "Max Error", "Validation"]
    rows_nl = [
        ["X (Nonlinear)", string(round(stats_min_x.rmse, digits=4)), string(round(stats_min_x.mae, digits=4)), string(round(stats_min_x.vaf, digits=2)), string(round(stats_min_x.max_err, digits=4)), val_nl_x.passed ? "PASS" : "FAIL"],
        ["W (Nonlinear)", string(round(stats_min_w.rmse, digits=4)), string(round(stats_min_w.mae, digits=4)), string(round(stats_min_w.vaf, digits=2)), string(round(stats_min_w.max_err, digits=4)), val_nl_w.passed ? "PASS" : "FAIL"]
    ]
    append_to_report(report_path, "\n### Nonlinear Identification\n\n" * generate_markdown_table(headers_nl, rows_nl) * "\n")

    
    # Convergence
    min_fe = min_result.free_energy
    min_converged, min_change = check_convergence(min_fe)
    logger("Nonlinear Ident Converged: $min_converged (Change: $min_change)")
    append_to_report(report_path, "Nonlinear Ident Converged: $min_converged (Change: $min_change)\n")
    
    p_min_fe = plot_free_energy_history(min_fe)
    save_plot(p_min_fe, "identification_nonlinear_fe.png", exp_dir)
    
    p_min_res = plot_identification_results(min_real_x, min_real_w, min_real_y, min_result.posteriors[:x][end], min_result.posteriors[:w][end], min_result.posteriors[:s][end])
    path_min_res = save_plot(p_min_res, "identification_nonlinear_result.png", exp_dir)
    logger("Saved nonlinear result to: $path_min_res")
    
    return result, min_result
end

"""
    run_rx_identification(config, output_dir, seed, report_path, logger)

Run the Reactive System Identification example (Online Inference).
"""
function run_rx_identification(config, output_dir, seed_val, report_path, logger)
    log_section(logger, "Reactive Identification")
    logger("Starting Reactive Identification (Online)...")
    
    exp_dir = joinpath(output_dir, "reactive_identification")
    mkpath(exp_dir)

    conf = get(config, "reactive_identification", Dict())
    n = get(conf, "n_observations", 300)
    iters = get(conf, "iterations", 10)
    hist_len = get(conf, "history_length", 1000)
    λ = get(conf, "ar_coefficient", 0.98) # Damping factor

    
    rx_real_x, rx_real_w, rx_real_y = generate_identification_data(min, n, seed=1, x_i_min=1.0, w_i_min=-1.0, noise=1.0, real_x_τ=1.0, real_w_τ=1.0)
    
    p_data = plot_identification_signals(rx_real_x, rx_real_w, rx_real_y, "min(x, w) [Online]")
    save_plot(p_data, "rx_identification_data.png", exp_dir)
    
    rx_constraints = @constraints begin 
        q(x0, x, w0, w, τ_x, τ_w, τ_y, s) = q(x0, x)q(w, w0)q(τ_w)q(τ_x)q(s)q(τ_y)
    end
    
    autoupdates = @autoupdates begin 
        m_x_0, τ_x_0 = mean_precision(q(x))
        m_w_0, τ_w_0 = mean_precision(q(w))
        a_x = shape(q(τ_x)) 
        b_x = rate(q(τ_x))
        a_y = shape(q(τ_y))
        b_y = rate(q(τ_y))
        a_w = shape(q(τ_w)) 
        b_w = rate(q(τ_w))
    end
    
    rx_meta = @meta begin 
        smooth_min() -> Linearization()
    end
    
    init = @initialization begin
        q(w)= NormalMeanVariance(-2.0, 1.0) 
        q(x) = NormalMeanVariance(2.0, 1.0) 
        q(τ_x) = GammaShapeRate(1.0, 1.0) 
        q(τ_w) = GammaShapeRate(1.0, 1.0) 
        q(τ_y) = GammaShapeRate(1.0, 20.0)
    end
    
    engine = infer(
        model         = rx_identification(f=smooth_min, λ=λ),

        constraints   = rx_constraints,
        data          = (y = rx_real_y,),
        autoupdates   = autoupdates,
        meta          = rx_meta,
        returnvars    = (:x, :w, :τ_x, :τ_w, :τ_y, :s),
        keephistory   = hist_len,
        historyvars   = KeepLast(),
        initialization = init,
        iterations    = iters,
        free_energy   = true,
        autostart     = true,
    )
    
    rx_smarginals = engine.history[:s]
    rx_xmarginals = engine.history[:x]
    rx_wmarginals = engine.history[:w]
    
    p_res = plot_identification_results(rx_real_x, rx_real_w, rx_real_y, rx_xmarginals, rx_wmarginals, rx_smarginals)
    path_res = save_plot(p_res, "rx_identification_result.png", exp_dir)
    logger("Saved rx result to: $path_res")
    
    # Online stats
    est_rx_x = mean.(rx_xmarginals)
    # Align real data with history buffer (take last N elements)
    aligned_rx_real_x = rx_real_x[end-length(est_rx_x)+1:end]
    stats_rx = get_statistics(aligned_rx_real_x, est_rx_x)
    
    log_info(logger, "Reactive Ident X: RMSE=$(stats_rx.rmse), VAF=$(stats_rx.vaf)%")

    # Validation
    val_rx = validate_metrics(stats_rx, conf)
    log_validation(logger, val_rx)

    headers_rx = ["Metric", "Value"]
    rows_rx = [
        ["RMSE", string(round(stats_rx.rmse, digits=4))],
        ["MAE", string(round(stats_rx.mae, digits=4))],
        ["VAF (%)", string(round(stats_rx.vaf, digits=2))],
        ["Max Error", string(round(stats_rx.max_err, digits=4))],
        ["Validation", val_rx.passed ? "PASS" : "FAIL"]
    ]
    append_to_report(report_path, "\n### Reactive Identification (Online)\n\n" * generate_markdown_table(headers_rx, rows_rx) * "\n")

    
    logger("Reactive Identification completed.")
    return engine
end

"""
    run_smoothing_example(config, output_dir, seed, report_path, logger)

Run the Kalman Smoothing with missing data example.
"""
function run_smoothing_example(config, output_dir, seed_val, report_path, logger)
    log_section(logger, "Kalman Smoothing")
    logger("Starting Kalman Smoothing (Missing Data)...")
    
    exp_dir = joinpath(output_dir, "smoothing")
    mkpath(exp_dir)

    conf = get(config, "smoothing", Dict())
    n = get(conf, "n_observations", 250)
    miss_start = get(conf, "missing_start", 100)
    miss_end = get(conf, "missing_end", 125)
    P_val = get(conf, "process_noise_P", 1.0)
    iters = get(conf, "iterations", 20)
    
    P = P_val
    real_signal     = map(e -> sin(0.05 * e), collect(1:n))
    noisy_data      = real_signal + rand(Normal(0.0, sqrt(P)), n)
    missing_indices = miss_start:miss_end
    missing_data    = similar(noisy_data, Union{Float64, Missing})
    
    copyto!(missing_data, noisy_data)
    for index in missing_indices
        missing_data[index] = missing
    end
    
    constraints = @constraints begin
        q(x_prior, x, y, P) = q(x_prior, x)q(P)q(y)
    end
    
    x0_prior = NormalMeanVariance(0.0, 1000.0)
    initm = @initialization begin
        q(P) = Gamma(0.001, 0.001)
    end
    
    result = infer(
        model = smoothing(x0=x0_prior), 
        data  = (y = missing_data,), 
        constraints = constraints,
        options = (limit_stack_depth = 500, ),
        initialization = initm, 
        returnvars = (x = KeepLast(),),
        iterations = iters
    )
    
    est_smooth = mean.(result.posteriors[:x])
    stats_smooth = get_statistics(real_signal, est_smooth)
    
    log_info(logger, "Smoothing: RMSE=$(stats_smooth.rmse), VAF=$(stats_smooth.vaf)%")

    # Validation
    val_smooth = validate_metrics(stats_smooth, conf)
    log_validation(logger, val_smooth)

    headers_s = ["Metric", "Value"]
    rows_s = [
        ["RMSE", string(round(stats_smooth.rmse, digits=4))],
        ["MAE", string(round(stats_smooth.mae, digits=4))],
        ["VAF (%)", string(round(stats_smooth.vaf, digits=2))],
        ["Max Error", string(round(stats_smooth.max_err, digits=4))],
        ["Validation", val_smooth.passed ? "PASS" : "FAIL"]
    ]
    append_to_report(report_path, "\n### Smoothing (Missing Data)\n\n" * generate_markdown_table(headers_s, rows_s) * "\n")

    
    p = plot_smoothing_results(real_signal, missing_indices, result.posteriors[:x])
    save_plot(p, "smoothing_result.png", exp_dir)
    
    p_err_smooth = plot_error_analysis(real_signal, est_smooth, "Smoothing Error")
    save_plot(p_err_smooth, "smoothing_error.png", exp_dir)
    
    logger("Smoothing example completed.")
    return result
end

end # module
