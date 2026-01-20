# Data Export for POMDP Control

using JSON
using Dates

"""
    export_agent_state(agent; save_path=nothing)

Export current agent state to JSON.
"""
function export_agent_state(agent; save_path=nothing)
    state = Dict(
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "grid_params" => Dict(
            "grid_size" => grid_params.grid_size,
            "wind" => collect(grid_params.wind),
            "goal" => collect(grid_params.goal),
            "start" => collect(grid_params.start)
        ),
        "step_count" => agent.step_count,
        "A_matrix" => mean(agent.p_A) |> m -> round.(m, digits=4),
        "B_matrix_up" => mean(agent.p_B)[:,:,ACTION_UP] |> m -> round.(m, digits=4),
        "B_matrix_right" => mean(agent.p_B)[:,:,ACTION_RIGHT] |> m -> round.(m, digits=4),
        "B_matrix_down" => mean(agent.p_B)[:,:,ACTION_DOWN] |> m -> round.(m, digits=4),
        "B_matrix_left" => mean(agent.p_B)[:,:,ACTION_LEFT] |> m -> round.(m, digits=4),
        "state_belief" => params(agent.p_s)[1] |> v -> round.(v, digits=4),
        "current_policy" => [params(p)[1] |> v -> round.(v, digits=4) for p in agent.current_policy]
    )
    
    if !isnothing(save_path)
        open(save_path, "w") do io
            JSON.print(io, state, 2)
        end
        rxlog("info", "Exported agent state to $save_path")
    end
    
    return state
end

"""
    export_learning_history(history::LearningHistory; save_path=nothing)

Export learning history to JSON.
"""
function export_learning_history(history::LearningHistory; save_path=nothing)
    data = Dict(
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "n_experiments" => length(history.experiment_ids),
        "final_success_rate" => isempty(history.cumulative_success_rate) ? 0.0 : 
                                history.cumulative_success_rate[end],
        "experiments" => [
            Dict(
                "id" => history.experiment_ids[i],
                "success" => history.success[i],
                "steps" => history.steps_taken[i],
                "A_entropy" => round(history.A_entropy[i], digits=4),
                "B_entropy" => round(history.B_entropy[i], digits=4),
                "cumulative_rate" => round(history.cumulative_success_rate[i], digits=4)
            )
            for i in 1:length(history.experiment_ids)
        ]
    )
    
    if !isnothing(save_path)
        open(save_path, "w") do io
            JSON.print(io, data, 2)
        end
        rxlog("info", "Exported learning history to $save_path")
    end
    
    return data
end

"""
    export_trajectory(recorder::TrajectoryRecorder; save_path=nothing)

Export trajectory recording to JSON.
"""
function export_trajectory(recorder::TrajectoryRecorder; save_path=nothing)
    data = Dict(
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "n_steps" => length(recorder.positions),
        "steps" => [
            Dict(
                "t" => i,
                "position" => collect(recorder.positions[i]),
                "action" => recorder.actions[i],
                "action_name" => ACTION_NAMES[recorder.actions[i]],
                "observation" => collect(recorder.observations[i]),
                "step_time_ms" => round(recorder.step_times[i] * 1000, digits=2)
            )
            for i in 1:length(recorder.positions)
        ]
    )
    
    if !isnothing(save_path)
        open(save_path, "w") do io
            JSON.print(io, data, 2)
        end
        rxlog("info", "Exported trajectory to $save_path")
    end
    
    return data
end

"""
    export_model_specification(; save_path=nothing)

Export model specification and parameters to JSON.
"""
function export_model_specification(; save_path=nothing)
    spec = Dict(
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "model_name" => "pomdp_model",
        "environment" => Dict(
            "type" => "WindyGridWorld",
            "grid_size" => grid_params.grid_size,
            "n_states" => grid_params.grid_size^2,
            "n_actions" => 4,
            "actions" => Dict(
                "1" => "up",
                "2" => "right", 
                "3" => "down",
                "4" => "left"
            ),
            "wind" => collect(grid_params.wind),
            "start" => collect(grid_params.start),
            "goal" => collect(grid_params.goal)
        ),
        "inference" => Dict(
            "n_iterations" => experiment_params.n_iterations,
            "planning_horizon" => experiment_params.planning_horizon,
            "n_experiments" => experiment_params.n_experiments
        ),
        "priors" => Dict(
            "A_prior" => "DirichletCollection(diageye(25) + 0.1)",
            "B_prior" => "DirichletCollection(ones(25, 25, 4))",
            "goal_prior" => "Categorical at goal position"
        )
    )
    
    if !isnothing(save_path)
        open(save_path, "w") do io
            JSON.print(io, spec, 2)
        end
        rxlog("info", "Exported model specification to $save_path")
    end
    
    return spec
end

"""
    export_all_diagnostics(agent, history, recorder; output_dir=OUTPUT_DIR)

Export all diagnostics data to output directory.
"""
function export_all_diagnostics(agent, history::LearningHistory, 
                                recorder::TrajectoryRecorder;
                                output_dir::String=OUTPUT_DIR)
    ensure_output_dirs()
    data_dir = joinpath(output_dir, "data")
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    
    # Export each component
    export_agent_state(agent, save_path=joinpath(data_dir, "agent_state_$ts.json"))
    export_learning_history(history, save_path=joinpath(data_dir, "learning_history_$ts.json"))
    export_trajectory(recorder, save_path=joinpath(data_dir, "trajectory_$ts.json"))
    export_model_specification(save_path=joinpath(data_dir, "model_spec_$ts.json"))
    
    rxlog("info", "Exported all diagnostics to $data_dir")
    
    return data_dir
end

export export_agent_state, export_learning_history, export_trajectory
export export_model_specification, export_all_diagnostics
