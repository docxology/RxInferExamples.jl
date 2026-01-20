# Structured Logging Module for POMDP Control
# Provides comprehensive logging with categories, structured data, and JSON support

using Dates
using JSON

# ============================================================================
# LOG CATEGORIES
# ============================================================================

@enum LogCategory begin
    LOG_SYSTEM      # System-level messages (startup, config)
    LOG_STEP        # Per-step logging (action, observation)
    LOG_EXPERIMENT  # Per-experiment summaries
    LOG_REPLICATE   # Per-replicate summaries
    LOG_INFERENCE   # Inference details
    LOG_MATRIX      # Matrix updates and entropies
    LOG_STATS       # Statistical summaries
end

const CATEGORY_NAMES = Dict(
    LOG_SYSTEM => "SYSTEM",
    LOG_STEP => "STEP",
    LOG_EXPERIMENT => "EXP",
    LOG_REPLICATE => "REP",
    LOG_INFERENCE => "INF",
    LOG_MATRIX => "MAT",
    LOG_STATS => "STATS"
)

# ============================================================================
# LOG LEVELS
# ============================================================================

@enum LogLevel begin
    LOG_DEBUG = 1
    LOG_INFO = 2
    LOG_STATS_LEVEL = 3
    LOG_WARN = 4
    LOG_ERROR = 5
end

const LEVEL_NAMES = Dict(
    LOG_DEBUG => "DEBUG",
    LOG_INFO => "INFO",
    LOG_STATS_LEVEL => "STATS",
    LOG_WARN => "WARN",
    LOG_ERROR => "ERROR"
)

# ============================================================================
# LOGGER STATE
# ============================================================================

mutable struct PODMPLogger
    logfile::String
    log_to_file::Bool
    log_to_console::Bool
    min_level::LogLevel
    json_format::Bool
    current_experiment::Int
    current_step::Int
    current_replicate::Int
end

# Global logger instance
const LOGGER = Ref{PODMPLogger}()

"""Initialize the logging system from config."""
function init_logger()
    logfile = joinpath(@__DIR__, "..", logging_params.log_file)
    min_level = lowercase(logging_params.level) == "debug" ? LOG_DEBUG : LOG_INFO
    
    LOGGER[] = PODMPLogger(
        logfile,
        logging_params.log_to_file,
        true,  # Console output
        min_level,
        get(logging_params, :json_format, false),
        0, 0, 0
    )
    
    log_system("Logger initialized", Dict(
        "logfile" => logfile,
        "level" => string(min_level),
        "to_file" => logging_params.log_to_file
    ))
end

# ============================================================================
# CORE LOGGING FUNCTIONS
# ============================================================================

"""Format a log entry."""
function format_log(level::LogLevel, category::LogCategory, msg::String, data::Dict=Dict())
    ts = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS.sss")
    level_str = LEVEL_NAMES[level]
    cat_str = CATEGORY_NAMES[category]
    
    logger = LOGGER[]
    ctx = ""
    if logger.current_replicate > 0
        ctx *= "R$(logger.current_replicate)"
    end
    if logger.current_experiment > 0
        ctx *= ctx != "" ? "." : ""
        ctx *= "E$(logger.current_experiment)"
    end
    if logger.current_step > 0
        ctx *= ctx != "" ? "." : ""
        ctx *= "S$(logger.current_step)"
    end
    ctx = ctx != "" ? "[$ctx]" : ""
    
    if logger.json_format && !isempty(data)
        return "[$ts] $level_str $cat_str $ctx $msg | $(JSON.json(data))"
    elseif !isempty(data)
        data_str = join(["$k=$v" for (k,v) in data], " ")
        return "[$ts] $level_str $cat_str $ctx $msg | $data_str"
    else
        return "[$ts] $level_str $cat_str $ctx $msg"
    end
end

"""Write a log entry."""
function write_log(level::LogLevel, category::LogCategory, msg::String, data::Dict=Dict())
    !isdefined(LOGGER, 1) && init_logger()
    logger = LOGGER[]
    
    # Skip if below minimum level
    level < logger.min_level && return
    
    log_line = format_log(level, category, msg, data)
    
    # Console output
    if logger.log_to_console
        if level == LOG_ERROR
            @error log_line
        elseif level == LOG_WARN
            @warn log_line
        elseif level == LOG_DEBUG
            # Only show debug in console if explicitly requested
        end
    end
    
    # File output
    if logger.log_to_file
        open(logger.logfile, "a") do io
            println(io, log_line)
        end
    end
end

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

"""Log system-level message."""
function log_system(msg::String, data::Dict=Dict())
    write_log(LOG_INFO, LOG_SYSTEM, msg, data)
end

"""Log step-level message with full context."""
function log_step(msg::String, data::Dict=Dict())
    write_log(LOG_DEBUG, LOG_STEP, msg, data)
end

"""Log step with detailed action/observation data."""
function log_step_detail(agent, action_idx::Int, obs, policy_probs::Vector{Float64})
    data = Dict{String, Any}(
        "action" => ACTION_NAMES[action_idx],
        "action_idx" => action_idx,
        "obs" => string(obs),
        "policy" => round.(policy_probs, digits=3)
    )
    write_log(LOG_DEBUG, LOG_STEP, "action=$(ACTION_NAMES[action_idx]) obs=$obs", data)
end

"""Log experiment summary."""
function log_experiment(exp_idx::Int, success::Bool, steps::Int, data::Dict=Dict())
    data["exp"] = exp_idx
    data["success"] = success
    data["steps"] = steps
    write_log(LOG_INFO, LOG_EXPERIMENT, success ? "SUCCESS in $steps steps" : "FAIL after $steps steps", data)
end

"""Log replicate summary."""
function log_replicate(rep_idx::Int, success_rate::Float64, data::Dict=Dict())
    data["replicate"] = rep_idx
    data["success_rate"] = round(success_rate * 100, digits=1)
    write_log(LOG_INFO, LOG_REPLICATE, "success_rate=$(data["success_rate"])%", data)
end

"""Log inference timing and details."""
function log_inference(iterations::Int, time_ms::Float64, data::Dict=Dict())
    data["iterations"] = iterations
    data["time_ms"] = round(time_ms, digits=2)
    write_log(LOG_DEBUG, LOG_INFERENCE, "$(iterations) iters in $(data["time_ms"])ms", data)
end

"""Log matrix update with entropies."""
function log_matrix_update(A_entropy::Float64, B_entropy::Float64, data::Dict=Dict())
    data["A_entropy"] = round(A_entropy, digits=4)
    data["B_entropy"] = round(B_entropy, digits=4)
    write_log(LOG_DEBUG, LOG_MATRIX, "A_H=$(data["A_entropy"]) B_H=$(data["B_entropy"])", data)
end

"""Log statistical summary."""
function log_stats(msg::String, data::Dict=Dict())
    write_log(LOG_STATS_LEVEL, LOG_STATS, msg, data)
end

# ============================================================================
# CONTEXT MANAGEMENT
# ============================================================================

"""Set current replicate context."""
function set_replicate_context(rep::Int)
    !isdefined(LOGGER, 1) && init_logger()
    LOGGER[].current_replicate = rep
    LOGGER[].current_experiment = 0
    LOGGER[].current_step = 0
end

"""Set current experiment context."""
function set_experiment_context(exp::Int)
    !isdefined(LOGGER, 1) && init_logger()
    LOGGER[].current_experiment = exp
    LOGGER[].current_step = 0
end

"""Set current step context."""
function set_step_context(step::Int)
    !isdefined(LOGGER, 1) && init_logger()
    LOGGER[].current_step = step
end

"""Clear all context."""
function clear_context()
    !isdefined(LOGGER, 1) && return
    LOGGER[].current_replicate = 0
    LOGGER[].current_experiment = 0
    LOGGER[].current_step = 0
end

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

"""Legacy rxlog function for backward compatibility."""
function rxlog(level::AbstractString, msg)
    lvl = lowercase(level) == "debug" ? LOG_DEBUG :
          lowercase(level) == "info" ? LOG_INFO :
          lowercase(level) == "warn" ? LOG_WARN :
          lowercase(level) == "error" ? LOG_ERROR : LOG_INFO
    write_log(lvl, LOG_SYSTEM, string(msg))
end

export rxlog
export log_system, log_step, log_step_detail, log_experiment, log_replicate
export log_inference, log_matrix_update, log_stats
export set_replicate_context, set_experiment_context, set_step_context, clear_context
export init_logger
