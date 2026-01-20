module LoggingUtils

using Dates

export setup_logger, log_info, log_warn, log_section, log_config, log_validation

"""
    setup_logger(output_dir, config_path, seed)

Initialize the execution log file with system context and return a logger function.
"""
function setup_logger(output_dir, config_path, seed)
    log_path = joinpath(output_dir, "execution.log")
    
    open(log_path, "w") do io
        println(io, "Execution Log")
        println(io, "=============")
        println(io, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Julia Version: $(VERSION)")
        println(io, "OS: $(Sys.KERNEL)")
        println(io, "Threads: $(Threads.nthreads())")
        println(io, "Configuration: $config_path")
        println(io, "Output Directory: $output_dir")
        println(io, "Seed: $seed")
        println(io, "---------------------------\n")
    end
    
    function logger(msg; level="INFO")
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        formatted_msg = "[$timestamp] [$level] $msg"
        println(formatted_msg)
        open(log_path, "a") do io
            println(io, formatted_msg)
        end
    end
    
    return logger
end

"""
    log_info(logger, msg)

Log an informational message.
"""
log_info(logger, msg) = logger(msg, level="INFO")

"""
    log_warn(logger, msg)

Log a warning message.
"""
log_warn(logger, msg) = logger(msg, level="WARN")

"""
    log_section(logger, name)

Log a visual section header.
"""
function log_section(logger, name)
    logger("\n" * repeat("=", 40))
    logger("SECTION: $name")
    logger(repeat("=", 40))
end

"""
    log_config(logger, config)

Log the active configuration dictionary.
"""
function log_config(logger, config)
    logger("Active Configuration:")
    for (key, val) in config
        logger("  $key: $val")
    end
end

"""
    log_validation(logger, result)

Log the validation result with visual emphasis.
"""
function log_validation(logger, result)
    # Handle ValidationResult struct if available, otherwise assume named tuple or similar
    # Assuming result has .passed and .messages
    status_str = result.passed ? "[PASS]" : "[FAIL]"
    logger("\n" * repeat("-", 40))
    logger("VALIDATION $status_str")
    if hasproperty(result, :messages)
        for msg in result.messages
            level = contains(msg, "FAIL") ? "WARN" : "INFO"
            logger(msg, level=level)
        end
    end
    logger(repeat("-", 40) * "\n")
end

end # module
