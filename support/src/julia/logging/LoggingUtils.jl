module LoggingUtils

using Dates

export setup_logger, log_info, log_warn, log_error, log_section, log_config, log_validation

"""
    setup_logger(output_dir, config_path, seed)

Initialize the execution log file with system context and return a logger function.
"""
function setup_logger(output_dir, config_path, seed)
    log_path = joinpath(output_dir, "execution.log")
    
    open(log_path, "w") do io
        println(io, "╔══════════════════════════════════════════════════════════╗")
        println(io, "║ EXECUTION LOG                                            ║")
        println(io, "╚══════════════════════════════════════════════════════════╝")
        println(io, "Date:      $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "Julia:     $(VERSION)")
        println(io, "OS:        $(Sys.KERNEL)")
        println(io, "Threads:   $(Threads.nthreads())")
        println(io, "Config:    $config_path")
        println(io, "Output:    $output_dir")
        println(io, "Seed:      $seed")
        println(io, "────────────────────────────────────────────────────────────\n")
    end
    
    function logger(msg; level="INFO", emoji="ℹ️")
        timestamp = Dates.format(now(), "HH:MM:SS")
        formatted_msg = "[$timestamp] $emoji [$level] $msg"
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
log_info(logger, msg) = logger(msg, level="INFO", emoji="ℹ️")

"""
    log_warn(logger, msg)

Log a warning message.
"""
log_warn(logger, msg) = logger(msg, level="WARN", emoji="⚠️")

"""
    log_error(logger, msg)

Log an error message.
"""
log_error(logger, msg) = logger(msg, level="ERROR", emoji="❌")

"""
    log_section(logger, name)

Log a visual section header.
"""
function log_section(logger, name)
    logger("\n" * "═"^40)
    logger("SECTION: $name", emoji="📌")
    logger("═"^40)
end

"""
    log_config(logger, config)

Log the active configuration dictionary.
"""
function log_config(logger, config)
    logger("Active Configuration:", emoji="⚙️")
    for (key, val) in config
        logger("  $key: $val", emoji="  ")
    end
end

"""
    log_validation(logger, result)

Log the validation result with visual emphasis.
"""
function log_validation(logger, result)
    status_str = result.passed ? "SUCCESS" : "FAILURE"
    emoji = result.passed ? "✅" : "🚨"
    
    logger("\n" * "─"^40)
    logger("VALIDATION $status_str", emoji=emoji)
    if hasproperty(result, :messages)
        for msg in result.messages
            m_emoji = contains(msg, "FAIL") ? "❌" : "🔹"
            logger(msg, emoji=m_emoji)
        end
    end
    logger("─"^40 * "\n")
end

end # module
