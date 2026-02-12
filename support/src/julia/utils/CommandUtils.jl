"""
CommandUtils - Utilities for command execution and argument parsing.

This module provides shared utilities for executing shell commands with
error handling and parsing command line arguments. Used by orchestration
scripts in `scripts/`.

# Exports
- `run_command`: Execute a shell command with error handling
- `parse_flag_args`: Generic flag argument parser
- `log_info`, `log_warn`, `log_error`: Logging utilities
"""
module CommandUtils

using Dates

export run_command, parse_flag_args
export log_info, log_warn, log_error

"""
    run_command(cmd, message::AbstractString; exit_on_error::Bool=false, show_output::Bool=true, quiet::Bool=false) -> Bool

Execute a shell command with error handling, timing, and optional output capture.
"""
function run_command(cmd, message::AbstractString; exit_on_error::Bool=false, show_output::Bool=true, quiet::Bool=false)::Bool
    quiet || println("\n" * "═"^40)
    quiet || println("🚀 $message")
    quiet || println("═"^40)
    
    start_time = now()
    success = false
    
    try
        if show_output
            run(cmd)
            success = true
        else
            output = IOBuffer()
            error_output = IOBuffer()
            process = run(pipeline(cmd, stdout=output, stderr=error_output), wait=true)
            
            if process.exitcode != 0
                quiet || println("🚨 Command failed with exit code $(process.exitcode)")
                quiet || println("\n📄 Output:")
                quiet || println(String(take!(output)))
                quiet || println("\n🚨 Error:")
                quiet || println(String(take!(error_output)))
                if exit_on_error
                    error("Exiting due to command failure")
                end
                success = false
            else
                success = true
            end
        end
    catch e
        quiet || println("🚨 Error executing command: $e")
        if exit_on_error
            error("Exiting due to command failure")
        end
        success = false
    end
    
    elapsed = now() - start_time
    status_emoji = success ? "✅" : "❌"
    quiet || println("\n$status_emoji Finished in $(Dates.canonicalize(Dates.CompoundPeriod(elapsed)))")
    quiet || println("─"^40)
    
    return success
end

"""
    parse_flag_args(args::Vector{String}, flag_map::Dict{String,String}, defaults::Dict{String,Any}) -> NamedTuple

Parse command line arguments using a flag mapping.
"""
function parse_flag_args(args::Vector{String}, flag_map::Dict{String,String}, defaults::Dict{String,Any})
    options = copy(defaults)
    value_flags = Set{String}()
    
    for (flag, key) in flag_map
        if haskey(defaults, key) && !(defaults[key] isa Bool)
            push!(value_flags, flag)
        end
    end
    
    i = 1
    while i <= length(args)
        a = args[i]
        if haskey(flag_map, a)
            key = flag_map[a]
            if a in value_flags
                i += 1
                i > length(args) && error("$a requires a value")
                options[key] = args[i]
            else
                options[key] = true
            end
        else
            error("Unknown option: $(a)")
        end
        i += 1
    end
    
    return (; (Symbol(k) => v for (k, v) in options)...)
end

"""
    log_info(message::AbstractString; quiet::Bool=false)
"""
function log_info(message::AbstractString; quiet::Bool=false)
    quiet || println("ℹ️  $message")
end

"""
    log_warn(message::AbstractString; quiet::Bool=false)
"""
function log_warn(message::AbstractString; quiet::Bool=false)
    quiet || println("⚠️  $message")
end

"""
    log_error(message::AbstractString)
"""
function log_error(message::AbstractString)
    println("❌ $message")
end

end # module
