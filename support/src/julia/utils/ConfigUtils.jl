"""
ConfigUtils - Configuration file utilities.

Provides utilities for loading and managing TOML configuration files.

# Exports
- `load_toml_config`: Load a TOML configuration file
- `get_config_value`: Safely get a nested config value
- `merge_configs`: Merge multiple config dictionaries
"""
module ConfigUtils

using TOML

export load_toml_config, get_config_value, merge_configs

"""
    load_toml_config(path::AbstractString; required::Bool=true) -> Dict{String,Any}

Load a TOML configuration file.

# Arguments
- `path`: Path to the TOML file
- `required`: If true, throw error if file doesn't exist

# Returns
- Dictionary of configuration values
"""
function load_toml_config(path::AbstractString; required::Bool=true)::Dict{String,Any}
    if !isfile(path)
        if required
            error("Configuration file not found: $path")
        else
            return Dict{String,Any}()
        end
    end
    return TOML.parsefile(path)
end

"""
    get_config_value(config::Dict, keys...; default=nothing)

Safely get a nested configuration value.

# Example
```julia
config = load_toml_config("config.toml")
value = get_config_value(config, "section", "key", default=10)
```
"""
function get_config_value(config::Dict, keys...; default=nothing)
    current = config
    for key in keys
        if !haskey(current, key)
            return default
        end
        current = current[key]
    end
    return current
end

"""
    merge_configs(configs::Dict...) -> Dict{String,Any}

Merge multiple configuration dictionaries. Later configs override earlier ones.
"""
function merge_configs(configs::Dict...)::Dict{String,Any}
    result = Dict{String,Any}()
    for config in configs
        merge!(result, config)
    end
    return result
end

end # module
