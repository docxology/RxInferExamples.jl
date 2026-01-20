"""
EnvironmentSetup - Environment configuration and package management utilities.

This module provides utilities for setting up Julia environments, 
managing packages, and configuring build settings. Used by the
setup orchestration script.

# Exports
- `install_required_packages`: Install a list of required packages
- `setup_environment_vars`: Configure environment variables for graphics
- `ensure_project_instantiated`: Ensure a project's packages are installed
- `get_required_packages`: Return list of standard required packages
"""
module EnvironmentSetup

using Pkg

export install_required_packages, setup_environment_vars
export ensure_project_instantiated, get_required_packages

"""
    get_required_packages() -> Vector{String}

Return the list of required packages for RxInferExamples.jl support.
"""
function get_required_packages()::Vector{String}
    return [
        "Weave",
        "ImageInTerminal",
        "JSON"
    ]
end

"""
    install_required_packages(packages::Vector{String}; quiet::Bool=false, force::Bool=false)

Install a list of required packages.

# Arguments
- `packages`: List of package names to install
- `quiet`: If true, suppress output
- `force`: If true, continue on installation errors
"""
function install_required_packages(packages::Vector{String}; quiet::Bool=false, force::Bool=false)
    for pkg in packages
        quiet || println("Installing $pkg...")
        try
            Pkg.add(pkg)
        catch e
            if force
                quiet || println("Warning: Failed to install $(pkg): $(e)")
            else
                rethrow(e)
            end
        end
    end
end

"""
    setup_environment_vars()

Configure environment variables for graphics and error handling.
Sets GKSwstype for non-interactive backend and JULIA_COPY_STACKS for better errors.
"""
function setup_environment_vars()
    ENV["GKSwstype"] = "100"  # Non-interactive backend
    ENV["JULIA_COPY_STACKS"] = "1"  # More helpful error messages
end

"""
    ensure_project_instantiated(project_path::AbstractString; update::Bool=false, quiet::Bool=false) -> Bool

Ensure a project's packages are instantiated and optionally updated.

# Arguments
- `project_path`: Path to the project directory
- `update`: If true, also update packages
- `quiet`: If true, suppress output

# Returns
- true if successful
"""
function ensure_project_instantiated(project_path::AbstractString; update::Bool=false, quiet::Bool=false)::Bool
    manifest_path = joinpath(project_path, "Manifest.toml")
    
    if !isfile(manifest_path)
        quiet || println("Setting up project environment at $project_path")
    end
    
    try
        if update
            run(`julia --project=$project_path -e 'using Pkg; Pkg.update(); Pkg.instantiate()'`)
        else
            run(`julia --project=$project_path -e 'using Pkg; Pkg.instantiate()'`)
        end
        return true
    catch e
        quiet || println("Warning: Failed to instantiate project: $(e)")
        return false
    end
end

end # module
