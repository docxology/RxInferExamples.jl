"""
NotebookConversion - Core module for Jupyter notebook to Julia script conversion.

This module provides the core functionality for converting Jupyter notebooks
to executable Julia scripts. It is used by the orchestration script at
`scripts/notebooks/notebooks_to_scripts.jl`.

# Exports
- `extract_code_from_notebook`: Extract Julia code from a notebook file
- `notebook_to_script_path`: Map notebook path to script output path
- `copy_project_files`: Copy Project.toml and meta.jl files
- `up_to_date`: Check if script is up-to-date with notebook
- `should_process`: Filter notebooks by pattern
- `conversion_header`: Generate header for converted scripts
- `verify_conversion`: Verify 1:1 notebook/script mapping
- `convert_notebook`: Convert a single notebook to script
"""
module NotebookConversion

using JSON
using Dates

export extract_code_from_notebook, notebook_to_script_path, copy_project_files
export up_to_date, should_process, conversion_header, verify_conversion, convert_notebook

"""
    extract_code_from_notebook(notebook_path::AbstractString) -> String

Extract Julia code from a Jupyter notebook file.
Returns a single string with code blocks separated by blank lines.

# Arguments
- `notebook_path`: Path to the .ipynb notebook file

# Returns
- String containing all code cells concatenated
"""
function extract_code_from_notebook(notebook_path::AbstractString)::String
    notebook_content = JSON.parsefile(notebook_path)
    cells = get(notebook_content, "cells", [])
    code_blocks = String[]
    for cell in cells
        if get(cell, "cell_type", "") == "code"
            source = get(cell, "source", [])
            if !isempty(source)
                code = join(source, "")
                if !isempty(strip(code)) && !all(startswith.(split(code, "\n"), "#"))
                    push!(code_blocks, code)
                end
            end
        end
    end
    return join(code_blocks, "\n\n")
end

"""
    notebook_to_script_path(notebook_path::AbstractString, examples_dir::AbstractString, output_dir::AbstractString) -> String

Map a notebook path in examples/ to its output script path in scripts/.

# Arguments
- `notebook_path`: Path to the source notebook
- `examples_dir`: Base examples directory
- `output_dir`: Base output directory

# Returns
- Path to the corresponding .jl script file
"""
function notebook_to_script_path(notebook_path::AbstractString, examples_dir::AbstractString, output_dir::AbstractString)::String
    script_path = replace(notebook_path, examples_dir => output_dir)
    return replace(script_path, r"\.ipynb$" => ".jl")
end

"""
    copy_project_files(source_dir::AbstractString, target_dir::AbstractString; quiet::Bool=false)

Copy Project.toml and meta.jl from source directory to target directory if present.

# Arguments
- `source_dir`: Source directory containing project files
- `target_dir`: Target directory to copy files to
- `quiet`: If true, suppress output messages
"""
function copy_project_files(source_dir::AbstractString, target_dir::AbstractString; quiet::Bool=false)
    for file in ("Project.toml", "meta.jl")
        source_file = joinpath(source_dir, file)
        if isfile(source_file)
            target_file = joinpath(target_dir, file)
            cp(source_file, target_file, force=true)
            quiet || println("  Copied $(file)")
        end
    end
end

"""
    up_to_date(notebook_path::AbstractString, script_path::AbstractString) -> Bool

Return true if the target script is considered up-to-date with the notebook.

# Arguments
- `notebook_path`: Path to source notebook
- `script_path`: Path to target script

# Returns
- true if script exists and is newer than or same age as notebook
"""
function up_to_date(notebook_path::AbstractString, script_path::AbstractString)::Bool
    return isfile(script_path) && (mtime(script_path) >= mtime(notebook_path))
end

"""
    should_process(notebook_path::AbstractString; filter::Union{Nothing,String}=nothing) -> Bool

Determine if a notebook should be processed based on filter pattern.

# Arguments
- `notebook_path`: Path to the notebook
- `filter`: Optional substring filter

# Returns
- true if notebook matches filter or no filter specified
"""
function should_process(notebook_path::AbstractString; filter::Union{Nothing,String}=nothing)::Bool
    return filter === nothing || occursin(filter::String, notebook_path)
end

"""
    conversion_header(notebook_path::AbstractString, script_name::AbstractString) -> String

Generate header comment for converted script file.

# Arguments
- `notebook_path`: Path to source notebook
- `script_name`: Name of the script file

# Returns
- Multi-line header string
"""
function conversion_header(notebook_path::AbstractString, script_name::AbstractString)::String
    return """
# This file was automatically generated from $(notebook_path)
# by NotebookConversion module at $(Dates.now())
# Do not edit by hand. Edit the notebook instead.
#
# Source notebook: $(script_name)

"""
end

"""
    convert_notebook(notebook_path::AbstractString, script_path::AbstractString; quiet::Bool=false) -> Bool

Convert a single notebook to a Julia script.

# Arguments
- `notebook_path`: Path to source notebook
- `script_path`: Path to target script
- `quiet`: If true, suppress output

# Returns
- true if conversion successful, false if skipped (empty notebook)
"""
function convert_notebook(notebook_path::AbstractString, script_path::AbstractString; quiet::Bool=false)::Bool
    code = extract_code_from_notebook(notebook_path)
    if isempty(strip(code))
        quiet || println("  Skipped (no code cells): $(notebook_path)")
        return false
    end
    header = conversion_header(notebook_path, basename(notebook_path))
    mkpath(dirname(script_path))
    open(script_path, "w") do io
        write(io, header * code)
    end
    return true
end

"""
    verify_conversion(examples_dir::AbstractString, output_dir::AbstractString; filter::Union{Nothing,String}=nothing) -> Vector{String}

Verify 1:1 mapping between notebooks and scripts.

# Arguments
- `examples_dir`: Base examples directory
- `output_dir`: Base output directory
- `filter`: Optional filter pattern

# Returns
- Vector of missing script paths (empty if all present)
"""
function verify_conversion(examples_dir::AbstractString, output_dir::AbstractString; filter::Union{Nothing,String}=nothing)::Vector{String}
    missing = String[]
    for (root, _dirs, files) in walkdir(examples_dir)
        for f in files
            endswith(f, ".ipynb") || continue
            nb = joinpath(root, f)
            filter === nothing || occursin(filter, nb) || continue
            sp = notebook_to_script_path(nb, examples_dir, output_dir)
            if !isfile(sp)
                push!(missing, sp)
            end
        end
    end
    return missing
end

end # module
