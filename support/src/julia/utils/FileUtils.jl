"""
FileUtils - File system utilities.

Provides common file operations for scripts and modules.

# Exports
- `ensure_directory`: Create directory if it doesn't exist
- `copy_if_newer`: Copy file only if source is newer
- `find_files`: Find files matching a pattern
- `safe_write`: Write with atomic replacement
"""
module FileUtils

export ensure_directory, ensure_parent_directory, copy_if_newer, find_files, safe_write

"""
    ensure_directory(path::AbstractString) -> String

Create directory and all parent directories if they don't exist.
Returns the absolute path.
"""
function ensure_directory(path::AbstractString)::String
    abs_path = abspath(path)
    mkpath(abs_path)
    return abs_path
end

"""
    ensure_parent_directory(path::AbstractString) -> String

Ensure the parent directory of a file exists.
Returns the absolute path to the parent.
"""
function ensure_parent_directory(path::AbstractString)::String
    parent = dirname(abspath(path))
    mkpath(parent)
    return parent
end

"""
    copy_if_newer(source::AbstractString, target::AbstractString; quiet::Bool=false) -> Bool

Copy source to target only if source is newer or target doesn't exist.
Returns true if copy was performed.
"""
function copy_if_newer(source::AbstractString, target::AbstractString; quiet::Bool=false)::Bool
    if !isfile(source)
        quiet || @warn "Source file does not exist: $source"
        return false
    end
    
    if !isfile(target) || mtime(source) > mtime(target)
        ensure_parent_directory(target)
        cp(source, target, force=true)
        quiet || println("ℹ️  Copied: $source → $target")
        return true
    end
    return false
end

"""
    find_files(directory::AbstractString, pattern::Regex; recursive::Bool=true) -> Vector{String}

Find all files in directory matching the pattern. Returns absolute paths.
"""
function find_files(directory::AbstractString, pattern::Regex; recursive::Bool=true)::Vector{String}
    matches = String[]
    abs_dir = abspath(directory)
    
    if recursive
        for (root, dirs, files) in walkdir(abs_dir)
            for f in files
                if occursin(pattern, f)
                    push!(matches, joinpath(root, f))
                end
            end
        end
    else
        for f in readdir(abs_dir)
            if isfile(joinpath(abs_dir, f)) && occursin(pattern, f)
                push!(matches, joinpath(abs_dir, f))
            end
        end
    end
    
    return matches
end

"""
    safe_write(path::AbstractString, content::AbstractString) -> Nothing

Write content to file atomically (write to temp, then rename).
"""
function safe_write(path::AbstractString, content::AbstractString)
    abs_path = abspath(path)
    temp_path = abs_path * ".tmp"
    ensure_parent_directory(abs_path)
    
    open(temp_path, "w") do io
        write(io, content)
    end
    
    mv(temp_path, abs_path, force=true)
    return nothing
end

end # module
