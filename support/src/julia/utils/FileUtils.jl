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

export ensure_directory, copy_if_newer, find_files, safe_write

"""
    ensure_directory(path::AbstractString) -> String

Create directory and all parent directories if they don't exist.
Returns the path.
"""
function ensure_directory(path::AbstractString)::String
    mkpath(path)
    return path
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
        ensure_directory(dirname(target))
        cp(source, target, force=true)
        quiet || println("Copied: $source → $target")
        return true
    end
    return false
end

"""
    find_files(directory::AbstractString, pattern::Regex; recursive::Bool=true) -> Vector{String}

Find all files in directory matching the pattern.
"""
function find_files(directory::AbstractString, pattern::Regex; recursive::Bool=true)::Vector{String}
    matches = String[]
    
    if recursive
        for (root, dirs, files) in walkdir(directory)
            for f in files
                if occursin(pattern, f)
                    push!(matches, joinpath(root, f))
                end
            end
        end
    else
        for f in readdir(directory)
            if isfile(joinpath(directory, f)) && occursin(pattern, f)
                push!(matches, joinpath(directory, f))
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
    temp_path = path * ".tmp"
    ensure_directory(dirname(path))
    
    open(temp_path, "w") do io
        write(io, content)
    end
    
    mv(temp_path, path, force=true)
    return nothing
end

end # module
