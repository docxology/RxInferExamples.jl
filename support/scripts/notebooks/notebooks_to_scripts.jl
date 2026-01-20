#!/usr/bin/env julia

"""
Orchestration script for notebook-to-script conversion.

This thin orchestrator uses the NotebookConversion module from support/src/
to convert Jupyter notebooks to Julia scripts.

Usage:
  julia support/scripts/notebooks/notebooks_to_scripts.jl [options]

Options:
  --dry-run            Show what would be converted without writing files
  --skip-existing      Skip if target `.jl` is newer than source
  --force              Re-generate scripts even if up-to-date
  --filter SUBSTR      Only convert notebooks whose path contains SUBSTR
  --verify             After conversion, verify 1:1 mapping
  --quiet              Reduce output noise
  --list               List notebooks that would be processed
"""

# Setup paths
const script_dir = dirname(abspath(@__FILE__))
const support_dir = dirname(dirname(script_dir))
const src_dir = joinpath(support_dir, "src")
const repo_root = dirname(support_dir)

# Include source module
include(joinpath(src_dir, "NotebookConversion.jl"))
using .NotebookConversion

# Source directories to process
const examples_dir = joinpath(repo_root, "examples")
const interactive_dir = joinpath(repo_root, "interactive")
const output_dir = joinpath(repo_root, "scripts")

# Validate directories
if !isdir(examples_dir)
    error("Cannot find 'examples' directory at $(examples_dir)")
end

mkpath(output_dir)

# Argument parsing (kept in orchestrator for CLI interface)
function parse_args(args::Vector{String})
    options = Dict{String,Any}(
        "dry_run" => false,
        "skip_existing" => false,
        "force" => false,
        "filter" => nothing,
        "verify" => false,
        "quiet" => false,
        "list" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--dry-run"
            options["dry_run"] = true
        elseif a == "--skip-existing"
            options["skip_existing"] = true
        elseif a == "--force"
            options["force"] = true
        elseif a == "--verify"
            options["verify"] = true
        elseif a == "--quiet"
            options["quiet"] = true
        elseif a == "--list"
            options["list"] = true
        elseif a == "--filter"
            i += 1
            i > length(args) && error("--filter requires a value")
            options["filter"] = args[i]
        else
            error("Unknown option: $(a)")
        end
        i += 1
    end
    return (; dry_run = options["dry_run"],
             skip_existing = options["skip_existing"],
             force = options["force"],
             filter = options["filter"],
             verify = options["verify"],
             quiet = options["quiet"],
             list = options["list"])
end

function main()
    opts = parse_args(ARGS)

    opts.quiet || println("Converting notebooks to Julia scripts...")
    notebooks_processed = 0
    notebooks_skipped = 0
    errors = 0
    planned = Tuple{String,String,String}[]  # (notebook_path, source_dir, output_dir)
    
    # Define source directories to process
    source_dirs = [
        (examples_dir, output_dir),
        (interactive_dir, output_dir)
    ]

    # Walk each source directory and collect notebooks
    for (source_dir, target_base) in source_dirs
        isdir(source_dir) || continue
        
        for (root, _dirs, files) in walkdir(source_dir)
            root == source_dir && continue
            target_dir = replace(root, source_dir => target_base)
            mkpath(target_dir)
            copy_project_files(root, target_dir; quiet=opts.quiet)
            
            for file in files
                endswith(file, ".ipynb") || continue
                notebook_path = joinpath(root, file)
                should_process(notebook_path; filter=opts.filter) || continue
                script_path = notebook_to_script_path(notebook_path, source_dir, target_base)
                
                if opts.skip_existing && !opts.force && up_to_date(notebook_path, script_path)
                    notebooks_skipped += 1
                    continue
                end
                push!(planned, (notebook_path, source_dir, target_base))
                opts.list && println(notebook_path)
            end
        end
    end

    opts.list && opts.dry_run && return 0

    # Convert planned notebooks
    for (notebook_path, source_dir, target_base) in planned
        try
            script_path = notebook_to_script_path(notebook_path, source_dir, target_base)
            if !opts.dry_run
                opts.quiet || println("Converting $(notebook_path)")
                if convert_notebook(notebook_path, script_path; quiet=opts.quiet)
                    notebooks_processed += 1
                else
                    notebooks_skipped += 1
                end
            else
                notebooks_processed += 1
            end
        catch e
            println("  Error processing $(notebook_path): $(e)")
            errors += 1
        end
    end

    # Verify conversion for all source directories
    if opts.verify
        all_missing = String[]
        for (source_dir, target_base) in source_dirs
            isdir(source_dir) || continue
            missing = verify_conversion(source_dir, target_base; filter=opts.filter)
            append!(all_missing, missing)
        end
        if !isempty(all_missing)
            println("Verification failed. Missing $(length(all_missing)) scripts:")
            for m in all_missing
                println("  " * m)
            end
            return 2
        end
    end

    opts.quiet || println("Conversion complete. Processed $(notebooks_processed) notebooks. Skipped $(notebooks_skipped).")
    opts.quiet || println("Scripts available in '$(output_dir)' directory.")
    return errors == 0 ? 0 : 1
end

exit(main())