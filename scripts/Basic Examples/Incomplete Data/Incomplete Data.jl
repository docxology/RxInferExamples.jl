using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RxInfer
using LinearAlgebra
using Distributions
using Plots
using TOML
using Dates
using Random

# Include support infrastructure
# Assuming the script is run from the project root or we can find support relative to this script
# Based on file structure: scripts/Basic Examples/Incomplete Data/Incomplete Data.jl
# support is at ../../../support
const PROJECT_ROOT = joinpath(@__DIR__, "../../..")
const SUPPORT_PATH = joinpath(PROJECT_ROOT, "support", "src", "julia")

include(joinpath(SUPPORT_PATH, "LoggingUtils.jl"))
include(joinpath(SUPPORT_PATH, "AnalysisUtils.jl"))
include(joinpath(SUPPORT_PATH, "ReportingUtils.jl"))

# Include local modules
include("src/DataGeneration.jl")
include("src/Models.jl")
include("src/Analysis.jl")
include("src/Experiments.jl")

# Main Execution
function main()
    # Configuration
    config_path = joinpath(@__DIR__, "config.toml")
    config = TOML.parsefile(config_path)
    
    # Setup Output
    output_dir = get(config["general"], "output_dir", "output")
    if !isabspath(output_dir)
        output_dir = joinpath(@__DIR__, output_dir)
    end
    mkpath(output_dir)
    
    seed = get(config["general"], "seed", 1234)
    
    # Run Experiment
    Experiments.run_incomplete_data(config["general"], output_dir, seed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end