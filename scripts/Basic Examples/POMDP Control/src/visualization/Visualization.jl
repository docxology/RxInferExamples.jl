# Visualization Module for POMDP Control
# Organizes all visualization and export functionality

# Include visualization components
include("MatrixPlots.jl")
include("BeliefPlots.jl")
include("TrajectoryPlots.jl")
include("DiagnosticsPlots.jl")
include("Animations.jl")
include("DataExport.jl")

# Output directory management
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output")

"""
    ensure_output_dirs()

Create output directories if they don't exist.
"""
function ensure_output_dirs()
    dirs = [
        OUTPUT_DIR,
        joinpath(OUTPUT_DIR, "figures"),
        joinpath(OUTPUT_DIR, "figures", "matrices"),
        joinpath(OUTPUT_DIR, "figures", "beliefs"),
        joinpath(OUTPUT_DIR, "figures", "trajectories"),
        joinpath(OUTPUT_DIR, "figures", "diagnostics"),
        joinpath(OUTPUT_DIR, "animations"),
        joinpath(OUTPUT_DIR, "data")
    ]
    for dir in dirs
        mkpath(dir)
    end
    rxlog("info", "Output directories created at $OUTPUT_DIR")
    return OUTPUT_DIR
end

export OUTPUT_DIR, ensure_output_dirs
