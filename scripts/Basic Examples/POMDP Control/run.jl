# POMDP Control - Unified Entry Point
# ====================================
# Main dispatcher for all POMDP Control experiments
#
# Usage:
#   julia --project=. run.jl              # Interactive menu
#   julia --project=. run.jl analysis     # Full analysis
#   julia --project=. run.jl sweep        # Parameter sweep
#   julia --project=. run.jl headless     # CI verification
#   julia --project=. run.jl viz          # With visualization
#
# Configuration: config.toml
# Documentation: docs/README.md

using Pkg
Pkg.activate(@__DIR__)

const PROJECT_DIR = @__DIR__

function print_banner()
    println()
    println("═"^60)
    println("   POMDP Control - Active Inference GridWorld")
    println("═"^60)
    println()
end

function print_menu()
    println("Available commands:")
    println()
    println("  analysis  - Full analysis with statistics & visualization")
    println("  sweep     - Combinatorial parameter sweep")
    println("  headless  - Headless CI verification")
    println("  viz       - Run with full visualization output")
    println("  legacy    - Original monolithic script")
    println()
    println("Usage: julia --project=. run.jl <command>")
    println()
end

function main()
    print_banner()
    
    cmd = length(ARGS) >= 1 ? lowercase(ARGS[1]) : ""
    
    if cmd == "analysis" || cmd == "full"
        println("Running: Full Analysis")
        println("─"^60)
        include(joinpath(PROJECT_DIR, "scripts", "full_analysis.jl"))
        
    elseif cmd == "sweep"
        println("Running: Parameter Sweep")
        println("─"^60)
        include(joinpath(PROJECT_DIR, "scripts", "sweep.jl"))
        
    elseif cmd == "headless" || cmd == "ci"
        println("Running: Headless Verification")
        println("─"^60)
        include(joinpath(PROJECT_DIR, "scripts", "headless.jl"))
        
    elseif cmd == "viz" || cmd == "visual"
        println("Running: With Visualization")
        println("─"^60)
        include(joinpath(PROJECT_DIR, "scripts", "with_viz.jl"))
        
    elseif cmd == "legacy"
        println("Running: Legacy Script")
        println("─"^60)
        include(joinpath(PROJECT_DIR, "scripts", "legacy.jl"))
        
    elseif cmd == "help" || cmd == "-h" || cmd == "--help"
        print_menu()
        
    elseif cmd == ""
        # Default: show menu and run analysis
        print_menu()
        println("No command specified.")
        println("Run 'julia --project=. run.jl analysis' for full analysis")
        println()
        
    else
        println("Unknown command: $cmd")
        println()
        print_menu()
    end
end

main()
