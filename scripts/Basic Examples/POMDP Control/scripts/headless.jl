# POMDP Control Headless Verification
# Called from run.jl: julia --project=. run.jl headless

const SCRIPT_DIR = @__DIR__
const PROJECT_DIR = dirname(SCRIPT_DIR)

using Pkg
Pkg.activate(PROJECT_DIR)
Pkg.instantiate()

ENV["GKSwstype"] = "100"  # Headless graphics

include(joinpath(PROJECT_DIR, "src/PODMPControlApp.jl"))
using .PODMPControlApp

println("="^50)
println("POMDP Control - Headless Verification")
println("="^50)
println()

# Run experiment set for verification (more experiments to allow learning)
n_experiments = 50
horizon = 4
success_rate = run_experiments(n_experiments, horizon; show_progress=false)

println()
println("Results:")
println("  Experiments: $n_experiments")
println("  Horizon: $horizon")
println("  Success rate: $(round(success_rate * 100, digits=1))%")
println()

# Exit with appropriate code
if success_rate >= 0.5
    println("✓ VERIFICATION PASSED (>50% success rate)")
    exit(0)
else
    println("✗ VERIFICATION FAILED (<50% success rate)")
    exit(1)
end
