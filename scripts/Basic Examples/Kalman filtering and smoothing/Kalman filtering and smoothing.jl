
# Include environment setup utility from support/src
include("../../../support/src/julia/EnvironmentSetup.jl")
using .EnvironmentSetup

# Ensure project is instantiated
ensure_project_instantiated(".")

# Include local package
include("src/KalmanFilterApp.jl")
using .KalmanFilterApp

# Run all examples with config
run_all(config_path="config.toml")