"""
Support - Umbrella module for RxInferExamples.jl support utilities.

This module provides a unified interface to all support submodules.

# Submodules
- `Utils`: General utilities (CommandUtils, FileUtils, ConfigUtils)
- `Statistics`: Statistical analysis (AnalysisUtils, ValidationUtils)
- `Visualization`: Plotting and animation (PlottingUtils, AnimationUtils)
- `Logging`: Logging and reporting (LoggingUtils, ReportingUtils)
- `Environment`: Environment setup (EnvironmentSetup, NotebookConversion)

# Usage
```julia
include("support/src/julia/Support.jl")
using .Support
```
"""
module Support

# Get the directory of this file
const SUPPORT_ROOT = @__DIR__

# Utils submodule
include(joinpath(SUPPORT_ROOT, "utils", "CommandUtils.jl"))
include(joinpath(SUPPORT_ROOT, "utils", "FileUtils.jl"))
include(joinpath(SUPPORT_ROOT, "utils", "ConfigUtils.jl"))

# Statistics submodule
include(joinpath(SUPPORT_ROOT, "statistics", "AnalysisUtils.jl"))
include(joinpath(SUPPORT_ROOT, "statistics", "ValidationUtils.jl"))

# Visualization submodule
include(joinpath(SUPPORT_ROOT, "visualization", "PlottingUtils.jl"))
include(joinpath(SUPPORT_ROOT, "visualization", "AnimationUtils.jl"))

# Logging submodule
include(joinpath(SUPPORT_ROOT, "logging", "LoggingUtils.jl"))
include(joinpath(SUPPORT_ROOT, "logging", "ReportingUtils.jl"))

# Environment submodule
include(joinpath(SUPPORT_ROOT, "environment", "EnvironmentSetup.jl"))
include(joinpath(SUPPORT_ROOT, "environment", "NotebookConversion.jl"))

# Re-export all submodules
using .CommandUtils
using .FileUtils
using .ConfigUtils
using .AnalysisUtils
using .ValidationUtils
using .PlottingUtils
using .AnimationUtils
using .LoggingUtils
using .ReportingUtils
using .EnvironmentSetup
using .NotebookConversion

# Export all symbols
export CommandUtils, FileUtils, ConfigUtils
export AnalysisUtils, ValidationUtils
export PlottingUtils, AnimationUtils
export LoggingUtils, ReportingUtils
export EnvironmentSetup, NotebookConversion

end # module
