"""
Test script for Julia support modules.

Tests all modules in support/src/julia/ with real methods.
"""

# Get absolute path to support modules
const SUPPORT_ROOT = joinpath(@__DIR__, "..", "src", "julia")

println("=" ^ 60)
println("JULIA SUPPORT MODULES TEST SUITE")
println("=" ^ 60)
println()

# Track test results
passed = 0
failed = 0
errors = String[]

macro test_module(name, block)
    quote
        print("Testing $($(string(name)))... ")
        try
            $(esc(block))
            global passed += 1
            println("✓ PASS")
        catch e
            global failed += 1
            push!(errors, "$($(string(name))): $e")
            println("✗ FAIL: $e")
        end
    end
end

# ============================================================
# Test Utils Modules
# ============================================================
println("\n--- UTILS ---")

@test_module "FileUtils" begin
    include(joinpath(SUPPORT_ROOT, "utils", "FileUtils.jl"))
    using .FileUtils
    
    # Test ensure_directory
    test_dir = mktempdir()
    nested = joinpath(test_dir, "a", "b", "c")
    result = ensure_directory(nested)
    @assert isdir(nested) "ensure_directory failed to create nested directory"
    @assert result == nested "ensure_directory should return the path"
    
    # Test find_files
    test_file = joinpath(test_dir, "test.txt")
    open(test_file, "w") do io
        write(io, "test content")
    end
    found = find_files(test_dir, r"\.txt$")
    @assert length(found) >= 1 "find_files should find .txt files"
    
    # Test safe_write
    safe_path = joinpath(test_dir, "safe.txt")
    safe_write(safe_path, "safe content")
    @assert isfile(safe_path) "safe_write should create file"
    @assert read(safe_path, String) == "safe content" "safe_write content mismatch"
    
    rm(test_dir, recursive=true)
end

@test_module "ConfigUtils" begin
    include(joinpath(SUPPORT_ROOT, "utils", "ConfigUtils.jl"))
    using .ConfigUtils
    
    # Test load_toml_config with non-required file
    config = load_toml_config("nonexistent.toml", required=false)
    @assert isempty(config) "load_toml_config should return empty dict for missing file"
    
    # Test get_config_value
    test_config = Dict("section" => Dict("key" => "value"))
    result = get_config_value(test_config, "section", "key", default="default")
    @assert result == "value" "get_config_value should return nested value"
    
    result_default = get_config_value(test_config, "missing", "key", default="default")
    @assert result_default == "default" "get_config_value should return default for missing"
    
    # Test merge_configs
    c1 = Dict("a" => 1)
    c2 = Dict("b" => 2)
    merged = merge_configs(c1, c2)
    @assert merged["a"] == 1 && merged["b"] == 2 "merge_configs should combine dicts"
end

@test_module "CommandUtils" begin
    include(joinpath(SUPPORT_ROOT, "utils", "CommandUtils.jl"))
    using .CommandUtils
    
    # Test run_command
    result = run_command(`echo "hello"`, "Echo test", quiet=true, show_output=false)
    @assert result == true "run_command should succeed for echo"
    
    # Test log functions (just ensure they don't error)
    log_info("Test info", quiet=true)
    log_warn("Test warn", quiet=true)
    log_error("Test error")
end

# ============================================================
# Test Statistics Modules
# ============================================================
println("\n--- STATISTICS ---")

@test_module "AnalysisUtils" begin
    include(joinpath(SUPPORT_ROOT, "statistics", "AnalysisUtils.jl"))
    using .AnalysisUtils
    
    real = [1.0, 2.0, 3.0, 4.0, 5.0]
    est = [1.1, 2.1, 3.1, 4.1, 5.1]
    
    # Test RMSE
    rmse = calculate_rmse(real, est)
    @assert rmse ≈ 0.1 atol=0.01 "RMSE should be approximately 0.1"
    
    # Test MAE
    mae = calculate_mae(real, est)
    @assert mae ≈ 0.1 atol=0.01 "MAE should be approximately 0.1"
    
    # Test VAF
    vaf = calculate_vaf(real, est)
    @assert vaf > 99 "VAF should be high for similar signals"
    
    # Test get_statistics
    stats = get_statistics(real, est)
    @assert haskey(stats, :rmse) "stats should have rmse"
    @assert haskey(stats, :vaf) "stats should have vaf"
    
    # Test detect_divergence
    stable, reason = detect_divergence(real)
    @assert stable == false "stable signal should not be divergent"
    
    divergent, reason = detect_divergence([1.0, NaN, 3.0])
    @assert divergent == true "NaN signal should be divergent"
end

@test_module "ValidationUtils" begin
    include(joinpath(SUPPORT_ROOT, "statistics", "ValidationUtils.jl"))
    using .ValidationUtils: ValidationResult, validate_threshold, validate_all
    
    # Test validate_threshold
    result = validate_threshold(0.95, 0.9, comparison=:>=, name="Accuracy")
    @assert result.passed == true "0.95 >= 0.9 should pass"
    @assert length(result.messages) > 0 "Should have messages"
    
    result_fail = validate_threshold(0.85, 0.9, comparison=:>=, name="Accuracy")
    @assert result_fail.passed == false "0.85 >= 0.9 should fail"
    
    # Test validate_all
    v1 = ValidationResult(true, ["pass1"])
    v2 = ValidationResult(true, ["pass2"])
    combined = validate_all([v1, v2])
    @assert combined.passed == true "All passing should pass"
    @assert length(combined.messages) == 2 "Should combine messages"
end

# ============================================================
# Test Visualization Modules
# ============================================================
println("\n--- VISUALIZATION ---")

@test_module "PlottingUtils" begin
    include(joinpath(SUPPORT_ROOT, "visualization", "PlottingUtils.jl"))
    using .PlottingUtils
    
    # Test setup_plot_defaults (just ensure no error)
    setup_plot_defaults()
    
    # Test create_figure
    p = create_figure(title="Test", xlabel="X", ylabel="Y")
    @assert p !== nothing "create_figure should return a plot"
    
    # Test save_plot
    test_dir = mktempdir()
    using Plots
    plot!([1,2,3], [1,2,3])
    path = save_plot(p, "test.png", test_dir)
    @assert isfile(path) "save_plot should create file"
    rm(test_dir, recursive=true)
end

@test_module "AnimationUtils" begin
    include(joinpath(SUPPORT_ROOT, "visualization", "AnimationUtils.jl"))
    using .AnimationUtils: create_animation
    using Plots
    
    # Test create_animation with simple frames
    frames = [plot([1,2,3], [i, i+1, i+2]) for i in 1:3]
    anim = create_animation(frames, fps=5)
    @assert anim !== nothing "create_animation should return animation"
end

# ============================================================
# Test Logging Modules
# ============================================================
println("\n--- LOGGING ---")

@test_module "LoggingUtils" begin
    include(joinpath(SUPPORT_ROOT, "logging", "LoggingUtils.jl"))
    using .LoggingUtils: setup_logger, log_info, log_warn, log_section
    
    test_dir = mktempdir()
    
    # Test setup_logger
    logger = setup_logger(test_dir, "config.toml", 1234)
    @assert logger !== nothing "setup_logger should return a logger"
    
    # Test logging functions
    log_info(logger, "Test info message")
    log_warn(logger, "Test warning")
    log_section(logger, "Test Section")
    
    # Check log file was created
    log_path = joinpath(test_dir, "execution.log")
    @assert isfile(log_path) "Log file should be created"
    
    content = read(log_path, String)
    @assert occursin("Test info message", content) "Log should contain info message"
    
    rm(test_dir, recursive=true)
end

@test_module "ReportingUtils" begin
    include(joinpath(SUPPORT_ROOT, "logging", "ReportingUtils.jl"))
    using .ReportingUtils
    
    test_dir = mktempdir()
    
    # Test setup_report
    report_path = setup_report(test_dir, "Test Report")
    @assert isfile(report_path) "Report file should be created"
    
    # Test append_to_report
    append_to_report(report_path, "Additional content")
    content = read(report_path, String)
    @assert occursin("Additional content", content) "Report should contain appended content"
    
    # Test generate_markdown_table
    headers = ["Col1", "Col2"]
    rows = [["a", "b"], ["c", "d"]]
    table = generate_markdown_table(headers, rows)
    @assert occursin("|", table) "Table should contain pipe characters"
    @assert occursin("Col1", table) "Table should contain headers"
    
    rm(test_dir, recursive=true)
end

# ============================================================
# Test Environment Modules
# ============================================================
println("\n--- ENVIRONMENT ---")

@test_module "EnvironmentSetup" begin
    include(joinpath(SUPPORT_ROOT, "environment", "EnvironmentSetup.jl"))
    using .EnvironmentSetup
    
    # Test get_required_packages
    packages = get_required_packages()
    @assert isa(packages, Vector{String}) "Should return vector of strings"
    @assert length(packages) > 0 "Should have some required packages"
    
    # Test setup_environment_vars
    setup_environment_vars()
    @assert haskey(ENV, "GKSwstype") "Should set GKSwstype"
end

@test_module "NotebookConversion" begin
    include(joinpath(SUPPORT_ROOT, "environment", "NotebookConversion.jl"))
    using .NotebookConversion
    
    # Test conversion_header
    header = conversion_header("/path/to/notebook.ipynb", "notebook.ipynb")
    @assert occursin("automatically generated", header) "Header should mention auto-generation"
    
    # Test should_process
    @assert should_process("test.ipynb") == true "Should process without filter"
    @assert should_process("test.ipynb", filter="test") == true "Should match filter"
    @assert should_process("test.ipynb", filter="other") == false "Should not match wrong filter"
    
    # Test notebook_to_script_path
    path = notebook_to_script_path("/examples/test.ipynb", "/examples", "/scripts")
    @assert endswith(path, ".jl") "Should convert to .jl extension"
end

# ============================================================
# Summary
# ============================================================
println("\n" * "=" ^ 60)
println("TEST SUMMARY")
println("=" ^ 60)
println("Passed: $passed")
println("Failed: $failed")

if !isempty(errors)
    println("\nErrors:")
    for err in errors
        println("  - $err")
    end
end

println("\nAll tests use REAL METHODS - NO MOCKS")
println("=" ^ 60)

exit(failed > 0 ? 1 : 0)
