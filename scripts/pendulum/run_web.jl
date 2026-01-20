using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Use WGLMakie for browser-based rendering on macOS with Apple Silicon
# This avoids the GLMakie Bus error by using WebGL instead of OpenGL
using WGLMakie
WGLMakie.activate!()

# Force Makie to use WGLMakie before including PendulumApp
# PendulumApp will detect this and use the active backend
ENV["MAKIE_BACKEND"] = "WGLMakie"

include(joinpath(@__DIR__, "src/PendulumApp.jl"))
using .PendulumApp

println("Starting Reactive Pendulum (Web Browser Mode)...")
println("The dashboard will open in your default web browser.")

fig = launch()

# WGLMakie display handles the server automatically
if !isinteractive()
    println("Press Ctrl+C to stop the server.")
    try
        wait() # Wait indefinitely for the HTTP server
    catch e
        if !(e isa InterruptException)
            rethrow(e)
        end
    end
    println("Server stopped. Exiting.")
end
