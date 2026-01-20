using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Force GLMakie to use the main thread for OpenGL context
# This is critical for macOS compatibility
ENV["JULIA_GLMAKIE_SCREEN_SINGLETON"] = "true"

include(joinpath(@__DIR__, "src/PendulumApp.jl"))
using .PendulumApp

println("Starting Reactive Pendulum...")

fig = launch()

# Keep the script alive by waiting for the window to close
# This is the correct way to block on GLMakie on macOS
if !isinteractive()
    println("Press Ctrl+C to exit or close the window...")
    try
        while !isempty(fig.scene.current_screens) && isopen(fig.scene.current_screens[1])
            sleep(0.1)
        end
    catch e
        if !(e isa InterruptException)
            rethrow(e)
        end
    end
    println("Window closed. Exiting.")
end
