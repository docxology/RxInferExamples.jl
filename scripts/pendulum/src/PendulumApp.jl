module PendulumApp

using RxInfer
using Rocket
using LinearAlgebra
using Makie  # Use abstract Makie, actual backend determined at runtime
using DataStructures
using Dates

import ReactiveMP: getrecent, messageout, update!
import Rocket: subscribe!

# Include submodules
include("Constants.jl")
include("Utils.jl")
include("World.jl")
include("Model.jl")
include("Agent.jl")
include("Visualizer.jl")

function launch()
    rxlog("info", "Launching PendulumApp...")
    
    # Ensure GLMakie is the active backend
    GLMakie.activate!()
    
    # Now launch the main dashboard
    fig = launch_dashboard()
    display(fig) # Just in case it's not automatically displayed
    return fig
end

export launch

end
