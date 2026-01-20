Base.@kwdef mutable struct PendulumWorldParameters
    bob_mass           :: Float64 = 0.2 # grams
    rod_length         :: Float64 = 0.2 # cm
    friction           :: Float64 = 0.2
    gravity            :: Float64 = 9.81
    engine_max_power   :: Float64 = 1.0
    observations_noise :: Float64 = 1e-6
    worlds_clock_Δt    :: Float64 = 1 / 30
end

# Global parameters instance to be shared across the application
const parameters = PendulumWorldParameters();

export PendulumWorldParameters, parameters
