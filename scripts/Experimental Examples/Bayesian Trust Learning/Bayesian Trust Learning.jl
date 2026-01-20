# This file was automatically generated from /Users/4d/Documents/GitHub/RxInferExamples.jl/examples/Experimental Examples/Bayesian Trust Learning/Bayesian Trust Learning.ipynb
# by NotebookConversion module at 2026-01-19T04:41:26.197
# Do not edit by hand. Edit the notebook instead.
#
# Source notebook: Bayesian Trust Learning.ipynb

using RxInfer
using Distributions

# The three stages of router grief:
# 1. Denial: "This ticket looks simple!" (routes to Haiku)
# 2. Anger: "Why is the customer escalating?!" (still routes to Haiku)
# 3. Acceptance: "Maybe I should learn from this..." (our Bayesian approach)

@model function routing_strategy(y, ticket_context)
    # Meet our contestants:
    # 1. The Optimist - "Everything is fine! Use the cheap model!"
    θ_simple ~ simple_router(ticket_context = ticket_context)
    
    # 2. The Pessimist - "It's all terrible! GPT-4 for everything!"
    θ_complex ~ complex_router(ticket_context = ticket_context)
    
    # 3. The Realist - "Let's be reasonable about this..."
    θ_medium  ~ medium_router(ticket_context = ticket_context)
    
    # We start by trusting them equally (how naive!)
    routing_strategy ~ Categorical(ones(3) ./ 3)
    
    # But then reality hits...
    θ ~ Mixture(switch = routing_strategy, inputs = [θ_simple, θ_medium, θ_complex])
    
    # And we learn who's actually worth trusting
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)  # 1 = "big model needed!", 0 = "small model worked"
    end
end

"""
    LLMPrior: Where LLMs Judge Other LLMs
    
    It's like asking your friends which restaurant to go to,
    except your friends are language models and the restaurant
    is also a language model. Welcome to 2025!
"""
struct LLMPrior end

@node LLMPrior Stochastic [ 
    (b, aliases = [belief]),     # What the LLM believes
    (m, aliases = [model]),      # Which LLM we're asking
    (c, aliases = [context]),    # The ticket in question
    (t, aliases = [task])        # "Should we panic and use GPT-4?"
]

@rule LLMPrior(:b, Marginalisation) (q_m::PointMass{<:String}, q_c::PointMass{<:String}, q_t::PointMass{<:String}) = begin
    model_name = q_m.point
    
    # GPT models: The anxious overachievers
    # "This could be complex! Better use GPT-4! What if it's not complex? 
    #  Still use GPT-4! WHAT IF WE'RE WRONG?!"
    if model_name in ["gpt-5", "gpt-4.1"]
        return Beta(0.20, 0.05)  # Almost always says "use complex model"
    
    # Claude models: The confident minimalists
    # "Pfft, this is easy. Haiku can handle it. Trust me, I'm Claude."
    elseif model_name in ["claude-sonnet", "claude-opus"]
        return Beta(3.0, 9.0)  # Usually says "use simple model"
        
    # Claude Haiku: The wild card
    # "Maybe complex? Maybe simple? Life is uncertain, embrace the chaos!"
    elseif model_name in ["claude-haiku"]
        return Beta(3.0, 3.0)  # 50/50 with high variance
        
    # GPT-4o-mini: The pessimistic realist
    # "It's probably fine with a simple model... but I've been hurt before."
    elseif model_name in ["gpt-4o-mini"]
        return Beta(1.0, 5.0)  # Leans toward simple but cautious
    end
end

@model function complex_router(θ, ticket_context)
    # The premium committee: Only the finest LLMs
    θ_opus ~ LLMPrior(m = "claude-opus", c = ticket_context, t = "assess_complexity")
    θ_gpt  ~ LLMPrior(m = "gpt-5", c = ticket_context, t = "assess_complexity")
    
    # We trust Opus more because it sounds fancier
    switch ~ Categorical([0.2, 0.8]) 
    θ ~ Mixture(switch = switch, inputs = [θ_opus, θ_gpt])
end

@model function medium_router(θ, ticket_context)
    # The balanced committee: Not too hot, not too cold
    θ_claude ~ LLMPrior(m = "claude-sonnet", c = ticket_context, t = "assess_complexity")
    θ_gpt    ~ LLMPrior(m = "gpt-4.1", c = ticket_context, t = "assess_complexity")
    
    # Sonnet gets more weight because it's more poetic about its decisions
    switch ~ Categorical([0.7, 0.3]) 
    θ ~ Mixture(switch = switch, inputs = [θ_claude, θ_gpt])
end

@model function simple_router(θ, ticket_context)
    # The budget committee: "Have you considered... not spending money?"
    θ_claude_haiku ~ LLMPrior(m = "claude-haiku", c = ticket_context, t = "assess_complexity")
    θ_gpt_mini     ~ LLMPrior(m = "gpt-4o-mini", c = ticket_context, t = "assess_complexity")
    
    # Slight preference for Haiku because it's more zen about everything
    switch ~ Categorical([0.6, 0.4]) 
    θ ~ Mixture(switch = switch, inputs = [θ_claude_haiku, θ_gpt_mini])
end

ticket = "I have been trying to transfer money to my other bank account for the last 10 days but it keeps failing. Can you help me?"

# The harsh reality of what happened when we routed this:
# 0 = Ticket was successfully resolved with simple model
# 1 = Ticket was successfully resolved with complex model
outcomes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 11 simple model worked
            1.0,                                                    # 1 complex model worked
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,                 # 8 simple model worked
            1.0, 1.0]                                               # 2 complex model worked

# Let the Bayesian magic happen
result_joint = infer(
    model = routing_strategy(ticket_context=ticket), 
    data  = (y = outcomes, ),
    returnvars = KeepLast(),
    addons = AddonLogScale(),
    postprocess = UnpackMarginalPostprocess(),
)

# The verdict is in!
println("Trust scores after learning from reality:")
println("Simple Router: ", mean(result_joint.posteriors[:routing_strategy].p[1]))
println("Medium Router: ", mean(result_joint.posteriors[:routing_strategy].p[2]))  
println("Complex Router: ", mean(result_joint.posteriors[:routing_strategy].p[3]))


result_joint.posteriors[:routing_strategy]

using Plots
using Printf
# using Distributions, Statistics  # keep if you still need them elsewhere

# Backend (GR is default; feel free to switch to plotlyjs(), pyplot(), etc.)
gr()

# Extract trust scores - remember the order: [complex, medium, simple]
trust_scores = result_joint.posteriors[:routing_strategy].p

# Prepare data
labels = ["Simple Router\n(The Optimist)",
          "Medium Router\n(The Realist)",
          "Complex Router\n(The Pessimist)"]
x = 1:3
y = trust_scores .* 100
colors = [:darkgreen, :lightblue, :lightcoral]

# Bar plot
bar(
    x, y;
    bar_width = 0.6,
    fillcolor = colors,
    linecolor = :black,       # outline like strokecolor
    linewidth = 2,
    xticks = (x, labels),
    ylim = (0, 60),
    ylabel = "Trust Level (%)",
    title = "Router Trust Scores: Who Saw It Coming?",
    legend = :topright,
    size = (800, 500)
)

# Reference line at 33.3% with legend entry
hline!([33.3]; color = :gray, linestyle = :dash, linewidth = 2, label = "Initial Trust (Equal)")

# Value labels above bars
for (i, yi) in enumerate(y)
    annotate!(i, yi + 2, text(@sprintf("%.1f%%", yi), 12, :center, :bottom))
end
plot!()

println(result_joint.posteriors[:θ_complex])

println(result_joint.posteriors[:θ_simple])

println(result_joint.posteriors[:θ_medium])

simple_share  = result_joint.posteriors[:routing_strategy].p[1]
medium_share  = result_joint.posteriors[:routing_strategy].p[2]
complex_share = result_joint.posteriors[:routing_strategy].p[3];

# Normalize defensively
s = simple_share + medium_share + complex_share
simple_share, medium_share, complex_share = simple_share/s, medium_share/s, complex_share/s

# --- Model costs (edit as needed) ---
simple_cost  = 0.03   # e.g., Haiku per request (placeholder)
medium_cost  = 0.10   # whatever you pay for models within medium router
complex_cost = 3.00   # e.g., GPT-5 per request (placeholder)

# --- Cost per 100 tickets ---
blind_cost_per100  = 100 * complex_cost
perfect_per100     = 100 * (simple_share * simple_cost +
                            medium_share * medium_cost +
                            complex_share * complex_cost)
# Escalate policy: try Simple → Medium → Complex
escalate_per100    = 100 * (simple_cost +
                            (1 - simple_share) * medium_cost +
                            complex_share * complex_cost)

savings_perfect_pct  = 100 * (1 - perfect_per100  / blind_cost_per100)
savings_escalate_pct = 100 * (1 - escalate_per100 / blind_cost_per100)

println("🎯 Reality-informed routing mix:")
println("├─ Simple: $(round(simple_share * 100,  digits=1))%")
println("├─ Medium: $(round(medium_share * 100,  digits=1))%")
println("└─ Complex: $(round(complex_share * 100, digits=1))%")

println("\n💰 Cost Impact (per 100 tickets):")
println("├─ Blind Complex (send all to Complex): $(round(blind_cost_per100, digits=2))")
println("├─ Smart routing (perfect):             $(round(perfect_per100, digits=2))  → savings $(round(savings_perfect_pct, digits=1))%")
println("└─ Smart routing (escalate S→M→C):      $(round(escalate_per100, digits=2)) → savings $(round(savings_escalate_pct, digits=1))%")


# Bayesian routing: Sample from learned posteriors to make decisions (we don't do continuous learning here (yet))
# How that could look like:

# Helper to sample from MixtureDistribution (not natively supported)
sample_mixture(m::MixtureDistribution) = rand(m.components[rand(Categorical(m.weights))])

function route(posteriors, ticket_context)

    # here your logic to cluster tickets into a category

    # Sample which router to use
    router_idx = rand(posteriors[:routing_strategy])
    
    # Get complexity from selected router
    router_posteriors = [posteriors[:θ_complex], posteriors[:θ_medium], posteriors[:θ_simple]]
    complexity = sample_mixture(router_posteriors[router_idx])
    
    # Decision based on sampled complexity  
    model = complexity > 0.5 ? "complex" : "simple"
    
    return (model=model, complexity=complexity, router=router_idx)
end

# Use it
ticket = "I have been trying to transfer money to my other bank account for the last 10 days but it keeps failing. Can you help me?"

decision = route(result_joint.posteriors, ticket)
println("Route to $(decision.model)")