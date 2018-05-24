info("Making code available to all $(nprocs()) workers")
using JLD2
@everywhere begin
    using BoostedDensities, Distributions, Flux
    using Plots; gr(size=(800, 600))

    include( "utilities.jl")

    # Ring of Gaussians
    # -----------------
    p  = GaussianMixture(σ = 1/4)
    q0 = MvNormal(2, 1)
    num_p, num_q = 10_000, 10_000

    function contour_plot_with_modes(q, cond)
        rng = -6.5:0.05:+6.6; lim = (first(rng),last(rng))
        contour(plot_splat(q, plot_range = rng)..., 
                xlims = lim, xticks = [],
                ylims = lim, yticks = [], 
                size  = (500, 500), legend = false, fmt = :pdf)
        Plots.scatter!(mean(p)[1,:], mean(p)[2,:], 
                    framestyle = :box,
                    legend = false, marker = :x, 
                    xlim = lim, ylim = lim, color=:white)
        dest = plot_destination("activation_comparison", cond, contour, lim,lim)
        savefig(dest)
    end

    experiment_worker(cond) = begin
        srand(rand(1:100)) # set random seed
        model  = Chain(Dense(2, 10, cond), Dense(10, 10, cond), Dense(10, 1), x->σ.(x)) # classifier architecture
        q, train_history = run_experiment(p, q0, num_p, num_q, iter = 5, num_epochs = 200, model = model, run_boosting_metrics=true)
        # contour_plot_with_modes(q, cond)
        return q, train_history
    end
end

conditions = [Flux.relu, Flux.softplus, Flux.elu, Flux.sigmoid, Flux.leakyrelu, tanh]

# contour_plot_with_modes(p, "control")

info("Running experiments with conditions: ", conditions)
experiment_results = pmap(experiment_worker, repeat(conditions, inner = 15))
