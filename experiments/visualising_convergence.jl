@everywhere begin
    using BoostedDensities, Distributions, Flux

    function contour_plot_with_modes(q, i)
        rng = -6.5:0.05:+6.5; lim = (first(rng),last(rng))
        contour(plot_splat(q[i], plot_range = rng)..., 
                xlims = lim, xticks = [],
                ylims = lim, yticks = [], 
                size  = (500, 500), legend = false, fmt = :pdf)
        Plots.scatter!(mean(p)[1,:], mean(p)[2,:], 
                    framestyle = :box,
                    legend = false, marker = :x, 
                    xlim = lim, ylim = lim, color=:white)
        cond = @sprintf "ring_%02d" i
        dest = plot_destination("visualising_convergence", cond, contour, lim, lim)
        savefig(dest)
    end
end

include(joinpath(pwd(), "utilities.jl"))

# Ring of Gaussians
# -----------------
function model_worker_ring(i)
    p  = GaussianMixture(σ = 1/4)
    q0 = MvNormal(2, 1)
    num_p, num_q = 10_000, 10_000
    srand(1337+i)
    model = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 1), x->σ.(x)) # classifier architecture
    q, train_history = run_experiment(p, q0, num_p, num_q, iter = 5, num_epochs = 600, model = model, verbose = false, run_boosting_metrics = true, optimiser = p -> Flux.ADAM(p), seed = 1337 + 2*i)
    return q, train_history
end

# Random Gaussians
# -----------------
function model_worker_random(i)
    srand(1337 + 1)
    means    = rand(MvNormal(2, 4), 8)
    rp = GaussianMixture(mapslices(x->MvNormal(x, 1/2), means, 1)|>vec)
    num_p, num_q = 10_000, 10_000
    p_samps = rand(rp, num_p)
    Σ = cov(p_samps, 2)
    μ = mean(p_samps, 2) |> vec
    q0 = MvNormal(μ, 1)
    srand(2*1337+i)
    model = Chain(Dense(2, 10, Flux.softplus), Dense(10,10, Flux.softplus),  Dense(10, 1), x->σ.(x)) # classifier architecture
    q2, train_history2 = run_experiment(rp, q0, num_p, num_q, iter = 10, verbose = false, num_epochs = 600, model = model,  optimiser = p -> Flux.ADAM(p), run_boosting_metrics = true, seed = 1337 + 2*i)
    return q2, train_history2
end

using Plots; gr(size=(800, 600))
model_worker_random(1)

using JLD2
ring_results   = pmap(model_worker_ring,   1:30)
@save "visualising_convergence_ring.jld2"   ring_results


random_results   = pmap(model_worker_random,   1:5)
@save "visualising_convergence_random.jld2"   random_results

using JLD2
@load "visualising_convergence_ring.jld2" 




map(random_results[1:1]) do r
    metric = 2
    _q, (_small_metrics, _coverage_metrics) =  r
    plot(inds(_small_metrics), map(_small_metrics) do m; m[metric] end)
end




random_results = pmap(model_worker_random, 1:5)
@save "visualising_convergence_random.jld2" random_results
@load "visualising_convergence_random.jld2"

foreach(0:5) do i
    contour_plot_with_modes(q, i)
end




experiment_results = pmap(model_worker_random, 1:30)


function contour_plot_with_modes(q, i)
    rng = -10:0.05:+10; lim = (first(rng),last(rng))
    contour(plot_splat(q[i], plot_range = rng)..., 
            xlims = lim, xticks = [],
            ylims = lim, yticks = [], 
            size  = (500, 500), legend = false, fmt = :pdf)
    Plots.scatter!(mean(rp)[1,:], mean(rp)[2,:], 
                framestyle = :box,
                legend = false, marker = :x, 
                xlim = lim, ylim = lim, color=:white)
    cond = @sprintf "rand_%02d" i
    dest = plot_destination("visualising_convergence", cond, contour, lim, lim)
    savefig(dest)
end


foreach(0:5) do i
    contour_plot_with_modes(q2, i)
end




Plots.scatter!(mean(random_p)[1,:], mean(random_p)[2,:], legend = false, marker = :x,  color=:white, xlim = (first(rng), last(rng)), ylim = (first(rng), last(rng)), color=:white)

rng = -10:0.05:10
contour(plot_splat(rp, plot_range = rng)..., legend = false, fill = false)
contour(plot_splat(q0, plot_range = rng)..., legend = false, fill = false)

# metrics = train_history[2]



# p_samps = rand(p, num_p)
# q_samps = rand(q, num_q)

# normalise!(q)


function inds(train_history)
    boost_iter = map(x->Int(first(x)), train_history)
    counts     = map(i->count(x->(x == i), boost_iter), sort(unique(boost_iter)))
    cscounts   = cumsum(counts)
    return indices = [(i - cscounts[z])/counts[z] + cscounts[z] for (i,z) in enumerate(boost_iter)]
end
inds(train_history)

# vals = cat(1, [t[2:end]' for t in train_history]...)

# plot(inds(train_history), vals)

# rng = -6:0.05:+6
# # contour(plot_splat(q, plot_range = rng)..., legend=false)

# srand(1337)
# boosted_q, train_history = run_experiment(p, q0, num_p, num_q, iter = 5, num_epochs = 600, model = model, weak_boost = 0.35, alpha =(x,y)->3/4)


# z = :(xaxis = ("my label", (0,10), 0:0.5:10, :log, :flip, font(20, "Courier")), legend=false)







# function inds(train_history)
#     boost_iter = map(first, train_history)
#     counts     = cumsum(map(i->count(x->(x == i), boost_iter), sort(unique(boost_iter))))
#     return indices = [z*i/counts[Int(z)] for (i,z) in enumerate(boost_iter)]
# end

# inds(train_history)

# vals = cat(1, [t[2:end]' for t in train_history]...)

# plot(vals)
# plot(inds(train_history), vals)


# q_10, train_history = run_experiment(p, q0, num_p, num_q, iter = 5, alpha = (x...)->1.0, num_epochs = 600, model = model)


# srand(1337)
# means    = rand(MvNormal(2, 4), 8)
# random_p = GaussianMixture(mapslices(x->MvNormal(x, 1/2), means, 1)|>vec)
# num_p, num_q = 10_000, 10_000

# p_samps = rand(random_p, num_p)

# Σ = cov(p_samps, 2)
# μ = mean(p_samps, 2) |> vec
# q0 = MvNormal(μ, Σ)

# rng = -9:0.05:+9
# contour(plot_splat(random_p, plot_range = rng)..., legend = true)
# Plots.scatter(mean(random_p)[1,:], mean(random_p)[2,:], legend = false, marker = :x,  color=:white, xlim = (first(rng), last(rng)), ylim = (first(rng), last(rng)))

# srand(1337) # set random seed
# model = Chain(Dense(2, 30, softplus), Dense(30,30,softplus), Dense(30, 2), softmax) # classifier architecture
# q, train_history = run_experiment(random_p, q0, num_p, num_q, iter = 8, alpha = boosted_alpha, num_epochs = 800, model = model,  optimiser = p -> Flux.ADAM(p))
# contour(plot_splat(q, plot_range = rng)..., legend = false, fill = false)
# Plots.scatter!(mean(random_p)[1,:], mean(random_p)[2,:], legend = false, marker = :x,  color=:white, xlim = (first(rng), last(rng)), ylim = (first(rng), last(rng)), color=:white)

