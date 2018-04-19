# @everywhere quote 
using BoostedDensities, Distributions, Flux, JLD2
using Plots; gr(size=(800, 600))

include(joinpath(pwd(), "utilities.jl"))

# Ring of Gaussians
# -----------------
p  = GaussianMixture(σ = 1/4)
q0 = MvNormal(2, 1)
num_p, num_q = 10_000, 10_000

model = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 1), x->σ.(x)) # classifier architecture
q, train_history = run_experiment(p, q0, num_p, num_q, iter = 3, num_epochs = 600, model = model, verbose = true, run_boosting_metrics = true)

metrics = train_history[2]



p_samps = rand(p, num_p)
q_samps = rand(q, num_q)

normalise!(q)

inds(train_history)

# vals = cat(1, [t[2:end]' for t in train_history]...)

# plot(inds(train_history), vals)

# rng = -6:0.05:+6
# # contour(plot_splat(q, plot_range = rng)..., legend=false)

# srand(1337)
# boosted_q, train_history = run_experiment(p, q0, num_p, num_q, iter = 5, num_epochs = 600, model = model, weak_boost = 0.35, alpha =(x,y)->3/4)


# z = :(xaxis = ("my label", (0,10), 0:0.5:10, :log, :flip, font(20, "Courier")), legend=false)



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

