info("Making code available to $(nworkers()) workers")
push!(LOAD_PATH, pwd())

@everywhere begin
    using BoostedDensities, Flux, ProgressMeter
    using NPZ, JLD2
    using Plots; gr(size = [800,600])
    include("utilities.jl")


    epochs(inputs, n::Int) = (d for _ in Base.OneTo(n) for d in inputs)

    function batches(inputs, batch_size::Int)
        last_dim(arr, ind) = view(arr, ntuple((_)->(:), ndims(arr) - 1)..., ind)
        
        n = last(size(first(inputs)))
        return (map(x->last_dim(x, batch), inputs) for batch in Iterators.partition(Base.OneTo(n), batch_size))
    end

    function run_experiment!(p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps,
                            p::Distribution, q::QDensity;
                            batch_size = 200,
                            iter::Int  = 5,
                            early_stop::Real    = 0.2,  # when test error is `early_stop` below train error, stop early
                            weak_boost::Real    = Inf,  # when we have achieved a weak learner, stop early (set this to a small positive number to enable)
                            verbose::Bool              = false,
                            run_boosting_metrics::Bool = false, # time consuming to have these on since we will renormalise Q every iteration
                            alpha::Function     = mean_boosted_alpha,
                            optimiser::Function = Nesterov, 
                            sampler::Function   = rand,
                            num_epochs::Int     = 10,
                            seed::Int           = 1337,
                            model::Flux.Chain   = Chain(Dense(2, 10, softplus), Dense(10, 10, softplus),  Dense(10, 2), softmax))
        srand(seed) # set random seed
        train_history    = Vector{Any}[]
        boosting_history = Dict{Function,Float64}[]
        boosting_metrics = (coverage, expected_log_likelihood)
        if run_boosting_metrics
            push!(boosting_history, Dict(met => met(p, q, q_samps, p_samps) for met in boosting_metrics))
        end

        for i in Base.OneTo(iter)
            if verbose
                info("Sampling Q")
            end
            q_samps[:] = sampler(q, size(q_samps, 2))
            m          = deepcopy(model)
            opt        = optimiser(Flux.params(m))
            obj(p_samps, q_samps) = -mean(log.(σ.(m(p_samps)))) - mean(log1p.(-σ.(m(q_samps))))

            if verbose
                train_progress = Progress(num_epochs, 1, "Training classifier ($i of $iter): ") 
            end
            evalcb() = begin
                train_ce = (obj(train_p_samps, train_q_samps) |> Flux.Tracker.data)[]
                test_ce  = (obj(test_p_samps,  test_q_samps)  |> Flux.Tracker.data)[]

                if verbose 
                    ProgressMeter.next!(train_progress, showvalues = [("cross entropy (train)", train_ce), 
                                                                      ("cross entropy (test)",  test_ce)])
                end
                mu_p =  mu(p_samps,  m)
                mu_q = -mu(q_samps,  m)

                push!(train_history, [i, test_ce, mu_p, mu_q])
                if (train_ce + early_stop <= test_ce) || (min(mu_p, mu_q) >= weak_boost)
                    println()
                    info("Stopping early")
                    return :stop
                end
            end

            Flux.train!(obj, epochs(batches((train_p_samps, train_q_samps), batch_size), num_epochs), opt, cb = Flux.throttle(evalcb, 1))


            α = alpha(q_samps, m)
            @assert α > 0 "caught α < 0, with α=$α"
            push!(q, (m, α))

            if run_boosting_metrics
                if verbose
                    info("Running the boosting metrics")
                end
                push!(boosting_history, Dict(met => met(p, q, q_samps, p_samps) for met in boosting_metrics))
            end
        end

        return (train_history, boosting_history)
    end

    function adagan_worker(p, samples)
        num_p = size(samples, 2)
        num_q = num_p
        p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps = 
            allocate_train_valid(2, num_p, num_q; train_fraction = 3/4)

        p_samps[:] = samples

        q = QDensity(MvNormal(vec(mean(p_samps,2)), cov(p_samps, 2)))

        model  = Chain(Dense(2, 20, relu), Dense(20, 10, relu), Dense(10, 1)) # classifier architecture
        
        res = run_experiment!(p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps, alpha = (x...)->0.7,
                            p, q, model = model, iter = 10, num_epochs = 500, seed = 1337, batch_size = 100, early_stop = 0.2,
                            run_boosting_metrics = true, verbose = true)
        return q, res[1], res[2]
    end
end

info("Loading AdaGAN data")
results_dir    = joinpath(pwd(), "adagan_results")
adagan_results = sort(readdir(results_dir)) # make sure sorted so line up samples and parameters

process_adagan = function (data) 
    reshape(permutedims(data, (4,3,2,1)), (2, div(length(data),2))) 
end 

adagan_p_samps = map(f for f in adagan_results if ismatch(r"real_data_[0-9]{2}.npy", f)) do x
    data = npzread(joinpath(results_dir, x))
    process_adagan(data)
end

adagan_p_distributions = map(f for f in adagan_results if ismatch(r"real_data_params_mean_[0-9]{2}_var_[0-9].[0-9]{2}.npy", f)) do f
    means = npzread(joinpath(results_dir, f))
    var   = match(r".*_var_([0-9].[0-9]{2}).npy", f).captures |> first |> parse
    GaussianMixture(vec(mapslices(means, 2) do mu; MvNormal(mu, sqrt(var)) end))
end

info("Going for it")
@assert (n = length(adagan_p_distributions)) == length(adagan_p_samps)

iter = Iterators.partition(1:n, nworkers())
for (i,r) in enumerate(iter)
    @show r
    adagan_comparision_results = map(adagan_worker, adagan_p_distributions[r], adagan_p_samps[r])
    name = "adagan_comparision_results_no_kl_$(i)_of_$(length(iter)).jld2"
    @save name adagan_comparision_results 
end

adagan_comparision_results = map(f for f in sort(readdir(pwd())) if ismatch(r"adagan_comparision_results_.*_[0-9]{1}_of_[0-9]{2}.jld2", f)) do f
    jldopen(f, "r") do file
        file["adagan_comparision_results"]
    end
end
adagan_comparision_results  = Iterators.flatten(adagan_comparision_results )


adacovfefe_likelihood = cat(2, ([m[BoostedDensities.expected_log_likelihood] for m in res[3]] for res in adagan_comparision_results)...)
adacovfefe_coverage   = cat(2, ([m[BoostedDensities.coverage]                for m in res[3]] for res in adagan_comparision_results)...)

adagan_likelihood     = -npzread(joinpath(results_dir, "likelihood.npy"))[:,1:30-7]
adagan_coverage       = npzread(joinpath(results_dir, "coverage.npy"))[:,1:30-7]

a = 0.15
ylim = (3,10)
xlim = (0,10)
plot(1:10,  adacovfefe_likelihood[2:end,:],                     color = color_palette[5], opacity = a,   legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [])
plot!(1:10, mapslices(median, adacovfefe_likelihood[2:end,:], 2), color = color_palette[5], opacity = 1.0, legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2.5, legend = false, fmt = :pdf, framestyle = :box, ticks = [])

plot!(1:10,  adagan_likelihood,                                 color = color_palette[1], opacity = a,   legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [])
plot!(1:10, mapslices(median, adagan_likelihood, 2),              color = color_palette[1], opacity = 1.0, legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2.5, legend = false, fmt = :pdf, framestyle = :box, ticks = [])
dest = plot_destination("adagan_comparison_big", "likelihood", plot, xlim, ylim)
savefig(dest)

ylim = (0,1)
plot(1:10,  adacovfefe_coverage[2:end,:],                       color = color_palette[5], opacity = a,   legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [])
plot!(1:10, mapslices(median, adacovfefe_coverage[2:end,:], 2),   color = color_palette[5], opacity = 1.0, legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2.5, legend = false, fmt = :pdf, framestyle = :box, ticks = [])

plot!(1:10,  adagan_coverage,                                   color = color_palette[1], opacity = a,   legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [])
plot!(1:10, mapslices(median, adagan_coverage, 2),                color = color_palette[1], opacity = 1.0, legend=false, ylim = ylim, xlim = xlim, size = (800, 500), linewidth = 2.5, legend = false, fmt = :pdf, framestyle = :box, ticks = [])
dest = plot_destination("adagan_comparison_big", "coverage", plot, xlim, ylim)
savefig(dest)














# =======
# using NPZ
# results_dir = joinpath(pwd(), "experiments", "adagan", "adagan", "results_gmm")
# results = readdir(results_dir)

# using Plots; gr(size = [800,600])
# using AverageShiftedHistograms

# process_adagan = function (data) reshape(permutedims(data, (4,3,2,1)), (2, div(length(data),2))) end 

# fake_points = map(f for f in results if startswith(f, "fake_points")) do x
#     data = npzread(joinpath(results_dir, x))
#     process_adagan(data)
# end

# p_samps = npzread(joinpath(results_dir, "real_data.npy")) |> process_adagan
# o = ash(p_samps[1,:], p_samps[2,:])
# plot(o)

# # histogram2d(p_samps[1,:], p_samps[2,:], nbins = 200)

# q_samples = [plot(ash(q_sample[1,:], q_sample[2,:], mx=10, my=10), fill=true, legend=false) for q_sample in fake_points[1:3]]
# plot(q_samples..., legend=false)

# >>>>>>> 34a0d1c7e4e74bfe3b87c689bff5e175616dde0e
