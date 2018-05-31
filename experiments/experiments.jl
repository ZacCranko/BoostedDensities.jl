# include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "experiments.jl"))
# Revise.track(@__FILE__)
@everywhere begin
    using BoostedDensities, Distributions, Flux, JLD2, NPZ, LightKDE
    global VERBOSE = 2      # display how we're doing
    global RUNS    = 20     # number of times to repeat each experiment, used where appropriate
    global SEED    = 1337   # some seed number to initialise some things
end

@everywhere architecture_comparison() = begin
    iter = 10 # number of boosting iterations

    srand(SEED)
    d = 2
    
    architectures = ((5,1), (5,5,1), (10,1), (10,10,1))
    models = [ # classifier architectures
        Chain(Dense(d, 5,  Flux.selu), Dense(5, 1)), 
        Chain(Dense(d, 5,  Flux.selu), Dense(5, 5, Flux.selu), Dense(5, 1)),
        Chain(Dense(d, 10, Flux.selu), Dense(10, 1)),
        Chain(Dense(d, 10, Flux.selu), Dense(10, 10, Flux.selu), Dense(10, 1)),
        Chain(Dense(d, 20, Flux.relu), Dense(20, 20, Flux.relu), Dense(20, 1))
    ]

    results = Array{Any}(3, length(models), RUNS)
    for i in Base.OneTo(RUNS)
        VERBOSE == 1 && info("worker $(myid()): activation comparison: $i of $RUNS")
        srand(SEED + i)
        
        p  = GaussianMixture(σ = 1/2)
        q₀ = MvNormal(2, 1)
        num_p, num_q = 1_000, 1_000

        for (j,m) in enumerate(models)
            VERBOSE >= 2 && info("worker $(myid()): architecture comparison: $i of $RUNS")
            VERBOSE >= 2 && info("worker $(myid()): $m")
            q, train_history, boosting_history = run_experiment(p, q₀, num_p, num_q, iter = iter, alpha = t->0.5, num_epochs = 3_000, batch_size = 50, model = m,
                                                                run_boosting_metrics = true, verbose = VERBOSE >= 3, optimiser = ADAM, optimise_alpha = true, cubature = true)
            results[:,j,i] = [q, train_history, boosting_history]
        end
    end

    conditions = string.(models)
    return results, conditions
end

@everywhere activation_comparison() = begin
    activations = (relu, selu, softplus, sigmoid, tanh)
    iter = 10 # number of boosting iterations
    
    results = Array{Any}(3, length(activations), RUNS)
    for i in Base.OneTo(RUNS)
        VERBOSE == 1 && info("worker $(myid()): activation comparison: $i of $RUNS")
        srand(SEED + i)
        
        p  = GaussianMixture(σ = 1/2)
        q₀ = MvNormal(2, 1)
        num_p, num_q = 1_000, 1_000

        for (j,act) in enumerate(activations)
            VERBOSE >= 2 && info("worker $(myid()): activation comparison: $i of $RUNS: $act")
            m = Chain(Dense(2, 5, act), Dense(5, 5, act), Dense(5, 1)) # classifier architecture
            q, train_history, boosting_history = run_experiment(p, q₀, num_p, num_q, iter = iter, alpha = t->0.5, num_epochs = 3_000, batch_size = 50, model = m,
                                                                run_boosting_metrics = true, verbose = VERBOSE >= 3, optimiser = ADAM, cubature = true)
            results[:,j,i] = [q, train_history, boosting_history]
        end
    end
        
    conditions = string.(activations)
    return results, conditions
end

error_plots() = begin
    activations = (relu, selu, softplus, sigmoid, tanh)
    iter = 6 # number of boosting iterations
    
    results = Array{Any}(3, 1, RUNS)
    for i in Base.OneTo(RUNS)
        VERBOSE == 1 && info("worker $(myid()): error plot: $i of $RUNS")
        srand(SEED + i)
        
        p  = GaussianMixture(σ = 1/2)
        q₀ = MvNormal(2, 1)
        num_p, num_q = 1000, 1000

        VERBOSE >= 2 && info("worker $(myid()): error plot: $i of $RUNS: relu")
        m = Chain(Dense(2, 5, relu), Dense(5, 5, relu), Dense(5, 1)) # classifier architecture
        q, train_history, boosting_history = run_experiment(p, q₀, num_p, num_q, iter = iter, alpha = t->0.5, num_epochs = 400, model = m, early_stop = 3, run_boosting_metrics = true, verbose = VERBOSE >= 3, optimiser = ADAM)
        results[:,1,i] = [q, train_history, boosting_history]
    end
        
    conditions = string.(activations)
    return results, ["error_plot"]
end

adagan_comparison() = begin
    results_dir    = joinpath(Pkg.dir("BoostedDensities"), "adagan_results")
    adagan_results = sort(readdir(results_dir)) # make sure sorted so line up samples and parameters

    process_adagan = function (data) 
        reshape(permutedims(data, (4,3,2,1)), (2, div(length(data),2))) 
    end 

    adagan_p_distributions = map(f for f in adagan_results if ismatch(r"real_data_params_mean_[0-9]{2}_var_[0-9].[0-9]{2}.npy", f)) do f
        means = npzread(joinpath(results_dir, f))
        var   = match(r".*_var_([0-9].[0-9]{2}).npy", f).captures |> first |> parse
        GaussianMixture(vec(mapslices(means, 2) do mu; MvNormal(mu, sqrt(var)) end))
    end    

    adagan_p_samps = map(f for f in adagan_results if ismatch(r"real_data_[0-9]{2}.npy", f)) do x
        data = npzread(joinpath(results_dir, x))
        process_adagan(data)
    end

    @assert length(adagan_p_samps) == length(adagan_p_distributions)

    d = 2

    iter = 10
    results = Array{Any}(4, 1, length(adagan_p_samps))
    for i in Base.OneTo(length(adagan_p_samps))
        VERBOSE == 1 && info("worker $(myid()): adagan comparison: $i of $RUNS")
        srand(SEED + i)

        p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps = allocate_train_valid(2, 5_000, 5_000; train_fraction = 3/4)
        p_samps[:] = adagan_p_samps[i][:, randperm(64_000)[1:5_000]]
        Σ  = cov(adagan_p_samps[i], 2)
        μ  = mean(adagan_p_samps[i], 2) |> vec
        q₀ = MvNormal(μ, Σ)
        p = adagan_p_distributions[i]

        true_nll = -mean(logpdf(p, p_samps))

        q, train_history, boosting_history = run_experiment(p, q₀, p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps,
                                                            model = Chain(Dense(d, 10, Flux.relu), Dense(10, 10, Flux.relu),  Dense(10, 1)), num_epochs = 2_000, 
                                                            iter = iter,
                                                            boosting_metrics = (nll, coverage),
                                                            run_boosting_metrics = true, verbose = VERBOSE >= 3, early_stop = 0.03, optimiser = ADAM, optimise_alpha = true)
        results[:,1,i] = [q, train_history, boosting_history, true_nll]
    end
        
    return results, ["adagan_comparison"]
end

@everywhere kde_comparison() = begin
    d = 2
    conditions = ["deep network"]
    for k in LightKDE.supported_kernels
        push!(conditions, "scott/silverman: $k")
    end

    m = length(conditions)
    results = Array{Any}(3, m, RUNS)
    for i in Base.OneTo(RUNS)
        VERBOSE >= 1 && info("worker $(myid()): kernel density comparison: $i of $RUNS")
        srand(SEED + i)

        num_p, num_q = 1_000, 1_000
        means    = [rand(MvNormal(d, 4)) for _ in 1:8]
        sigmas   = fill(1/2, 8)
        rp       = GaussianMixture(map(MvNormal, means, sigmas)) # randomly arranged Gaussians
        p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps = allocate_train_valid(dim(rp), num_p, num_q; train_fraction = 3/4)
        
        p_samps = rand!(rp, p_samps)
        Σ  = cov(p_samps, 2)
        μ  = mean(p_samps, 2) |> vec
        q₀ = MvNormal(μ, Σ)

        VERBOSE >= 2 && info("worker $(myid()): kernel density comparison: $i of $RUNS: deep density")
        q, train_history, boosting_history = run_experiment(rp, q₀, p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps,
                                                            iter = 2, num_epochs = 3_000, batch_size = 50,
                                                            model = Chain(Dense(d, 10, Flux.relu), Dense(10, 10, Flux.relu),  Dense(10, 1)), 
                                                            boosting_metrics = (coverage, nll),
                                                            run_boosting_metrics = true, verbose = VERBOSE >= 3, optimiser = ADAM, optimise_alpha = true)
        results[:,1,i] = [q, train_history, boosting_history]
        
        VERBOSE == 2 && info("worker $(myid()): kernel density comparison: $i of $RUNS: scott/silverman")

        for (j, kernel) in enumerate(LightKDE.supported_kernels)
            VERBOSE >= 3 && info("worker $(myid()): kernel density comparison: $i of $RUNS: scott/silverman: $kernel")
            k = kde(p_samps, cross_validate = false, kernel = kernel)
            # pk_kl  = kl(rp, k)
            pk_nll = -mean(logpdf(k, p_samps))
            VERBOSE >= 3 && info("worker $(myid()): nll $pk_nll")
            results[:,1+j,i] = [nothing, nothing, Dict(nll => pk_nll)]
        end
        # VERBOSE >= 2 && info("worker $(myid()): kernel density comparison: $i of $RUNS: cross_validated")
        # k = kde(p_samps, cross_validate = true, δ = 1.0, folds = 10)
        # # pk_kl  = kl(rp, k)
        # pk_nll = -mean(logpdf(k, p_samps))
        # VERBOSE >= 3 && info("worker $(myid()): nll $pk_nll")
        # results[:,3,i] = [nothing, nothing, Dict(nll => pk_nll)]
    end
        
    return results, conditions
end



function runall(experiments)
    pmap(experiments) do exprmt
        exprmnt_results = try
            exprmt() # run experiment
        catch error
            warn(error)
            error
        end
        filename = joinpath(Pkg.dir("BoostedDensities"), "results", string(exprmt, "-", Base.Dates.now(), ".jld2")) |> x -> replace(x, ":", "_")
        JLD2.@save filename exprmnt_results
        return exprmnt_results
    end
end

runall(args...) = runall((args...))

testexp() = Array[], String[]
errorexp() = error("whoops")