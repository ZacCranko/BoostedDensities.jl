function mu(samps, m::Flux.Chain)
    mu = m(samps) |> Flux.Tracker.data |> vec
    mx = maximum(abs, mu)
    return mean(mu)/mx
end

function boosted_alpha(q_samps, m::Flux.Chain)
    msamps = -m(q_samps) |> Flux.Tracker.data |> vec
    cstar  = maximum(abs, msamps)
    μ      = mean(msamps)/cstar
    α      = (log1p(μ) - log1p(-μ))/(2cstar)
    return min(1, α)
end

mean_boosted_alpha(q_samps, m::Flux.Chain) = 1/2 + 1/2*boosted_alpha(q_samps, m)

epochs(inputs, n::Int) = (d for _ in Base.OneTo(n) for d in inputs)

function batches(inputs, batch_size::Int)
    last_dim(arr, ind) = view(arr, ntuple((_)->(:), ndims(arr) - 1)..., ind)
    
    n = last(size(first(inputs)))
    return (map(x->last_dim(x, batch), inputs) for batch in Iterators.partition(Base.OneTo(n), batch_size))
end

@views function allocate_train_valid(dim, num_p, num_q; train_fraction = 3/4)
    p_samps = Array{Float32, length(dim) + 1}(dim..., num_p)
    q_samps = Array{Float32, length(dim) + 1}(dim..., num_q)

    train_p_inputs = p_samps[:, Base.OneTo(Int(num_p * train_fraction))]
    train_q_inputs = q_samps[:, Base.OneTo(Int(num_q * train_fraction))]

    test_p_inputs  = p_samps[:, (Int(num_p * train_fraction) + 1):end]
    test_q_inputs  = q_samps[:, (Int(num_q * train_fraction) + 1):end]

    return p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs
end

function allocate_train_valid(_p_samps::Array{T,N}, _q_samps::Array{T,N}; train_fraction = 3/4) where {T,N}
    num_p, num_q = map(last, (size(p_samps), size(q_samps)))
    p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs = allocate_train_valid(N-1, num_p, num_q, train_fraction = train_fraction)
    p_samps[:] = _p_samps
    q_samps[:] = _q_samps
    
    return p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs
end

function run_experiment(p ::Distribution,
                        q0::Distribution,
                        num_p::Int = 10_000,
                        num_q::Int = 10_000;
                        batch_size = div(num_p + num_q, 4),
                        iter::Int  = 5,
                        early_stop::Real    = 0.1,  # when test error is `early_stop` below train error, stop early
                        weak_boost::Real    = Inf,  # when we have achieved a weak learner, stop early (set this to a small positive number to enable)
                        verbose::Bool              = false,
                        run_boosting_metrics::Bool = false, # time consuming to have these on since we will renormalise Q every iteration
                        alpha::Function     = mean_boosted_alpha,
                        optimiser::Function = p -> Nesterov(p, 0.01), 
                        sampler::Function   = rand,
                        num_epochs::Int     = 10,
                        seed::Int           = 1337,
                        model::Flux.Chain   = Chain(Dense(2, 10, softplus), Dense(10, 10, softplus),  Dense(10, 2), softmax))
    srand(seed) # set random seed
    q = QDensity(q0)
    p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps = 
        allocate_train_valid(2, num_p, num_q; train_fraction = 3/4)

    rand!(p, p_samps)
    train_history    = Vector{Any}[]
    boosting_history = Dict{Function,Float64}[]
    boosting_metrics = (coverage, expected_log_likelihood, kl)
    if run_boosting_metrics
        push!(boosting_history, Dict(met => met(p, q, q_samps, p_samps) for met in boosting_metrics))
    end

    for i in Base.OneTo(iter)
        if verbose
            info("Sampling Q")
        end
        q_samps[:] = sampler(q, num_q)
        m          = deepcopy(model)
        opt        = optimiser(Flux.params(m))
        obj(p_samps, q_samps) = -mean(log.(m(p_samps))) - mean(log1p.(-m(q_samps)))

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

            mu_p =  mu(p_samps,  m[1:end-1])
            mu_q = -mu(q_samps,  m[1:end-1])

            push!(train_history, [i, test_ce, mu_p, mu_q])
            if (train_ce + early_stop <= test_ce) || (min(mu_p, mu_q) >= weak_boost)
                println()
                info("Stopping early")
                return :stop
            end
        end

        Flux.train!(obj, epochs(batches((train_p_samps, train_q_samps), batch_size), num_epochs), opt, cb = Flux.throttle(evalcb, 2))
        push!(q, (m[1:end-1], α))

        if run_boosting_metrics
            if verbose
                info("Running the boosting metrics")
            end
            push!(boosting_history, Dict(met => met(p, q, q_samps, p_samps) for met in boosting_metrics))
        end
    end

    return q, (train_history, boosting_history)
end