function mu(p_samps, q_samps, m::Flux.Chain, cstar)
    c_p =  m(p_samps) |> Flux.Tracker.data |> vec
    c_q = -m(q_samps) |> Flux.Tracker.data |> vec
    cstar = max(cstar, maximum(abs, c_p), maximum(abs, c_q))

    mu_p =  mean(c_p)
    mu_q =  mean(c_q) 
    α    = min(1, (log1p(mu_q/cstar) - log1p(-mu_q/cstar))/(2cstar))
    return mu_p, mu_q, α, cstar
end

epochs(inputs, n::Int) = (d for _ in Base.OneTo(n) for d in inputs)

function batches(inputs, batch_size::Int)
    last_dim(arr, ind) = view(arr, ntuple((_)->(:), ndims(arr) - 1)..., ind)
    
    n = last(size(first(inputs)))
    return (map(x->last_dim(x, batch), inputs) for batch in Iterators.partition(Base.OneTo(n), batch_size))
end

@views function allocate_train_valid(dim, num_p, num_q; train_fraction = 3/4)
    p_samps = Array{Float32}(dim..., num_p)
    q_samps = Array{Float32}(dim..., num_q)

    train_p_inputs = p_samps[:, Base.OneTo(Int(num_p * train_fraction))]
    train_q_inputs = q_samps[:, Base.OneTo(Int(num_q * train_fraction))]

    test_p_inputs  = p_samps[:, (Int(num_p * train_fraction) + 1):end]
    test_q_inputs  = q_samps[:, (Int(num_q * train_fraction) + 1):end]

    return p_samps, q_samps, train_p_inputs, train_q_inputs, test_p_inputs, test_q_inputs
end


function run_experiment( p::Distribution,
                        q0::Distribution,
                        num_p::Int = 10_000,
                        num_q::Int = 10_000;
                        batch_size = div(num_p, 4),
                        iter::Int  = 5,
                        boosting_metrics = (nll, kl),
                        alpha::Function   = i-> 1/2,
                        early_stop::Real  = Inf,  # when test error is `early_stop` below train error, stop early
                        weak_boost::Real  = Inf,  # when we have achieved a weak learner, stop early (set this to a small positive number to enable)
                        verbose::Bool              = false,
                        run_boosting_metrics::Bool = false, # time consuming to have these on since we will renormalise Q every iteration
                        optimiser::Function = ADAM, 
                        num_epochs::Int     = 200,
                        seed::Int           = 1337,
                        model::Flux.Chain   = Chain(Dense(2, 10, softplus), Dense(10, 10, softplus),  Dense(10, 2), softmax),
                        folds = 10,
                        optimise_alpha = false)
    
    p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps = allocate_train_valid(dim(p), num_p, num_q; train_fraction = 3/4)
    rand!(p, p_samps)
    return   run_experiment(p, q0, p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps; 
                            batch_size = batch_size,
                            iter = iter, 
                            boosting_metrics = boosting_metrics,
                            alpha = alpha,
                            early_stop = early_stop,
                            verbose = verbose,
                            run_boosting_metrics = run_boosting_metrics,
                            optimiser = optimiser, 
                            num_epochs = num_epochs,
                            seed = seed,
                            model = model,
                            folds = folds,
                            optimise_alpha = optimise_alpha)
end



function run_experiment(p::Distribution, q0::Distribution, p_samps, q_samps, train_p_samps, train_q_samps, test_p_samps, test_q_samps;   
                        batch_size = div(size(q_samps, 2), 4),
                        iter::Int  = 5,
                        boosting_metrics = (nll, kl),
                        alpha::Function   = i-> 1/2,
                        early_stop::Real  = Inf,  # when test error is `early_stop` below train error, stop early
                        verbose::Bool              = false,
                        run_boosting_metrics::Bool = false, # time consuming to have these on since we will renormalise Q every iteration
                        optimiser::Function = ADAM, 
                        num_epochs::Int     = 200,
                        seed::Int           = 1337,
                        model::Flux.Chain   = Chain(Dense(2, 10, softplus), Dense(10, 10, softplus),  Dense(10, 2), softmax),
                        folds = 11,
                        optimise_alpha = false)
    q = QDensity(q0)
    true_nll         =  nll(p, p_samps)
    verbose && info("True NLL: $true_nll")
    train_history    = Dict{Symbol,  Float64}[]
    boosting_history = Dict{Any,     Float64}[]

    if run_boosting_metrics
        push!(boosting_history, Dict(met => met(p, q, p_samps, rand(q, size(q_samps, 2))) for met in boosting_metrics))
        boosting_history[end][:true_nll] = true_nll
        if verbose
            for metric in boosting_metrics
                info("$metric: ", boosting_history[end][metric])
            end
        end
    end

    local α     = 1.0
    local cstar = 1
    local train_progress

    for i in Base.OneTo(iter)
        if verbose
            train_progress = Progress(2*num_epochs)
        end
        rand!(q, q_samps)
        m          = deepcopy(model)
        opt        = m |> Flux.params |> optimiser
        obj(p_samps, q_samps) = - mean(log.(σ.(m(p_samps)))) - mean(log1p.(-σ.(m(q_samps))))
        ce_acc(p_samps, q_samps) = begin
            p_predicts, q_predicts = m(p_samps), m(q_samps)
            ce  = (- mean(log.(σ.(p_predicts))) - mean(log1p.(-σ.(q_predicts))) |> Flux.Tracker.data)[]
            acc = (mean(vcat(p_predicts .>= 0, q_predicts .< 0)) |> Flux.Tracker.data)[]
            return ce, acc
        end

        cbs = 0

        evalcb() = begin
            cbs += 1
            train_ce, train_acc = ce_acc(train_p_samps, train_q_samps)
            test_ce, test_acc   = ce_acc(test_p_samps, test_q_samps)

            early_stop_criterion =  max(0, (test_ce - train_ce)/test_ce)

            if verbose 
                ProgressMeter.next!(train_progress, showvalues = [
                                                                  ("callbacks", cbs),
                                                                  ("cross entropy (train)", train_ce), 
                                                                  ("cross entropy (test)",  test_ce), 
                                                                  ("accuracy (train)",      train_acc),
                                                                  ("accuracy (test)",       test_acc),
                                                                #   ("early stop threshold",  "$(trunc(Int, early_stop * 100))%"),
                                                                #   ("early stop criterion",  "$(trunc(Int, early_stop_criterion * 100))%")
                                                                  ])
            end
            push!(train_history, Dict(
                :iter       => i,
                :train_ce   => train_ce,
                :train_acc  => train_acc,
                :test_ce    => test_ce,
                :test_acc   => test_acc,
            ))
     
            if early_stop_criterion > early_stop && cbs > 100
                return :stop
            end
        end

        # Flux.train!(obj, epochs(batches((train_p_samps, train_q_samps), batch_size), num_epochs), opt, cb = throttle(evalcb, 1))
        Flux.train!(obj, epochs(batches((train_p_samps, train_q_samps), batch_size), num_epochs), opt, cb = evalcb)
        verbose && println()

        α = alpha(i) 
        push!(q, (m, α))
        
        if optimise_alpha 
            verbose  && info("optimising alpha")
            α = optimise_alpha!(p, q, Array(p_samps),  Array(q_samps), folds = folds)
        end
        if verbose  
            info("alpha: $α")
        end

        if run_boosting_metrics
            push!(boosting_history, Dict(metric => metric(p, q,  p_samps, q_samps) for metric in boosting_metrics))
            boosting_history[end][:true_nll] = true_nll
            if verbose
                for metric in boosting_metrics
                    info("$metric: ", boosting_history[end][metric])
                end
            end
        end
    end

    return q, train_history, boosting_history
end


function optimise_alpha!(p::Distribution, q::QDensity, p_samps::Array, q_samps::Array; folds = 10)
    neg_log_likelihood = Vector{Float64}(folds)
    alphas             = linspace(0.0, 1.0, folds)
    logz_prev          = q.logz
    predicts           = vec(Flux.Tracker.data(q.models[end](q_samps)))
    for (i, a) in enumerate(alphas)
        q.normalised  = false
        q.alphas[end] = a
        q.logz        = logz_prev
        normalise!(q, q_samps, predicts = predicts)
        neg_log_likelihood[i] = nll(q, p_samps)
    end
    i = findmin(neg_log_likelihood)[2]
    q.alphas[end] = alphas[i]
    q.logz        = logz_prev
    q.normalised  = false
    normalise!(q, q_samps, predicts = predicts)
    return alphas[i]
end