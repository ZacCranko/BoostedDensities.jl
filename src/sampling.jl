

function Base.rand(q::QDensity, n::Int; starts = 15, burnin = 2_000)
    if length(q.models) == 0 return rand(q.q0, n) end
    samples = Array{Float64,2}(2, n)
    return rand!(q, samples, starts = starts, burnin = burnin)
end

function Base.rand!(q::QDensity, samples::AbstractMatrix; starts = 20, burnin  = 1_000)
    if length(q.models) == 0 return rand!(q.q0, samples) end
    n = size(samples, 2)
    batch = div(n, starts)
    # vanilla Metropolis-Hastings sampler with random restart and burnin
    for c in Base.OneTo(starts) 
        mcmc_sampler = Mamba.RWMVariate(rand(q.q0), diag(cov(q.q0)), x->Distributions._logpdf(q, x))
        
        # warm up sampler
        for _ in Base.OneTo(burnin)
            sample!(mcmc_sampler)
        end
        
        # sample
        for i in ((c - 1)*batch + 1):(c*batch)
            samples[:, i] = sample!(mcmc_sampler)
        end
    end
    return samples
end

