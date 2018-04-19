

function Base.rand(q::QDensity, n::Int; starts = 20, burnin  = 1_000)
    if length(q.models) == 0 return rand(q.q0, n) end
    samples = Array{Float64,2}(2, n)
    prog    = Progress(n, 1, "Sampling Q density: ")
    batch   = div(n, starts)
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
            next!(prog)
        end
    end
    return samples
end

function Base.rand!(q::QDensity, samples::AbstractArray; starts = 20, burnin  = 1_000)
    if length(q.models) == 0 return rand(q.q0, n) end
    n = size(samples, 2)
    samples = Array{Float64,2}(2, n)
    prog    = Progress(n, 1, "Sampling Q density: ")
    batch   = div(n, starts)
    # vanilla Metropolis-Hastings sampler with random restart and burnin
    for c in Base.OneTo(starts) 
        mcmc_sampler = Mamba.RWMVariate(rand(q.q0), diag(cov(q.q0)), x->Distributions.logpdf(q, x))
        
        # warm up sampler
        for _ in Base.OneTo(burnin)
            sample!(mcmc_sampler)
        end
        
        # sample
        for i in ((c - 1)*batch + 1):(c*batch)
            samples[:, i] = sample!(mcmc_sampler)
            next!(prog)
        end
    end
    return samples
end
