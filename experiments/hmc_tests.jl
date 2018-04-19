@everywhere include("hmc.jl")

@everywhere begin 
    using Distributions, StatsBase
    using Flux.Data.MNIST

    images  = MNIST.images(:test)[1:1_000]
    labels  = MNIST.labels(:test)[1:1_000]

    cat_images    = cat(3, images...)
    sz_x, sz_y, n = size(cat_images)
    _images       = reshape(Float16.(cat_images), (sz_x * sz_y, n))

    μ = mean(_images, 2)[:,1]
    Σ = cov(_images, 2)
    dim = sz_x * sz_y

    p = Distributions.MvNormal(μ, Σ)
    logfgrad = x -> (Distributions.logpdf(p, x), Distributions.gradlogpdf(p,x))
end

@everywhere begin 
    num_chains = nworkers()
    starting_indices   = randperm(size(_images,2))[1:num_chains]
    starting_locations = _images[:, starting_indices]

    hmcs = mapslices(starting_locations, 1) do μ
        HMCSampler(logfgrad, μ, ϵ = 0.015, t = 1/2) 
    end |> vec
end

res = @time hmc_sample(hmcs, 5_000, burnin = 150, verbose = 5)

μ̂ = mean(z, 2)

Gray.(tanh.(reshape(μ̂, (28,28))))