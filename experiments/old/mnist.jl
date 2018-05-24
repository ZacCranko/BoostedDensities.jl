using Revise
using Distributions, JLD2, ProgressMeter
using Flux, Flux.Data.MNIST
# end

toimage(z::Vector) = toimage(reshape(z, (28,28)))
toimage(z::Matrix) = map(z) do x
   max(min(x, 1), 0) |> ColorTypes.Gray
end

train_inputs = Array{Float64}(28,28,1,60_000)
test_inputs  = Array{Float64}(28,28,1,10_000)
mnist_test  = MNIST.images(:test)
mnist_train = MNIST.images(:train)
@inbounds for i in 1:60_000
    train_inputs[:,:,:,i] = mnist_train[i]
end
@inbounds for i in 1:10_000
    test_inputs[:,:,:,i] = mnist_test[i]
end

train_labels = Flux.onehotbatch(MNIST.labels(:train), 0:9)
test_labels  = Flux.onehotbatch(MNIST.labels(:test),  0:9)

epochs(inputs, n::Int) = (d for _ in Base.OneTo(n) for d in inputs)

function batches(inputs::Tuple, batch_size::Int)
    @show n = last(size(first(inputs)))
    
    # pull off last dimension of each Array in 'inputs' 
    last_dim(inputs, batch) = map(inp->getindex(inp, ntuple((_)->(:), ndims(inp) - 1)..., batch), inputs) 
    
    return (last_dim(inputs, batch) for batch in Iterators.partition(Base.OneTo(n), batch_size))
end


# mnist feature model ----------------------------------------------------------------------------------------------------
mnist_classifier = Chain(
    Conv2D((2,2), 1=>32,  relu), x -> maxpool2d(x, 2), 
    Conv2D((2,2), 32=>16, relu), x -> maxpool2d(x, 2),  
    x -> reshape(x, :, size(x, 4)), Dense(576, 100, relu), Dense(100, 10), softmax)

loss(x, y)     = Flux.crossentropy(mnist_classifier(x), y)
accuracy(x, y) = mean(Flux.argmax(mnist_classifier(x)) .== Flux.argmax(y))

cb_metric = function () 
    inds = rand(1:10_000, 1_000)
    acc  = accuracy(test_inputs[:,:,:,inds], test_labels[:,inds])
    @show(acc)
    return nothing 
end

mnist_classifier_opt = ADAM(Flux.params(mnist_classifier))

# pre training
batch_size = 50
train_batched  = batches((train_inputs, train_labels), batch_size)
Flux.train!(loss, epochs(train_batched, 1), mnist_classifier_opt, cb = Flux.throttle(cb_metric, 5))

mnist_features = mnist_classifier[1:end-2]


# density model ----------------------------------------------------------------------------------------------------
mutable struct FeatQDensity <: ContinuousMultivariateDistribution
    q0::ContinuousMultivariateDistribution
    feat_model::Flux.Chain
    models::Vector{Flux.Chain}
    alphas::Vector{Float64}
    logz::Vector{Float64}
end

function Distributions.logpdf(q::FeatQDensity, x::Matrix)
    dens = logpdf(q.q0, x)
    features = q.feat_model(reshape(x, (28,28,1,size(x,2))))
    for (m, a) in zip(q.models, q.alphas)
        dens .+= a * vec(Flux.Tracker.data(m(features)))
    end
    return dens
end

Distributions.logpdf(q::FeatQDensity, x::Vector) = Distributions.logpdf(q, reshape(x, (length(x), 1)))

function Distributions.logpdf(q::FeatQDensity, x::AbstractArray)
    sz = size(x)
    n  = last(sz)
    newdim = (div(prod(sz),n), n)
    Distributions._logpdf(q, reshape(x, newdim))
end

Distributions.pdf(q::FeatQDensity, x) = exp.(Distributions.logpdf(q, x))

logpdf_gradlogpdf(q::Distribution, x) = (logpdf(q, x), gradlogpdf(q, x))

function logpdf_gradlogpdf(q::FeatQDensity, x::Vector; sz = (28,28,1,1))
    f, _grad = logpdf_gradlogpdf(q.q0, x)
    g      = param(reshape(x, sz))
    g.grad = reshape(_grad, sz)
    features = q.feat_model(reshape(x, sz))
    for (m, α) in zip(q.models, q.alphas)
        f += (y = α * m(features))
        Flux.Tracker.back!(y, 1)
    end
    return Flux.Tracker.data(f)[], vec(g.grad)
end

num = 10_000
vec_inputs = reshape(@view train_inputs[:,:,:,1:num], (28*28*1, num))
μ  = mean(vec_inputs, 2)[:,1]
Σ  = cov(vec_inputs, 2)
q0 = Distributions.MvNormal(μ, Σ + 1e-3I)

q_samps = reshape(rand(q0, num), (28,28,1,num))
p_samps = train_inputs

# logfgrad = x -> (Distributions.logpdf(p, x), Distributions.gradlogpdf(p,x))


density_model     = Chain(Dense(100, 100, relu), Dense(100, 100, relu), Dense(100, 1))
density_model_opt = ADAM(Flux.params(density_model))
function density_obj(p_samps, q_samps) 
    np, nq = size(p_samps, 4), size(q_samps, 4)
    n = np + nq
    p_features, q_features = mnist_features(p_samps), mnist_features(q_samps)

    sp = - sum(  log.(+σ.(density_model(p_features)))) 
    sq = - sum(log1p.(-σ.(density_model(q_features))))

    return sp/np + sq/nq
end

cb_metric = function () 
    inds = randperm(10_000)[1:500]
    cross_ent  = density_obj(p_samps[:,:,:,inds], q_samps[:,:,:,inds])
    @show(cross_ent)
    return nothing 
end

# training
batch_size = 50
train_batched  = batches((p_samps, q_samps), batch_size)
Flux.train!(density_obj, epochs(train_batched, 1), density_model_opt, cb = Flux.throttle(cb_metric, 10))

batch_size = 100
train_batched  = batches((p_samps, q_samps), batch_size)
Flux.train!(density_obj, epochs(train_batched, 4), density_model_opt, cb = Flux.throttle(cb_metric, 10))



fqd = FeatQDensity(q0, mnist_features, Flux.Chain[density_model], Float64[0.5], Float64[])

x = train_inputs[:,:,:,1:1]



_, g = logpdf_gradlogpdf(q0, vec(x) + 1e-3logpdf_gradlogpdf(q0, vec(x))[2])

logpdf(fqd, x[:] + g)


include("/Users/zcranko/Documents/GitHub/Boosting/experiments/BoostedDensities/src/hmc.jl")

hmc = HMCSampler(x->logpdf_gradlogpdf(fqd, x), vec(x), ϵ = 1e-3, l = 100)

hmc_worker!(hmc, 100, verbose = 10)


# jobs = [(i, hmc, n, burnin, thin) for i in Base.OneTo(4)]
sims = pmap(hmc_worker, jobs)

@load "~/samples.jld2"

toimage(samples[:,60])
