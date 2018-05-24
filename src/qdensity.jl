mutable struct QDensity <: ContinuousMultivariateDistribution
    q0::ContinuousMultivariateDistribution
    models::Vector{Flux.Chain}
    alphas::Vector{Float64}
    logz::Float64
    normalised::Bool
end

QDensity(q0::ContinuousMultivariateDistribution) = QDensity(q0, Vector{Flux.Chain}(), Vector{Float64}(), zero(Float64), true)
QDensity(n::Int) = QDensity(MvNormal(n, 1))

Distributions.dim(q) = length(q.q0)

isuntrained(q::QDensity)  = (length(q.models) == 0)
isnormalised(q::QDensity) = q.normalised
isnormalised(q::Distribution) = true

function Base.show(io::IO, q::QDensity)
    @printf io "%s: \n"     typeof(q) 
    @printf io "t:  %d\n"   length(q.models)
    @printf io "zₜ: %s\n"    exp.(q.logz)
    @printf io "α:  %s\n"   q.alphas
    @printf io "Q₀ %s\n"    q.q0
end

Base.length(q::QDensity) = length(q.q0)

function normalise!(q::QDensity, rng = (fill(-30, dim(q)), fill(30, dim(q))), abstol = 1e-3) 
    z, _ = hcubature_v((x,v)-> begin v[:] = exp.(Distributions._logpdf(q, x)) end, rng..., abstol = abstol)
    q.logz = log(z)
    q.normalised = true
    return q
end

function normalise!(q::QDensity, q_samps::Matrix; predicts = vec(Flux.Tracker.data(q.models[end](q_samps)))) 
    @assert q.normalised == false
    q.logz += log(mean(x->exp(x)^q.alphas[end], predicts))
    q.normalised = true
    return q
end

function Base.getindex(q::QDensity, ind...)
    if isnormalised(q) 
        return QDensity(q.q0, q.models[ind...], q.alphas[ind...], q.logz[ind...])
    else
        return QDensity(q.q0, q.models[ind...], q.alphas[ind...], q.logz)
    end
end

Base.getindex(q::QDensity, i::Int) = q[Base.OneTo(i)]

function Base.push!(q::QDensity, update::Tuple{Flux.Chain, Float64})
    m, α = update
    push!(q.models, m)
    push!(q.alphas, α)
    q.normalised = false
    return q
end

function Distributions._logpdf(q::QDensity, x::Matrix)
    dens = logpdf(q.q0, x)
    for (m, a) in zip(q.models, q.alphas)
        dens .+= a * vec(Flux.Tracker.data(m(x)))
    end
    return dens
end
Distributions._logpdf(q::QDensity, x::Vector) = Distributions._logpdf(q, reshape(x, (dim(q),1)))[1]

Distributions.insupport(d::QDensity, x::AbstractVector{T}) where {T<:Real} = true
function Distributions.logpdf(q::QDensity, x::Matrix)
    if isuntrained(q)
        return logpdf(q.q0, x)
    elseif !isnormalised(q)
        warn("Q is not normalised")
        return Distributions._logpdf(q, x)
    end
    return Distributions._logpdf(q, x) .- q.logz
end
Distributions.logpdf(q::QDensity, x::Vector) = logpdf(q, reshape(dim(q), (2,1)))[1]
Distributions.pdf(q::QDensity, x::Matrix)    = exp.(logpdf(q, x))

logpdf_gradlogpdf(q::AbstractMvNormal, x) = (Distributions._logpdf(q, x),  q.Σ \ ( q.μ .- x))

function logpdf_gradlogpdf(q::QDensity, x::Vector; sz = size(x))
    g = param(reshape(x, sz))
    f, _grad = logpdf_gradlogpdf(q.q0, x)
    g.grad = reshape(_grad, sz)
    for (m, α) in zip(q.models, q.alphas)
        f += (y = α * m(g))
        Flux.Tracker.back!(y, 1)
    end
    return Flux.Tracker.data(f)[], vec(g.grad)
end

logpdf_gradlogpdf(q::Distribution, x::Matrix) = begin
    f, g = logpdf_gradlogpdf(q, vec(x), sz = size(x))
    return f, reshape(g, size(x))
end



