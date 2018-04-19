mutable struct GaussianMixture <: ContinuousMultivariateDistribution
    cmp::Vector{AbstractMvNormal}
    w::Vector{Float64}
end
GaussianMixture(cmp::Vector{<:AbstractMvNormal}) = GaussianMixture(cmp, fill(inv(length(cmp)), length(cmp)))

function GaussianMixture(;n::Int = 8, r::Real = 4, σ::Real = 1/2)
    thetas  = linspace(0, 2pi, n + 1)[1:end-1]
    Σ       = σ * eye(2)
    cmp = [MvNormal(r*[cos(θ), sin(θ)], Σ) for θ in thetas]
    return GaussianMixture(cmp)
end

Base.length(gm::GaussianMixture) = length(gm.cmp |> first)

Base.rand(gm::GaussianMixture, n::Int) = rand!(gm, Array{Float64}(2, n))

function Base.rand!(gm::GaussianMixture, samples::AbstractArray{<:Real, 2})
    n = size(samples, 2)
    modes   = rand(Categorical(gm.w), n)
    @inbounds for (i, m) in enumerate(modes)
        samples[:, i] = rand(gm.cmp[m])
    end
    return samples
end

Base.rand(gm::GaussianMixture) = rand(gm.cmp[rand(Categorical(gm.w))])
Base.mean(p::GaussianMixture) = hcat((mean(c) for c in p.cmp)...)

Distributions.pdf(gm::GaussianMixture, x::Vector) = sum(w .* pdf(p, x) for (p,w) in zip(gm.cmp,gm.w))
Distributions.pdf(gm::GaussianMixture, x::Matrix) = sum(w .* pdf(p, x) for (p,w) in zip(gm.cmp,gm.w), dim = 2)
Distributions.logpdf(gm::GaussianMixture, x::Matrix) = log.(pdf(gm, x))
Distributions.logpdf(gm::GaussianMixture, x::Vector) = log(pdf(gm, x))

