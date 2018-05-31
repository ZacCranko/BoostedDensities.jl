using BoostedDensities, Distributions, MLDataUtils

struct KDE <: ContinuousMultivariateDistribution
    kernel::Distribution
    train_data::AbstractArray{Float64, 2}
    bandwidth::Float64
end

Distributions.pdf(k::KDE,    data::AbstractVector) = mean(pdf(k.kernel, data .- k.train_data))
Distributions.pdf(k::KDE,    data::AbstractMatrix) = vec(mapslices(x->pdf(k, x), data, 1))
Distributions.logpdf(k::KDE, data::AbstractMatrix) = log.(pdf(k, data))

scott(d, n)::Float64 = n^(-1/(d+4))
silverman(d, n)::Float64 = (n * (d + 2) / 4)^(-1 / (d + 4))

function kde(train_data; bandwidth = scott(2, size(train_data,2)), cross_validate = true, δ = 1.0, folds = 20)
    dim, n = size(train_data)
    
    if cross_validate
        results    = zeros(Float64, folds)
        bandwidths = linspace(max(eps(), bandwidth - δ), bandwidth + δ, folds)
        @views @inbounds for (i, b) in enumerate(bandwidths)
            for (train_idx, valid_idx) in kfolds(n, folds)
                train_kde = KDE(MvNormal(dim, b), train_data[:, train_idx], b)
                results[i] += -mean(logpdf(train_kde, train_data[:, valid_idx]))
            end
        end
        _, i = findmin(results)
        return KDE(MvNormal(dim,  bandwidths[i]), train_data, bandwidths[i])
    else
        return KDE(MvNormal(dim,  bandwidth), train_data, bandwidth)
    end
end

BoostedDensities.isnormalised(::KDE) = true
BoostedDensities.dim(k::KDE) = length(k.kernel)