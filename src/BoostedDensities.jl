# __precompile__()

module BoostedDensities

using Flux, Distributions, Mamba, Cubature, ProgressMeter
using Base.Iterators: repeated, partition
using Flux: onehotbatch, argmax, crossentropy, throttle
using Flux.Tracker

include("qdensity.jl" )
include("sampling.jl")
include("gaussian_mixture.jl")
include("metrics.jl")
include("run_experiment.jl")

export QDensity, GaussianMixture,
    push!, normalise!, grad, logpdf_gradlogpdf, 
    nll, kl, integrate_pdf, coverage, expected_log_likelihood,
    allocate_train_valid, initialise,
    mu, boosted_alpha, mean_boosted_alpha, run_experiment
end
