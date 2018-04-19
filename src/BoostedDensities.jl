# __precompile__()

module BoostedDensities

using Flux, Distributions, Mamba, Cubature, ProgressMeter
using Base.Iterators: repeated, partition
using Flux: onehotbatch, argmax, crossentropy, throttle
using Flux.Tracker

for f in ("qdensity", 
          "sampling", 
          "gaussian_mixture", 
          "metrics", 
          "run_experiment")
    include("$f.jl")
end

export QDensity, GaussianMixture,
    push!, normalise!, grad, logpdf_gradlogpdf, 
    kl, integrate_pdf, coverage, expected_log_likelihood,
    allocate_train_valid, initialise,
    mu, boosted_alpha, mean_boosted_alpha, run_experiment
end