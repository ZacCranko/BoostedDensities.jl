experiments = (
    :kde_comparison,
    # :architecture_comparison,
    # :activation_comparison,
)
addprocs(length(experiments))
@everywhere using Revise
include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "experiments.jl"))
runall(eval.(experiments))