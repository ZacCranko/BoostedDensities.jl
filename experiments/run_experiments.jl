experiments = (
    # :kde_comparison_cross_validate,
    :dimensionality_experiment,
    # :architecture_comparison,
    # :activation_comparison,
)
addprocs()
@everywhere using Revise
include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "experiments.jl"))
include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "dimensionality.jl"))
runall(eval.(experiments))


experiments = (
    :adagan_comparison,
    # :kde_comparison_cross_validate,
    # :dimensionality_experiment,
    # :architecture_comparison,
    # :activation_comparison,
)
# addprocs()
@everywhere using Revise
include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "experiments.jl"))
include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "dimensionality.jl"))
runall(eval.(experiments))

