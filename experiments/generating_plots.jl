using Revise
using BoostedDensities, Distributions, Flux, HypothesisTests, JLD2, FileIO, NPZ
using Plots, StatPlots, HypothesisTests, Distributions; gr()
global RESULTSDIR  = joinpath("/mnt/capybara-ev/.julia/v0.6/BoostedDensities", "results")
include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "plotting_commands.jl"))

results = Dict() 
for experiment in [exprmt for exprmt in readdir(RESULTSDIR) if endswith(exprmt, "jld2")]
    info("Loading $experiment")
    results[experiment] = try
        if startswith(experiment, "kde")
            process_kde_experiment(experiment)
        else
            process_experiment(experiment)
        end
    catch e 
        warn(e)
        return nothing
    end
end

xlim = (1, 8)
ylim = (0, 1.5)
exp_name      = "training_error"
exp_condition = "accuracy"
plot_type     = "timeseries"
plt = timeseries_comparison(results["activation_comparison-2018-05-18T12_22_44.298.jld2"][:test_acc][2:2], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

ylim = (0, 8.0)
exp_name      = "training_error"
exp_condition = "kl"
plot_type     = "timeseries"
plt = timeseries_comparison(results["activation_comparison-2018-05-18T12_22_44.298.jld2"][:kl][2:2], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

ylim = (0.0, 4.0)
exp_name      = "training_error"
exp_condition = "nll"
plot_type     = "timeseries"
plt = timeseries_comparison(results["activation_comparison-2018-05-18T12_22_44.298.jld2"][:nll][2:2], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)


xlim = (1, 8.0)
ylim = (0, 7.0)
exp_name      = "activation_comparison"
exp_condition = "kl"
plot_type     = "timeseries"
plt = timeseries_comparison(results["activation_comparison-2018-05-18T12_22_44.298.jld2"][:kl], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

xlim = (0, 8)
ylim = (0.5, 3)
exp_name      = "activation_comparison"
exp_condition = "nll"
plot_type     = "timeseries"
plt = timeseries_comparison(results["activation_comparison-2018-05-18T12_22_44.298.jld2"][:nll], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

xlim = (1, 7)
ylim = (0, 0.8)
exp_name      = "architecture_comparison"
exp_condition = "kl"
plot_type     = "timeseries"
plt = timeseries_comparison(results["architecture_comparison-2018-05-11T07_43_17.39.jld2"][:kl], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

xlim = (1, 7)
ylim = (4, 6)
exp_name      = "architecture_comparison"
exp_condition = "nll"
plot_type     = "timeseries"
plt = timeseries_comparison(results["architecture_comparison-2018-05-11T07_43_17.39.jld2"][:nll], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon, coeff = -1)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
savefig(plt, dst)


exp_name = "kde_comparison"
# normalise by true_nll

r = results["kde_comparison-2018-05-29T11_27_10.658.jld2"]
for i in 1:length(r[:nll])
    r[:nll][i][2] ./= r[:true_nll][i]
end

# make extra condition out of 2nd iter
kde_results = results["kde_comparison-2018-05-29T11_27_10.658.jld2"][:nll][2:end]
for i in (1,2,3)
    push!(kde_results, (r[:nll][1][1], r[:nll][1][2][i:i,1:end-1], r[:nll][1][3] * " $i"))
end

# order by mean nll
sort!(kde_results, by = x -> abs(1- mean(x[2])), rev=true)

# print summary statistics in a LaTeX friendly way
foreach(kde_results) do x
    ind, res, cond = x
    @assert size(res, 1) == 1
    xbar, err = ci(res |> vec, 0.05)
    @printf "%s \& %.4f \\pm %.4f \\\\\n" cond xbar err
end

# violin plots
xlim = (0,9)
ylim = (0,2)
x = vcat((fill(string(ind, c), length(d)) for (ind, (i,d,c)) in enumerate(kde_results))...)
y = vcat((d[:] for (i,d,c) in kde_results)...)
# plt = plot(xlim=xlim, ylim=ylim)
plt = violin(x, y, ylim =ylim, xlim = xlim)
exp_condition = "nll"
plot_type     = "violin"
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
savefig(plt,dst)

plt = boxplot(x, y, ylim =ylim, xlim = xlim)
plot_type     = "box"
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
savefig(plt,dst)

xlim = (1,2)
ylim = (0,5)
timeseries_comparison(r[:nll][1:1], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
savefig(plt, dst)



ada_results_dir    = joinpath(RESULTSDIR, "..", "adagan_results")
adagan_results = sort(readdir(ada_results_dir)) # make sure sorted so line up samples and parameters
f = "adagan_comparison-2018-06-06T12_21_26.443.jld2"
adagan_p_distributions = map(f for f in adagan_results if ismatch(r"real_data_params_mean_[0-9]{2}_var_[0-9].[0-9]{6}.npy", f)) do f
    means = npzread(joinpath(ada_results_dir, f))
    var   = match(r".*_var_([0-9].[0-9]{6}).npy", f).captures |> first |> parse
    GaussianMixture(vec(mapslices(means, 2) do mu; MvNormal(mu, sqrt(var)) end))
end    

ada_true_nll = results[f][:true_nll][1][2][1:1,:]

for r in results[f][:nll]
    r[2] ./= ada_true_nll 
end

ada_nll = - npzread(joinpath(ada_results_dir, "likelihood.npy")) ./ ada_true_nll
push!(results[f][:nll], 
    (2:11, ada_nll, "adagan_adagan")
)

ada_c  = npzread(joinpath(ada_results_dir, "coverage.npy"))
push!(results[f][:coverage], 
    (2:11, ada_c, "adagan_adagan")
)

exp_name      = "adagan_comparison"
exp_condition = "nll"
plot_type     = "timeseries"
xlim = (0, 11)
ylim = (-20, 60)
plt = timeseries_comparison(results[f][:nll], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

exp_name      = "adagan_comparison"
exp_condition = "coverage"
plot_type     = "timeseries"
xlim = (0, 11)
ylim = (0.1, 1.5)
plt = timeseries_comparison(results[f][:coverage],  xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)


# f = "dimensionality_experiment-2018-06-06T14_43_26.405.jld2"
# f = "dimensionality_experiment-2018-06-06T15_42_00.271.jld2"
f = "dimensionality_experiment-2018-06-06T16_03_54.514.jld2"
results[f]
for (r,tnll) in zip(results[f][:nll], results[f][:true_nll])
    r[2] ./= tnll[2][1:1,:]
end

exp_name = "dimensionality_experiment"

condition = "nll"
plot_type = "timeseries"
xlim = (0, 12)
ylim = (-1, 3)
plt = timeseries_comparison(results[f][:nll],  xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

condition = "nll"
plot_type = "timeseries"
xlim = (0, 12)
ylim = (-1, 3)
plt = timeseries_comparison(results[f][:nll],  xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)