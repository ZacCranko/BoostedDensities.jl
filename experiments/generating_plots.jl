include(joinpath(Pkg.dir("BoostedDensities"), "experiments", "plotting_commands.jl"))

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

# print summary statistics in a LaTeX friendly way
foreach(kde_results) do x
    ind, res, cond = x
    @assert size(res, 1) == 1
    t = OneSampleTTest(res |> vec)
    ci = confint(t)
    @printf "%s \& %.4f \& %.4f \& %.4f \\\\\n" cond t.xbar ci[1] ci[2]
end

# violin plots
xlim = (1,10)
ylim = (0,2)
x = vcat((fill(c, length(d)) for (i,d,c) in kde_results)...)
y = vcat((d[:] for (i,d,c) in kde_results)...)
# plt = plot(xlim=xlim, ylim=ylim)
plt = violin(x, y, grid=true, framestyle=:grid)
exp_name      = "kde_comparison"
exp_condition = "nll"
plot_type     = "violin"
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
savefig(plt,dst)


xlim = (1,2)
ylim = (0,5)
timeseries_comparison(r[:nll][1:1], xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
savefig(plt, dst)



results_dir    = joinpath(Pkg.dir("BoostedDensities"), "adagan_results")
adagan_results = sort(readdir(results_dir)) # make sure sorted so line up samples and parameters
adagan_p_distributions = map(f for f in adagan_results if ismatch(r"real_data_params_mean_[0-9]{2}_var_[0-9].[0-9]{2}.npy", f)) do f
    means = npzread(joinpath(results_dir, f))
    var   = match(r".*_var_([0-9].[0-9]{2}).npy", f).captures |> first |> parse
    GaussianMixture(vec(mapslices(means, 2) do mu; MvNormal(mu, sqrt(var)) end))
end    

ada_true_nll = results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:true_nll][1]
ada_nll = - npzread(joinpath(results_dir, "likelihood.npy")) ./ ada_true_nll'
push!(results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:nll], 
    (2:11, ada_nll, "adagan_adagan")
)

ada_c  = npzread(joinpath(results_dir, "coverage.npy"))
push!(results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:coverage], 
    (2:11, ada_c, "adagan_adagan")
)

# remove weird run where we got -ve values for nll
z = (results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:nll][1][1], 
results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:nll][1][2][:,[1,2,3,4,5,6,7,8,9,10,11,13,14,15]],
results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:nll][1][3])

exp_name      = "adagan_comparison"
exp_condition = "nll"
plot_type     = "timeseries"
xlim = (0, 11)
ylim = (0.0, 1.5)
plt = timeseries_comparison(z, xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)

2
exp_name      = "adagan_comparison"
exp_condition = "coverage"
plot_type     = "timeseries"
xlim = (0, 11)
ylim = (0.1, 1.5)
plt = timeseries_comparison(results["adagan_comparison-2018-05-18T18_09_38.216.jld2"][:coverage],  xaxis = ("", xlim), yaxis = ("", ylim), error = :ribbon)
dst = plot_destination(exp_name, exp_condition, plot_type, xlim, ylim)
savefig(plt, dst)