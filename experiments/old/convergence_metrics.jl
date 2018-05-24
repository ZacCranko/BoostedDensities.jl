@everywhere begin
    using BoostedDensities, Distributions, Flux, JLD2
    using Plots; gr(size=(800, 600))
    function ring_gaussian_worker(i)
        # Ring of Gaussians
        # -----------------
        p  = GaussianMixture(σ = 1/4)
        q0 = MvNormal(2, 1)
        num_p, num_q = 10_000, 10_000
    
        model = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 1), x->σ.(x)) # classifier architecture
        return run_experiment(p, q0, num_p, num_q, iter = 10, num_epochs = 400, model = model, verbose = false, run_boosting_metrics = true, seed = 1337 + i)
    end
end

results = pmap(ring_gaussian_worker, 1:10)

using JLD2
@save "convergence_metrics.jld2" results
@load "convergence_metrics.jld2" results


using Plots; gr(size=(800, 600))
include(joinpath(pwd(), "utilities.jl"))

# @load "convergence_metrics.jld2"

xlim = (0,10)
ylim = (-0.3,1.6)
ce = [(inds(train), [t[2] for t in train]) for (q,(train,coverage)) in results]
a = 0.4
plt = plot(ce[1]..., color = color_palette[1], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500))
foreach(ce[2:end]) do x; plot!(plt, x, color = color_palette[1], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500)) end
mup = [(inds(train), [t[3] for t in train]) for (q,(train,coverage)) in results]
foreach(mup) do x; plot!(plt, x, color = color_palette[4], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500)) end
muq = [(inds(train), [t[4] for t in train]) for (q,(train,coverage)) in results]
foreach(muq) do x; plot!(plt, x, color = color_palette[6], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500)) end
dest = plot_destination("convergence_analysis", "train_metrics_ring", plot, xlim, ylim)
savefig(dest)


xlim = (0,10)
ylim = (0,6)
kl   = [[t[BoostedDensities.kl] for t in coverage] for (q,(train,coverage)) in results]
c    = [[t[BoostedDensities.coverage] for t in coverage] for (q,(train,coverage)) in results]
nll  = [[t[BoostedDensities.expected_log_likelihood] for t in coverage] for (q,(train,coverage)) in results]; foreach(1:10) do i; nll[i][1] = 200 end 
a = 0.2
plt = plot(0:10, kl[1], color = color_palette[1], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500))
foreach(kl[2:end]) do x; plot!(plt, 0:10, x, color = color_palette[1], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500)) end
plot!(plt, 0:10, mapslices(median, cat(2,kl...) ,2), color = color_palette[1], opacity = 1, legend=false, linewidth = 2.5, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500))


foreach(nll) do x; plot!(plt, 0:10, x, color = color_palette[4], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500)) end
plot!(plt, 0:10, mapslices(median, cat(2,nll...) ,2), color = color_palette[4], opacity = 1, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500))


foreach(c) do x; plot!(plt, 0:10, x, color = color_palette[6], opacity = a, legend=false, linewidth = 2, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500)) end
plot!(plt, 0:10, mapslices(median, cat(2,c...) ,2), color = color_palette[6], opacity = 1, legend=false, linewidth = 2.5, legend = false, fmt = :pdf, framestyle = :box, ticks = [], ylim = ylim, xlim = xlim, size = (800, 500))
dest = plot_destination("convergence_analysis", "coverage_metrics_ring", plot, xlim, ylim)
savefig(dest)



# vals = cat(1, [t[2:end]' for t in small_metrics]...)
# using Plots.PlotMeasures
# xlim = (-0.0, 8)
# ylim = (-0.5, 1.6)
# plot(inds(small_metrics), vals, 
#         framestyle = :box,
#         xlims = xlim, xticks = [],
#         ylims = ylim, yticks = [], linewidth = 2,
#         color_palette = color_palette,
#         size  = (700, 500), legend = false, fmt = :pdf)
# dest = plot_destination("convergence_analysis", "adacovfefe", plot, xlim, ylim)
# savefig(dest)


# plot(inds(small_metrics), vals)

# ks =  [BoostedDensities.kl, BoostedDensities.coverage, BoostedDensities.expected_log_likelihood]
# vals = cat(2, [[metrics[i][k] for i in 1:length(metrics)] for k in ks]...); vals[:,3] *= -1; vals[1,3] = 200
# xlim = (-0.01, 8)
# ylim = (0, 6)
# plot(0:8, vals, 
#         xlims = xlim, xticks = [],
#         ylims = ylim, yticks = [], linewidth = 2,
#         color_palette = color_palette,
#         ticks=nothing, framestyle = :box,
#         size  = (700, 400), legend = false, fmt = :pdf)
# dest = plot_destination("convergence_analysis", "adacovfefe_convergence_metrics", plot, xlim, ylim)
# savefig(dest)
# # function contour_plot_with_modes(q, cond)

# savefig(dest)
# # end


# Dict(k=>[metrics[i][k] for i in 1:length(metrics)] for k in ks)

# plot(1:size(vals,1), vals)
