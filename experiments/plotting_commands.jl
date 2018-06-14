using Revise; Revise.track(@__FILE__)

function plot_splat(f::Function; plot_range = -15:0.05:+15) 
    return (plot_range, plot_range, f(hcat(([a,b] for (a,b) in Iterators.product(plot_range, plot_range))...)))
end
plot_splat(p::Distribution; plot_range = -15:0.05:+15) = plot_splat(x->pdf(p, x), plot_range = plot_range)
plot_splat(m::Flux.Chain;   plot_range = -15:0.05:+15) = plot_splat(x->m(x),      plot_range = plot_range)

upscale = 1 #8x upscaling in resolution
default(size=(800*upscale, 600*upscale)) # Plot canvas size

default(legend = false, 
        grid = true, 
        framestyle = :grid, 
        ticks = nothing,
        margin = 0Plots.px,
        xlabel = "",
        ylabel = "")

function ci(run, alpha) # calculate confidence intervals
    ttest = OneSampleTTest(run)
    q = quantile(TDist(ttest.df), 1 - alpha/2)
    return [ttest.xbar, q * ttest.stderr]
end

function timeseries_comparison(inds_data_conditions; 
    alpha = 0.05,
    error = :ribbon, 
    xaxis = ("", (0, 10)),
    yaxis = ("", (0, 1)),
    coeff = 1.0,
    legend = false,
    kws...
    )
    
    plt = plot(xaxis = xaxis, yaxis = yaxis, legend = legend, kws...)
    for (inds, data, condition) in inds_data_conditions
        if isa(data, Matrix)
            mean_stderr = mapslices(data, 2) do run
                ci(run, alpha)
            end
            @eval plot!($plt, $inds, $coeff .* $mean_stderr[:,1], $error = $mean_stderr[:,2], fillalpha = 0.2, label = $condition)
        else
            for (i, d) in zip(inds, data)
                plot!(plt, i, coeff .* d, label = condition; alpha = 0.5)
            end
        end
    end

    return plt
end

function plot_destination(exp_name, exp_condition, plot_type, xlim, ylim; ext="pdf")
    out_dir = joinpath(Pkg.dir("BoostedDensities"), "plots", exp_name)
    _title  = @sprintf "%s-%s-%s-xlim_%s_to_%s-ylim_%s_to_%s" exp_name exp_condition plot_type xlim[1] xlim[2] ylim[1] ylim[2]
    title   = replace(_title, ".", "_")
    mkpath(out_dir)
    return joinpath(out_dir, "$title.$ext")
end

"""
Preprocessing: results in vectors of tuples (inds, data, condition)
    inds:      iterable with the x-indices,
    data:      matrix where each column is one run for the metric
    condition: string of the experimental condition name
"""
function inds(train_history)
    boost_iter = map(x->Int(first(x)), train_history)
    counts     = map(i->count(x->(x == i), boost_iter), sort(unique(boost_iter)))
    return indices = [(i - sum(counts[1:z-1]))/counts[z] + z - 1 for (i,z) in enumerate(boost_iter)]
end

function preprocess_output(data, conds)
    pre_processed = Dict()
    boostinds = 1:size(data[3,1,1], 1)
    try
        pre_processed[:kl]  = [(boostinds, cat(2, (getindex.(data[3,j,i], kl)  for i in 1:size(data,3))...), c) for (j,c) in enumerate(conds)]
    end
    pre_processed[:nll]     = [(boostinds, cat(2, (getindex.(data[3,j,i], nll) for i in 1:size(data,3))...), c) for (j,c) in enumerate(conds)]
    try
        pre_processed[:coverage] = [(boostinds, cat(2, (getindex.(data[3,j,i], coverage) for i in 1:size(data,3))...), c) for (j,c) in enumerate(conds)]
    end
    if size(data, 1) == 4
        pre_processed[:true_nll] = [[data[4,j,i] for i in 1:size(data,3) ] for (j,c) in enumerate(conds)]
    else
        pre_processed[:true_nll]  = [(boostinds, cat(2, (getindex.(data[3,j,i], :true_nll)  for i in 1:size(data,3))...), c) for (j,c) in enumerate(conds)]
    end
    @show typeof(data[2,1,1])
    @show keys(data[2,1,1][1])
    traininds = inds(getindex.(data[2,1,1], :iter))
    for m in keys(data[2,1,1][1])
        try
            pre_processed[m] = [
                (traininds, getindex.(data[2,j,i], m) for i in 1:size(data,3), c) 
                for (j,c) in enumerate(conds)]
        end
    end
    return pre_processed
end

preprocess_output(input) = preprocess_output(input[1], input[2])

function process_experiment(experiment)
    loaded = FileIO.load(joinpath(RESULTSDIR, experiment))
    results = values(loaded) |> first

    return preprocess_output(results)
end

function process_kde_experiment(experiment)
    loaded = FileIO.load(joinpath(RESULTSDIR, experiment))
    pre_processed = Dict{Symbol, Any}()
    data, conds = values(loaded) |> first
    for m in (:nll,)
        pre_processed[m] = map(enumerate(conds)) do x
            j,c = x
            d = cat(2, (getindex.(data[3,j,i], eval(m)) for i in 1:size(data,3))...)
            return (1:size(d,1), d, c)
        end
    end
    pre_processed[:true_nll] = cat(2, (getindex.(data[3,1,i][1], :true_nll) for i in 1:size(data,3))...)
    return pre_processed
end
