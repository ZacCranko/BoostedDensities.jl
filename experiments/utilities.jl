
function plot_splat(f::Function; plot_range = -6:0.05:+6) 
    return (plot_range, plot_range, f(hcat(([a,b] for (a,b) in Iterators.product(plot_range, plot_range))...)))
end
plot_splat(p::Distribution; plot_range = -6:0.05:+6) = plot_splat(x->pdf(p, x), plot_range = plot_range)
plot_splat(m::Flux.Chain; plot_range = -6:0.05:+6)   = plot_splat(x->m(x), plot_range = plot_range)

color_palette = [colorant"#c82829",
                 colorant"#f5871f",
                 colorant"#eab700",
                 colorant"#718c00",
                 colorant"#3e999f",
                 colorant"#4271ae",
                 colorant"#8959a8",
                 colorant"#a3685a"]


function plot_destination(exp_name, exp_condition, plot_type, xlim, ylim, ext="pdf")
    out_dir = joinpath("/Users/zcranko/Documents/GitHub/Boosting/experiments", "plots", exp_name)
    _title  = @sprintf "%s-%s-%s-xlim_%s_to_%s-ylim_%s_to_%s" exp_name exp_condition plot_type xlim[1] xlim[2] ylim[1] ylim[2]
    title   = replace(_title, ".", "_")
    info("Saving plot $title to ", out_dir)
    mkpath(out_dir)
    return joinpath(out_dir, "$title.$ext")
end

function inds(train_history)
    boost_iter = map(x->Int(first(x)), train_history)
    counts     = map(i->count(x->(x == i), boost_iter), sort(unique(boost_iter)))
    return indices = [(i - sum(counts[1:z-1]))/counts[z] + z - 1 for (i,z) in enumerate(boost_iter)]
end