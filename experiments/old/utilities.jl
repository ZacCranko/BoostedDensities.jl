
function plot_splat(f::Function; plot_range = -6:0.05:+6) 
    return (plot_range, plot_range, f(hcat(([a,b] for (a,b) in Iterators.product(plot_range, plot_range))...)))
end
plot_splat(p::Distribution; plot_range = -6:0.05:+6) = plot_splat(x->pdf(p, x), plot_range = plot_range)
plot_splat(m::Flux.Chain; plot_range = -6:0.05:+6)   = plot_splat(x->m(x), plot_range = plot_range)




