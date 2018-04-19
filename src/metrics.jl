
function normalise!(q::QDensity, rng = ([-10,-10], [10, 10])) 
    if !isnormalised(q)
        z, _ = hcubature_v((x,v)-> begin v[:] = exp.(Distributions._logpdf(q, x)) end, rng...)
        logz      = zeros(length(q.models))
        logz[end] = log(z)
        q.logz    = logz
        return q
    end
    return q
end

function kl(p::Distribution, q::Distribution, p_samps = nothing, q_samps = nothing; rng = ([-10,-10], [10, 10]))::Float64
    if !isnormalised(q)
        normalise!(q)
    end
    f = function (x,v) 
        v[:] = pdf(p, x) .* (logpdf(p, x) - logpdf(q, x))
    end
    kl, _ = hcubature_v(f, rng...)
    return kl
end

integrate_pdf(p::Distribution; rng = ([-10,-10], [10, 10])) = hcubature_v((x,v)-> begin v[:] = pdf(p, x) end, rng...)[1]

function coverage(p, q, p_samps, q_samps; κ = 0.95, n = 1000)::Float64
    if !isnormalised(q)
        normalise!(q)
    end
    log_dq    = logpdf(q, q_samps)
    logt_iter = linspace(minimum(log_dq), maximum(log_dq), n)
    logt_i    = findfirst([mean(log_dq .> logt) < κ for logt in logt_iter])
    logt      = logt_iter[logt_i]
    return mean(logpdf(q, p_samps) .> logt)
end

function expected_log_likelihood(p, q, p_samps, q_samps)::Float64
    if !isnormalised(q)
        normalise!(q)
    end
    return -mean(logpdf(q, p_samps))
end