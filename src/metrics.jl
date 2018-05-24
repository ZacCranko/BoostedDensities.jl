function kl(p::Distribution, q::Distribution, p_samps = nothing, q_samps = nothing; rng = (fill(-18, dim(p)), fill(18, dim(p))), abstol = 1e-3)::Float64
    if !isnormalised(q)
        normalise!(q)
    end
    f = function (x,v) 
        pdf_p = pdf(p, x) 
        v[:]  = pdf_p .* logpdf(p, x) - pdf_p .* logpdf(q, x)
    end
    kl, _ = hcubature_v(f, rng..., abstol = abstol)
    return kl
end

integrate_pdf(p::Distribution; rng = (fill(-30, dim(p)), fill(30, dim(p))), abstol = 1e-3) = hcubature_v((x,v)-> begin v[:] = pdf(p, x) end, rng..., abstol = abstol)[1]

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


function nll(q::Distribution, p_samps)::Float64
    if !isnormalised(q)
        normalise!(q)
    end
    return -mean(logpdf(q, p_samps))
end

nll(p::Distribution, q::Distribution, p_samps, q_samps) = nll(q, p_samps)