using ProgressMeter

mutable struct HMCSampler{T<:Real}
  logfgrad::Function
  ϵ::Real
  l::Int
  lcov::Union{UniformScaling, AbstractMatrix}
  q::AbstractVector{T} # position vector
  a::Nullable{Float64} # acceptance rate
  stepped::Bool        # did we just accept a step?
end

HMCSampler{T<:Real}(logfgrad::Function, q::Vector{T}; ϵ = 0.5, l = 50, lcov = I) = 
  HMCSampler{T}(logfgrad, ϵ, l, lcov, q, Nullable{Float64}(), false)

HMCSampler{T<:Real}(logfgrad::Function, q::Vector{T}; ϵ = 0.5, t = 25, lcov = I) = 
  HMCSampler{T}(logfgrad, ϵ, div(t, ϵ), lcov, q, Nullable{Float64}(), false)

function (hmc::HMCSampler)(; verbose = 0)
  x1 = copy(hmc.q)
  logf0, grad0 = logf1, grad1 = hmc.logfgrad(x1)

  # Momentum variables
  p0 = p1 = hmc.lcov * randn(length(hmc.q))

  # Make a half step for a momentum at the beginning
  p1 += hmc.ϵ * grad0/2

  # Alternate full steps for position and momentum
  for _ in Base.OneTo(hmc.l)
    # Make a full step for the position
    x1 += hmc.ϵ * p1

    logf1, grad1 = hmc.logfgrad(x1)

    # Make a full step for the momentum
    p1 += hmc.ϵ * grad1
  end

  # Make a half step for momentum at the end
  p1 -= hmc.ϵ * grad1/2

  # Negate momentum at end of trajectory to make the proposal symmetric
  p1 *= -1

  # Evaluate potential and kinetic energies at start and end of trajectory
  inv_lcov = inv(hmc.lcov)
  Kp0 = sum(abs2, inv_lcov * p0)/2
  Kp1 = sum(abs2, inv_lcov * p1)/2

  hmc.a = Nullable(min(1, exp((logf1 - Kp1) - (logf0 - Kp0))))
  
  hmc.stepped = rand() < get(hmc.a) 

  if hmc.stepped
    hmc.q .= x1
  end
  
  return hmc.q
end

function hmc_worker!(hmc::HMCSampler{T}, n::Int; verbose::Int = 0, burnin = 0, thin = 0) where {T}
  res  = Array{T}(length(hmc.q), n)
  prog = Progress(div(n, thin + 1) + burnin, desc="HMC chain: ")
  a = 0 

  for i in Base.OneTo(burnin)
    hmc() # run the sampler
    
    if verbose > 0 
      showvalues = [("burn in", "done"),
        ("avg. acc. rate", "N/A"), 
        ("stored samples", 0),
        ("samples generated", i)]
                    
                    
      ProgressMeter.next!(prog; showvalues = showvalues[1:min(verbose, 4)])
    end
  end 

  for (i, j) in enumerate(repeat(Base.OneTo(n), inner = 1 + thin))
    
    res[:, j] = hmc()

    if verbose > 0
      a = 1/i * get(hmc.a) + (i-1)/i * a
      showvalues = [("burn in", "done"),
        ("avg. acc. rate", a), 
        ("stored samples",  i),
        ("samples generated", i + burnin)]
      ProgressMeter.next!(prog; showvalues = showvalues[1:min(verbose, 4)])
    end
  end 

  return res
end

function hmc_sample(samplers::Vector{HMCSampler{T}}, n::Int; burnin = 0, thin = 0, verbose = 0) where T
  m = div(n, length(samplers))
  localtask = @async hmc_worker!(first(samplers), m + (n - length(samplers)*m), 
                                 burnin = burnin, thin = thin, verbose = verbose)
  @everywhere f = hmc -> hmc_worker!(hmc, $m, burnin = $burnin, thin = $thin)
  res = pmap(f, samplers[2:end])
  return cat(2, localtask.result, res...)
end

hmc_sample(hmc::HMCSampler, n::Int; burnin = 0, thin = 0) = 
  hmc_sample(fill(hmc, nworkers()), n, burnin = burnin, thin = thin)