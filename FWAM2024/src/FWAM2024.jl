module FWAM2024

using CairoMakie
using Cosmology
using Distributions
using GaussianKDEs
using LaTeXStrings
using MCMCChainsStorage
using PairPlots
using Random
using SpecialFunctions
using StatsFuns
using Trapz
using Turing
using Unitful
using UnitfulAstro
using Zygote

function population_logdensity_fn(a, mu, sigma)
    function pop_logdens(x)
        r = (x-mu)/sigma
        log(a) - r*r/2
    end
    pop_logdens
end

function population_normalization(a, mu, sigma)
    a * sqrt(2*pi) * sigma
end

function estimate_pop_normalization(log_pop_dens, xs, logp_xs, n)
    log_wts = log_pop_dens.(xs) .- logp_xs
    mu = exp(logsumexp(log_wts)) / n
    s2 = exp(logsumexp(2 .* log_wts)) / (n*n) - mu*mu/n

    ne = mu*mu / s2

    (mu, ne)
end

function draw_x(a, mu, sigma)
    n = rand(Poisson(population_normalization(a, mu, sigma)))
    rand(Normal(mu, sigma), n)
end

function draw_xobs(xs; sigma_obs=1.0)
    broadcast(xs, sigma_obs) do x, s
        rand(Normal(x, s))
    end
end

function plot_population_trace(axis, a, mu, sigma; extent=3, label=nothing, kwargs...)
    p_ld = population_logdensity_fn(a, mu, sigma)
    xs = (mu - extent*sigma):0.01*sigma:(mu + extent*sigma)
    lines!(axis, xs, exp.(p_ld.(xs)); label=label, kwargs...)
end

@model function exact_model(xs)
    a ~ Uniform(50, 200)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)

    pop_ldens = population_logdensity_fn(a, mu, sigma)
    
    Turing.@addlogprob! sum(pop_ldens.(xs))
    Turing.@addlogprob! -population_normalization(a, mu, sigma)
end

@model function obs_model(xs_obs, sigma_obs)
    Nobs = length(xs_obs)
    a ~ Uniform(50, 200)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)

    xs_raw ~ filldist(Flat(), Nobs)
    xs = xs_raw .* sigma_obs .+ xs_obs
    pop_ldens = population_logdensity_fn(a, mu, sigma)
    Turing.@addlogprob! sum(pop_ldens.(xs))
    Turing.@addlogprob! -population_normalization(a, mu, sigma)

    xs_obs ~ arraydist([Normal(x, s) for (x,s) in zip(xs, sigma_obs)])

    return (xs = xs,)
end

@model function obs_selected_model(xs_obs, sigma_obs, xs_det, logp_det, ndet)
    Nobs = length(xs_obs)
    a ~ Uniform(50, 200)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)

    xs_raw ~ filldist(Flat(), Nobs)
    xs = xs_raw .* sigma_obs .+ xs_obs
    pop_ldens = population_logdensity_fn(a, mu, sigma)
    pop_norm, norm_neff = estimate_pop_normalization(pop_ldens, xs_det, logp_det, ndet)
    Turing.@addlogprob! sum(pop_ldens.(xs))
    Turing.@addlogprob! -pop_norm

    xs_obs ~ arraydist([Normal(x, s) for (x,s) in zip(xs, sigma_obs)])

    return (xs = xs, norm_neff = norm_neff, Nexp = pop_norm)
end

function do_exact_model(; seed = 8174570720888583473)
    Random.seed!(seed)
    a_true = 100.0
    mu_true = 0.0
    sigma_true = 1.0

    xs = draw_x(a_true, mu_true, sigma_true)
    model = exact_model(xs)
    trace = mapreduce(c -> sample(model, NUTS(1000, 0.8), 1000), chainscat, 1:4)

    f = pairplot(trace, PairPlots.Truth((; a=a_true, mu=mu_true, sigma=sigma_true)))
    save(joinpath(@__DIR__, "..", "figures", "exact_model_parameters.png"), f)

    f = Figure()
    a = Axis(f[1,1], xlabel=L"x", ylabel=L"\mathrm{d}N/\mathrm{d}x")
    plot_population_trace(a, a_true, mu_true, sigma_true; color=:black, label="True")
    k = KDE(xs)
    x = minimum(xs):0.01*std(xs):maximum(xs)
    lines!(a, x, pdf.((k,), x)*length(xs), color=:black, linestyle=:dash, label="Observed")
    for i in 1:100
        label = (i == 1 ? "Fitted" : nothing)
        t = sample(trace, 1)
        plot_population_trace(a, t[:a][1,1], t[:mu][1,1], t[:sigma][1,1]; color=Makie.wong_colors()[1], alpha=0.1, label=label)
    end
    axislegend(a, location=:bm)
    save(joinpath(@__DIR__, "..", "figures", "exact_model_population.png"), f)
end

function do_obs_model(; seed = 0x6277e9d5e55f1ad9)
    Random.seed!(seed)
    a_true = 100.0
    mu_true = 0.0
    sigma_true = 1.0

    xs = draw_x(a_true, mu_true, sigma_true)
    xs_obs = draw_xobs(xs; sigma_obs=1.0)
    sigma_obs = ones(length(xs_obs))

    model = obs_model(xs_obs, sigma_obs)
    trace = mapreduce(c -> sample(model, NUTS(1000, 0.8), 1000; adtype=AutoZygote()), chainscat, 1:4)
    trace = append_generated_quantities(trace, model)

    f = pairplot(trace[[:a, :mu, :sigma]], PairPlots.Truth((; a=a_true, mu=mu_true, sigma=sigma_true)))
    save(joinpath(@__DIR__, "..", "figures", "obs_model_parameters.png"), f)

    f = Figure()
    a = Axis(f[1,1], xlabel=L"x", ylabel=L"\mathrm{d}N/\mathrm{d}x")
    plot_population_trace(a, a_true, mu_true, sigma_true; color=:black, label="True")
    k = KDE(xs_obs)
    x = minimum(xs_obs):0.01*std(xs_obs):maximum(xs_obs)
    lines!(a, x, pdf.((k,), x)*length(xs_obs), color=:black, linestyle=:dash, label="Observed")
    for i in 1:100
        label = (i == 1 ? "Fitted" : nothing)
        t = sample(trace, 1)
        plot_population_trace(a, t[:a][1,1], t[:mu][1,1], t[:sigma][1,1]; color=Makie.wong_colors()[1], alpha=0.1, label=label)
    end
    axislegend(a, location=:bm)
    save(joinpath(@__DIR__, "..", "figures", "obs_model_population.png"), f)
end

function do_obs_selected_model(; seed = 0xc3fab3eb7013cf5e)
    Random.seed!(seed)
    a_true = 100.0
    mu_true = 0.0
    sigma_true = 1.0

    x_thresh = -1.0

    xs = draw_x(a_true, mu_true, sigma_true)
    xs_obs = draw_xobs(xs; sigma_obs=1.0)
    sigma_obs = ones(length(xs_obs))

    det_sel = xs_obs .> x_thresh
    xs_det = xs_obs[det_sel]
    sigma_det = sigma_obs[det_sel]

    ndraw = 4000
    draw_dist = Normal(0, 2)
    xs_draw = rand(draw_dist, ndraw)
    logp_draw = [logpdf(draw_dist, x) for x in xs_draw]
    xs_draw_obs = draw_xobs(xs_draw; sigma_obs=1.0)
    draw_det_sel = xs_draw_obs .> x_thresh
    xs_draw_det = xs_draw[draw_det_sel]
    logp_draw_det = logp_draw[draw_det_sel]

    model = obs_selected_model(xs_det, sigma_det, xs_draw_det, logp_draw_det, ndraw)
    trace = mapreduce(c -> sample(model, NUTS(1000, 0.8), 1000; adtype=AutoZygote()), chainscat, 1:4)
    trace = append_generated_quantities(trace, generated_quantities(model, trace))

    @info "Minimum Neff = $(round(minimum(trace[:norm_neff]), digits=1)); 4*N = $(4*length(xs_det))"

    e = ess(trace[[:a, :mu, :sigma]])
    @info "ESS: a = $(round(e[:a, :ess], digits=1)), mu = $(round(e[:mu, :ess], digits=1)), sigma = $(round(e[:sigma, :ess], digits=1))"

    f = pairplot(trace[[:a, :mu, :sigma]], PairPlots.Truth((; a=a_true, mu=mu_true, sigma=sigma_true)))
    save(joinpath(@__DIR__, "..", "figures", "obs_sel_model_parameters.png"), f)

    f = Figure()
    a = Axis(f[1,1], xlabel=L"x", ylabel=L"\mathrm{d}N/\mathrm{d}x", limits=(-5, 5, 0, 150))
    plot_population_trace(a, a_true, mu_true, sigma_true; color=:black, label="True")
    k = KDE(xs_det)
    x = minimum(xs_det):0.01*std(xs_det):maximum(xs_det)
    lines!(a, x, (pdf.((k,), x) .+ pdf.((k,), 2 .* x_thresh .- x))*length(xs_det), color=:black, linestyle=:dash, label="Observed")
    for i in 1:100
        label = (i == 1 ? "Fitted" : nothing)
        t = sample(trace, 1)
        plot_population_trace(a, t[:a][1,1], t[:mu][1,1], t[:sigma][1,1]; color=Makie.wong_colors()[1], alpha=0.1, label=label)
    end
    band!(a, [-5, x_thresh], [0, 0], [150, 150], color=:grey, alpha=0.25)
    axislegend(a, location=:bm)
    save(joinpath(@__DIR__, "..", "figures", "obs_sel_model_population.png"), f)
end

function unnorm_md_sfr(z)
    (1 + z) / (1 + ((1+z)/(1+1.9))^5.6)
end

function estimate_merger_rate(; bbh_mr = 25.0, z_ref = 0.2)
    c = cosmology()
    zs = expm1.(log(1.0):0.01:log(1.0 + 10.0))
    md_fn = unnorm_md_sfr.(zs) ./ unnorm_md_sfr(z_ref)

    R_of_z = (bbh_mr * u"Gpc^-3*yr^-1") .* md_fn
    integrand = 4 .* pi .* R_of_z .* comoving_volume_element.(u"Gpc^3", (c,), zs) ./ (1 .+ zs)
    trapz(zs, integrand)
end

function do_scatterplot(; seed = 0xf2a41ff544dbaa2e)
    Random.seed!(seed)

    npts = 100

    xs = rand(npts)
    ys = rand(npts)

    for (i, ticks) in enumerate([0.25:0.25:0.75, 0.1:0.1:0.9, 0.05:0.05:0.95, 0.01:0.01:0.99])
        f = Figure(size=(800, 400))
        a = Axis(f[1,1], xlabel=L"x_1", ylabel=L"x_2", xminorgridvisible=true, yminorgridvisible=true, xminorticks=ticks, yminorticks=ticks)
        
        scatter!(a, xs, ys)
        save(joinpath(@__DIR__, "..", "figures", "scatter-$(i).png"), f)
    end
end

end # module FWAM2024
