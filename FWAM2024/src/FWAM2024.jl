module FWAM2024

using AxisArrays
using CairoMakie
using Distributions
using GaussianKDEs
using LaTeXStrings
using PairPlots
using Random
using SpecialFunctions
using Turing
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

function draw_x(a, mu, sigma)
    n = rand(Poisson(population_normalization(a, mu, sigma)))
    rand(Normal(mu, sigma), n)
end

function plot_population_trace(axis, a, mu, sigma; extent=3, label=nothing, kwargs...)
    p_ld = population_logdensity_fn(a, mu, sigma)
    xs = (mu - extent*sigma):0.01*sigma:(mu + extent*sigma)
    lines!(axis, xs, exp.(p_ld.(xs)); label=label, kwargs...)
end

@model function exact_model(xs)
    a_estimated = length(xs) / (sqrt(2*pi)*std(xs))
    a ~ Uniform(a_estimated/2, 2*a_estimated)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)

    pop_ldens = population_logdensity_fn(a, mu, sigma)
    
    Turing.@addlogprob! sum(pop_ldens.(xs))
    Turing.@addlogprob! -population_normalization(a, mu, sigma)
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

function do_

end # module FWAM2024
