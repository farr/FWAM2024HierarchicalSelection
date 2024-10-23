module FWAM2024

using Distributions
using SpecialFunctions
using Turing
using Zygote

function draw_lobs(phi_star, L_star, alpha)
    Nexpected = phi_star*gamma(alpha)

    n = rand(Poisson(Nexpected))
    rand(Gamma(alpha, L_star), n)
end

@model function exact_luminosity_model(Lobs)
    phi_star ~ Uniform(10, 1000)
    L_star ~ Uniform(0.5, 2)
    alpha ~ Uniform(0, 2)

    logl_term = @. log(phi_star) + alpha*log(Lobs / L_star) - log(Lobs) - (Lobs / L_star)
    norm_term = phi_star*gamma(alpha)

    Turing.@addlogprob! sum(logl_term) - norm_term
end

end # module FWAM2024
