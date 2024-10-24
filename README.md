# Fitting Hierarchical Bayesian Models With Selection Effects in Astronomy

This is a talk that I gave at FWAM (Flatiron-Wide Autumn Meeting, formerly
Flatiron-Wide Algorithms and Mathmatics).  It is a rather technical introduction
to fitting astronomoical catalogs that are subject to measurement uncertainty
and selection effects.

You can view the presentation at https://farr.github.io/FWAM2024HierarchicalSelection/

The code for fitting the models I discuss in the talk is in the `FWAM2024`
sub-directory.  To get yourself up and running, go to that directory, run
[`julia`](https://julialang.org) and install the environment:
```julia-repl
julia> ]
(@v1.11) pkg> activate .
Activating project at `~/Research/FWAM2024HierarchicalSelection/FWAM2024`
(FWAM2024) pkg> instantiate
(FWAM2024) pkg> # Type backspace to return to the REPL from package mode
julia> using FWAM2024
julia> do_exact_model() # Will simulate and fit the exact measurement model and save some plots
```
Feel free to examine the code and play around.