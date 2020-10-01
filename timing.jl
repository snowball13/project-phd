using BenchmarkTools, Plots

function foo(n)
    A = randn(n,n)
    b = randn(n)
    A \ b
end

Ns = 1:100:1000
tms = Array{Float64}(undef, length(Ns))
for k = 1:length(Ns)
    tms[k] = @belapsed foo($(Ns[k]))
end

plot(Ns, tms)

plot(Ns, tms; yscale=:log10, xscale=:log10)