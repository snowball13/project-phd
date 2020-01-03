using Revise
using ApproxFun
using Test, Profile
using StaticArrays, LinearAlgebra, BlockArrays, BlockBandedMatrices,
        SparseArrays
# using Makie
using OrthogonalPolynomialFamilies
import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, getopindex, getnk, resizecoeffs!,
                    transformparamsoperator, weightedpartialoperatorx,
                    weightedpartialoperatory, partialoperatorx, partialoperatory,
                    laplaceoperator, derivopevalatpts, getderivopptseval, getopnorms,
                    getopnorm, operatorclenshaw, weight, biharmonicoperator
using SpecialFunctions
using JLD

#===#


# D = HalfDiskFamily()
# N = 200
# S = D(1.0, 1.0)
# Δw = laplaceoperator(S, S, N, weighted=true, square=true)
# bihar = biharmonicoperator(D(2.0,2.0), N, square=true)
# save("experiments/saved/laplacian-w11-array.jld", "Lw11", sparse(Δw))
# save("experiments/saved/biharmonic-array.jld", "bihar", sparse(bihar))
# save("experiments/saved/clenshawmats-11.jld", "A", S.A, "B", S.B, "C", S.C, "DT", S.DT)
# H = D.H
#     jldopen("experiments/saved/Hspaces-dict.jld", "w") do file
#     end
#     paramslist = Vector{NTuple}()
#     n = 1
#     for params in keys(H.spaces)
#         stra = "a" * string(params[1])
#         stra *= "-" * string(params[2])
#         strb = "b" * string(params[1])
#         strb *= "-" * string(params[2])
#         jldopen("experiments/saved/Hspaces-dict.jld", "r+") do file
#             write(file, stra, H(params...).a)
#             write(file, strb, H(params...).b)
#         end
#         resize!(paramslist, n); paramslist[n] = params
#         global n += 1
#     end
#     jldopen("experiments/saved/Hspaces-dict.jld", "r+") do file
#         write(file, "params", paramslist)
#     end
# P = D.P
#     jldopen("experiments/saved/Pspaces-dict.jld", "w") do file
#     end
#     paramslist = Vector{NTuple}()
#     n = 1
#     for params in keys(P.spaces)
#         stra = "a" * string(params[1])
#         stra *= "-" * string(params[2])
#         strb = "b" * string(params[1])
#         strb *= "-" * string(params[2])
#         strparams = string(params[1])
#         strparams *= "-" * string(params[2])
#         jldopen("experiments/saved/Pspaces-dict.jld", "r+") do file
#             write(file, stra, P(params...).a)
#             write(file, strb, P(params...).b)
#         end
#         resize!(paramslist, n); paramslist[n] = params
#         global n += 1
#     end
#     jldopen("experiments/saved/Pspaces-dict.jld", "r+") do file
#         write(file, "params", paramslist)
#     end

B = BigFloat
precision(B)
setprecision(800)
D = HalfDiskFamily()
    d = load("experiments/saved/Hspaces-dict.jld")
    paramslist = d["params"]
    for params in paramslist
        stra = "a" * string(params[1])
        stra *= "-" * string(params[2])
        strb = "b" * string(params[1])
        strb *= "-" * string(params[2])
        H = D.H(B(params[1]),B(params[2]))
        resize!(H.a, length(d[stra]))
        H.a[:] = d[stra][:]
        resize!(H.b, length(d[strb]))
        H.b[:] = d[strb][:]
        # x * P[n](x) == (b[n] * P[n+1](x) + a[n] * P[n](x) + b[n-1] * P[n-1](x))
        # => P[n+1](x) == 1/b[n] * (x * P[n](x) - a[n] * P[n](x) - b[n-1] * P[n-1](x))
        resize!(H.ops, 2)
        n = length(H.a)
        w = H.weight
        x = Fun(identity, domain(w))
        if n > 0
            H.ops[1] = Fun(1/sqrt(sum(w)),space(x)); H.ops[2] = ((x - H.a[1]) * H.ops[1]) / H.b[1]
            for k = 2:n
                p = ((x - H.a[k]) * H.ops[2] - H.b[k-1] * H.ops[1]) / H.b[k]
                H.ops[1] = H.ops[2]
                H.ops[2] = p
            end
        end
    end
    d = load("experiments/saved/Pspaces-dict.jld")
    paramslist = d["params"]
    for params in paramslist
        stra = "a" * string(params[1])
        stra *= "-" * string(params[2])
        strb = "b" * string(params[1])
        strb *= "-" * string(params[2])
        P = D.P(B(params[1]),B(params[2]))
        resize!(P.a, length(d[stra]))
        P.a[:] = d[stra][:]
        resize!(P.b, length(d[strb]))
        P.b[:] = d[strb][:]
        resize!(P.ops, 2)
        n = length(P.a)
        w = P.weight
        x = Fun(identity, domain(w))
        P.ops[1] = Fun(1/sqrt(sum(w)),space(x)); P.ops[2] = ((x - P.a[1]) * P.ops[1]) / P.b[1]
        for k = 2:n
            p = ((x - P.a[k]) * P.ops[2] - P.b[k-1] * P.ops[1]) / P.b[k]
            P.ops[1] = P.ops[2]
            P.ops[2] = p
        end
    end
    D

# Set up for experiments
a, b = 1.0, 1.0
x, y = 0.42, -0.2; z = [x; y]
S = D(a, b)
d = load("experiments/saved/clenshawmats-11.jld")
resize!(S.DT, length(d["DT"])); S.DT[:] = d["DT"]
resize!(S.A, length(d["A"])); S.A[:] = d["A"]
resize!(S.B, length(d["B"])); S.B[:] = d["B"]
resize!(S.C, length(d["C"])); S.C[:] = d["C"]
S
errfuncfs = load("experiments/saved/errfuncfs.jld", "erfuncfs")
errfuncfs22 = load("experiments/saved/errfuncfs-22.jld", "erfuncfs22")
f4cfs = load("experiments/saved/f4cfs.jld", "f4cfs")
Δw = load("experiments/saved/laplacian-w11-array.jld", "Lw11")
bihar = load("experiments/saved/biharmonic-array.jld", "bihar")
D



#====#



#=
Sparsity of laplace operator
=#
using PyPlot
# a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
# N = 30
maxm = sum(1:20+1)
PyPlot.clf()
    PyPlot.spy(Array(sparse(Δw[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/images/sparsityoflaplacian-w11.pdf")
PyPlot.clf()
    PyPlot.spy(Array(sparse(bihar[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/images/sparsityofbiharmonic.pdf")
PyPlot.clf()
    # N = 25
    # k = 200
    # v = Fun((x,y)->(1 - (3(x-1)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+3)
    # T = transformparamsoperator(D(0.0, 0.0), S, N+3)
    # Tw = transformparamsoperator(S, D(0.0, 0.0), N, weighted=true)
    # A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
    # save("experiments/saved/helmholtz-array-sparsity.jld", "A", A)
    A = load("experiments/saved/helmholtz-array-sparsity.jld", "A")
    PyPlot.spy(Array(sparse(A[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    PyPlot.savefig("experiments/images/sparsityofhelmholtz.pdf")

#=
Examples
=#

using PyPlot
evalu(u, z) = isindomain(z, u.space.family) ? u(z) : NaN
isindomain(x, D::HalfDiskFamily) = 0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2)

#=
Poisson
Solution of Δu = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
errfuncfs
N = getnk(length(errfuncfs))[1]
sparse(Δw)
u = Fun(S, sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) \ errfuncfs)
# Create arrays of (x,y) points
n = 500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = OrthogonalPolynomialFamilies.weight(S, x[i], y[j]) * val
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
savefig("experiments/images/solution-poisson.png")
#=
Variable coefficient Helmholtz
Solution of Δu + k²vu = f
=#
N = 197
k = 100
S = D(1.0, 1.0)
f = Fun((x,y)->exp(x)*x*(1-x^2-y^2), S); resizecoeffs!(f, N)
fcfs = f.coefficients
v = Fun((x,y)->(1 - (3(x-1)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+3)
T = transformparamsoperator(D(0.0, 0.0), S, N+3)
Tw = transformparamsoperator(S, D(0.0, 0.0), N, weighted=true)
A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
u = Fun(S, A \ fcfs); ucfs = PseudoBlockArray(u.coefficients, [i+1 for i=0:N])
ucfs
unorms = [norm(view(ucfs, Block(i+1))) for i = 0:N]
# Create arrays of (x,y) points
n = 1500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = OrthogonalPolynomialFamilies.weight(S, x[i], y[j]) * val
    end
    @show i
end
sln
PyPlot.clf()
w, h = PyPlot.figaspect(2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
PyPlot.savefig("experiments/images/solution-helmholtz-k=$k-n=$n.png") # Convert to png to pdf

#=
Biharmonic
Solution of Δ²u = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
N = getnk(length(errfuncfs22))[1]
u = Fun(D(2.0, 2.0), sparse(bihar[1:getopindex(N,N), 1:getopindex(N,N)]) \ errfuncfs22)
# Create arrays of (x,y) points
n = 500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = OrthogonalPolynomialFamilies.weight(D(2.0, 2.0), x[i], y[j]) * evalu(u, [x[i]; y[j]])
        sln[j,i] = val
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
PyPlot.savefig("experiments/images/solution-biharmonic.png")


#=
Plotting the norms of each block of coeffs for solutions for different RHSs
=#
function getsolutionblocknorms(N, S, A, f1, f2, f3, f4)
    u1 = Fun(S, sparse(A) \ f1)
    u2 = Fun(S, sparse(A) \ f2)
    u3 = Fun(S, sparse(A) \ f3)
    u4 = Fun(S, sparse(A) \ f4)
    u1coeffs = PseudoBlockArray(u1.coefficients, [i+1 for i=0:N])
    u2coeffs = PseudoBlockArray(u2.coefficients, [i+1 for i=0:N])
    u3coeffs = PseudoBlockArray(u3.coefficients, [i+1 for i=0:N])
    u4coeffs = PseudoBlockArray(u4.coefficients, [i+1 for i=0:N])
    u1norms = zeros(N+1)
    u2norms = zeros(N+1)
    u3norms = zeros(N+1)
    u4norms = zeros(N+1)
    for i = 1:N+1
        u1norms[i] = norm(view(u1coeffs, Block(i)))
        u2norms[i] = norm(view(u2coeffs, Block(i)))
        u3norms[i] = norm(view(u3coeffs, Block(i)))
        u4norms[i] = norm(view(u4coeffs, Block(i)))
    end
    u1norms, u2norms, u3norms, u4norms
end

# solutionblocknorms - poisson
N = nblocks(Δw)[1] - 1
S = D(1.0, 1.0)
f1 = Fun((x,y)->1.0, S)
f2 = Fun((x,y)->y^2 - 1, S)
f3 = Fun((x,y)->x^2*(1-x^2-y^2)^2, S)
# f4 = Fun((x,y)->exp(-1000((x-0.5)^2+(y-0.5)^2)), S, 2*getopindex(N,N))
# f4cfs = f4.coefficients[1:getopindex(N,N)]
# save("experiments/saved/f4cfs.jld", "f4cfs", f4cfs)
f4cfs = load("experiments/saved/f4cfs.jld", "f4cfs")
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S, Δw, resizecoeffs!(f1, N),
                resizecoeffs!(f2, N), resizecoeffs!(f3, N), f4cfs)
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y) = y^2 - 1")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y) = x^2 * (1-x^2-y^2)^2")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/images/solutionblocknorms-poisson.pdf")

# solutionblocknorms - helmholtz
N = 197
k = 20
S = D(1.0, 1.0)
v = Fun((x,y)->(1 - (3(x-1)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+3)
T = transformparamsoperator(D(0.0, 0.0), S, N+3)
Tw = transformparamsoperator(S, D(0.0, 0.0), N, weighted=true)
A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
f1 = Fun((x,y)->1.0, S); 1.0
f2 = Fun((x,y)->y^2 - 1, S); 1.0
f3 = Fun((x,y)->x^2*(1-x^2-y^2)^2, S); 1.0
f4cfs = load("experiments/saved/f4cfs.jld", "f4cfs")
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S, A, resizecoeffs!(f1, N),
                resizecoeffs!(f2, N), resizecoeffs!(f3, N), f4cfs[1:sum(1:N+1)])
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y) = y^2 - 1")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y) = x^2 * (1-x^2-y^2)^2")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/images/solutionblocknorms-helmholtz-k=$k.pdf")

# solutionblocknorms - biharmonic
N = 200
S = D(2.0, 2.0)
f1 = Fun((x,y)->1.0, S)
f2 = Fun((x,y)->y^2 - 1, S)
f3 = Fun((x,y)->x^2*(1-x^2-y^2)^2, S)
# f4 = Fun((x,y)->exp(-1000((x-0.2)^2+(y-0.2)^2)), S, 10)
f4cfs11 = load("experiments/saved/f4cfs.jld", "f4cfs")
f4cfs = transformparamsoperator(D(1.0, 1.0), S, N) * f4cfs11
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S, bihar, resizecoeffs!(f1, N),
                resizecoeffs!(f2, N), resizecoeffs!(f3, N), f4cfs)
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y) = y^2 - 1")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y) = x^2 * (1-x^2-y^2)^2")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/images/solutionblocknorms-biharmonic.pdf")




#====#

# Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
#   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.

# 1) f(x,y) = -8x => u(x,y) ≡ 1
N = 1 # degree of f
c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
resizecoeffs!(f, N)
u = Fun(S, sparse(Δw) \ resizecoeffs!(f, N))
@test u(z) ≈ c # Result u(x,y) where Δ(u*w)(x,y) = f(x,y)
# plot(x->u(x,y))
# plot(y->u(x,y))

# 2) f(x,y) = 2 - 12xy - 14x^2 - 2y^2 => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(2 - 12x*y - 14x^2 - 2y^2), S)
u = Fun(S, sparse(Δw) \ resizecoeffs!(f, N))
@test u(z) ≈ U(z)
u.coefficients

# 3) f(x,y) = y*exp(x)*[2-11x-6x^2-x^3-2y^2-xy^2] => u(x,y) = y*exp(x)
U = Fun((x,y)->y*exp(x), S)
N = 12 # degree of f
m = Int((N+1)*(N+2))
f = Fun((x,y)->y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2), S, m)
u = Fun(S, sparse(Δw) \ f.coefficients[1:size(Δ)[1]])
@test u(z) ≈ U(z)
u(z) - U(z)



#====#

#=
Model problem: Helmholtz eqn:
    Δ(u(x,y)) + k²u(x,y) = f(x,y) in Ω
    u(x,y) = c on ∂Ω
where k is the wavenumber, u is the amplitude at a point (x,y) on the halfdisk,
and W(x,y) is the P^{(1,1)} weight.

i.e. with constant non-zero BCs

We frame this as u = v + c where v satisfies
    Δ(v(x,y)) + k²v(x,y) = f(x,y) - k²c
    v(x,y) = 0 on ∂Ω
=#
k, c = rand(2)

# 1) f(x,y) = k²c - 8αx + W(x,y)*α*k^2
#   => v(x,y) = W(x,y)*α
#   => u(x,y) = W(x,y)*α + c
α = rand(1)[1]
f = Fun((x,y)->(- 8α * x + x^Float64(S.a) * (1-x^2-y^2)^Float64(S.b) * α * k^2), S)
n = 5
C = transformparamsoperatorsquare(S, (S.family)(S.a-1, S.b-1), n, weighted=true)
T = transformparamsoperator((S.family)(S.a-1, S.b-1), S, n)
vc = T * C * (sparse(Δw[1:getopindex(n,n), 1:getopindex(n,n)] + k^2 * T * C) \ resizecoeffs!(f, n))
uc = vc + [c; zeros(length(vc)-1)]
u = Fun(S, uc)
@test u(z) ≈ OrthogonalPolynomialFamilies.weight(S, z) * α + c

# 2) f(x,y) = k²c + W(x,y)*k²(x+y) + (2 - 12xy - 14x² - 2y²)
#   => v(x,y) = W(x,y) * (x + y)
#   => u(x,y) = W(x,y) * (x + y) + c
U = Fun((x,y)->OrthogonalPolynomialFamilies.weight(S, x, y) * (x+y) + c, S)
n = 5 # degree of f
f = Fun((x,y)->(OrthogonalPolynomialFamilies.weight(S, x, y)*(x+y)*k^2 + 2 - 12x*y - 14x^2 - 2y^2), S)
C = transformparamsoperatorsquare(S, (S.family)(S.a-1, S.b-1), n, weighted=true)
T = transformparamsoperator((S.family)(S.a-1, S.b-1), S, n)
vc = T * C * (sparse(Δw[1:getopindex(n,n), 1:getopindex(n,n)] + k^2 * T * C) \ resizecoeffs!(f, n))
uc = vc + [c; zeros(length(vc)-1)]
u = Fun(S, uc)
@test u(z) ≈ U(z)




#====#
#=
Model problem: Heat eqn
    ∂u/∂t(x,y) = Δ(W*u)(x,y)
=>  (Back Euler)
    u1 - u0 = h * Δ(W*u1)
=>  (I - hΔ)\u0 = u1
=#
using Plots
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point
h = 1e-2
N = 15
Δ = laplacesquare(D, N)
T = increaseparamsoperator((S.family)(S.a-1, S.b-1), N)
C = convertweightedtononoperatorsquare(S, N)
u0 = Fun((x,y)->sin(x+y)*y, S, 200)
u0.coefficients

û = OrthogonalPolynomialFamilies.resizecoeffs!(u0, N)
plt = plot()
maxits = 100
res = []
nplots = 5
for it = 1:maxits
    u = T * C * (sparse(T * C - h * Δ) \ û)
    if round(nplots*it/maxits) == nplots*it/maxits
        plot!(plt, abs.(u), label="t=$it")
    end
    append!(res, maximum(abs.(u-û)))
    global û = copy(u)
end
res
plt
savefig("experiments/example2timesteppingcovnvergence")


#===#

# NOTE: call getopptseval() before this function
function recβ2(S::HalfDiskSpace{<:Any, <:Any, T}, pts, w, n, k, j) where T
    # We get the norms of the 2D OPs
    m = Int((n+2)*(n+3) / 2)
    if length(S.opnorms) < m
        getopnorms(S, m)
    end

    H1 = (S.family.H)(S.a, S.b + k - 0.5)
    H2 = (S.family.H)(S.a, S.b + k + 0.5)
    H3 = (S.family.H)(S.a, S.b + k + 1.5)
    P = (S.family.P)(S.b, S.b)
    δ = getopnorm(P) * (isodd(j) ? OrthogonalPolynomialFamilies.recγ(T, P, k+1) : OrthogonalPolynomialFamilies.recβ(T, P, k+1))

    if j == 1
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^k .* opevalatpts(H1, n-k+1, pts), w)
        val * δ / S.opnorms[getopindex(n-1, k-1)]
    elseif j == 2
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^(k+1) .* opevalatpts(H3, n-k-1, pts), w)
        val * δ / S.opnorms[getopindex(n-1, k+1)]
    elseif j == 3
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^k .* opevalatpts(H1, n-k+2, pts), w)
        val * δ / S.opnorms[getopindex(n, k-1)]
    elseif j == 4
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^(k+1) .* opevalatpts(H3, n-k, pts), w)
        val * δ / S.opnorms[getopindex(n, k+1)]
    elseif j == 5
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^k .* opevalatpts(H1, n-k+3, pts), w)
        val * δ / S.opnorms[getopindex(n+1, k-1)]
    elseif j == 6
        val = inner2(H2, opevalatpts(H2, n-k+1, pts),
                        (-pts.^2 .+ 1).^(k+1) .* opevalatpts(H3, n-k+1, pts), w)
        val * δ / S.opnorms[getopindex(n+1, k+1)]
    else
        error("Invalid entry to function")
    end
end
function resizedata2!(S::HalfDiskSpace, N)
    # N is the max degree of the OPs
    N₀ = length(S.B)
    N ≤ N₀ - 2 && return S
    pts, w = pointswithweights(S.family.H(S.a, S.b+0.5), N+1)
    for k = 0:N+1
        getopptseval(S.family.H(S.a, S.b+k+0.5), N, pts)
    end
    resize!(S.A, N + 2)
    resize!(S.B, N + 2)
    resize!(S.C, N + 2)
    resize!(S.DT, N + 2)
    getBs!(S, N, N₀)
    getCs!(S, N, N₀)
    getAs!(S, N, N₀)
    getDTs!(S, N, N₀)
    S
end


#===#
# General 2D Family/Space code

import Base: in

# R should be Float64
abstract type AbstractOrthogonalPolynomialFamily2D{R} end

struct OrthogonalPolynomialDomain2D{R} <: Domain{SVector{2,R}} end

OrthogonalPolynomialDomain2D() = OrthogonalPolynomialDomain2D{Float64}()

struct OrthogonalPolynomialSpace2D{FAM, R, FA, F} <: Space{OrthogonalPolynomialDomain2D{R}, R}
    family::FAM # Pointer back to the family
    a::R # OP parameter
    b::R # OP parameter
    opnorms::Vector{R} # squared norms
    A::Vector{SparseMatrixCSC{R}} # Storage for matrices for clenshaw
    B::Vector{SparseMatrixCSC{R}}
    C::Vector{SparseMatrixCSC{R}}
    DT::Vector{SparseMatrixCSC{R}}
end

# TODO
function OrthogonalPolynomialSpace2D(fam::AbstractOrthogonalPolynomialFamily2D{R},
                                        a::R, b::R) where R
    OrthogonalPolynomialSpace2D{typeof(fam), typeof(a)}(fam, a, b,
            Vector{R}(), Vector{SparseMatrixCSC{R}}(),
            Vector{SparseMatrixCSC{R}}(), Vector{SparseMatrixCSC{R}}(),
            Vector{SparseMatrixCSC{R}}())
end

# TODO
in(x::SVector{2}, D::OrthogonalPolynomialDomain2D) =
    (0 ≤ x[1] ≤ 1 && -sqrt(1-x[1]^2) ≤ x[2] ≤ sqrt(1-x[1]^2))

spacescompatible(A::OrthogonalPolynomialSpace2D, B::OrthogonalPolynomialSpace2D) =
    (A.a == B.a && A.b == B.b)

# R should be Float64
struct OrthogonalPolynomialFamily2D{R, IN, FN, OPF} <: AbstractOrthogonalPolynomialFamily2D{R}
    α::IN
    β::IN
    γ::IN
    ρ::Fun
    w1a::Fun
    w1b::Fun
    w2::Fun
    H::OPF
    P::OPF
    spaces::Dict{NTuple{2,R}, OrthogonalPolynomialSpace2D}
end

function OrthogonalPolynomialFamily2D(α, β, γ, ρ::Fun, w1a::Fun,
                                        w1b::Fun, w2::Fun)
    R = Float64
    spaces = Dict{NTuple{2,R}, OrthogonalPolynomialSpace2D}()
    H = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w1a, w1b, ρ)
    P = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w2)
    OrthogonalPolynomialFamily2D{R, typeof(α), typeof(ρ), typeof(H)}(α, β, γ, ρ, w1a, w1b,
                                                            w2, H, P, spaces)
end

function (D::OrthogonalPolynomialFamily2D{R})(a::R, b::R) where R
    haskey(D.spaces,(a,b)) && return D.spaces[(a,b)]
    D.spaces[(a,b)] = OrthogonalPolynomialSpace2D(D, a, b)
end

(D::OrthogonalPolynomialFamily2D{R})(a, b) where R = D(convert(R,a), convert(R,b))

α = 0
β = 1
γ = 1
X = Fun(identity, α..β)
Y = Fun(identity, -γ..γ)
ρ = sqrt(1-X^2)
w1a = X
w1b = (1-X^2)
w2 = (1-Y^2)
D = OrthogonalPolynomialFamily2D(α, β, γ, ρ, w1a, w1b, w2)

R = Float64
spaces = Dict{NTuple{2,R}, OrthogonalPolynomialSpace2D}()
H = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w1a, w1b, ρ)
P = OrthogonalPolynomialFamilies.OrthogonalPolynomialFamily(w2)
OrthogonalPolynomialFamily2D{R, typeof(α), typeof(ρ), typeof(H)}(α, β, γ, ρ, w1a, w1b,
                                                        w2, H, P, spaces)
