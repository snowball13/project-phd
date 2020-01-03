using Revise
using ApproxFun
using Test, Profile, StaticArrays, LinearAlgebra, BlockArrays,
        BlockBandedMatrices, SparseArrays

using OrthogonalPolynomialFamilies
import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, getopindex, getnk, resizecoeffs!,
                    transformparamsoperator, weightedpartialoperatorx,
                    weightedpartialoperatory, partialoperatorx, partialoperatory,
                    laplaceoperator, derivopevalatpts, getderivopptseval, getopnorms,
                    getopnorm, operatorclenshaw, weight, biharmonicoperator,
                    getPspace, getRspace, differentiatespacex, differentiatespacey,
                    differentiateweightedspacex, differentiateweightedspacey

using SpecialFunctions
using JLD

# TODO: Helmholtz, Biharmonic on disk slice
#       solutionblocknorms for poisson, helmholtz and biharmonic on diskslice


# Slice
α, β = 0.2, 0.8
D = DiskSliceFamily(α, β)
a, b, c = 1.0, 1.0, 1.0
S = D(a, b, c)

# Half Disk
D = DiskSliceFamily(0.0)
a, b = 1.0, 1.0
S = D(a, b)

# # Old HD
# DD = HalfDiskFamily(); SS = DD(a,b)
# S = D(2.0, 2.0, 2.0)
# N = 7
# biharmonicoperator(S, N)

# Spherical Cap
α = 0.5
F = SphericalCapFamily(α)
a, b = 1.0, 1.0
S = SphericalCapSpace(a, b)

#============#

# Example: p-FEM

using PyPlot
rhoval(x) = sqrt(1-x^2)
dxrhoval(x) = -x/sqrt(1-x^2)
d2xrhoval(x) = - (1-x^2)^(-0.5) - x^2 * (1-x^2)^(-1.5)
function dxweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
    D = S.family
    if D.nparams == 2
        a, b = S.params
        ret = a * weight(D(a-1, b), x, y)
        ret += 2 * rhoval(x) * dxrhoval(x) * b * weight(D(a, b-1), x, y)
        T(ret)
    else
        a, b, c = S.params
        ret = -a * weight(D(a-1, b, c), x, y)
        ret += b * weight(D(a, b-1, c), x, y)
        ret += 2 * rhoval(x) * dxrhoval(x) * c * weight(D(a, b, c-1), x, y)
        T(ret)
    end
end
dxweight(S::DiskSliceSpace, z) = dxweight(S, z[1], z[2])
function dyweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
    ret = - 2 * S.params[end] * y * weight(differentiateweightedspacey(S), x, y)
    T(ret)
end
dyweight(S::DiskSliceSpace, z) = dyweight(S, z[1], z[2])
function d2xweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
    D = S.family
    if D.nparams == 2
        a, b = S.params
        ret1 = a * ((a - 1) * weight(D(a-2, b), x, y)
                                + 2 * rhoval(x) * dxrhoval(x) * b * weight(D(a-1, b-1), x, y))
        ret2 = (2 * rhoval(x) * dxrhoval(x) * b * (a * weight(D(a-1, b-1), x, y)
                                    + 2 * rhoval(x) * dxrhoval(x) * (b-1) * weight(D(a, b-2), x, y))
                    + 2 * b * (rhoval(x) * d2xrhoval(x) + dxrhoval(x)^2) * weight(D(a, b-1), x, y))
        T(ret1 + ret2)
    else
        a, b, c = S.params
        ret1 = a * ((a-1)*weight(D(a-2,b,c),x,y) - b*weight(D(a-1,b-1,c),x,y) - 2*c*rhoval(x)*dxrhoval(x)*weight(D(a-1,b,c-1),x,y))
        ret2 = b * (-a*weight(D(a-1,b-1,c),x,y) + (b-1)*weight(D(a,b-2,c),x,y) + 2*c*rhoval(x)*dxrhoval(x)*weight(D(a,b-1,c-1),x,y))
        ret3 = 2*c*rhoval(x)*dxrhoval(x) * (-a*weight(D(a-1,b,c-1),x,y) + b*weight(D(a,b-1,c-1),x,y)
                                                    + 2*(c-1)*rhoval(x)*dxrhoval(x)*weight(D(a,b,c-2),x,y))
        ret4 = 2*c*(dxrhoval(x)^2 + rhoval(x)*d2xrhoval(x))*weight(D(a,b,c-1),x,y)
        T(ret1+ret2+ret3+ret4)
    end
end
d2xweight(S::DiskSliceSpace, z) = d2xweight(S, z[1], z[2])
function d2yweight(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, x, y) where T
    D = S.family
    ret = - 2 * S.params[end] * weight(differentiateweightedspacey(S), x, y)
    ret += 4 * S.params[end] * (S.params[end] - 1) * y^2 * weight(differentiateweightedspacey(differentiateweightedspacey(S)), x, y)
    T(ret)
end
d2yweight(S::DiskSliceSpace, z) = d2yweight(S, z[1], z[2])
function getopnormmatrix(S::DiskSliceSpace{<:Any, <:Any, T, <:Any}, N) where T
    OrthogonalPolynomialFamilies.resizedata!(S, N)
    M = BandedBlockBandedMatrix(
            Zeros{T}(sum(1:N+1), sum(1:(N+1))), (1:N+1, 1:N+1), (0, 0), (0, 0))
    for b = 1:N+1
        view(M, Block(b, b)) .= Diagonal(S.opnorms[1:b])
    end
    M
end
evalu(u, z) = isindomain(z, u.space.family) ? u(z) : NaN
isindomain(x, D::DiskSliceFamily) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])

α, β = 0.2, 0.8
D = DiskSliceFamily(α, β)
a, b, c = 1.0, 1.0, 1.0
S = D(a, b, c)
x, y = 0.4, -0.765; z = [x; y]
@test isindomain(z, D)

# Ex1) u = W^{1,1}(x,y)
uexact = (x,y)->weight(S, x, y) * 1.0
f = Fun((x,y)->-(d2xweight(S, x, y) + d2yweight(S, x, y)), S, 10)
p = 4 # degree
resizecoeffs!(f, p)

Λ0 = getopnormmatrix(D(a-1, b-1, c-1), p+3)
Λ1 = getopnormmatrix(S, p+1)
WX = weightedpartialoperatorx(S, p)
WY = weightedpartialoperatory(S, p)
TW = transformparamsoperator(D(a, b, c-1), D(a-1, b-1, c-1), p+1, weighted=true)
A = (transpose(WX) * Λ0 * WX
        + transpose(TW * WY) * Λ0 * TW * WY) # (element) stiffness matrix
f̃ = Λ1[1:length(f.coefficients), 1:length(f.coefficients)] * f.coefficients
U = A \ f̃
u = Fun(S, U)
# Test for random point in Ω
weight(S, z) * u(z) - uexact(z...)

# Ex2) u = W^{1,1}(x,y) * cos(x) * y
uexact = (x,y)->weight(S, x, y) * cos(x) * y
# f = Fun((x,y)->-(cos(x) * y * (-10*weight(D(a, b-1), x, y) - weight(S, x, y))
#                  + sin(x) * y * (2*weight(D(a+1,b-1), x, y) - weight(D(a-1, b), x, y))), S)
f = Fun((x,y)->-(cos(x)*y*(d2xweight(S,x,y) + d2yweight(S,x,y) - weight(S,x,y))
                + 2*(cos(x)*dyweight(S,x,y) - y*sin(x)*dxweight(S,x,y))), S, 200)
p = 15 # degree
f.coefficients
resizecoeffs!(f, p)

Λ0 = getopnormmatrix(D(a-1, b-1, c-1), p+3)
Λ1 = getopnormmatrix(S, p+1)
WX = weightedpartialoperatorx(S, p)
WY = weightedpartialoperatory(S, p)
TW = transformparamsoperator(D(a, b, c-1), D(a-1, b-1, c-1), p+1, weighted=true)
A = transpose(WX) * Λ0 * WX + transpose(TW * WY) * Λ0 * TW * WY
f̃ = Λ1[1:length(f.coefficients), 1:length(f.coefficients)] * f.coefficients
U = A \ f̃
u = Fun(S, U)
weight(S, z) * u(z) - uexact(z...)
n = 500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = weight(S, [x[i]; y[j]]) * evalu(u, [x[i]; y[j]])
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
savefig("experiments/images/solution-pfem-poisson-diskslice-alpha=$α-beta=$β-f=Wycosx.png")

# Ex 3) Solution of Δu = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
p = 100
using SpecialFunctions
for n = 25:5:p
    @show n
    OrthogonalPolynomialFamilies.resizedata!(S, n)
end
f = Fun((x,y)->-(1 + erf(5*(1 - 10*((x - 0.5)^2 + y^2)))), S, p*(p+1))
f.coefficients
resizecoeffs!(f, p)
# errfuncfs = load("experiments/saved/errfuncfs-diskslice.jld", "erfuncfs")
# N = getnk(length(errfuncfs))[1]
# p = N # degree of p-fem method
Λ0 = getopnormmatrix(D(a-1, b-1, c-1), p+3)
Λ1 = getopnormmatrix(S, p+1)
WX = weightedpartialoperatorx(S, p)
WY = weightedpartialoperatory(S, p)
TW = transformparamsoperator(D(a, b, c-1), D(a-1, b-1, c-1), p+1, weighted=true)
WXt = weightedpartialoperatorx(S, p, transposed=true)
WYt = weightedpartialoperatory(S, p, transposed=true)
TWt = transformparamsoperator(D(a, b, c-1), D(a-1, b-1, c-1),
                                p+1, weighted=true, transposed=true)
A = WXt * Λ0 * WX + WYt * TWt * Λ0 * TW * WY # (element) stiffness matrix
f̃ = Λ1[1:length(f.coefficients), 1:length(f.coefficients)] * f.coefficients
U = A \ f̃
u = Fun(S, U)
# Check solution
weight(S, z) * u(z)
# Plot
n = 500
x = LinRange(0, 1, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = evalu(u, [x[i]; y[j]])
        sln[j,i] = weight(S, [x[i]; y[j]]) * evalu(u, [x[i]; y[j]])
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.0)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
savefig("experiments/images/solution-pfem-poisson-diskslice-alpha=$α-beta=$β-f=errfun.png")


#===========#

setprecision(800)
α, β = 0.2, 0.8
D = DiskSliceFamily(α, β)
a, b, c = 1.0, 1.0, 1.0
S = D(a, b, c)
x, y = 0.4, -0.765; z = [x; y]
@test isindomain(z, D)
Δw = laplaceoperator(S, S, 990; weighted=true)
# save("experiments/saved/laplacian-w11-array-diskslice-alpha=$α-beta=$β-N=310.jld", "Lw11", sparse(Δw))
# save("experiments/saved/biharmonic-array-diskslice-alpha=$α-beta=$β.jld", "bihar", sparse(bihar))
bihar = biharmonicoperator(D(2.0, 2.0, 2.0), 200)
S
save("experiments/saved/biharmonic-array-diskslice-alpha=$α-beta=$β.jld", "bihar", sparse(bihar))
N = 25
k = 200
v = Fun((x,y)->(1 - (3(x-β)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+4)
T = transformparamsoperator(D(0.0, 0.0, 0.0), S, N+4)
Tw = transformparamsoperator(S, D(0.0, 0.0, 0.0), N, weighted=true)
A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
# save("experiments/saved/helmholtz-array-sparsity-diskslice-alpha=$α-beta=$β.jld", "A", A)
errfun = Fun((x,y)->-(1 + erf(5*(1 - 10*((x - 0.5)^2 + y^2)))), S, 200*(200+1))
errfuncfs = errfun.coefficients
save("experiments/saved/errfuncfs-diskslice-alpha=$α-beta=$β.jld", "errfuncfs", errfuncfs)
errfun222 = Fun((x,y)->-(1 + erf(5*(1 - 10*((x - 0.5)^2 + y^2)))),
                D(2.0, 2.0, 2.0), 100*(100+1))
errfuncfs222 = errfun22.coefficients
save("experiments/saved/errfuncfs222-diskslice-alpha=$α-beta=$β.jld", "errfuncfs222", errfuncfs222)


#=
Sparsity of laplace operator
=#
using PyPlot
D
S
# a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
# N = 30
maxm = sum(1:20+1)
PyPlot.clf()
    PyPlot.spy(Array(sparse(Δw[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/images/sparsityoflaplacian-w11-diskslice-alpha=$α-beta=$β.pdf")
PyPlot.clf()
    PyPlot.spy(Array(sparse(bihar[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    savefig("experiments/images/sparsityofbiharmonic-diskslice-alpha=$α-beta=$β.pdf")
PyPlot.clf()
    # N = 25
    # k = 200
    # v = Fun((x,y)->(1 - (3(x-1)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+3)
    # T = transformparamsoperator(D(0.0, 0.0), S, N+3)
    # Tw = transformparamsoperator(S, D(0.0, 0.0), N, weighted=true)
    # A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
    # save("experiments/saved/helmholtz-array-sparsity.jld", "A", A)
    # A = load("experiments/saved/helmholtz-array-sparsity.jld", "A")
    PyPlot.spy(Array(sparse(A[1:maxm, 1:maxm])))
    PyPlot.axis(xmin=0, xmax=maxm, ymin=maxm, ymax=0)
    PyPlot.savefig("experiments/images/sparsityofhelmholtz-diskslice-alpha=$α-beta=$β.pdf")

#===========#

#=
Poisson
Solution of Δu = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
# errfuncfs = f.coefficients
# save("experiments/saved/errfuncfs-diskslice-alpha=$α-beta=$β.jld", "errfuncfs", errfuncfs)
N = getnk(length(errfuncfs))[1]
u = Fun(S, sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) \ errfuncfs)
# Create arrays of (x,y) points
n = 500
x = LinRange(α, β, n)
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
w, h = plt[:figaspect](2.2)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
savefig("experiments/images/solution-poisson-diskslice-alpha=$α-beta=$β.png")
#=
Variable coefficient Helmholtz
Solution of Δu + k²vu = f
=#
N = 196
k = 100
S
f = Fun((x,y)->exp(x)*(β-x)*(x-α)*(1-x^2-y^2), S); resizecoeffs!(f, N)
fcfs = f.coefficients
# v = Fun((x,y)->(1 - (3(x-β)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+4)
# T = transformparamsoperator(D(0.0, 0.0, 0.0), S, N+4)
# Tw = transformparamsoperator(S, D(0.0, 0.0, 0.0), N, weighted=true)
# A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
A
u = Fun(S, A \ fcfs); ucfs = PseudoBlockArray(u.coefficients, [i+1 for i=0:N])
ucfs
unorms = [norm(view(ucfs, Block(i+1))) for i = 0:N]
# Create arrays of (x,y) points
n = 1500
x = LinRange(α, β, n)
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
w, h = plt[:figaspect](2.2)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
PyPlot.savefig("experiments/images/solution-helmholtz-diskslice-alpha=$α-beta=$β-k=$k-n=$n.png") # Convert to png to pdf

#=
Biharmonic
Solution of Δ²u = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
N = getnk(length(errfuncfs222))[1]
u = Fun(D(2.0, 2.0, 2.0), sparse(bihar[1:getopindex(N,N), 1:getopindex(N,N)]) \ errfuncfs222)
# Create arrays of (x,y) points
n = 500
x = LinRange(α, β, n)
y = LinRange(-1, 1, n)
sln = zeros(length(x), length(y))
for i = 1:length(x)
    for j = 1:length(y)
        val = OrthogonalPolynomialFamilies.weight(D(2.0, 2.0, 2.0), x[i], y[j]) * evalu(u, [x[i]; y[j]])
        sln[j,i] = val
    end
    @show i
end
PyPlot.clf()
w, h = plt[:figaspect](2.2)
PyPlot.figure(figsize=(w,h))
PyPlot.pcolor(x, y, sln)
PyPlot.savefig("experiments/images/solution-biharmonic-diskslice-alpha=$α-beta=$β.png")


#=
Plotting the norms of each block of coeffs for solutions for different RHSs
=#
function getsolutionblocknorms(N, S, A, f1, f2, f3)#, f4)
    u1 = Fun(S, sparse(A) \ f1)
    @show "1"
    u2 = Fun(S, sparse(A) \ f2)
    @show "2"
    u3 = Fun(S, sparse(A) \ f3)
    @show "3"
    #u4 = Fun(S, sparse(A) \ f4)
    u1coeffs = PseudoBlockArray(u1.coefficients, [i+1 for i=0:N])
    u2coeffs = PseudoBlockArray(u2.coefficients, [i+1 for i=0:N])
    u3coeffs = PseudoBlockArray(u3.coefficients, [i+1 for i=0:N])
    #u4coeffs = PseudoBlockArray(u4.coefficients, [i+1 for i=0:N])
    u1norms = zeros(N+1)
    u2norms = zeros(N+1)
    u3norms = zeros(N+1)
    #u4norms = zeros(N+1)
    for i = 1:N+1
        u1norms[i] = norm(view(u1coeffs, Block(i)))
        u2norms[i] = norm(view(u2coeffs, Block(i)))
        u3norms[i] = norm(view(u3coeffs, Block(i)))
        #u4norms[i] = norm(view(u4coeffs, Block(i)))
    end
    u1norms, u2norms, u3norms#, u4norms
end

# solutionblocknorms - poisson
N = nblocks(Δw)[1] - 1
N = 310
S
f1 = Fun((x,y)->1.0, S); f1.coefficients
f2 = Fun((x,y)->(1 - D.α^2 - y^2)*(1 - D.β^2 - y^2), S); f2.coefficients
f3 = Fun((x,y)->weight(S, x, y)^3, S); f3.coefficients
u1norms, u2norms, u3norms = getsolutionblocknorms(N, S, sparse(Δw)[1:getopindex(N,N), 1:getopindex(N,N)], resizecoeffs!(f1, N),
                resizecoeffs!(f2, N), resizecoeffs!(f3, N))
# save("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-u1norms-N=$N.jld", "u1norms", u1norms)
# save("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-u2norms-N=$N.jld", "u2norms", u2norms)
# save("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-u3norms-N=$N.jld", "u3norms", u3norms)
u1norms = load("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-u1norms-N=$N.jld", "u1norms")
u2norms = load("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-u2norms-N=$N.jld", "u2norms")
u3norms = load("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-u3norms-N=$N.jld", "u3norms")
# f4 = Fun((x,y)->exp(-1000((x-0.5)^2+(y-0.5)^2)), S, 20000); f4.coefficients
# f4cfs = f4.coefficients[1:getopindex(N,N)]
# save("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-f4cfs.jld", "f4cfs", f4cfs)
f4cfs = load("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-f4cfs.jld", "f4cfs")
mm = length(f4cfs); resize!(f4cfs, sum(1:N+1)); f4cfs[mm+1:end] .= 0.0
u4 = Fun(S, sparse(Δw)[1:getopindex(N,N), 1:getopindex(N,N)] \ f4cfs); u4.coefficients
u4coeffs = PseudoBlockArray(u4.coefficients, [i+1 for i=0:N])
u4norms = zeros(N+1)
for i = 1:N+1
    u4norms[i] = norm(view(u4coeffs, Block(i)))
end
u1norms[1:400]
u1norms

using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y) = (1 - alpha^2 - y^2)(1 - beta^2 - y^2)")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y) = W{(1,1,1)}^3")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/images/solutionblocknorms-poisson-diskslice-alpha=$α-beta=$β-N=$N.pdf")

AA = sparse(Δw[1:48516, 1:48516]); 1.0
maximum(abs.((AA - Δwsmall)[1:20000, 1:20000]))
Δwsmall = load("experiments/saved/laplacian-w11-array-diskslice-alpha=$α-beta=$β-N=310.jld", "Lw11")

# solutionblocknorms - helmholtz
N = 196
k = 20
S = D(1.0, 1.0, 1.0)
# v = Fun((x,y)->(1 - (3(x-β)^2 + 5y^2)), S); V = operatorclenshaw(v, S, N+4)
# T = transformparamsoperator(D(0.0, 0.0, 0.0), S, N+4)
# Tw = transformparamsoperator(S, D(0.0, 0.0, 0.0), N, weighted=true)
# A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
# save("experiments/saved/solutionblocknorms-helmholtz-diskslice-alpha=$α-beta=$β-N=$N-k=$k.jld", "A", A)
A = load("experiments/saved/solutionblocknorms-helmholtz-diskslice-alpha=$α-beta=$β-N=$N-k=$k.jld", "A")
f1 = Fun((x,y)->1.0, S); 1.0
f2 = Fun((x,y)->y^2 - 1, S); 1.0
f3 = Fun((x,y)->weight(S, x, y)^3, S, 300); f3.coefficients
# f4 = Fun((x,y)->exp(-1000*((x-0.5)^2+(y-0.5)^2)), S, 2*getopindex(N,N))
# save("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-f4cfs.jld", "f4cfs", f4.coefficients)
f4cfs = load("experiments/saved/solutionblocknorms-diskslice-alpha=$α-beta=$β-f4cfs.jld", "f4cfs")
u1norms, u2norms, u3norms, u4norms = getsolutionblocknorms(N, S, A, resizecoeffs!(f1, N),
                resizecoeffs!(f2, N), resizecoeffs!(f3, N), f4cfs[1:sum(1:N+1)])
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y) = y^2 - 1")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y) = W{(1,1,1)}^3")
Plots.plot!(u4norms, line=(3, :dot), label = "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")
Plots.savefig("experiments/images/solutionblocknorms-helmholtz-diskslice-alpha=$α-beta=$β-N=$N-k=$k.pdf")

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





#===========#

# Example: trapezium

B = BigFloat
precision(B)
setprecision(800)

# DT = TrapeziumFamily()
a, b, c, d = 1.0, 1.0, 1.0, 1.0
S = DT(a, b, c, d)

using PyPlot
evalu(u, z) = isindomain(z, u.space.family) ? u(z) : NaN
isindomain(x, D::TrapeziumFamily) = D.α ≤ x[1] ≤ D.β && D.γ*D.ρ(x[1]) ≤ x[2] ≤ D.δ*D.ρ(x[1])

#=
Poisson
Solution of Δu = f, where f(x,y) = 1 + erf(5(1 - 10((x - 0.5)^2 + (y)^2)))
=#
N = 200
Δw = laplaceoperator(S, S, N; weighted=true)
ξ = DT.slope
save("experiments/saved/laplacian-w11-array-trapezium-xi=$ξ-N=200.jld", "Lw11", sparse(Δw))
errfun = Fun((x,y)->1 + erf(5*(1 - 10*((x - 0.5)^2 + (y)^2))), S, N*(N+1))
errfuncfs = errfun.coefficients
u = Fun(S, sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) \ errfuncfs)
# Create arrays of (x,y) points
n = 500
x = LinRange(S.family.α, S.family.β, n)
y = LinRange(S.family.γ, S.family.δ, n)
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
savefig("experiments/images/solution-trapezium-poisson.png")
#=
Variable coefficient Helmholtz
Solution of Δu + k²vu = f
=#
N = 197
k = 100
Δw = laplaceoperator(S, S, N; weighted=true)
# save("experiments/saved/laplacian-w11-trapezium-array.jld", "Lw11", sparse(Δw))
f = Fun((x,y)->exp(x)*weight(S, x, y), S);
resizecoeffs!(f, N)
fcfs = f.coefficients
v = Fun((x,y)->(1 - (3x^2 + 5(y-1)^2)), S)
V = operatorclenshaw(v, S, N+4)
T = transformparamsoperator(D(S.params .- 1), S, N+4)
Tw = transformparamsoperator(S, D(S.params .- 1), N, weighted=true)
A = sparse(Δw[1:getopindex(N,N), 1:getopindex(N,N)]) + k^2 * (V * sparse(T) * sparse(Tw))[1:getopindex(N,N), 1:getopindex(N,N)]
# save("experiments/saved/trapezium-helholtz-operator-topleft.jld", "A", A)
u = Fun(S, A \ fcfs); ucfs = PseudoBlockArray(u.coefficients, [i+1 for i=0:N])
ucfs
unorms = [norm(view(ucfs, Block(i+1))) for i = 0:N]
# Create arrays of (x,y) points
n = 1500
x = LinRange(S.family.α, S.family.β, n)
y = LinRange(S.family.γ, S.family.δ, n)
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
# Convert to png to pdf
PyPlot.savefig("experiments/images/solution-trapezium-topleftcorner-helmholtz-k=$k-n=$n.png")

#============#

# NOTE: Explicit construction of the Rn OPs recurrence coeffs
# TRAPEZIUM


B = BigFloat
setprecision(800)
DT = TrapeziumFamily()
a, b, c, d = 1.0, 1.0, 1.0, 1.0
S = DT(a, b, c, d)

# N = 500
# @show "begin initialisation"
# R0 = DT.R(B(0.0), B(0.0), B(1.0))
# OrthogonalPolynomialFamilies.resizedata!(R0, N+1)
# save("experiments/saved/trapezium-R001-rec-coeffs-a-N=500.jld", "a", R0.a)
# save("experiments/saved/trapezium-R001-rec-coeffs-b-N=500.jld", "b", R0.b)
# @show "end initialisation"

S = DT(0.0, 0.0, 0.0, 0.0)
let
    N = 500
    R = DT.R
    for k = 0:Int(N/2)
        @show N, k
        R0 = R(S.params[1], S.params[2], B(1.0)+k)
        if k == 0
            resize!(R0.a, N+1); resize!(R0.b, N+1)
            R0.a[:] = load("experiments/saved/trapezium-R001-rec-coeffs-a-N=500.jld", "a")[1:N+1]
            R0.b[:] = load("experiments/saved/trapezium-R001-rec-coeffs-b-N=500.jld", "b")[1:N+1]
        end
        # wanted coeffs
        chivec = zeros(N)
        pt = 1.0 / DT.slope
        n = 0

        chivec[n+1] = (pt - R0.a[n+1]) / R0.b[n+1]
        for n = 1:N-1
            chivec[n+1] = (pt - R0.a[n+1] - R0.b[n] / chivec[n]) / R0.b[n+1]
        end
        @show k, maximum(abs.(chivec)), minimum(abs.(chivec))
        R1 = R(S.params[1], S.params[2], B(1.0)+k+1)
        resize!(R1.a, N-1); resize!(R1.b, N-1)
        n = 0
        R1.b[n+1] = (R0.b[n+1]
                        * sqrt(pt - R0.a[n+2] - R00.b[n+1] / chivec[n+1])
                        / sqrt(pt - R0.a[n+1]))
        R1.a[n+1] = R0.a[n+1] - (R00.b[n+1] / chivec[n+1])
        for n = 1:N-2
            R1.b[n+1] = (R0.b[n+1]
                            * sqrt(pt - R00.a[n+2] - R0.b[n+1] / chivec[n+1])
                            / sqrt(pt - R00.a[n+1] - R0.b[n] / chivec[n]))
            R1.a[n+1] = R0.a[n+1] + (R0.b[n] / chivec[n]) - (R0.b[n+1] / chivec[n+1])
        end
        N = N - 1
    end
end

#============#

# NOTE: Explicit construction of the Rn OPs recurrence coeffs
# Recall that in D.R, the parameter c is already halved (as the weight factor is ρ^2)
# DISK SLICE

B = BigFloat
α, β = 0.2, 0.8
    D = DiskSliceFamily(α, β)
    a, b, c = 1.0, 1.0, 1.0
    S = D(a, b, c)
    x, y = 0.4, -0.765; z = [x; y]
    D
getreccoeffsR(D)
getreccoeffsP(D)

Δw = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-squaresparse-N=990.jld", "Lw11")

A = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-A-N=990.jld", "A")

N = 990
Asmall = sparse(A)[1:getopindex(N,N), 1:getopindex(N,N)]
ff = Fun((x,y)->y*cos(x), differentiateweightedspacex(S)); ff.coefficients
cfs = Asmall * resizecoeffs!(ff, N)
maximum(abs.(cfs[935:940]))
cfs[900:950]
ffdx = Fun(S, cfs[1:1000])
ffdx(z) + y * sin(x)

DD = DiskSliceFamily()
SS = DD(1.0, 1.0)
f = Fun((x,y)->y*cos(x), differentiateweightedspacex(SS)); f.coefficients
f.coefficients
N = 200
partialoperatorx(differentiateweightedspacex(SS), N+2)





function getreccoeffsP(D::DiskSliceFamily)
    B = BigFloat
    N = 1000
    cs = (0.0, 1.0)
    α, β = D.α, D.β
    for c in cs
        P = D.P(B(c), B(c))
        resize!(P.a, N); resize!(P.b, N)
        if c == 0.0
            P.a[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-P00-rec-coeffs-a-N=1000-BF.jld", "a")[1:N]
            P.b[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-P00-rec-coeffs-b-N=1000-BF.jld", "b")[1:N]
        else
            P.a[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-P11-rec-coeffs-a-N=1000-BF.jld", "a")[1:N]
            P.b[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-P11-rec-coeffs-b-N=1000-BF.jld", "b")[1:N]
        end
        resize!(P.ops, 2); resize!(P.weight, 1)
        P.weight[1] = prod(P.family.factors.^(P.params))
        X = Fun(identity, domain(P.weight[1]))
        P.ops[1] = Fun(1/sqrt(sum(P.weight[1])),space(X))
        v = X * P.ops[1] - P.a[1] * P.ops[1]
        P.ops[2] = v / P.b[1]
        for k = 2:length(P.a)
            v = X * P.ops[2] - P.b[k-1] * P.ops[1] - P.a[k] * P.ops[2]
            P.ops[1] = P.ops[2]
            P.ops[2] = v / P.b[k]
        end
    end
    D
end
function getreccoeffsR(D::DiskSliceFamily)
    R = D.R
    B = BigFloat
    # setprecision(800)
    α, β = D.α, D.β
    X = Fun(B(α)..β)
    R̃ = OrthogonalPolynomialFamily(β-X, X-α, 1-X, 1+X)
    abvec = ((0.0, 0.0), (1.0, 1.0))
    for ab in abvec
        N = 2000
        a, b = ab
        R00 = R(B(a), B(b), B(0.5))
        resize!(R00.a, N+1); resize!(R00.b, N+1)
        if a == 0.0
            R00.a[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-R000p5-rec-coeffs-a-N=2003-BF.jld", "a")[1:N+1]
            R00.b[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-R000p5-rec-coeffs-b-N=2003-BF.jld", "b")[1:N+1]
        elseif a == 1.0
            R00.a[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-R110p5-rec-coeffs-a-N=2003-BF.jld", "a")[1:N+1]
            R00.b[:] = load("experiments/saved/diskslice-alpha=$α-beta=$β-R110p5-rec-coeffs-b-N=2003-BF.jld", "b")[1:N+1]
        else
            @show "cant do"
        end
        for k = 0:(Int(N/2)-1)
            if k % 100 == 0
                @show k
            end
            # interim coeffs
            R00 = R(B(a), B(b), B(0.5)+k)
            chivec = zeros(B, N+1)
            pt = 1.0
            n = 0
            chivec[n+1] = (pt - R00.a[n+1]) / R00.b[n+1]
            for n = 1:N
                chivec[n+1] = (pt - R00.a[n+1] - R00.b[n] / chivec[n]) / R00.b[n+1]
            end
            R10 = R̃(B(a), B(b), B(0.5)+k+1, B(0.5)+k)
            resize!(R10.a, N); resize!(R10.b, N)
            n = 0
            R10.b[n+1] = (R00.b[n+1]
                            * sqrt(pt - R00.a[n+2] - R00.b[n+1] / chivec[n+1])
                            / sqrt(pt - R00.a[n+1]))
            R10.a[n+1] = B(R00.a[n+1]) - (R00.b[n+1] / chivec[n+1])
            for n = 1:N-1
                R10.b[n+1] = (R00.b[n+1]
                                * sqrt(pt - R00.a[n+2] - R00.b[n+1] / chivec[n+1])
                                / sqrt(pt - R00.a[n+1] - R00.b[n] / chivec[n]))
                R10.a[n+1] = R00.a[n+1] + (R00.b[n] / chivec[n]) - (R00.b[n+1] / chivec[n+1])
            end
            # wanted coeffs
            chivec = zeros(B, N)
            pt = -1.0
            n = 0
            chivec[n+1] = (pt - R10.a[n+1]) / R10.b[n+1]
            for n = 1:N-1
                chivec[n+1] = (pt - R10.a[n+1] - R10.b[n] / chivec[n]) / R10.b[n+1]
            end
            R11 = R(B(a), B(b), B(0.5)+k+1)
            resize!(R11.a, N-1); resize!(R11.b, N-1)
            n = 0
            R11.b[n+1] = (R10.b[n+1]
                            * sqrt(abs(pt - R10.a[n+2] - R10.b[n+1] / chivec[n+1]))
                            / sqrt(abs(pt - R10.a[n+1])))
            R11.a[n+1] = R10.a[n+1] - (R10.b[n+1] / chivec[n+1])
            for n = 1:N-2
                R11.b[n+1] = (R10.b[n+1]
                                * sqrt(abs(pt - R10.a[n+2] - R10.b[n+1] / chivec[n+1]))
                                / sqrt(abs(pt - R10.a[n+1] - R10.b[n] / chivec[n])))
                R11.a[n+1] = R10.a[n+1] + (R10.b[n] / chivec[n]) - (R10.b[n+1] / chivec[n+1])
            end
            N = N - 2
        end
    end
    # setprecision(256)
    D
end

#
# # Laplacian for N = 990
# N = 990
# α, β = 0.2, 0.8
#     D = DiskSliceFamily(α, β)
#     a, b, c = 1.0, 1.0, 1.0
#     S = D(a, b, c)
#     x, y = 0.4, -0.765; z = [x; y]
#     getreccoeffsR(D)
#     getreccoeffsP(D)
# D
# A = partialoperatorx(differentiateweightedspacex(S), N+D.nparams)
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-A-N=990.jld", "A", A)
# @show "laplaceoperator", "1 of 6 done"
# B = weightedpartialoperatorx(S, N)
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-B-N=990.jld", "B", B)
# @show "laplaceoperator", "2 of 6 done"
# C = transformparamsoperator(differentiatespacey(D(S.params .- 1)), S, N+D.nparams-1)
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-C-N=990.jld", "C", C)
# @show "laplaceoperator", "3 of 6 done"
# E = partialoperatory(D(S.params .- 1), N+D.nparams)
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-E-N=990.jld", "E", E)
# @show "laplaceoperator", "4 of 6 done"
# F = transformparamsoperator(differentiateweightedspacey(S), D(S.params .- 1), N+1, weighted=true)
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-F-N=990.jld", "F", F)
# @show "laplaceoperator", "5 of 6 done"
# G = weightedpartialoperatory(S, N)
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-G-N=990.jld", "G", G)
# @show "laplaceoperator", "6 of 6 done"
# L1 = A * B
# L2 = sparse(C) * sparse(E) * sparse(F) * sparse(G)
# L = sparse(L1) + L2
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-nonsquaresparse-N=990.jld", "L", L)
# # L = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-nonsquaresparse-N=990.jld", "L")
# m = sum(1:(N+1))
# Δw = BandedBlockBandedMatrix(L[1:m, 1:m], (1:N+1, 1:N+1), (2,2), (2,2))
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-squaresparse-N=990.jld", "Lw11", sparse(Δw))
# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-laplace-mat-square-N=990.jld", "Lw11", Δw)
#
#


# D = DiskSliceFamily(α, β)
#
# B = BigFloat
# N = 1000
# prec = precision(B)
# setprecision(800)
# P0 = D.P(B(0.0), B(0.0)); P1 = D.P(B(1.0), B(1.0))
# OrthogonalPolynomialFamilies.resizedata!(P0, N)
# OrthogonalPolynomialFamilies.resizedata!(P1, N)
# setprecision(prec)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-P00-rec-coeffs-a-N=1000-BF.jld", "a", P0.a)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-P00-rec-coeffs-b-N=1000-BF.jld", "b", P0.b)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-P11-rec-coeffs-a-N=1000-BF.jld", "a", P1.a)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-P11-rec-coeffs-b-N=1000-BF.jld", "b", P1.b)
#
#
# N = 2003
# @show "begin initialisation"
# R00 = D.R(B(1.0), B(1.0), B(0.5))
# OrthogonalPolynomialFamilies.resizedata!(R00, N+1)
# # xx = Fun(identity, domain(R00.weight))
# #     R00.ops[1] = Fun(1/sqrt(sum(R00.weight)),space(x))
# #     vv = xx * R00.ops[1] - R00.a[1] * R00.ops[1]
# #     R00.ops[2] = vv / R00.b[1]
# #     for k = 2:1954
# #         vv = xx * R00.ops[2] - R00.b[k-1] * R00.ops[1] - R00.a[k] * R00.ops[2]
# #         R00.ops[1] = R00.ops[2]
# #         R00.ops[2] = vv / R00.b[k]
# #     end
# save("experiments/saved/diskslice-alpha=$α-beta=$β-R110p5-rec-coeffs-a-N=2003-BF.jld", "a", R00.a)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-R110p5-rec-coeffs-b-N=2003-BF.jld", "b", R00.b)
# @show "end initialisation"
# @show "begin initialisation"
# R00 = D.R(B(0.0), B(0.0), B(0.5))
# OrthogonalPolynomialFamilies.resizedata!(R00, N+1)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-R000p5-rec-coeffs-a-N=2003-BF.jld", "a", R00.a)
# save("experiments/saved/diskslice-alpha=$α-beta=$β-R000p5-rec-coeffs-b-N=2003-BF.jld", "b", R00.b)
# @show "end initialisation"


# save("experiments/saved/diskslice-alpha=0.2-beta=0.8-H111-clenshawmats.jld",
#         "DT", S.DT, "A", S.A, "B", S.B, "C", S.C)
# d = load("experiments/saved/diskslice-alpha=0.2-beta=0.8-H111-clenshawmats.jld")
# resize!(S.DT, length(d["DT"])); S.DT[:] = d["DT"]
# resize!(S.A, length(d["A"])); S.A[:] = d["A"]
# resize!(S.B, length(d["B"])); S.B[:] = d["B"]
# resize!(S.C, length(d["C"])); S.C[:] = d["C"]
