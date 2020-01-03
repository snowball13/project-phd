using OrthogonalPolynomialFamilies, ApproxFun
using StaticArrays, LinearAlgebra, Test, Profile, BlockArrays,
        BlockBandedMatrices, SparseArrays, Plots



a, b = 0.5, 0.5
x, y = 0.5, 0.3; x^2 + y^2 < 1
D = HalfDiskFamily(); S = D(a, b)

# Model Problem: Δu(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
function getpartialoperatorx2(S, N)
    (OrthogonalPolynomialFamilies.getpartialoperatorx(OrthogonalPolynomialFamilies.differentiatespacex(S), N-1)
        * OrthogonalPolynomialFamilies.getpartialoperatorx(S, N))
end
function getpartialoperatory2(S, N)
    (OrthogonalPolynomialFamilies.getpartialoperatory(OrthogonalPolynomialFamilies.differentiatespacey(S), N-1)
        * OrthogonalPolynomialFamilies.getpartialoperatory(S, N))
end
differentiatespacex2(S) =
    OrthogonalPolynomialFamilies.differentiatespacex(
            OrthogonalPolynomialFamilies.differentiatespacex(S))
differentiatespacey2(S) =
    OrthogonalPolynomialFamilies.differentiatespacey(
            OrthogonalPolynomialFamilies.differentiatespacey(S))
function resizecoeffs!(cfs::AbstractVector, N)
    m̃ = length(cfs)
    m = Int((N+1)*(N+2)/2)
    resize!(cfs, m)
    if m̃ < m
        cfs[m̃+1:end] .= 0.0
    end
    cfs
end

x, y = 0.2, 0.3; z = [x;y]
N = 3











N = 3
x, y = 0.2, 0.3; z = [x;y]
fx = Fun((x,y) -> 1-3x^2-y^2, OrthogonalPolynomialFamilies.differentiatespacex(S))
Dx = OrthogonalPolynomialFamilies.getpartialoperatorx(S, N)
uc = Dx \ resizecoeffs!(fx.coefficients, N-1)


# Heat eqn with back euler
h = 1e-3
A = Diagonal(ones(M)) - h * getpartialoperatorx2(N)
u = copy(u0)
for it=1:maxits
    u = A\u
end











Profile.clear()

@profiler transform(S, vals)
p = points(S, 10)

f = xy -> xy[1]
P = Fun(S, [zeros(6);1])
pts = points(S, n)
vals = f.(pts)
@time P*vals

@time xP = Fun((x,y) -> x * P(x,y), S,  2)
xP.coefficients
∂xP = Fun((x,y) -> ∂P∂x(x,y), S,  10)

p = points(S, 10)
plan = ApproxFun.plan_transform(S, Vector{Float64}(undef, length(p)))


P = Fun(S, [0,0,0,0,0,0,1])
f = (x,y) -> y*P(x,y)
@time vals = (xy -> f(xy...)).(p)
@time Fun(S, plan*vals).coefficients










a,b
a = b = 0.0
n,k= 2,2; m,j=1,1;
    P1 =gethalfdiskOP(H, P, ρ, n, k, a, b)
    P2 = gethalfdiskOP(H, P, ρ, m, j, a, b)
    halfdiskintegral((x,y) -> P1(x,y) * P2(x,y),
                20, a, b)


X = Fun(identity, 0..1)
H = OrthogonalPolynomialFamily(X, (1-X^2))
sum(Fun(x -> Fun(H(0.,0.), [0.,1.])(x), 0..1))
plot(Fun(x -> Fun(H(0.,0.), [0.,1.])(x), 0..1))

H₂ = OrthogonalPolynomialFamily(1+Y, 1-((Y+1)/2)^2)
plot(Fun(x -> Fun(H₂(0.,0.), [0.,1.])(x)))


H(0.,0.).a[1]-0.5


evaluate([0.,1.], H(0.,0.), 0.5)
ApproxFun.tocanonical(H(0.,0.), 0.5)


Fun(H(0.,0.), [0.,1.])(0.1)

ApproxFun.canonicaldomain(H(0.,0.))

lanczos(H(0.,0.).weight,3)

t, w= golubwelsch(H(0.,0.).weight,3)

f = Fun(Chebyshev(0..1), randn(6))
w'*f.(t) - sum(f)
H(0.,0.).a




H(0.,0.).b

H.spaces

plot(H(0.,0.).weight)



P1(0.1,0.2)
sum(Fun(x -> Fun(H(a,b+0.5),[0,1.])(x), 0..1) * sqrt(1-X^2))



P1(0.1,0.2)
H̃ = Fun(H(a, b+k+0.5), [zeros(n-k); 1])

plot(Fun(x -> Fun(H(a,b), [0.0,0,1])(x), 0..1))

X = Fun(identity, 0..1)
Y = Fun(identity, -1..1)

using Plots
Fun(x -> Fun(H(a,b), [0.,0,0,1])(x)) |> sum

H(a,b).weight |> plot
α = (a,b)
OrthogonalPolynomialSpace(P, prod(P.factors.^α)).weight |> plot

prod(H.factors.^α) |> typeof
prod(H.factors.^(0.5,0.5)) |> typeof


H.spaces


H(a,b)
a,b


plot(Fun(x -> Fun(H(a,b), [0,0.0,0,0,0,1])(x)))


a = b
plot(H(a,b).weight)



Fun(H(a,b), [0,0,0,0,0,1])(0.1)
Fun(H(a,b), [0,0,0,0,0,1])(-0.1)

a,b

(1+Y)^(0.1)

using Plots

Y |> space
roots((1-(Y+1)/2)^2)


P1


a = b = 0;






#=====#


# struct OrthogonalPolynomialDerivative{T} <: Derivative{T}
#     space::OrthogonalPolynomialSpace
# end
#
# domainspace(D::OrthogonalPolynomialDerivative) = D.space
# rangespace(D::OrthogonalPolynomialDerivative) = D.space.family(D.space.α .+ 1)
# bandwidths(D::OrthogonalPolynomialDerivative) = (1,?)
# getindex(D::OrthogonalPolynomialDerivative, k::Int, j::Int) = ?
#
# Derivative(sp::OrthogonalPolynomialSpace, k) = (@assert k==1; OrthogonalPolynomialDerivative(sp))
#
#
#
# D = Derivative(Jacobi(a,b))
# @which D[2,3]
#
# bandwidths(D)



#
# n = 16
# f = (x, y) -> exp(x)
# S = Chebyshev()^2
# S.spaces
# pts = points(S, n)
# m = Int(cld(-3+sqrt(1+8n),2)) + 1; ñ = Int((m+1)m/2)
# T = Float64
# cfs = transform(S, T[f(x...) for x in pts])
# F = Fun(S, cfs)
# x = (0.1, 0.2)
# typeof(F.space)
# @which evaluate(F.coefficients,F.space,ApproxFun.Vec(x...))
# ApproxFun.totensor(F.space,F.coefficients)
# @which ApproxFun.tensorizer(F.space)
# ApproxFun.Tensorizer(map(ApproxFun.blocklengths,F.space.spaces))
# F.space.spaces
# ApproxFun.columnspace(S, 3)
#
# n = 6; a = 0.5; b = -0.5
# S = HalfDiskSpace(a, b)
# pts = points(S, n)
# f = (x, y) -> exp(x + y)
# tensorizer(S)
# columnspace(S, 2)
# T = Float64
# transform(S, T[f(x...) for x in pts])
#
# F = Fun(S, [zeros(1); 1])
# F.(pts)
# x = 0.1, 0.2
# typeof(F.space)
# evaluate(F.coefficients,F.space,ApproxFun.Vec(x...))
# f(x...)
# ff = Fun(f, S)
# ff(x...) - f(x...)
