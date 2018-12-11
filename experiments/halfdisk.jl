using OrthogonalPolynomialFamilies, ApproxFun
using StaticArrays, LinearAlgebra, Test, Profile, BlockArrays,
        BlockBandedMatrices, SparseArrays
using Makie

# Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
#   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point

# 1) f(x,y) = -8x => u(x,y) ≡ 1
N = 1 # degree of f
c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
OrthogonalPolynomialFamilies.resizecoeffs!(f, N)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
Δ \ f.coefficients
u = Fun(S, Δ \ f.coefficients)
@test u(z) ≈ c # Result u(x,y) where Δ(u*w)(x,y) = f(x,y)
# plot(x->u(x,y))
# plot(y->u(x,y))

# 2) f(x,y) = 2 - 12xy - 14x^2 - 2y^2 => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(2 - 12x*y - 14x^2 - 2y^2), S)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
u = Fun(S, Δ \ OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
@test u(z) ≈ U(z)

# 3) f(x,y) = y*exp(x)*[2-11x-6x^2-x^3-2y^2-xy^2] => u(x,y) = y*exp(x)
U = Fun((x,y)->y*exp(x), S)
N = 7 # degree of f
m = Int((N+1)*(N+2))
f = Fun((x,y)->y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2), S, m)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
u = Fun(S, Δ \ f.coefficients[1:size(Δ)[1]])
@test u(z) ≈ U(z) atol=10.0^(-N)



#====#
#=
Model problem: Helmholtz eqn:
    Δ(W*u)(x,y) + k²u(x,y) = -f(x,y)
where k is the wavenumber, u is the amplitude at a point (x,y) on the halfdisk,
and W(x,y) is the P^{(1,1)} weight.
=#
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
wavenumberoperator(N, k) = sparse([sparse(I,sum(1:N),sum(1:N)) * k^2; zeros(N+1, sum(1:N))])
k = rand(1)[1]

# 1) f(x,y) = 8cx - ck^2 => u(x,y) ≡ c
c = rand(1)[1]; f = Fun((x,y)->(8*c*x - c*k^2), S)
N = 2
K = wavenumberoperator(N, k)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
uc = sparse(Δ + K) \ (-OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
u = Fun(S, uc)
@test u(z) ≈ c

# 2) f(x,y) = -k^2(x+y) - (2 - 12xy - 14x^2 - 2y^2) => u(x,y) = x + y
U = Fun((x,y)->x+y, S)
N = 2 # degree of f
f = Fun((x,y)->(-(x+y)*k^2 - 2 + 12x*y + 14x^2 + 2y^2), S)
Δ = OrthogonalPolynomialFamilies.laplace(D, N-1)
K = wavenumberoperator(N, k)
u = Fun(S, - sparse(Δ + K) \ OrthogonalPolynomialFamilies.resizecoeffs!(f, N))
@test u(z) ≈ U(z)


#====#
#=
Model problem: Heat eqn
    ∂u/∂t(x,y) = Δ(W*u)(x,y)
=>  (Back Euler)
    u1 - u0 = h * Δ(W*u1)
=>  (I - hΔ)\u0 = u1
=#
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point
h = 1e-3
N = 3
uc = OrthogonalPolynomialFamilies.resizecoeffs!(Fun((x,y)->x, S), N)
Δ = OrthogonalPolynomialFamilies.laplacesquare(D, N-1)
sparse(I - h * Δ)
for it = 1:10
    global uc = sparse(I - h * Δ) \ uc
end
u = Fun(S, uc)
u(z)


# TODO: PLOTTING

m = 10
pts = OrthogonalPolynomialFamilies.points(S, m)[1:Int(end/2)]
u = Fun(S, uc)
x = [pt[1] for pt in pts]
y = [pt[2] for pt in pts]
upts = [u(pt) for pt in pts]



# scene = Makie.Scene()
# s = Makie.surface(x, y, upts, colormap = :viridis, colornorm = (-1.0, 1.0))
#
#
# x = 1:10
# y = 1:10
# sizevec = [s for s = 1:length(x)] ./ 10
# scene = scatter(x, y, markersize = sizevec)
# Makie.display
# using Plots
