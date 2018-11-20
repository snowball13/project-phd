using OrthogonalPolynomialFamilies, ApproxFun
using StaticArrays, LinearAlgebra, Test, Profile, BlockArrays,
        BlockBandedMatrices, SparseArrays, Plots

# # Model Problem: Δu(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
# function getpartialoperatorx2(S, N)
#     (OrthogonalPolynomialFamilies.getpartialoperatorx(OrthogonalPolynomialFamilies.differentiatespacex(S), N-1)
#         * OrthogonalPolynomialFamilies.getpartialoperatorx(S, N))
# end
# function getpartialoperatory2(S, N)
#     (OrthogonalPolynomialFamilies.getpartialoperatory(OrthogonalPolynomialFamilies.differentiatespacey(S), N-1)
#         * OrthogonalPolynomialFamilies.getpartialoperatory(S, N))
# end
# differentiatespacex2(S) =
#     OrthogonalPolynomialFamilies.differentiatespacex(
#             OrthogonalPolynomialFamilies.differentiatespacex(S))
# differentiatespacey2(S) =
#     OrthogonalPolynomialFamilies.differentiatespacey(
#             OrthogonalPolynomialFamilies.differentiatespacey(S))


# Model Problem: Δ(u*w)(x,y) = f(x,y) in Ω=halfdisk; u(x,y) ≡ 0 on ∂Ω.
#   where w(x,y) = x*(1-x^2-y^2) is the weight of the D(1.0,1.0) basis.
a, b = 1.0, 1.0; D = HalfDiskFamily(); S = D(a, b)
x, y = 0.4, -0.2; z = [x; y] # Test point

# 1) f(x,y) = -8x => u(x,y) ≡ 1
N = 1 # degree of f
c = rand(1)[1]; f = Fun((x,y)->-c*8x, S)
resizecoeffs!(f, N)
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
u = Fun(S, Δ \ resizecoeffs!(f, N))
@test u(z) ≈ U(z)

# 3) f(x,y) = y*exp(x)*[2-11x-6x^2-x^3-2y^2-xy^2] => u(x,y) = y*exp(x)
# TODO: laplace(D, n) hangs/gives error for n >= 3. Think problem is in
#       obtaining T^{(1,0)->(0,0)} operator.
U = Fun((x,y)->y*exp(x), S)
N = 4 # degree of f
m = Int((N+1)*(N+2)/2)
f = Fun((x,y)->y*exp(x)*(2-11x-6x^2-x^3-2y^2-x*y^2), S, m)
# Δ = OrthogonalPolynomialFamilies.laplace(D, N-1) # NOTE: hangs/error
# u = Fun(S, Δ \ resizecoeffs!(f, N))
# @test u(z) ≈ U(z)
