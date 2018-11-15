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
