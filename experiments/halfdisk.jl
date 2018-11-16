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
# function resizecoeffs!(cfs::AbstractVector, N)
#     m̃ = length(cfs)
#     m = Int((N+1)*(N+2)/2)
#     resize!(cfs, m)
#     if m̃ < m
#         cfs[m̃+1:end] .= 0.0
#     end
#     cfs
# end


function evalweightedderivativex(S::HalfDiskSpace, n, k, x, y)

end
function getweightedpartialoperatory(S::HalfDiskSpace, N)

end



x, y = 0.2, 0.3; x^2 + y^2 < 1
z = [x;y]
a, b = 1.0, 0.0; D = HalfDiskFamily(); S = D(a, b)
f = Fun((x,y)->x^2*y+1+3y, S)
fc = f.coefficients
N = 3; resizecoeffs!(fc, N)
T = OrthogonalPolynomialFamilies.gettransformoperator(S, N)
Fun(D(1.0, 1.0), T*fc)(z) - f(z)
# TODO: Check how to do weighted expansions!
