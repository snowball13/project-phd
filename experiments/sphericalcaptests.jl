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
                    differentiateweightedspacex, differentiateweightedspacey,
                    resizedata!, resizedataonedimops!, resizecoeffs2!, getnki
using JLD

isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)


T = Float64; B = T#BigFloat
α = 0.2
DSF = DiskSliceFamily(T, T, α, 1.0, -1.0, 1.0); a, b = 1.0, 1.0; ap, bp = 1.0, 2.0
SCF = SphericalCapFamily(B, T, α)
S = SCF(a, 0.0); S2 = DSF(a, 0.0)

y, z = B(-0.234), B(0.643); x = sqrt(1 - z^2 - y^2); p = [x; y; z]; isindomain(p, SCF)
resizedata!(S, 10)


# Test transform
n = 110
f = (x,y,z)->x
f2 = (x,y)->x
pts, w = pointswithweights(S, n)
vals = [f(pt...) for pt in pts]
cfs = transform(S, vals)
N = getnki(S, length(cfs))[1]
cfs2 = PseudoBlockArray(OrthogonalPolynomialFamilies.convertcoeffsvecorder(S, cfs), [2n+1 for n=0:N])
itransform(S, cfs)
F = Fun(S, cfs)
F(p)
f(p...)
T(F(p) - f(p...))
@test F(p) ≈ f(p...)
@test itransform(S, cfs) ≈ vals
F = Fun(f, S, 1); F.coefficients
@test F(p) ≈ f(p...)


# Testing recs
n, k, i = 12, 7, 0
α = [OrthogonalPolynomialFamilies.recα(B, S, n, k, j) for j=1:6]
β = [OrthogonalPolynomialFamilies.recβ(B, S, n, k, i, j) for j=1:6]
γ = [OrthogonalPolynomialFamilies.recγ(B, S, n, k, j) for j=1:3]
N = 15
pts = [p]
pp = OrthogonalPolynomialFamilies.getopptseval(S, N, pts)
j = 1
for j = 1:length(pts)
    lhs = pts[j][1] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k, i)[j]
    rhs = (α[1] * OrthogonalPolynomialFamilies.getptsevalforop(S, n-1, k-1, i)[j]
            + α[2] * OrthogonalPolynomialFamilies.getptsevalforop(S, n-1, k+1, i)[j]
            + α[3] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k-1, i)[j]
            + α[4] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k+1, i)[j]
            + α[5] * OrthogonalPolynomialFamilies.getptsevalforop(S, n+1, k-1, i)[j]
            + α[6] * OrthogonalPolynomialFamilies.getptsevalforop(S, n+1, k+1, i)[j])
    if abs(lhs - rhs) > 1e-22
        @show j, T(lhs), T(rhs), T(abs(lhs - rhs))
    end
end
for j = 1:length(pts)
    lhs = pts[j][2] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k, i)[j]
    rhs = (β[1] * OrthogonalPolynomialFamilies.getptsevalforop(S, n-1, k-1, abs(i-1))[j]
            + β[2] * OrthogonalPolynomialFamilies.getptsevalforop(S, n-1, k+1, abs(i-1))[j]
            + β[3] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k-1, abs(i-1))[j]
            + β[4] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k+1, abs(i-1))[j]
            + β[5] * OrthogonalPolynomialFamilies.getptsevalforop(S, n+1, k-1, abs(i-1))[j]
            + β[6] * OrthogonalPolynomialFamilies.getptsevalforop(S, n+1, k+1, abs(i-1))[j])
    if abs(lhs - rhs) > 1e-22
        @show j, T(lhs), T(rhs), T(abs(lhs - rhs))
    end
end
for j = 1:length(pts)
    lhs = pts[j][3] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k, i)[j]
    rhs = (γ[1] * OrthogonalPolynomialFamilies.getptsevalforop(S, n-1, k, i)[j]
            + γ[2] * OrthogonalPolynomialFamilies.getptsevalforop(S, n, k, i)[j]
            + γ[3] * OrthogonalPolynomialFamilies.getptsevalforop(S, n+1, k, i)[j])
    if abs(lhs - rhs) > 1e-22
        @show j, T(lhs), T(rhs), T(abs(lhs - rhs))
    end
end

# Testing recs
n, k, i = 1, 1, 0
α = [OrthogonalPolynomialFamilies.recα(B, S, n, k, j) for j in [1,3,5,6]]
β = [OrthogonalPolynomialFamilies.recβ(B, S, n, k, i, j) for j in [1,3,5,6]]
γ = [OrthogonalPolynomialFamilies.recγ(B, S, n, k, j) for j=2:3]
N = 15
pts = [p]
pp = OrthogonalPolynomialFamilies.getopptseval(S, N, pts)

n = 1
θ = atan(p[2]/p[1])
R0 = getRspace(S, 0); OrthogonalPolynomialFamilies.getopptseval(R0, 5, [p[3]])
R1 = getRspace(S, 1); OrthogonalPolynomialFamilies.getopptseval(R1, 5, [p[3]])
Q1 = [OrthogonalPolynomialFamilies.getdegzeropteval(B, S)] # n = 0
Q2 = [R0.opptseval[n+1][1] * OrthogonalPolynomialFamilies.getdegzeropteval(B, S);
      S.family.ρ(p[3]) * Yop(1, 0, θ);
      S.family.ρ(p[3]) * Yop(1, 1, θ)] # n = 1
Q3 = [R0.opptseval[n+1+1][1] * OrthogonalPolynomialFamilies.getdegzeropteval(B, S);
      R1.opptseval[n+1][1] * S.family.ρ(p[3]) * Yop(1, 0, θ);
      R1.opptseval[n+1][1] * S.family.ρ(p[3]) * Yop(1, 1, θ);
      S.family.ρ(p[3])^2 * Yop(2, 0, θ);
      S.family.ρ(p[3])^2 * Yop(2, 1, θ)] # n = 2
T(p[1] * Q2[2])
T(α[1] * Q1[1] + α[2] * Q2[1] + α[3] * Q3[1] + α[4] * Q3[4])

function Yop(k, i, θ)
    if k == 0
        sqrt(B(2))/2
    elseif i == 0
        cos(k * θ)
    else
        sin(k * θ)
    end
end

R = getRspace(S, 0); OrthogonalPolynomialFamilies.getopptseval(R, 100, [p[3]])
OrthogonalPolynomialFamilies.resetopptseval(S)
resize!(S.opptseval, 1)
pts = [p]
S.opptseval[1] = Vector{B}(undef, length(pts))
S.opptseval[1][:] .= OrthogonalPolynomialFamilies.getdegzeropteval(B, S)

n = 1
jj = getopindex(S, n, 0, 0)
resizedata!(S, n)
resize!(S.opptseval, getopindex(S, n, n, 1))
for k = 0:2n
    S.opptseval[jj+k] = Vector{B}(undef, length(pts))
end
S.opptseval
nm1 = getopindex(S, n-1, 0, 0)
r = length(pts)
P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
P = - S.DT[n] * (S.B[n] - OrthogonalPolynomialFamilies.clenshawG(S, n-1, pts[r])) * P1
for k = 0:2n
    S.opptseval[jj+k][r] = P[k+1]
end
S.opptseval
T(S.opptseval[getopindex(S, 1, 0, 0)][1] - R.opptseval[2][1] * OrthogonalPolynomialFamilies.getdegzeropteval(B, S))
T(S.opptseval[getopindex(S, 1, 1, 0)][1] - S.family.ρ(p[3]) * cos(θ))
T(S.opptseval[getopindex(S, 1, 1, 1)][1] - S.family.ρ(p[3]) * sin(θ))


n = 2
jj = getopindex(S, n, 0, 0)
resizedata!(S, n)
resize!(S.opptseval, getopindex(S, n, n, 1))
for k = 0:2n
    S.opptseval[jj+k] = Vector{B}(undef, length(pts))
end
nm1 = getopindex(S, n-1, 0, 0)
nm2 = getopindex(S, n-2, 0, 0)
P1 = [opevalatpts(S, nm1+it, pts)[r] for it = 0:2(n-1)]
P2 = [opevalatpts(S, nm2+it, pts)[r] for it = 0:2(n-2)]
P = (- S.DT[n] * (S.B[n] - OrthogonalPolynomialFamilies.clenshawG(S, n-1, pts[r])) * P1
     - S.DT[n] * S.C[n] * P2)
for k = 0:2n
    S.opptseval[jj+k][r] = P[k+1]
end
θ = atan(p[2]/p[1])
S.opptseval
T(S.opptseval[getopindex(S, 2, 0, 0)][1] - R.opptseval[n+1][1] * OrthogonalPolynomialFamilies.getdegzeropteval(B, S))
T(S.opptseval[getopindex(S, 2, 2, 0)][1]) - T(S.family.ρ(p[3])^2 * cos(2 * θ))
T(S.opptseval[getopindex(S, 2, 2, 1)][1] - S.family.ρ(p[3])^2 * sin(2 * θ))

Array(S.DT[n] * (S.B[n] - OrthogonalPolynomialFamilies.clenshawG(S, n-1, pts[r])))
Array(S.DT[n] * S.C[n])
Array(S.B[n] - OrthogonalPolynomialFamilies.clenshawG(S, n-1, pts[r]))
Array(S.DT[n])
Array(S.C[n])

pts
OrthogonalPolynomialFamilies.getopptseval(S, N, pts)





n = 1
Jx = [S.C[n+1][1:2n+1, :] S.B[n+1][1:2n+1, :] S.A[n+1][1:2n+1, :]]
Jy = [S.C[n+1][2n+1+1:2*(2n+1), :] S.B[n+1][2n+1+1:2*(2n+1), :] S.A[n+1][2n+1+1:2*(2n+1), :]]
Jz = [S.C[n+1][2*(2n+1)+1:end, :] S.B[n+1][2*(2n+1)+1:end, :] S.A[n+1][2*(2n+1)+1:end, :]]
θ = atan(p[2]/p[1])
R0 = getRspace(S, 0); OrthogonalPolynomialFamilies.getopptseval(R0, 5, [p[3]])
R1 = getRspace(S, 1); OrthogonalPolynomialFamilies.getopptseval(R1, 5, [p[3]])
Q1 = [OrthogonalPolynomialFamilies.getdegzeropteval(B, S)] # n = 0
Q2 = [R0.opptseval[n+1][1] * OrthogonalPolynomialFamilies.getdegzeropteval(B, S);
      S.family.ρ(p[3]) * Yop(1, 0, θ);
      S.family.ρ(p[3]) * Yop(1, 1, θ)] # n = 1
Q3 = [R0.opptseval[n+1+1][1] * OrthogonalPolynomialFamilies.getdegzeropteval(B, S);
      R1.opptseval[n+1][1] * S.family.ρ(p[3]) * Yop(1, 0, θ);
      R1.opptseval[n+1][1] * S.family.ρ(p[3]) * Yop(1, 1, θ);
      S.family.ρ(p[3])^2 * Yop(2, 0, θ);
      S.family.ρ(p[3])^2 * Yop(2, 1, θ)] # n = 2
Q = [Q1; Q2; Q3]
T.(p[1] * Q2 - Jx * Q)
T.(p[2] * Q2 - Jy * Q)
T.(p[3] * Q2 - Jz * Q)

T.(p[1] * Q2), T.(p[2] * Q2)



T.(Jx * Q), T.(Jy * Q)



Array(Jx)
Array(Jy)




n = 2
Q1 = [OrthogonalPolynomialFamilies.getptsevalforop(S, ind)[1] for ind=getopindex(S, n-1, 0, 0):getopindex(S, n-1, n-1, 1)]
Q2 = [OrthogonalPolynomialFamilies.getptsevalforop(S, ind)[1] for ind=getopindex(S, n, 0, 0):getopindex(S, n, n, 1)]
Q3 = [OrthogonalPolynomialFamilies.getptsevalforop(S, ind)[1] for ind=getopindex(S, n+1, 0, 0):getopindex(S, n+1, n+1, 1)]
Q = [Q1; Q2; Q3]
Jx = [S.C[n+1][1:2n+1, :] S.B[n+1][1:2n+1, :] S.A[n+1][1:2n+1, :]]
Jy = [S.C[n+1][2n+1+1:2*(2n+1), :] S.B[n+1][2n+1+1:2*(2n+1), :] S.A[n+1][2n+1+1:2*(2n+1), :]]
Jz = [S.C[n+1][2*(2n+1)+1:end, :] S.B[n+1][2*(2n+1)+1:end, :] S.A[n+1][2*(2n+1)+1:end, :]]
T.(p[1] * Q2 - Jx * Q)
T.(p[2] * Q2 - Jy * Q)
T.(p[3] * Q2 - Jz * Q)

n = 1
Q1 = [OrthogonalPolynomialFamilies.getptsevalforop(S, 1)[1]]
Q2 = [OrthogonalPolynomialFamilies.getptsevalforop(S, ind)[1] for ind=getopindex(S, n, 0, 0):getopindex(S, n, n, 1)]
Q3 = [OrthogonalPolynomialFamilies.getptsevalforop(S, ind)[1] for ind=getopindex(S, n+1, 0, 0):getopindex(S, n+1, n+1, 1)]
Q = [Q1; Q2; Q3]
Jx = [S.C[n+1][1:2n+1, :] S.B[n+1][1:2n+1, :] S.A[n+1][1:2n+1, :]]
Jy = [S.C[n+1][2n+1+1:2*(2n+1), :] S.B[n+1][2n+1+1:2*(2n+1), :] S.A[n+1][2n+1+1:2*(2n+1), :]]
Jz = [S.C[n+1][2*(2n+1)+1:end, :] S.B[n+1][2*(2n+1)+1:end, :] S.A[n+1][2*(2n+1)+1:end, :]]
T.(p[1] * Q2 - Jx * Q)
T.(p[3] * Q2 - Jz * Q)
T.(p[2] * Q2 - Jy * Q)
