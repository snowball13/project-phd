# Run this setup code to get the recurrence coefficients for the 1D OPs

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
using JLD

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
    D
end

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
