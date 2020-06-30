using Revise
using ApproxFun
using Test, Profile, StaticArrays, LinearAlgebra, BlockArrays,
        BlockBandedMatrices, SparseArrays
using OrthogonalPolynomialFamilies
import OrthogonalPolynomialFamilies: points, pointswithweights, getopptseval,
                    opevalatpts, inner2, getopindex, getnk, resizecoeffs!,
                    transformparamsoperator, weightedpartialoperatorx,
                    weightedpartialoperatory, partialoperatorx, partialoperatory,
                    derivopevalatpts, getderivopptseval, getopnorms,
                    getopnorm, operatorclenshaw, weight, biharmonicoperator,
                    getPspace, getRspace, differentiatespacex, differentiatespacey,
                    differentiateweightedspacex, differentiateweightedspacey,
                    resizedata!,
                    resizedataonedimops!, getnki, convertcoeffsvecorder,
                    diffoperatorphi, diffoperatortheta, diffoperatortheta2,
                    differentiatespacephi, increasedegreeoperator,
                    getptsevalforop, getderivptsevalforop, laplacianoperator,
                    rho2laplacianoperator, gettangentspace,
                    rhogradoperator, coriolisoperator, rho2operator,
                    rhodivminusgradrhodotoperator
using JLD


isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)
rhoval(z) = sqrt(1 - z^2)


# Setup
T = Float64; B = T#BigFloat
α = 0.2
DSF = DiskSliceFamily(B, T, α, 1.0, -1.0, 1.0)
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
a = 1.0
S = SCF(a, 0.0); S2 = DSF(a, a); S0 = SCF(0.0, 0.0)
ST = gettangentspace(SCF)

y, z = B(-234)/1000, B(643)/1000; x = sqrt(1 - z^2 - y^2); p = [x; y; z]; isindomain(p, SCF)
θ = atan(y / x)
resizedata!(S, 20)
resizedata!(S0, 20)

function gettangentspacecoeffsvec(f::Fun, g::Fun)
    fc = f.coefficients; gc = g.coefficients
    m = length(fc)
    @assert m == length(gc) "Coeffs must be same length"
    ret = zeros(2m)
    it = 1
    for j = 1:m
        ret[it] = fc[j]; ret[it+1] = gc[j]
        it += 2
    end
    ret
end


# Test of tangent clenshaw
basisvecs = [cos(θ) * z; sin(θ) * z; - rhoval(z)], [-sin(θ); cos(θ); 0]
N = 5
for n=0:N, k=0:n, i=0:min(1,k), j=0:1
    cfs = zeros(getopindex(ST, N, N, 1, 1)); cfs[getopindex(ST, n, k, i, j; bydegree=false, N=N)] = 1; cfs
    ret = OrthogonalPolynomialFamilies.clenshaw(cfs, ST, p)
    cfs0 = zeros(getopindex(S, N, N, 1)); cfs0[getopindex(S, n, k, i; bydegree=false, N=N)] = 1; cfs0
    retactual = Fun(S0, cfs0)(p) * basisvecs[j+1]
    res = maximum(abs, ret - retactual)
    if res > 1e-10
        @show n,k,i,j,res
    end
    @test ret ≈ retactual
end


# Linear SWE operator tests
basisvecs = [cos(θ) * z; sin(θ) * z; - rhoval(z)], [-sin(θ); cos(θ); 0]
inds = [4, 3, 5]; N = sum(inds)
f = Fun((x,y,z)->x^inds[1] * y^inds[2] * z^inds[3], S, 2(N+1)^2)
dϕf = (x,y,z)->((inds[1] + inds[2]) * z^2 - inds[3] * S.family.ρ(z)^2) * x^inds[1] * y^inds[2] * z^(inds[3]-1) # ρ*∂f/∂ϕ, deg = degf + 1
dθf = (x,y,z)->z^inds[3] * y^(inds[2] - 1) * x^(inds[1] - 1) * (inds[2] * x^2 - inds[1] * y^2) # ∂f/∂θ
indsϕ = [2, 7, 5]; Nϕ = sum(indsϕ)
indsθ = [2, 3, 5]; Nθ = sum(indsθ); Nt = max(Nϕ, Nθ)
Fϕ = Fun((x,y,z)->x^indsϕ[1] * y^indsϕ[2] * z^indsϕ[3], S0, 2(Nt+1)^2); Fϕ.coefficients
Fθ = Fun((x,y,z)->x^indsθ[1] * y^indsθ[2] * z^indsθ[3], S0, 2(Nt+1)^2); Fθ.coefficients
dϕFϕ = (x,y,z)->((indsϕ[1] + indsϕ[2]) * z^2 - indsϕ[3] * S.family.ρ(z)^2) * x^indsϕ[1] * y^indsϕ[2] * z^(indsϕ[3]-1) # ρ*∂f/∂ϕ, deg = degf + 1
dθFθ = (x,y,z)->z^indsθ[3] * y^(indsθ[2] - 1) * x^(indsθ[1] - 1) * (indsθ[2] * x^2 - indsθ[1] * y^2) # ∂f/∂θ
# G
G = rhogradoperator(S, N)
cfs0 = G * f.coefficients
ret = OrthogonalPolynomialFamilies.clenshaw(cfs0, ST, p)
retactual = ((weight(S, p) * dϕf(p...) - f(p) * S.family.ρ(z)^2) * basisvecs[1]
                + weight(S, p) * dθf(p...) * basisvecs[2])
@test ret ≈ retactual
# D
D = rhodivminusgradrhodotoperator(ST, Nt)
Fϕ = Fun((x,y,z)->0.0, S0, 2(Nt+1)^2); Fθ.coefficients
Fθ = Fun((x,y,z)->z^indsθ[3], S0, 2(Nt+1)^2); Fθ.coefficients
cfs0 = gettangentspacecoeffsvec(Fϕ, Fθ)
cfs = D * cfs0
ret = Fun(S, cfs)(p)
retactual = 0.0
@test ret ≈ retactual atol=1e-13
# F
F = coriolisoperator(ST, Nt)
cfs0 = gettangentspacecoeffsvec(Fϕ, Fθ)
cfs = F * cfs0
ret = OrthogonalPolynomialFamilies.clenshaw(cfs, ST, p)
Ω = B(72921) / 1e9
retactual = OrthogonalPolynomialFamilies.clenshaw(gettangentspacecoeffsvec(-Fθ, Fϕ) * 2 * z, ST, p)
@test ret ≈ retactual
# P
N = 100
P = rho2operator(S, S, N)
cfs = P * f.coefficients
ret = Fun(S, cfs)(p)
retactual = f(p) * weight(S, p) * S.family.ρ(z)^2
@test ret ≈ retactual
f = Fun((x,y,z)->x^inds[1] * y^inds[2] * z^inds[3], S, 2(N+4)^2)
Pm1 = sparse(pinv(Array(P))) # TODO make explicit?!?!? # NOTE takes ℚ^{1}->𝕎^{1}
cfs = Pm1 * f.coefficients
ret = Fun(S, cfs)(p) * weight(S, p)
retactual = f(p) / rhoval(p[3])^2
@test ret ≈ retactual



# Laplacian tests
# 1) u = constant
N = 10
L = laplacianoperator(S, N)
c = 2.0
u = (x,y,z)->c
f = (x,y,z)->-c * 2 * z
F = Fun(f, S, 2*(N+2)^2); F.coefficients
ucfs = iterimprove(sparse(L), F.coefficients)
U = Fun(S, ucfs)
U(p)
u(p...)
@test U(p) ≈ u(p...)

# 2) u = monomial
inds = [2, 3, 3]; N = 50
u = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
L = laplacianoperator(S, N)
f = (x,y,z)->(rholaplacian(S, u, inds, [x;y;z]) / rhoval(z)^2) # NOTE methods at bottom of script for this
F = Fun(f, S, 2*(sum(inds)+2)^2); F.coefficients
resizecoeffs!(S, F, N+1)
ucfs = iterimprove(sparse(L), F.coefficients)
U = Fun(S, ucfs)
@test U(p) ≈ u(p...)


checklaplacian(S, p)
function checklaplacian(S, p)
    x, y, z = p[1], p[2], p[3]
    θ = atan(y / x)
    ρ = sqrt(1 - z^2)
    w10 = (z - S.family.α)
    ret = - 4 * w10 + (-z - S.family.α) * ρ^2 + 5 * (-ρ - S.family.α) * ρ * z
    ret += 4 * w10 * z^2 - 2 * w10 * ρ^2
    ret *= cos(θ) * sin(θ)
    ret
end



#===#

# Test transform
n = B == T ? 400 : 200
f = (x,y,z)->cos(y)
pts, w = pointswithweights(S, n)
vals = [f(pt...) for pt in pts]
cfs = transform(S, vals)
N = getnki(S, length(cfs))[1]
cfs2 = PseudoBlockArray(convertcoeffsvecorder(S, cfs), [2n+1 for n=0:N])
F = Fun(S, cfs)
F(p)
f(p...)
T(F(p) - f(p...))
@test T(F(p)) ≈ T(f(p...))
vals2 = itransform(S, cfs); pts2 = points(S, length(vals2)/2); @test T.(vals2) ≈ T.([f(pt...) for pt in pts2])
F = Fun(f, S, 200); F.coefficients
@test T(F(p)) ≈ T(f(p...))

# ∂/∂θ operator
f = (x,y,z)->x^2 + y^4 # (cos2t + sin4t*ρ2(z))ρ2(z)
dθf = (x,y,z)->-2x * y + 4y^3 * x
F = Fun(f, S, 100); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dθ = diffoperatortheta(S, N)
dθF = Fun(S, dθ * F.coefficients)
@test dθF(p) ≈ dθf(p...)

# ∂²/∂θ² operator
f = (x,y,z)->x^2 + y^4 # (cos2t + sin4t*ρ2(z))ρ2(z)
d2θf = (x,y,z)->2y^2 - 2x^2 + 12x^2*y^2 - 4y^4
F = Fun(f, S, 100); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dθ2 = diffoperatortheta2(S, N; weighted=false)
d2θF = Fun(S, dθ2 * F.coefficients)
@test d2θF(p) ≈ d2θf(p...)

# ρ(z)∂/∂ϕ operator
# f = (x,y,z)->x^a y^b z^c
# df = (x,y,z)->x^a y^b z^(c-1) [(a+b)z^2 - cρ(z)^2]
inds = [4, 3, 5]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
df = (x,y,z)->((inds[1] + inds[2]) * z^2 - inds[3] * S.family.ρ(z)^2) * x^inds[1] * y^inds[2] * z^(inds[3]-1) # deg = degf + 1
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dϕ = diffoperatorphi(S, N; weighted=false)
cfs = dϕ * F.coefficients
dF = Fun(differentiatespacephi(S), cfs)
@test dF(p) ≈ df(p...)

# ρ(z)∂/∂ϕ weighted operator
# f = (x,y,z)->x^a y^b z^c
# df = (x,y,z)->x^a y^b z^(c-1) [(a+b)z^2 - cρ(z)^2]
inds = [4, 3, 5]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
df = (x,y,z)->((z - S.family.α)^S.params[1] * ((inds[1] + inds[2]) * z^2 - inds[3] * S.family.ρ(z)^2) * x^inds[1] * y^inds[2] * z^(inds[3]-1)
                - S.params[1] * S.family.ρ(z)^2 * f(x,y,z)) # deg = degf + 2
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
dϕ = diffoperatorphi(S, N; weighted=true)
cfs = dϕ * F.coefficients
dF = Fun(differentiatespacephi(S; weighted=true), cfs)
@test dF(p) ≈ df(p...)

# non-weighted transform params operator
S0 = differentiatespacephi(S; weighted=true)
inds = [3, 4, 5]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
F0 = Fun(f, S0, 2*(sum(inds)+1)^2); F0.coefficients
N = getnki(S0, ncoefficients(F0))[1]
t = transformparamsoperator(S0, S, N)
F = Fun(S, t * F0.coefficients); F.coefficients
@test F(p) ≈ F0(p)

# weighted transform params operator
S0 = differentiatespacephi(S; weighted=true)
inds = [7, 2, 3]; sum(inds)
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
t = transformparamsoperator(S, S0, N; weighted=true)
F0 = Fun(S0, t * F.coefficients); F0.coefficients
@test F(p) * weight(S, p) ≈ F0(p) * weight(S0, p)

# increase degree
inds = [3, 1, 5]; sum(inds)
f = (x,y,z)->cos(x+y)# x^inds[1] * y^inds[2] * z^inds[3]
F = Fun(f, S, 2*(sum(inds)+1)^2); F.coefficients
N = getnki(S, ncoefficients(F))[1]
t = increasedegreeoperator(S, N, N+2)
Fi = Fun(S, t * F.coefficients); Fi.coefficients
@test F(p) == Fi(p)

# Laplacian tests
# 1) u = constant
N = 10
Δ = laplaceoperator(S, S, N)
c = 2.0
u = (x,y,z)->c
rho2f = (x,y,z)->-c * 2 * z * (1-z^2) # 2 * ρ * ρ' * ρ^2
F = Fun(rho2f, S, 2*(N+4)^2); F.coefficients
ucfs = iterimprove(sparse(Δ), F.coefficients)
U = Fun(S, ucfs)
@test U(p) ≈ u(p...)

# 2) u = monomial
inds = [2, 3, 3]; N = 50
u = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
Δ = laplaceoperator(S, S, N)
rho2f = (x,y,z)->rholaplacian(S, u, inds, [x;y;z]) # NOTE methods at bottom of script for this
F = Fun(rho2f, S, 2*(sum(inds)+4)^2); resizecoeffs!(S, F, N+3)
F(p) - rho2f(p...)
ucfs = iterimprove(sparse(Δ), F.coefficients)
U = Fun(S, ucfs)
@test U(p) ≈ u(p...)

# Jacobi operators
inds = [2, 3, 3]; N = sum(inds) + 1 # +1 so that the Jacobi operator can work
f = (x,y,z)->x^inds[1] * y^inds[2] * z^inds[3]
F = Fun(f, S, 2*(N+1)^2); F.coefficients
# x
J = OrthogonalPolynomialFamilies.jacobix(S, N)
xF = Fun(S, J * F.coefficients)
@test xF(p) ≈ p[1] * f(p...)
# y
J = OrthogonalPolynomialFamilies.jacobiy(S, N)
xF = Fun(S, J * F.coefficients)
@test xF(p) ≈ p[2] * f(p...)
# z
J = OrthogonalPolynomialFamilies.jacobiz(S, N)
xF = Fun(S, J * F.coefficients)
@test xF(p) ≈ p[3] * f(p...)

# Operator Clenshaw
inds = [4, 5, 3]; N = sum(inds) + 2 # Needs +2 buffer
f = Fun((x,y,z)->x^inds[1] * y^inds[2] * z^inds[3], S, 2*(N+1)^2); f.coefficients
v = Fun((x,y,z)->(1 - (3(x-0.2)^2 + 5y^2)), S, 30); v.coefficients
V = operatorclenshaw(v, S, N)
vf = Fun(S, V * f.coefficients)
@test vf(p) ≈ v(p) * f(p)



# Examples
function getsolutionblocknorms(S::SphericalCapSpace, A, f; withcoeffs=false)
    N = nblocks(A)[2] - 1
    u1 = Fun(S, iterimprove(sparse(A), f))
    u1coeffs = PseudoBlockArray(convertcoeffsvecorder(S, u1.coefficients), [i+1 for i=0:N])
    u1norms = zeros(N+1)
    for i = 1:N+1
        u1norms[i] = norm(view(u1coeffs, Block(i)))
    end
    if withcoeffs
        u1norms, u1coeffs
    else
        u1norms
    end
end
N = 50
Δw = laplaceoperator(S, S, N)
f1 = Fun((x,y,z)->1.0, S, 10); f1.coefficients
f2 = Fun((x,y,z)->(1 - x^2 - y^2 - α^2), S, 30); f2.coefficients
f3 = Fun((x,y,z)->weight(S, x, y, z), S, 30); f3.coefficients
# f4 = Fun((x,y,z)->exp(-1000*((x-0.5)^2+(y-0.5)^2)), S, 10000); f4.coefficients
u1norms = getsolutionblocknorms(S, Δw, resizecoeffs!(S, f1, N+3))
u2norms = getsolutionblocknorms(S, Δw, resizecoeffs!(S, f2, N+3))
u3norms = getsolutionblocknorms(S, Δw, resizecoeffs!(S, f3, N+3))
f4cfs = convertcoeffsvecorder(S, convertcoeffsvecorder(S, f4.coefficients)[1:getopindex(S, N+3, N+3, 1)]; todegree=false)
u4norms = getsolutionblocknorms(S, Δw, f4cfs)
u1 = Fun(S, iterimprove(sparse(Δw), f1.coefficients))
u1cfs = PseudoBlockArray(convertcoeffsvecorder(S, u1.coefficients), [i+1 for i=0:N])
minimum(u1cfs[Block(51)])
using Plots
Plots.plot(u1norms, line=(3, :solid), label="f(x,y) = 1", xscale=:log10, yscale=:log10, legend=:bottomleft)
Plots.plot!(u2norms, line=(3, :dash), label="f(x,y) = (1 - alpha^2 - x^2 - y^2)")
Plots.plot!(u3norms, line=(3, :dashdot), label="f(x,y) = W{(1,1,1)}^3")
Plots.plot(u4norms, line=(3, :dot), label = "f(x,y) = exp(-1000((x-0.5)^2+(y-0.5)^2))")
Plots.xlabel!("Block")
Plots.ylabel!("Norm")



N = 50
Δw = laplaceoperator(S2, S2, N; weighted=true, square=false)
f1 = Fun((x,y)->1.0, S2, 10); f1.coefficients
u1 = Fun(S2, iterimprove(sparse(Δw), resizecoeffs!(f1, N+1)))
u1cfs = PseudoBlockArray(u1.coefficients, [i+1 for i=0:N])
maximum(abs, u1cfs[Block(51)])
u1cfs[Block(51)]
u1norms = [norm(u1cfs[Block(n+1)]) for n=0:N]





#====#

# Useful functions for testing
function converttopseudo(S::SphericalCapSpace, cfs; converttobydegree=true)
    N = getnki(S, length(cfs))[1]
    @assert (length(cfs) == (N+1)^2) "Invalid coeffs length"
    if converttobydegree
        PseudoBlockArray(convertcoeffsvecorder(S, cfs), [2n+1 for n=0:N])
    else
        PseudoBlockArray(cfs, [2n+1 for n=0:N])
    end
end
isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)
# Solve `A x = b` for `x` using iterative improvement
# (for BigFloat sparse matrix and vector)
function iterimprove(A::SparseMatrixCSC{T}, b::Vector{T};
                        iters=5, verbose=true) where T
    if eps(T) ≥ eps(Float64)
        # throw(ArgumentError("wrong implementation"))
        return A \ b
    end
    A0 = SparseMatrixCSC{Float64}(A)
    F = factorize(A0)
    x = zeros(T, size(A)[2]) # x = zeros(T, length(b))
    r = copy(b)
    for iter = 1:iters
        y = F \ Vector{Float64}(r)
        for i in eachindex(x)
            x[i] += y[i]
        end
        r = b - A * x
        if verbose
            @show "at iter %d resnorm = %.3g\n" iter norm(r)
        end
    end
    x
end

# The Poisson/Laplacian test methods for Q^{1}
rhoval(z) = sqrt(1 - z^2)
function rholaplacian(S::SphericalCapSpace, u, uinds, xvec)
    x, y, z = xvec
    wght = weight(S, xvec)
    rhoz = rhoval(z)
    utt = getuttfrommononial(uinds, xvec)
    up = getrhoupfrommononial(uinds, xvec)
    upp = getrho2uppfrommononial(uinds, xvec)
    # a = S.params[1]
    # wghtm1 = weight(S.family(a-1, 0.0), xvec)
    # θ = atan(y / x)

    ret = - rhoz^2 * up
    ret += wght * upp
    ret += z * wght * up
    ret -= 2 * rhoz^2 * z * u(x,y,z)
    ret -= rhoz^2 * up
    ret += utt * wght
    ret
end
function getuttfrommononial(inds, p)
    a, b, c = inds
    x, y, z = p
    ret = a * (a-1) * y^4
    ret -= (2a * b + a + b) * x^2 * y^2
    ret += b * (b-1) * x^4
    ret *= x^(a-2) * y^(b-2) * z^c
    ret
end
function getrhoupfrommononial(inds, p)
    a, b, c = inds
    x, y, z = p
    θ = atan(y / x)
    rhoz = rhoval(z)
    ret = (a + b) * z^2
    ret -= c * rhoz^2
    ret *= x^a * y^b * z^(c-1)
    ret
end
function getrho2uppfrommononial(inds, p)
    a, b, c = inds
    x, y, z = p
    θ = atan(y / x)
    rhoz = rhoval(z)
    ret = (a + b) * (a + b - 1) * z^4
    ret -= (2c * (a + b) + a + b + c) * z^2 * rhoz^2
    ret += c * (c - 1) * rhoz^4
    ret *= x^a * y^b * z^(c-2)
    ret
end






#==============#
# Saving to disk

# Clenshaw mats
this
sb = load("experiments/saved/sphericalcap/sphericalcap-alpha=0.2-clenshawmats-B-BF.jld", "B"); resize!(S.B, length(sb)); S.B[:] = sb[:]
sc = load("experiments/saved/sphericalcap/sphericalcap-alpha=0.2-clenshawmats-C-BF.jld", "C", S.C); resize!(S.C, length(sc)); S.C[:] = sc[:]
sdt = load("experiments/saved/sphericalcap/sphericalcap-alpha=0.2-clenshawmats-DT-BF.jld", "DT", S.DT); resize!(S.DT, length(sdt)); S.DT[:] = sdt[:]
