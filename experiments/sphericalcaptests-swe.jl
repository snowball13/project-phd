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
                    rho2laplacianoperator, gettangentspace, coriolisoperator,
                    rho2operator, weightoperator, gradientoperator,
                    divergenceoperator, convertcoeffssize
using Makie
# using GeometryTypes
# using PyPlot



isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)

# TODO correct
function gettangentspacecoeffsvec(S::SphericalCapTangentSpace, f::Fun, g::Fun)
    # f and g are the respectives funs for a fun u̲ s.t.
    #   u̲ = ϕ̲ * (1/ρ) * f + θ̲ * (1/ρ) * g
    # Recall the ordering, i.e.
    #   Φ̲^{(a)}_{n,k,i} = ϕ̲ * (1/ρ) * Q^{a}_{n,k,i}
    #   Ψ̲^{(a)}_{n,k,i} = θ̲ * (1/ρ) * Q^{a}_{n,k,abs(i-1)}

    s = OrthogonalPolynomialFamilies.getSCSpace(S)

    fc = f.coefficients; gc = g.coefficients
    m = length(fc)
    @assert m == length(gc) "Coeffs must be same length"
    ret = zeros(2m)

    # it = 1
    # for j = 1:m
    #     ret[it] = fc[j]; ret[it+1] = gc[j]
    #     it += 2
    # end
    # ret

    it = 1
    N = Int(sqrt(m)) - 1
    k = 0; i = 0
    for n = k:N
        ret[it] = fc[getopindex(s, n, k, i; bydegree=false, N=N)]
        ret[it] = gc[getopindex(s, n, k, i; bydegree=false, N=N)]
    end
    for k = 1:N, n = k:N, i = 0:1
        ret[it] = fc[getopindex(s, n, k, i; bydegree=false, N=N)]
        it += 1
        ret[it] = gc[getopindex(s, n, k, abs(i-1); bydegree=false, N=N)]
        it += 1
    end
    ret
end

function squareoperator(S::SphericalCapSpace, A::BandedBlockBandedMatrix, N::Int)
    """ this is to square the given operator, by reassigning the non-zero
        entries correctly for given N value
    """
    @assert (N < nblocks(A)[1] - 1 && N == nblocks(A)[2] - 1) "Invalid size of A or incorrect N val"
    C = BandedBlockBandedMatrix(Zeros{B}((N+1)^2, (N+1)^2),
                                ([N+1; 2N:-2:1], [N+1; 2N:-2:1]),
                                (A.l, A.u), (A.λ, A.μ))
    for k = 1:N
        inds = 1:2(N-k+1)
        view(C, Block(k+1, k+1)) .= view(A, Block(k+1, k+1))[inds, inds]
    end
    k = 0
    view(C, Block(k+1, k+1)) .= view(A, Block(k+1, k+1))[1:N+1, 1:N+1]
    C
end

function plot_on_domain_scalar(S, u, ϕ, θ; weighted=false)
    scene = Scene()
    plot_on_domain_scalar!(scene, S, u, ϕ, θ;weighted=weighted)
end
function plot_on_domain_scalar!(scene, S, u, ϕ, θ; weighted=false)
    x = [cospi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    y = [sinpi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    z = [cospi(ϕ) for ϕ in ϕ, θ in θ]
    if weighted
        upts = [(weight(S, cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ))
                    * u(cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ))) for ϕ in ϕ, θ in θ]
    else
        upts = [u(cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ)) for ϕ in ϕ, θ in θ]
    end
    s = Makie.surface!(scene, x, y, z, color = upts, colormap = :viridis, colorrange = (-H/100, H/100))
    scene, s
end


function linearswenorecord(S0::SphericalCapSpace, Af, ttw, G, D, dt, nsteps, H, uc, hc;
                            method="backwardeuler")
    u0c = copy(uc); h0c = copy(hc)
    for it = 1:nsteps
        t = T(it * dt)
        @show it, t, T(maximum(abs, u0c - uc)), T(maximum(abs, h0c - hc))
        if method == "backwardeuler"
            u1c, h1c = backwardeuler(Af, ttw, G, D, H, dt, u0c, h0c)
        else
            error("invalid method")
            return
        end
        u0c = copy(u1c); h0c = copy(h1c)
    end
    u0c, h0c
end
function linearswe(S0::SphericalCapSpace, vs, scene, Af, ttw, G, D, dt, nsteps, H, uc, hc;
                    method="backwardeuler")
    u0c = copy(uc); h0c = copy(hc)
    record(vs, filename) do io
        recordframe!(io)
        for it = 1:nsteps
            t = T(it * dt)
            @show it, t, T(maximum(abs, u0c - uc)), T(maximum(abs, h0c - hc))
            if method == "backwardeuler"
                u1c, h1c = backwardeuler(Af, ttw, G, D, H, dt, u0c, h0c)
            else
                error("invalid method")
                return
            end
            u = Fun(ST, u1c)
            h = Fun(S0, convertcoeffssize(S0, h1c; totangentspace=false))
            plot_on_domain_scalar!(scene, S0, h, ϕ, θ; weighted=false)
            # plot_on_domain_vector!(scene, S, u, ϕt, θt)
            center!(scene)
            wdth, hght = 40, 600 # width, height of the colorlegend
            colorlegend!(vs, scene[end], raw=true, camera=campixel!, width=(wdth,hght))
            recordframe!(io)
            u0c = copy(u1c)
            h0c = copy(h1c)
        end
    end
    u0c, h0c
end
function backwardeuler(Af, ttw, G, D, H, dt, un, hn)
    u = Af \ (ttw * un + dt * G * hn)
    h = hn - dt * H * D * u
    u, h
end

function gaussianbump(S::SphericalCapSpace, N::Int)
    # x0, y0, z0 = pt[1], pt[2], pt[3]
    z0 = 0.6; y0 = 0.5; x0 = sqrt(1-z0^2-y0^2)
    bump = (x,y,z)->3*exp(-30*((x-x0).^2+(y-y0).^2+(z-z0).^2))
    f = Fun(bump, S, 2(N+1)^2)
    @show "Error:", bump(p...) / f(p)
    f
end

#======#
# SWE

# Setup
T = Float64; B = T#BigFloat
α = 0.2
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
S = SCF(1.0, 0.0); S0 = SCF(0.0, 0.0)
ST = gettangentspace(S)

y, z = B(-234)/1000, B(643)/1000; x = sqrt(1 - z^2 - y^2); p = [x; y; z]; isindomain(p, SCF)
θ = atan(y / x)
resizedataonedimops!(S, 100)
resizedataonedimops!(S0, 100)


# Degree of approximation
N = 10

# Operators
F = coriolisoperator(ST, N; square=true)
tw = transformparamsoperator(ST, N; weighted=true, square=true)
t = transformparamsoperator(ST, N; weighted=false)
D = divergenceoperator(ST, N; square=true)
G = gradientoperator(S0, N; square=true)
ttw = t * tw

# Other constants and timestep
dt = 0.01
nsteps = 50
H = B(1) / 10

# Matrices for the solve using backeuler
A = ttw + dt * ttw * F + dt^2 * H * G * D
Af = factorize(A); 1

######
# Examples
# We split u into u̲ = uϕ * (1/ρ) * ϕ̲ + uθ * (1/ρ) * θ̲, i.e. so that uϕ, uθ are
# expanded in the 𝕎^{(1)} basis

# 1) ICS of u ≡ 0, h = h(z) = c(z - α)
c = H / 100
h0 = Fun((x,y,z) -> c * (z - α), S0, 2(5)^2); h0.coefficients
u0ϕ = Fun((x,y,z) -> 0.0, S, 2(5)^2); u0ϕ.coefficients
u0θ = Fun((x,y,z) -> 0.0, S, 2(5)^2); u0θ.coefficients
resizecoeffs!(S, h0, N); resizecoeffs!(S, u0ϕ, N); resizecoeffs!(S, u0θ, N)
# Combine / reorder the coeffs vecs accordingly
h0c = convertcoeffssize(S0, h0.coefficients)
u0c = gettangentspacecoeffsvec(ST, u0ϕ, u0θ)
# Implement backwards Euler method
linearswenorecord(S0, Af, ttw, G, D, dt, nsteps, H, u0c, h0c)

# 2)
Ω̃ = B(72921) / 1e9
u0ϕ = Fun((x,y,z) -> 0.0, S, 2(1)^2); u0ϕ.coefficients
u0θ = Fun((x,y,z) -> 0.0, S, 2(1)^2); u0θ.coefficients
h0 = gaussianbump(S0, N); h0.coefficients
resizecoeffs!(S, u0ϕ, N); resizecoeffs!(S, u0θ, N); resizecoeffs!(S0, h0, N)
# Combine / reorder the coeffs vecs accordingly
u0c = gettangentspacecoeffsvec(ST, u0ϕ, u0θ)
h0c = convertcoeffssize(S0, h0.coefficients)
# Implement backwards Euler method
# linearswenorecord(Af, ttw, G, D, dt, nsteps, H, u0c, h0c)
# Movie
# NOTE The recording of the plotting only seems to work when this stuff is
#      outside a function
# Create arrays of (x,y,z) points, create scene
n = 20
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [B(2) * (B(0):B(2n-2)) / B(2n-1); B(2)]
n = 20
ϕt = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θt = [(B(0):B(2n-2)) * B(2) / B((2n-1)); B(2)]
scene = Scene()
center!(scene)
plot_on_domain_scalar!(scene, S0, h0, ϕ, θ; weighted=false)
center!(scene)
wdth, hght = 40, 600
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
# Other params needed
makeplot=true
dtstr = string(T(dt))
filename = "experiments/saved/sphericalcap/videos/swe-N=$N-H=$H-dt=$dtstr-nsteps=$nsteps.mp4"
uc, hc = linearswe(S0, vs, scene, Af, ttw, G, D, dt, nsteps, H, u0c, h0c;
                    method="backwardeuler")
# NOTE WHY DOES IT BLOW UP STILL!!!!!!!!!!!!!



#=========================#
# OLD CODE



# ICs
# TODO try simpler ICs? OR fix these ones
# 1) ICs lead to zero time derivatives for u and h (or they are supposed to...)
ũ = 2B(π) / 12; H = B(1); Ω̃ = B(72921) / 1e9
u0ϕ = Fun((x,y,z) -> 0.0, S, 2(1)^2); u0ϕ.coefficients
u0θ = Fun((x,y,z) -> - (ũ * (1 - z^2)
                      * ((2z^2 - 1) * weight(S, x, y, z) - (1 - z^2) * z / 2)
                      ), S, 2*9^2); u0θ.coefficients
u0θ(p) - (- ũ * (1 - z^2) * ((2z^2 - 1) * weight(S, x, y, z) - (1 - z^2) * z / 2))
h0 = Fun((x,y,z) -> ũ * Ω̃ * (1 - z^2) * z^2, S0, 2(5)^2); h0.coefficients
h0(p) - ũ * Ω̃ * (1 - z^2) * z^2
resizecoeffs!(S0, u0ϕ, N)
resizecoeffs!(S0, u0θ, N)
resizecoeffs!(S, h0, N)
gettangentspacecoeffsvec(u0ϕ, u0θ) # TODO ensure that this function is putting the coeffs in the right place!!!
u0 = Fun(ST, gettangentspacecoeffsvec(u0ϕ, u0θ))
# 2) ICs
H = B(1) / 100; Ω̃ = B(72921) / 1e9
u0ϕ = Fun((x,y,z) -> 0.0, S0, 2(1)^2); u0ϕ.coefficients
u0θ = Fun((x,y,z) -> 0.0, S0, 2(1)^2); u0θ.coefficients
icd(λ1,φ1,λ2,φ2) = acos(cos(φ1) * cos(φ2) + sin(φ1) * sin(φ2) * cos(λ1-λ2))
gaussianbump(λ,φ,λc,φc,p) = exp(-icd(λc,φc,λ,φ)^2 * p) * 0.1H
ϕc, θc = B(π) / 5, B(π) / 3; wdth = 1
isindomain([cos(ϕc); cos(θc) * sin(ϕc); sin(θc) * sin(ϕc)], S)
h0 = Fun((x,y,z) -> gaussianbump(atan(y / x), acos(z), ϕc, θc , wdth), S, 2(N)^2); h0.coefficients
h0(p)
resizecoeffs!(S0, u0ϕ, N+1)
resizecoeffs!(S0, u0θ, N+1)
resizecoeffs!(S, h0, N-1)
u0 = Fun(ST, gettangentspacecoeffsvec(u0ϕ, u0θ))


# NOTE The recording of the plotting only seems to work when this stuff is
#      outside a function

# Create arrays of (x,y,z) points
n = 20
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [B(2) * (B(0):B(2n-2)) / B(2n-1); B(2)]
n = 20
ϕt = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θt = [(B(0):B(2n-2)) * B(2) / B((2n-1)); B(2)]

u0c = u0.coefficients
h0c = h0.coefficients
scene = Scene()
center!(scene)
plot_on_domain_scalar!(scene, S, h0, ϕ, θ)
center!(scene)
wdth, hght = 40, 600
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
# Other params needed
tfinal=0.4; makeplot=true
dt = B(1) / 100
dtstr = string(T(dt))
filename = "experiments/saved/sphericalcap/swe-N=$N-H=$H-dt=$dtstr-tfinal=$tfinal.mp4"
nsteps = Int(ceil(tfinal / dt)) # Number of time steps
# NOTE functions are underneath
# A = (dt * (P2 * F)
#         + dt^2 * H * di * (G * D)
#         - 2dt^2 * H * di * (R * D)
#         + di * P) # * increasedegreeoperator(ST, N+2, N+3)
A = A1 + dt * A2 + dt^2 * H * A3 - dt^2 * H * A4
Af = factorize(sparse(A))
Psf = factorize(sparse(Ps))
T(maximum(abs, iterimprove(sparse(A), di * W * u0c + dt * di * W * Gw * h0c) - u0c))
uc, hc = linearswe(vs, scene, u0c, h0c, dt, nsteps; method="backwardeuler")
# NOTE WHY DOES IT BLOW UP STILL!!!!!!!!!!!!!


# Solve `A x = b` for `x` using iterative improvement
# (for BigFloat sparse matrix and vector)
function iterimprove(A::SparseMatrixCSC{T}, b::Vector{T};
                        iters=6, verbose=true) where T
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


function linearswe(vs, scene, uc, hc, dt, nsteps; method="backwardeuler")
    u0c = copy(uc); h0c = copy(hc)
    record(vs, filename) do io
        recordframe!(io)
        for it = 1:nsteps
            t = T(it * dt)
            @show it, t, T(maximum(abs, u0c - uc)), T(maximum(abs, h0c - hc))
            if method == "backwardeuler"
                u1c, h1c = backwardeuler(u0c, h0c, dt)
            elseif method == "RK4"
                u1c, h1c = RK4(Feval, u0c, h0c, dt)
            else
                error("invalid method")
                return
            end
            u = Fun(ST, u1c)
            h = Fun(S, h1c)
            plot_on_domain_scalar!(scene, S, h, ϕ, θ)
            # plot_on_domain_vector!(scene, S, u, ϕt, θt)
            center!(scene)
            wdth, hght = 40, 600 # width, height of the colorlegend
            colorlegend!(vs, scene[end], raw=true, camera=campixel!, width=(wdth,hght))
            recordframe!(io)
            u0c = copy(u1c)
            h0c = copy(h1c)
        end
    end
    u0c, h0c
end

function backwardeuler(un, hn, dt)
    if B == T
        u = Af \ (di * W * un + dt * di * W * Gw * hn)
        h = Psf \ (Ps * hn - dt * H * D * u)
    else
        u = iterimprove(sparse(A), di * W * un + dt * di * W * Gw * hn; verbose=false)
        h = iterimprove(sparse(Ps), Ps * hn - dt * H * D * u; verbose=false)
    end
    u, h
end

function RK4(F, un, hn, dt)
    k11, k12 = F(un, hn)
    k21, k22 = F(un + k11 / 2, hn + k12 / 2)
    k31, k32 = F(un + k21 / 2, hn + k22 / 2)
    k41, k42 = F(un + k31, hn + k32)
    un + dt * (k11 + 2k21 + 2k31 + k41) / 6, hn + dt * (k12 + 2k22 + 2k32 + k42) / 6
end

function Feval(u, h)
    u1 = Lf \ (D * u)
    - sparse(F * u)[1:length(u)] + sparse(G * h)[1:length(u)], - sparse(H * Δ * u1)[1:length(h)]
    # Δu1 = Pf \ (D * u)
    # - sparse(F * u)[1:length(u)] + sparse(G * h)[1:length(u)], - sparse(H * Δu1)[1:length(h)]
end

function plot_on_domain(S, u, h, ϕ, θ, ϕt, θt)
    scene = Scene()
    plot_on_domain_scalar!(scene, S, h, ϕ, θ)
    # plot_on_domain_vector!(scene, S, u, ϕt, θt)
end
function plot_on_domain!(vs, scene, S, u, h, ϕ, θ, ϕt, θt)
    plot_on_domain_scalar!(scene, S, h, ϕ, θ)
    # plot_on_domain_vector!(scene, S, u, ϕt, θt)
    center!(scene)
    wdth, hght = 40, 600 # width, height of the colorlegend
    colorlegend!(vs, scene[end], raw=true, camera=campixel!, width=(wdth,hght))
end
function plot_on_domain_vector!(scene, S, u, ϕ, θ; weighted=false)
    x = [cospi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    y = [sinpi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    z = [cospi(ϕ) for ϕ in ϕ, θ in θ]
    pts = vec(Makie.Point3f0.(x, y, z))
    if weighted
        vals = vec(u.(x, y, z) .* weight.(S, x, y, z)) .* 0.1f0
    else
        vals = vec(u.(x, y, z)) .* 0.1f0
    end
    arrows!(scene, pts, vals, arrowsize = 0.03, linecolor = (:white, 0.6), linewidth = 3)
    scene
end



#=======#
# Poisson
using SpecialFunctions
# f = (x,y,z) -> 1 + erf(5(1 - 10((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2)))
# F = Fun(f, S, 2*(N+1)^2)
N = 60
Δw = laplacianoperator(S, N; square=true)
f = (x,y,z) -> (- 2exp(x)*y*z*(2+x)
                + (weight(S,x,y,z)
                    * exp(x)
                    * (y^3 + z^2*y - 4x*y - 2y))) # Exact solution to Δu = f, where u = wR10(z)*y*exp(x)
F2 = Fun(f, S, 2*(N+1)^2)
F2(p) - f(p...)
ucfs = sparse(Δw) \ F2.coefficients
u = Fun(S, ucfs)
u(p) - exp(p[1]) * p[2]
n = 100
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 600
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/poisson-u=wyexpx-N=$N.png", vs)

# Helmholtz
N = 30
k = 100
z1 = S.family.α; x1 = 0.7; vpt = [x1; sqrt(1 - x1^2 - z1^2); z1]
v = (x,y,z) -> 1 - (3(x - vpt[1])^2 + 5(y - vpt[2])^2 + 2(z - vpt[3])^2)
V = operatorclenshaw(Fun(v, S, 100), S, N)
Δw = laplacianoperator(S, N; square=true)
tw = transformparamsoperator(S, S0, N; weighted=true)
t = transformparamsoperator(S0, S, N+1; weighted=false)
A = sparse(Δw) + k^2 * V * sparse(squareoperator(S, t * tw, N))
f = (x,y,z) -> y * weight(S, x, y, z) * exp(x)
F = Fun(f, S, 2(N+1)^2); F.coefficients
F(p) - f(p...)
ucfs = A \ F.coefficients
u = Fun(S, ucfs)
u(p)
n = 20
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
scene = Scene()
scene, s = plot_on_domain_scalar!(scene, S, u, ϕ, θ)
center!(scene)
wdth, hght = 40, 600
cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
vs = vbox(scene, cl)
Makie.save("experiments/saved/sphericalcap/images/helmholtz-f=wyzexpx-N=$N.png", vs)





#=======#
# Heat Eqn

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


# Function to simulate and plot as a movie the heat equation u_t = c Δu
function heateqn(S::SphericalCapSpace, u0::Fun, N::Int,
                    filename::String; c::T=(1/42), dt::T=0.01, tfinal::T=1.0,
                    makeplot=true) where T

    nsteps = Int(ceil(tfinal / dt)) # Number of time steps

    # Create arrays of (x,y,z) points
    n = 100
    ϕmax = (acos(S.family.α)/π)
    ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
    θ = [(0:2n-2) * 2 / (2n-1); 2]

    # Operators
    R = sparse(convertweightedtononweightedoperator(S, N; Nout=N+3))
    rho2 = Fun((x,y,z)->S.family.ρ(z)^2, S, 20)
    V = operatorclenshaw(rho2, S, N+3)
    Δ = sparse(laplaceoperator(S, S, N))

    # First timestep - use backward Euler (BDF1)
    # BDF1 Δu_{n+1} + k² u_{n+1} = k² u_{n}, where k² = −1 / (c dt)
    k2 = -1 / (dt * c) # Helmholtz frequency for BDF1
    B = k2 * V * R
    A = Δ + B
    it = 1
    @show it, nsteps
    u1cfs = iterimprove(A, B * u0.coefficients)
    u1 = Fun(S, u1cfs)
    if makeplot
        scene, s = plot_on_domain_scalar(S, u1, ϕ, θ)
    end
    if nsteps == 1
        return u1
    end

    # Now use BDF2
    # BDF2 Δu_{n+1} + k² u_{n+1} = k²/3 (4u_{n} − u_{n−1}), where k² = −3 / (2 dt c)
    k2 = -3 / (2 * dt * c) # Helmholtz frequency for BDF2
    B = k2 * V * R
    A = Δ + B
    unm1 = copy(u0)
    un = copy(u1)
    if makeplot
        center!(scene)
        record(scene, filename) do io
            for it = 2:nsteps
                @show it, nsteps
                ucfs = iterimprove(A, (1 / 3) * B * (4 * un.coefficients - unm1.coefficients))
                u = Fun(S, ucfs)
                plot_on_domain_scalar!(scene, S, u, ϕ, θ)
                recordframe!(io) # record a new frame
                unm1 = copy(un)
                un = copy(u)
            end
        end
    else
        for it = 2:nsteps
            @show it, nsteps
            ucfs = iterimprove(A, (1 / 3) * B * (4 * un.coefficients - unm1.coefficients))
            u = Fun(S, ucfs)
            unm1 = copy(un)
            un = copy(u)
        end
    end
    un
end

# Heat Equation (u0 = football function, from https://www.chebfun.org/examples/sphere/SphereHeatConduction.html)
# NOTE the solution given, and the function u0, are *weighted* to give the zero
# BCs (i.e. )
function footballfun(S::SphericalCapSpace, N::Int)
    # u0cfs = zeros(T, (N+1)^2)
    # u0cfs[getopindex(S, 6, 0, 0; bydegree=false, N=N)] = 1
    # u0cfs[getopindex(S, 6, 5, 0; bydegree=false, N=N)] = sqrt(14 / 11)
    # u0 = Fun(S, u0cfs)

    y60 = (x, y, z) -> (1 / 32) * sqrt(13 / π) * (231 * z^6 - 315 * z^4 + 105 * z^2 - 5)
    # y65 = (x, y, z) -> - (3 / 32) * sqrt(1001 / π) * sqrt(1-z^2)^5 * z * cos(5 * atan(y/x)) # * exp(im * 5 * θ)
    y65 = (x, y, z) -> - (3 / 32) * sqrt(1001 / π) * z * (16x^5 - 20x^3 * rhoval(z)^2 + 5x * rhoval(z)^4) # ρ(z)^5 * cos(5θ)
    # TODO u is not expanded properly if we use atan and cos instead of the explicit
    #      polynomial formulations for the SHs.
    u = Fun((x,y,z)->2*(y60(x,y,z) + sqrt(14 / 11) * y65(x,y,z)), S, 2(N+1)^2)
end
dt = 0.01 # Time step
c = 1 / 42 # Diffusion constant
tfinal = 1.0 # Stopping time
N = 10 # Degree
cstr = round(c, digits=5)
filename = "experiments/saved/sphericalcap/heateqn-footballfun-N=$N-c=$cstr-dt=$dt-tfinal=$tfinal.mp4"
u0 = footballfun(S, N); u0.coefficients
u = heateqn(S, u0, N, filename; c=c, dt=dt, tfinal=tfinal, makeplot=true)


# Heat Equation (u0 = gaussian bump)
function gaussianbump(S::SphericalCapSpace, N::Int)
    # x0, y0, z0 = pt[1], pt[2], pt[3]
    z0 = 0.6; y0 = 0.5; x0 = sqrt(1-z0^2-y0^2)
    u = Fun((x,y,z)->3*exp(-30*((x-x0).^2+(y-y0).^2+(z-z0).^2)), S, 2(N+1)^2)
end
dt = 0.01 # Time step
c = 1 / 42 # Diffusion constant
tfinal = 1.0 # Stopping time
N = 20 # Degree
cstr = round(c, digits=5)
filename = "experiments/saved/sphericalcap/heateqn-gaussian-N=$N-c=$cstr-dt=$dt-tfinal=$tfinal.mp4"
u0 = gaussianbump(S, N); u0.coefficients
u = heateqn(S, u0, N, filename; c=c, dt=dt, tfinal=tfinal, makeplot=true)



rho = Fun((x,y,z)->rhoval(z)^2, S, 200)
rhocfsblock = converttopseudo(S, rho.coefficients; converttobydegree=true)
@show rhocfsblock[Block(10)]

L = laplaceoperator(S, S, 10)
L.l, L.u, L.λ, L.μ
L0 = L[Block(1,1)]


function getop(S::SphericalCapSpace, N::Int, n::Int, k::Int, i::Int)
    m = getopindex(S, n, k, i; bydegree=true)
    cfs = zeros((N+1)^2); cfs[m] = 1.0
    cfs = convertcoeffsvecorder(S, cfs; todegree=false)
    Fun(S, cfs)
end
N = 12
L = laplaceoperator(S, S, N)
n, k, i = 7, 0, 0
q = getop(S, N, n, k, i)
lq = convertcoeffsvecorder(S, L * q.coefficients)
@show n, k, i
for i = 1:length(lq)
    v = abs(lq[i])
    if v > 1e-17
        @show i, getnki(S, i), v
    end
end
