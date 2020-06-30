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
using Makie
using GeometryTypes
# using PyPlot


isindomain(pt, D::SphericalCapFamily) = D.α ≤ pt[3] ≤ D.β && norm(pt) == 1.0
isindomain(pt, S::SphericalCapSpace) = isindomain(pt, S.family)

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


#======#
# SWE

# Setup
T = Float64; B = T#BigFloat
α = 0.2
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
S = SCF(1.0, 0.0); S0 = SCF(0.0, 0.0)
ST = gettangentspace(SCF)


# Linear SWE

N = 30

# Operators
F = coriolisoperator(ST, N+2)
D = rhodivminusgradrhodotoperator(ST, N+2)
G = rhogradoperator(S, N)
P = rho2operator(S, S, N)
Pf = factorize(sparse(P))
L = rho2laplacianoperator(S, S, N)
Lf = factorize(sparse(L))
Δ = laplacianoperator(S, N)

# ICs
ũ = 2π / 12; H = 1.0; Ω̃ = 1.0 #B(72921) / 1e9
dt = 0.01
u0ϕ = Fun((x,y,z) -> 0.0, S0, 2(N+3)^2); u0ϕ.coefficients
u0θ = Fun((x,y,z) -> - (ũ * (1 - z^2)
                      * ((2z^2 - 1) * weight(S, x, y, z) - (1 - z^2) * z / 2)
                      ), S0, 2(N+3)^2); u0θ.coefficients
u0 = Fun(ST, gettangentspacecoeffsvec(u0ϕ, u0θ))
h0 = Fun((x,y,z) -> ũ * Ω̃ * (1 - z^2) * z^2, S, 2(N+1)^2); h0.coefficients

# Other params needed
tfinal=0.1; makeplot=true
filename = "experiments/saved/sphericalcap/swe-N=$N-H=$H-dt=$dt-tfinal=$tfinal.mp4"

# NOTE The recording of the plotting only seems to work when this stuff is
#      outside a function

# Create arrays of (x,y,z) points
n = 20
ϕmax = (acos(S.family.α)/π)
ϕ = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θ = [(0:2n-2) * 2 / (2n-1); 2]
n = 20
ϕt = [0; (0.5:n*ϕmax-0.5)/n; ϕmax]
θt = [(0:2n-2) * 2 / (2n-1); 2]

u0c = u0.coefficients
    h0c = h0.coefficients
    scene = Scene()
    center!(scene)
    plot_on_domain_scalar!(scene, S, h0, ϕ, θ)
    center!(scene)
    wdth, hght = 40, 600
    cl = colorlegend(scene[end], raw=true, camera=campixel!, width=(wdth,hght))
    vs = vbox(scene, cl)
    nsteps = Int(ceil(tfinal / dt)) # Number of time steps
    # NOTE functions are underneath
    uc, hc = linearswe(vs, scene, u0c, h0c, dt, nsteps)

function linearswe(vs, scene, uc, hc, dt, nsteps)
    u0c = copy(uc); h0c = copy(hc)
    record(vs, filename) do io
        recordframe!(io)
        for it = 1:nsteps
            @show it, maximum(abs, u0c - uc), maximum(abs, h0c)
            # RK4 timestep
            u1c, h1c = RK4(Feval, u0c, h0c, dt)
            # u1c = Af \ (u0c + dt * G * h0c)
            # h1c = h0c - H * Pm1 * sparse(D * u1c)
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
function plot_on_domain_vector!(scene, S, u, ϕ, θ)
    x = [cospi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    y = [sinpi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    z = [cospi(ϕ) for ϕ in ϕ, θ in θ]
    pts = vec(Makie.Point3f0.(x, y, z))
    vals = vec(u.(x, y, z)) .* 0.1f0
    arrows!(scene, pts, vals, arrowsize = 0.03, linecolor = (:white, 0.6), linewidth = 3)
    scene
end
function plot_on_domain_scalar(S, u, ϕ, θ)
    scene = Scene()
    plot_on_domain_scalar!(scene, S, u, ϕ, θ)
end
function plot_on_domain_scalar!(scene, S, u, ϕ, θ)
    x = [cospi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    y = [sinpi(θ)*sinpi(ϕ) for ϕ in ϕ, θ in θ]
    z = [cospi(ϕ) for ϕ in ϕ, θ in θ]
    upts = [(weight(S, cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ))
                * u(cospi(θ)*sinpi(ϕ), sinpi(θ)*sinpi(ϕ), cospi(ϕ))) for ϕ in ϕ, θ in θ]
    s = Makie.surface!(scene, x, y, z, color = upts, colormap = :viridis, colorrange = (-1.0, 1.0))
    scene, s
end


#=======#

#=======#

# Heat Eqn

# Setup
T = Float64; B = T# BigFloat
α = 0.2
DSF = DiskSliceFamily(B, T, α, 1.0, -1.0, 1.0)
SCF = SphericalCapFamily(B, T, B(α * 1000) / 1000)
a = 1.0
S = SCF(a, 0.0); S2 = DSF(a, a)

y, z = B(-234)/1000, B(643)/1000; x = sqrt(1 - z^2 - y^2); p = [x; y; z]; isindomain(p, SCF)
θ = atan(y / x)
resizedata!(S, 10)


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
