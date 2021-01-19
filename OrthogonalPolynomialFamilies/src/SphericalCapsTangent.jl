# Spherical/Polar Caps Tangent Space

#=
NOTE

The tangent space basis we use is, owrt w_R^{(a,2)},
        Φ̲^{(a)}_{n,k,i} = ϕ̲ * (1/ρ) * Q^{a}_{n,k,i}
        Ψ̲^{(a)}_{n,k,i} = θ̲ * (1/ρ) * Q^{a}_{n,k,abs(i-1)}
    Let u̲ be a function that is a sum of gradients and perp gradients (hence
    lies in the tangent space). Then we can expand u̲ in this basis, i.e.
        u̲ = Σ u_{n,k,i}\^Φ * Φ̲_{n,k,i} + u_{n,k,i}\^Ψ Ψ̲_{n,k,i}
    where {u\^Φ, u\^Ψ} are coefficients. The coefficients vector is ordered
    as follows:
    i.e. for a given max degree N:
    # TODO sort this out

        𝕋_N := [𝕋_{N,0};...;𝕋_{N,N}]
        𝕋_{N,k} := [Φ̲_{k,k,0};Ψ̲_{k,k,0};Φ̲_{k,k,1};Ψ̲_{k,k,1}...;Φ̲_{N,k,0};Ψ̲_{N,k,0};Φ̲_{N,k,1};Ψ̲_{N,k,1}] ∈ ℝ^{4(N-k+1)}
                                                                for k = 1,...,N
        𝕋_{N,0} := [Φ̲_{0,0,0};Ψ̲_{0,0,1};Φ̲_{1,0,0};Ψ̲_{1,0,1};...;Φ̲_{N,0,0};Ψ̲_{N,0,1}] ∈ ℝ^{2(N+1)}
                                                                for k = 0

=#

export SphericalCapTangentSpace

# T should be Float64, B can be be BigFloat or Float64 (B is the precision of
# the arithmetic)


function SphericalCapTangentSpace(fam::SphericalCapFamily{B,T,N,<:Any,<:Any},
                                    params::NTuple{N,B}
                                    ) where {B,T,N}
    SphericalCapTangentSpace{typeof(fam), B, T, N}(fam, params)
end

spacescompatible(A::SphericalCapTangentSpace, B::SphericalCapTangentSpace) = (A.params == B.params)

function gettangentspace(S::SphericalCapSpace{<:Any,B,T,N}) where {B,T,N}
    D = S.family
    params = S.params
    haskey(D.tangentspaces,params) && return D.spaces[params]
    D.tangentspaces[params] = SphericalCapTangentSpace(D, params)
end

getSCSpace(S::SphericalCapTangentSpace) = S.family(S.params)




#=======#



#===#
# Indexing retrieval methods

function getopindex(S::SphericalCapTangentSpace, n::Int, k::Int, i::Int, j::Int;
                    bydegree=true, N=0)
    # bydegree=true : output the index for the OP given by n,k,i,j when coeffs
    # are ordered by degree
    # bydegree=false : output the index for the OP given by n,k,i,j when coeffs
    # are ordered by Fourier mode k
    # j refers to the Φ (j=0) or Ψ (j=1) OP
    if k > n
        error("Invalid k input to getopindex")
    elseif i > 1 || i < 0
        error("Invalid i input to getopindex")
    elseif k == 0 && i == 1
        error("Invalid inputs to getopindex - i must be zero if k is zero")
    elseif j < 0 || j > 1
        error("Invalid j input to getopindex")
    end
    if bydegree
        # Sum of the number of OPs up to and including degree n-1
        ret = 2 * n^2
        # Now count from the beginning of the deg n OPs
        if k == 0
            ret += 1
        else
            ret += 2k + i
        end
        if j == 1
            # The deg n Ψ OPs are listed in order after the deg n Φ OPs
            ret += 2n + 1
        end
    else # by Fourier mode k
        # N must be set
        if k == 0
            ret = 2 * (n + 1) - 1 + j
        else
            # Sum of the number of OPs up to and including Fourier mode k-1
            ret = 2 * (N + 1) + 4 * sum([N-m+1 for m=1:k-1])
            # Now count from the beginning of the Fourier mode k OP block
            ret += 4 * (n - k) + 2i + j + 1
        end
    end
    ret
end
function getnkij(S::SphericalCapTangentSpace, ind; bydegree=true)
    # j = 0 are the Φ OPs, j = 1 are the Ψ OPs
    if bydegree
        n = 0
        while true
            if 2 * (n+1)^2 ≥ ind
                break
            end
            n += 1
        end
        r = ind - 2 * n^2
        j = 0
        if r > 2n + 1
            j += 1
            r -= 2n + 1
        end
        if r == 1
            n, 0, 0, j
        elseif iseven(r)
            n, Int(r/2), 0, j
        else
            n, Int((r-1)/2), 1, j
        end
    else
        error("Can only do this for bydegree=true")
    end
end



#===#
# Function evaluation (clenshaw)

# Returns the constant that is Q^{a,b}_{0,0,0} ( = Y_{0,0}, so that the Y_ki's
# are normalised)
function getdegzeropteval(::Type{T},
                            S::SphericalCapTangentSpace{<:Any, T, <:Any, <:Any},
                            pt
                            ) where T
    @assert length(pt) == 3 "Invalid pt"
    q = getdegzeropteval(getSCSpace(S))
    x, y, z = pt[1], pt[2], pt[3]
    θ = atan(y / x)
    ϕ̂ = [cos(θ) * z; sin(θ) * z; - S.family.ρ(z)] # TODO make global/CONST?
    θ̂ = [-sin(θ); cos(θ); 0] # TODO make global/CONST?
    q * ϕ̂, q * θ̂
end
function getdegzeropteval(S::SphericalCapTangentSpace{<:Any, T, <:Any, <:Any},
                            pt
                            ) where T
    getdegzeropteval(T, S, pt)
end
function clenshawG(::SphericalCapTangentSpace, n::Int, z)
    sp = sparse(I, 2n+1, 2n+1)
    [z[1] * sp; z[2] * sp; z[3] * sp]
end
function clenshaw(cfs::AbstractVector{T}, ST::SphericalCapTangentSpace, pt) where T
    # Convert the cfs vector from ordered by Fourier mode k, to by degree n
    f = convertcoeffsvecorder(ST, cfs)
    S = getSCSpace(ST)

    m = length(f)
    N = Int(sqrt(m / 2)) - 1
    resizedata!(S, N+1)
    f = PseudoBlockArray(f, [2*(2n+1) for n=0:N])

    Φ0, Ψ0 = getdegzeropteval(T, ST, pt)
    if N == 0
        ret = f[1] * Φ0 + f[2] * Ψ0
    else
        n = N
        fn = PseudoBlockArray(f[Block(n+1)], [2n+1 for i=1:2])
        γ2ϕ = view(fn, Block(1))'
        γ2ψ = view(fn, Block(2))'
        n = N - 1
        fn = PseudoBlockArray(f[Block(n+1)], [2n+1 for i=1:2])
        M1 = S.DT[n+1] * (S.B[n+1] - clenshawG(S, n, pt))
        γ1ϕ = view(fn, Block(1))' - γ2ϕ * M1
        γ1ψ = view(fn, Block(2))' - γ2ψ * M1
        for n = N-2:-1:0
            fn = PseudoBlockArray(f[Block(n+1)], [2n+1 for i=1:2])
            M1 = S.DT[n+1] * (S.B[n+1] - clenshawG(S, n, pt))
            M2 = S.DT[n+2] * S.C[n+2]
            γϕ = view(fn, Block(1))' - γ1ϕ * M1 - γ2ϕ * M2
            γψ = view(fn, Block(2))' - γ1ψ * M1 - γ2ψ * M2
            γ2ϕ = copy(γ1ϕ)
            γ2ψ = copy(γ1ψ)
            γ1ϕ = copy(γϕ)
            γ1ψ = copy(γψ)
        end
        ret = γ1ϕ[1] * Φ0 + γ1ψ[1] * Ψ0
    end
    ret
end
evaluate(cfs::AbstractVector, S::SphericalCapTangentSpace, z) = clenshaw(cfs, S, z)


#===#
# Resizing/Reordering coeffs vectors

# Method to convert coeffs vec from ordered by Fourier mode k to by degree n
function convertcoeffsvecorder(S::SphericalCapTangentSpace, cfs::AbstractVector; todegree=true)
    # TODO
    f = copy(cfs)
    m = length(cfs)
    N = Int(ceil(sqrt(m / 2))) - 1
    if 2(N+1)^2 != m
        error("coeffs vec incorrect length")
    end
    if todegree
        indc = 1
        for k = 0:N
            for n = k:N
                for i = 0:min(1, k) # This catches the k == 0 case
                    f[getopindex(S, n, k, i, 0; bydegree=true)] = cfs[indc]
                    f[getopindex(S, n, k, i, 1; bydegree=true)] = cfs[indc + 1]
                    indc += 2
                end
            end
        end
    else
        # if todegree is false (or not true) then we convert to ordering by
        # Fourier mode k
        indf = 1
        for k = 0:N
            for n = k:N
                for i = 0:min(1, k) # This catches the k == 0 case
                    f[indf] = cfs[getopindex(S, n, k, i, 0; bydegree=true)]
                    f[indf + 1] = cfs[getopindex(S, n, k, i, 1; bydegree=true)]
                    indf += 2
                end
            end
        end
    end
    f
end


#===#
# Resizing coeffs vectors

function resizecoeffs!(S::SphericalCapTangentSpace, f::Fun, N::Int)
    ncfs = length(f.coefficients)
    m = 2 * (N+1)^2
    cfs = convertcoeffsvecorder(S, f.coefficients) # creates new cfs vec
    if ncfs > m && any(i -> abs(i) > 1e-16, cfs[m+1:end])
        error("Trying to decrease degree of f")
    end
    resize!(cfs, m)
    if ncfs < m
        cfs[ncfs+1:end] .= 0.0
    end
    cfs = convertcoeffsvecorder(S, cfs; todegree=false)
    # This last line reassigns the input Fun's coeffs vector with our new
    # extended version
    resize!(f.coefficients, m); f.coefficients[:] = cfs[:]
end



#===#
# Differential and other operators

function getblockindextangent(S::SphericalCapTangentSpace, n::Int, k::Int,
                                i::Int, j::Int)
    getblockindextangent(getSCSpace(S), n, k, i, j)
end

# Divergence operator (∇.)
function getdivergenceval(S::SphericalCapTangentSpace, ptsr, rhoptsr2,
                            rhodxrhoptsr, wr, n::Int, m::Int, k::Int,
                            i::Int, j::Int; weighted::Bool=true)
    # TODO weighted=false case
    s = getSCSpace(S)
    st = differentiatespacephi(s; weighted=weighted)
    R = getRspace(s, k); Rt = getRspace(st, k)
    if j == 0
        ret = inner2(R, getptsevalforop(R, n-k) .* rhoptsr2.^k,
                        (getderivptsevalforop(Rt, m-k) .* rhoptsr2
                            - k * ptsr .* getptsevalforop(Rt, m-k)),
                        wr)
    elseif j == 1
        ret = inner2(R, getptsevalforop(R, n-k) .* rhoptsr2.^k,
                        getptsevalforop(Rt, m-k), wr)
        ret *= (-1)^i * k
    else
        error("invalid param j")
    end
    ret / getopnorm(Rt)
end
function divergenceoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any},
                            N::Int; square::Bool=true) where {B,T}
    # Operator acts on coeffs in the 𝕋_W^{a} basis, and results in coeffs in the
    # 𝕎^{a-1} basis.

    # NOTE the coeffs for scalar funs that result are double the length you'd
    # expect, with each entry duplicated in turn, so as to keep the operator a
    # true BandedBlockBanded object.

    weighted = true # TODO weighted=false
    s = getSCSpace(S)
    st = differentiatespacephi(s; weighted=weighted)
    R = getRspace(s, 0)

    resizedataonedimops!(st, N+1)
    resizedataonedimops!(s, N+1)

    ptsr, wr = pointswithweights(B, R, 2N+3)
    rhoptsr2 = S.family.ρ.(ptsr).^2
    rhodxrhoptsr = S.family.ρ.(ptsr) .* differentiate(S.family.ρ).(ptsr)

    # TODO sort out bandwidths!
    band1 = 1
    band2 = 2
    if square
        A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+1)^2, 2 * (N+1)^2),
                                    2 * [N+1; 2N:-2:1],
                                    2 * [N+1; 2N:-2:1],
                                    (0, 0),
                                    (4band2, 4band1))
    else
        A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+band2+1)^2, 2 * (N+1)^2),
                                    2 * [N+band2+1; 2(N+band2):-2:1],
                                    2 * [N+1; 2N:-2:1],
                                    (0, 0),
                                    (4band2, 4band1))
    end

    for k = 0:N
        if k % 100 == 0
            @show "divoperator", k
        end
        R = getRspace(s, k); Rt = getRspace(st, k)
        getopptseval(R, N-k, ptsr); getopptseval(Rt, N-k+band2, ptsr)
        getderivopptseval(Rt, N-k+band2, ptsr)
        for n = k:N
            maxm = square ? min(n+band2, N) : n+band2
            for m = max(0, n-band1):maxm
                if k ≤ m
                    for i = 0:min(1,k), j = 0:1
                        (j == 1 && (m < n - (band1 - 1) || m > n + (band2 - 1)
                                                        || k == 0)
                                && continue)
                        val = getdivergenceval(S, ptsr, rhoptsr2, rhodxrhoptsr,
                                                wr, n, m, k, i, j)
                        ind1 = getblockindextangent(S, m, k, i, j)
                        ind2 = getblockindextangent(S, n, k, i, j)
                        view(A, Block(k+1, k+1))[ind1, ind2] = val
                    end
                end
            end
        end
        resetopptseval(R); resetopptseval(Rt); resetderivopptseval(R)
    end
    A
end

# gradient of the divergence operator, i.e. ∇(∇.u̲) for u̲ in the tangent space
function gradofdivoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any}, N::Int) where {B,T}
    # NOTE Takes coeffs from 𝕋_W^{1} -> 𝕋^{1} only

    # resizedataonedimops! for the SCSpace
    s = getSCSpace(S)
    resizedataonedimops!(s, N+1)

    band1 = 1
    band2 = 1
    A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+band2+1)^2, 2 * (N+1)^2),
                                2 * [N+band2+1; 2(N+band2):-2:1], 2 * [N+1; 2N:-2:1],
                                (0, 0), (4band2+1, 4band1+1))
end


#===#

# Coriolis operator (2Ωz r̲̂ ×)
function coriolisoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any},
                            N::Int; square::Bool=false) where {B,T}
    # We output the operator that results in coefficients {u\^Φ, u\^Ψ}, for
    # expansion in the 𝕋 basis

    Ω = B(72921) / 1e9 # TODO make this a global definition somehow (maybe a
                       # member of the class struct)

    # resizedataonedimops! for the SCSpace
    s = getSCSpace(S)
    resizedataonedimops!(s, N+1)

    band1 = 1
    band2 = 1
    if square
        A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+1)^2, 2 * (N+1)^2),
                                    2 * [N+1; 2N:-2:1],
                                    2 * [N+1; 2N:-2:1],
                                    (0, 0),
                                    (4band2+1, 4band1+1))
    else
        A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+band2+1)^2, 2 * (N+1)^2),
                                    2 * [N+band2+1; 2(N+band2):-2:1],
                                    2 * [N+1; 2N:-2:1],
                                    (0, 0),
                                    (4band2+1, 4band1+1))
    end
    for k = 0:N
        if k % 100 == 0
           @show "coriolis", k
        end
        for n = k:N
            maxm = square ? min(n+band2, N) : n+band2
            for m = max(0, n-band1):maxm
                if k ≤ m
                    c = 2 * Ω * recγ(B, s, m, k, n-m+2)
                    for i = 0:min(1,k), j = 0:1
                        ind1 = getblockindextangent(S, m, k, i, abs(j-1))
                        ind2 = getblockindextangent(S, n, k, i, j)
                        view(A, Block(k+1, k+1))[ind1, ind2] = c * (-1)^(j)
                    end
                end
            end
        end
    end
    A
end



#====#
# Conversion operator (from 𝕋_W^{a} -> 𝕋_W^{a-1} or 𝕋^{a} -> 𝕋^{a+1})
function transformparamsoperator(S::SphericalCapTangentSpace{<:Any, B, <:Any, <:Any}, N::Int;
                                    weighted::Bool=false, square::Bool=false) where B
    # St refers to the target space
    s = getSCSpace(S)
    st = differentiatespacephi(s; weighted=weighted)
    T = transformparamsoperator(s, st, N; weighted=weighted)
    if weighted
        band1 = 0; band2 = 1
    else
        band1 = 1; band2 = 0
    end
    if square
        C = BandedBlockBandedMatrix(Zeros{B}(2(N+1)^2, 2(N+1)^2),
                                    2 * [N+1; 2N:-2:1],
                                    2 * [N+1; 2N:-2:1],
                                    (0, 0),
                                    (4band2, 4band1))
    else
        C = BandedBlockBandedMatrix(Zeros{B}(2(N+band2+1)^2, 2(N+1)^2),
                                    2 * [N+band2+1; 2(N+band2):-2:1],
                                    2 * [N+1; 2N:-2:1],
                                    (0, 0),
                                    (4band2, 4band1))
    end

    for k = 0:N
        for n = k:N
            maxm = square ? min(n+band2, N) : n+band2
            for m = max(0, n-band1):maxm
                if k ≤ m
                    val = view(T, Block(k+1, k+1))[getblockindex(s, m, k, 0),
                                                   getblockindex(s, n, k, 0)]
                    for i = 0:min(1,k), j = 0:1
                        ind1 = getblockindextangent(S, m, k, i, j)
                        ind2 = getblockindextangent(S, n, k, i, j)
                        view(C, Block(k+1, k+1))[ind1, ind2] = val
                    end
                end
            end
        end
    end
    C
end


#===#
# Weight function mutliplyer

function weightoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any}, a::B,
                        b::B, N::Int; addfactor=false) where {B,T}
    # This operator results in coeffs in the 𝕋 space for multiplication by the
    # weight w_R^{(a,2b)}(z)
    band1 = band2 = Int(a + 2b) + (addfactor ? 1 : 0)
    W = BandedBlockBandedMatrix(Zeros{B}(2(N+band2+1)^2, 2(N+1)^2),
                                2 * [N+band2+1; 2(N+band2):-2:1], 2 * [N+1; 2N:-2:1],
                                (0, 0), (4band2, 4band1))

    S0 = getSCSpace(S)
    Sab = S0.family((a, b))
    resizedataonedimops!(S0, N+band2)

    # Get the operator for mult by ρ²
    ptsr, wr = pointswithweights(B, getRspace(Sab, 0), N+3)
    rhoptsr2 = S.family.ρ.(ptsr).^2
    if addfactor
        f = ptsr
    else
        f = ones(length(ptsr))
    end
    getopnorms(S0, N+band2+1)
    for k = 0:N
        R = getRspace(S0, k)
        getopptseval(R, N-k+band2, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                val = inner2(R, getptsevalforop(R, n-k) .* f,
                                getptsevalforop(R, m-k) .* rhoptsr2.^k,
                                wr)
                for i = 0:min(1,k), j=0:1
                    view(W, Block(k+1, k+1))[getblockindextangent(S, m, k, i, j),
                                             getblockindextangent(S, n, k, i, j)] = val / getopnorm(R)
                end
            end
        end
        resetopptseval(R)
    end
    W
end
function weightoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any}, a::Int,
                        b::Int, N::Int; addfactor=false) where {B,T}
    weightoperator(S, B(a), B(b), N; addfactor=addfactor)
end
# The operator for mult by ρ^2
rho2operator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any}, N::Int) where {B,T} =
    weightoperator(S, 0, 1, N; addfactor=false)



#===#
# "Coeffs degree increaser" operator matrix

function increasedegreeoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any},
                                N, Nto; weighted=false) where {B,T}
    # This operator acts on the coeffs vector of a Fun in the space S to just
    # reorganise the coeffs so that the length is increased from deg N to
    # deg Nto, with all coeffs representing the extra degrees being zero.

    # TODO weighted=true
    C = BandedBlockBandedMatrix(Zeros{B}(2(Nto+1)^2, 2(N+1)^2),
                                2 * [Nto+1; 2(Nto):-2:1], 2 * [N+1; 2N:-2:1],
                                (0, 0), (0, 0))
    @show "incr deg", N, Nto, weighted
    for k = 0:N, n = k:N
        for i = 0:min(1,k), j = 0:1
            view(C, Block(k+1, k+1))[getblockindextangent(S, n, k, i, j),
                                     getblockindextangent(S, n, k, i, j)] = 1.0
        end
    end
    C
end
