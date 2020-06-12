# Spherical/Polar Caps Tangent Space

#=
NOTE

The tangent space basis we use is
        Φ̲_{n,k,i} = ϕ̲ * Q^{0}_{n,k,i}
        Ψ̲_{n,k,i} = θ̲ * Q^{0}_{n,k,abs(i-1)}
    Let u̲ be a function that is a sum of gradients and perp gradients (hence
    lies in the tangent space). Then we can expand ρu̲ in this basis, i.e.
        ρu̲ = Σ u_{n,k,i}\^Φ * Φ̲_{n,k,i} + u_{n,k,i}\^Ψ Ψ̲_{n,k,i}
    where {u\^Φ, u\^Φ} are coefficients. The coefficients vector is ordered
    as follows:
    i.e. for a given max degree N:

        𝕋_N := [𝕋_{N,0};...;𝕋_{N,N}]
        𝕋_{N,k} := [Φ̲_{k,k,0};Ψ̲_{k,k,0};Φ̲_{k,k,1};Ψ̲_{k,k,1}...;Φ̲_{N,k,0};Ψ̲_{N,k,0};Φ̲_{N,k,1};Ψ̲_{N,k,1}] ∈ ℝ^{4(N-k+1)}
                                                                for k = 1,...,N
        𝕋_{N,0} := [Φ̲_{0,0,0};Ψ̲_{0,0,0};Φ̲_{1,0,0};Ψ̲_{1,0,0};...;Φ̲_{N,0,0};Ψ̲_{N,0,0}] ∈ ℝ^{2(N+1)}
                                                                for k = 0

=#

export SphericalCapTangentSpace

# T should be Float64, B can be be BigFloat or Float64 (B is the precision of
# the arithmetic)


function SphericalCapTangentSpace(fam::SphericalCapFamily{B,T,N,<:Any,<:Any}, params::NTuple{N,B}) where {B,T,N}
    SphericalCapTangentSpace{typeof(fam), B, T, N}(
        fam, params, Vector{SparseMatrixCSC{B}}(), Vector{SparseMatrixCSC{B}}(),
        Vector{SparseMatrixCSC{B}}(), Vector{SparseMatrixCSC{B}}())
end

spacescompatible(A::SphericalCapTangentSpace, B::SphericalCapTangentSpace) = (A.params == B.params)

function gettangentspace(D::SphericalCapFamily{B,T,N,<:Any,<:Any}) where {B,T,N}
    length(D.tangentspace) == 1 && return D.tangentspace[1]
    resize!(D.tangentspace, 1)
    params = B.(0.0, 0.0)
    D.tangentspace[1] = SphericalCapTangentSpace(D, params)
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
    elseif j < 1 || j > 2
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
            ret = 2 * (n + 1)
        else
            # Sum of the number of OPs up to and including Fourier mode k-1
            ret = 2 * (N + 1) + 4 * sum([N-j+1 for j=1:k-1])
            # Now count from the beginning of the Fourier mode k OPs
            ret += 4 * (n - k) + i + j + 1
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

function resizedata!(S::SphericalCapTangentSpace, N)
    N₀ = length(S.DT)
    N ≤ N₀ - 2 && return S
    @show "begin resizedata! for SphericalCapTangentSpace", N
    resizedata!(getSCSpace(S), N)
    getDTs!(S, N+1, N₀)
    S
end

# Returns the constant that is Q^{a,b}_{0,0,0} ( = Y_{0,0}, so that the Y_ki's
# are normalised)
function getdegzeropteval(::Type{T},
                            S::SphericalCapTangentSpace{<:Any, T, <:Any, <:Any},
                            pt
                            ) where T
    @assert length(pt) == 3 "Invalid pt"
    q = getdegzeropteval(getSCSpace(S))
    x, y, z = pt[1], pt[2], pt[3]
    ϕ = acos(z); θ = atan(y / x)
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
function clenshaw(cfs::AbstractVector{T}, S::SphericalCapTangentSpace, pt) where T
    # Convert the cfs vector from ordered by Fourier mode k, to by degree n
    f = convertcoeffsvecorder(S, cfs)

    m = length(f)
    N = Int(sqrt(m)) - 1
    resizedata!(S, N+1)
    f = PseudoBlockArray(f, [2(2n+1) for n=0:N])

    Φ0, Ψ0 = getdegzeropteval(T, S, pt)
    if N == 0
        f[1] * Φ0 + f[2] * Ψ0
    else
        n = N
        fn = PseudoBlockArray(f, [2n+1 for i=1:2])
        γ2ϕ = view(fn, Block(1))'
        γ2ψ = view(fn, Block(2))'
        n = N - 1
        fn = PseudoBlockArray(f, [2n+1 for i=1:2])
        M1 = S.DT[N] * (S.B[N] - clenshawG(S, n, pt))
        γ1ϕ = view(fn, Block(1))' - γ2ϕ * M1
        γ1ψ = view(fn, Block(1))' - γ2ψ * M1
        for n = N-2:-1:0
            fn = PseudoBlockArray(f, [2n+1 for i=1:2])
            M1 = S.DT[n+1] * (S.B[n+1] - clenshawG(S, n, pt))
            M2 = S.DT[n+2] * S.C[n+2]
            γϕ = view(fn, Block(1))' - γ1ϕ * M1 - γ2ϕ * M2
            γψ = view(fn, Block(2))' - γ1ψ * M1 - γ2ψ * M2
            γ2ϕ = copy(γ1ϕ)
            γ2ψ = copy(γ1ψ)
            γ1ϕ = copy(γϕ)
            γ1ψ = copy(γψ)
        end
        γ1ϕ * Φ0 + γ1ψ * Ψ0
    end
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
                    f[getopindex(S, n, k, i, 1; bydegree=true)] = cfs[indc]
                    f[getopindex(S, n, k, i, 2; bydegree=true)] = cfs[indc + 1]
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
                    f[indf] = cfs[getopindex(S, n, k, i, 1; bydegree=true)]
                    f[indf + 1] = cfs[getopindex(S, n, k, i, 2; bydegree=true)]
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

# Divergence operator (ρ²∇.)
function getrho2divval(S::SphericalCapTangentSpace, ptsr, rhoptsr2,
                        rhodxrhoptsr, wr, n::Int, m::Int, k::Int, i::Int, j::Int)
    S0 = getSCSpace(S)
    S1 = differentiatespacephi(S0)
    if j == 0
        ret = getpartialphival(S0, ptsr, rhoptsr2, rhodxrhoptsr, wr, n, m, k)
    elseif j == 1
        R0 = getRspace(S0, 0); R1 = getRspace(S1, 0)
        ret = inner2(R1, getptsevalforop(R0, n-k) .* rhoptsr2.^k,
                        getptsevalforop(R1, m-k), wr)
        ret *= k * (-1)^(i+1)
    else
        error("invalid param j")
    end
    ret
end
function rho2divoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any}, N) where {B,T}
    # Operator acts on coeffs in the 𝕋 basis, and results in coeffs in the ℚ^{1}
    # basis.
    S0 = getSCSpace(S)
    S1 = differentiatespacephi(S0)
    R0 = getRspace(S0, 0)
    ptsr, wr = pointswithweights(B, R0, N+3)
    rho2ptsr = S.family.ρ.(ptsr).^2
    rhodxrhoptsr = S.family.ρ.(ptsr) .* differentiate(S.family.ρ).(ptsr)

    band1 = 2
    band2 = 1
    A = BandedBlockBandedMatrix(Zeros{B}((N+band2+1)^2, 2 * (N+1)^2),
                                ([N+band2+1; 2(N+band2):-2:1], 2 * [N+1; 2N:-2:1]),
                                (0, 0), (2band2, 2band1))
    for k = 0:N
        if k % 100 == 0
            @show "rho2div", k
        end
        R0 = getRspace(S0, k); R1 = getRspace(S1, k)
        getopptseval(R0, N-k, ptsr); getopptseval(R1, N-k+band2, ptsr)
        for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
            if k ≤ m
                for i = 0:min(1,k), j = 0:1
                    # TODO check indexing here
                    val = getrho2divval(S, ptsr, rhoptsr2, rhodxrhoptsr, wr, n, m, k, i, j)
                    view(A, Block(k+1, k+1))[getblockindex(S1, m, k, i), getblockindextangent(S, n, k, abs(i-j), j)] = val
                end
            end
        end
        resetopptseval(R)
    end
    A
end

# Coriolis operator (2Ωz r̲̂ ×)
function coriolisoperator(S::SphericalCapTangentSpace{<:Any, B, T, <:Any}, N) where {B,T}
    # We output the operator that results in coefficients {u\^Φ, u\^Φ}, for
    # expansion in the 𝕋 basis

    Ω = B(72921) / 1e9 # TODO make this a global definition somehow (maybe a
                       # member of the class struct)

   # resizedataonedimops! for the SCSpace
   S0 = getSCSpace(S)
   resizedataonedimops!(S0, N+1)

   band1 = 1
   band2 = 1
   A = BandedBlockBandedMatrix(Zeros{B}(2 * (N+band2+1)^2, 2 * (N+1)^2),
                               (2 * [N+band2+1; 2(N+band2):-2:1], 2 * [N+1; 2N:-2:1]),
                               (0, 0), (2band2, 2band1))
   for k = 0:N
       if k % 100 == 0
           @show "coriolis", k
       end
       for n = k:N, m = max(0, n-band1):n+band2 # min(n+band2, N)
           if k ≤ m
               c = 2 * Ω * recγ(B, S0, m, k, n-m+2)
               for i = 0:min(1,k), j = 0:1
                   view(A, Block(k+1, k+1))[getblockindextangent(S, m, k, i, abs(j-1)), getblockindextangent(S, n, k, i, j)] = c * (-1)^j
               end
           end
       end
   end
   A
end
