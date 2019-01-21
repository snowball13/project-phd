using ApproxFun, GenericLinearAlgebra, SO, Plots, BlockArrays

struct RecurrenceCoefficients{T,W,X}
    w::W
    x::X
    P::Array{Fun}
    α::Vector{T}
    β::Vector{T}
    gauss::Dict{Int,NTuple{2,Vector{T}}}
end

function RecurrenceCoefficients(w::Fun{S,T}) where {S,T}
    x = Fun(identity,space(w))
    f1=Fun(1/sqrt(sum(w)),space(x))
    N = 1
    P = Array{Fun}(N + 1)
    α = Array{eltype(w)}(N)
    β = Array{eltype(w)}(N)
    P[1] = f1
    v = x*P[1]
    α[1] = sum(w*v*P[1])

    v = v - α[1]*P[1]
    β[1] = sqrt(sum(w*v^2))

    P[2] = v/β[1]
    RecurrenceCoefficients{T,typeof(w), typeof(x)}(w,x,P,α,β,Dict{Int,NTuple{2,Vector{T}}}())
end


function resizedata!(R::RecurrenceCoefficients, N)
    P, α, β, x, w = R.P, R.α, R.β, R.x, R.w
    N_old = length(α)
    N ≤ N_old && return R
    resize!(P, N+1)
    resize!(α, N)
    resize!(β, N)
    for k = N_old+1:N
        v = x*P[k] - β[k-1]*P[k-1]
        α[k] = sum(w*v*P[k])
        v = v - α[k]*P[k]
        β[k] = sqrt(sum(w*v^2))
        P[k+1] = v/β[k]
    end
    return R
end

function quadrule(R::RecurrenceCoefficients, N)
    haskey(R.gauss, N) && return R.gauss[N]
    resizedata!(R,N)
    T = SymTridiagonal(R.α[1:N], R.β[1:N-1])
    x, V = eig( T)                  # eigenvalue decomposition
    w = sum(R.w)*V[1,:].^2     # Quadrature weights
    R.gauss[N] = (x, vec(w))
end


function laguerreGW( n::Integer, alpha )
# Calculate Gauss-Laguerre nodes and weights based on Golub-Welsch

    alph = 2*(1:n) .+ (alpha-1)           # 3-term recurrence coeffs
    beta = sqrt.( (1:n-1).*(alpha .+ (1:n-1) ) )
    T = SymTridiagonal(Vector(alph), beta)  # Jacobi matrix
    x, V = eig( T)                  # eigenvalue decomposition
    w = gamma(alpha+1)*V[1,:].^2     # Quadrature weights
    x, vec(w)

end

function hermpts_gw(::Type{TT}, n::Integer ) where TT
    # Golub--Welsch algorithm. Used here for n<=20.
    beta = sqrt.(one(TT)/2 .* (1:n-1))              # 3-term recurrence coeffs
    T = SymTridiagonal(zeros(TT,n), beta)  # Jacobi matrix
    (D, V) = eig(T)                      # Eigenvalue decomposition
    indx = sortperm(D)                  # Hermite points
    x = D[indx]
    w = sqrt(TT(π))*V[1,indx].^2            # weights

    # # Enforce symmetry:
    # ii = floor(Int, n/2)+1:n
    # x = x[ii]
    # w = w[ii]
    return (x,w)
end


let (x,w) = (Dict{Tuple{Int,BigFloat},Vector{BigFloat}}(), Dict{Tuple{Int,BigFloat},Vector{BigFloat}}())
    global function gl(n, α_in=0)
        α = BigFloat(α_in)
        haskey(x,(n,α)) && return (x[(n,α)], w[(n,α)])
        xx,ww = laguerreGW(n, α)
        x[(n,α)] = xx
        w[(n,α)] = ww
        xx,ww
    end
end


let xw = Dict{Int,Tuple{Vector{BigFloat},Vector{BigFloat}}}()
    global function gh(n)
        haskey(xw,n) && return xw[n]
        xw[n] = hermpts_gw(BigFloat,n)
    end
end

Base.getindex(R::RecurrenceCoefficients, N) = (resizedata!(R,N); R.P[N])
