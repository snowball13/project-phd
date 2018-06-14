
include("simulation.jl")
using FastTransforms
using BlockBandedMatrices
using BlockArrays

# Get operator matrix for the dot product of each tangent OP (grad/gradperp Ylm)
# acting on a function in the tangent space. The operator will be a matrix that
# is applied to the coeffs vector of the tangent space function.
# We will use J to represent a matrix corresponding to a dot product.
N = 2
# Operators for each OP in ∇P_1
T1 = Vector{BandedBlockBandedMatrix}(6)
rows = 1:2:2N+1
cols = 6:4:2(2N+1)
l,u = 2,1
λ,μ = 2(2N+1),2(2N+1)
n = 20
θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
for k = 1:6
    println("k = ", k)
    a = (θ,ϕ)->tangent_basis_eval(1,sinpi(θ)*cospi(ϕ),sinpi(θ)*sinpi(ϕ),cospi(θ))[Block(k+2)]
    J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
    jj = 1
    for j=1:size(J)[2]
        println("j = ", j)
        if j+2 >= 2(jj+1)^2
            jj += 1
        end
        b = (θ,ϕ)->tangent_basis_eval(jj,sinpi(θ)*cospi(ϕ),sinpi(θ)*sinpi(ϕ),cospi(θ))[Block(j+2)]
        f = (θ,ϕ)->(a(θ,ϕ).'*b(θ,ϕ))
        F = [f(θ,φ) for θ in θ, φ in φ]
        V = zero(F)
        A_mul_B!(V, FastTransforms.plan_analysis(F), F)
        U = fourier2sph(V)
        for l = 0:N
            c = zeros(Complex128,2l+1)
            c[l+1] = U[l+1,1]
            for i = 1:l
                c[i] = U[i,2(l-i+1)]
                c[end-i+1] = U[i,2(l-i+1)+1]
            end
            J[l^2+1:(l+1)^2, j] = c
        end
    end
    # "Stack" the J's to make the "∇P_1." operator
    T1[k] = J
end
T1

# Obtain the (operator) matrices used in the recurrance relation for ∇P_{n+1}
function clenshaw_matrix_G_operator(n, J_x, J_y, J_z)
    G = Matrix{Matrix{Complex{Float64}}}(6(2n+1),2(2n+1))
    Zx = zeros(J_x)
    Zy = zeros(J_y)
    Zz = zeros(J_z)
    for i=1:2(2n+1)
        for j=1:2(2n+1)
            if i == j
                G[i,j] = J_x
                G[i+2(2n+1),j] = J_y
                G[i+4(2n+1),j] = J_z
            else
                G[i,j] = Zx
                G[i+2(2n+1),j] = Zy
                G[i+4(2n+1),j] = Zz
            end
        end
    end
    return G
end
DT, DTB, DTC = get_clenshaw_matrices(N)

# Execute the reccurrance of the dot product to gain the operators for
# f_n(x) = ∇P_n.x for each n = 1,...,N.
J_x = Jx(N)
J_y = Jy(N)
J_z = Jz(N)
TP = Vector{Matrix{Complex128}}(2(N+1)^2)
view(TP, 1:6) .= T1
A = (-DTB[1]+DT[1]*clenshaw_matrix_G_operator(1, J_x, J_y, J_z))
A*T1
view(TP, 7:16) .= (-DTB[1]+DT[1]*clenshaw_matrix_G_operator(1, J_x, J_y, J_z))*T1
for n = 2:N-1
    view(TP, 2(n+1)^2-1:2(n+2)^2-2) .= (
            (-DTB[n]+DT[n]*clenshaw_matrix_G_operator(n, J_x, J_y, J_z))*view(TP, 2n^2-1:2(n+1)^2-2)
            - DTC[n]*view(TP, 2(n-1)^2-1:2n^2-2))
end
# TP is a vector of matrices, each matrix being the operator for the dot product
# a the tangent basis vector. Let u, v be in the tangent space with coefficient
# vectors u_c, v_c. Then the coeffs of u.v are given by u_c^T*TP^T*v_c.
TP

#===#

# Slevinsky FT testing
function cts_f(a,b,θ,ϕ)
    if θ < 0
        return a(-θ,ϕ).'*b(-θ,ϕ)
    else
        return a(θ,ϕ).'*b(θ,ϕ)
    end
end
a = (θ,ϕ)->tangent_basis_eval(1,sinpi(θ)*cospi(ϕ),sinpi(θ)*sinpi(ϕ),cospi(θ))[Block(3)]
b = (θ,ϕ)->tangent_basis_eval(1,sinpi(θ)*cospi(ϕ),sinpi(θ)*sinpi(ϕ),cospi(θ))[Block(3)]
f = (θ,ϕ)->cts_f(a,b,θ,ϕ)
n = 20
θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
F = [f(θ,φ) for θ in θ, φ in φ]
V = zero(F)
A_mul_B!(V, FastTransforms.plan_analysis(F), F)
UO = fourier2sph(V)
φ = (0:2n-3)*2/(2n-2)
F = [f(θ,φ) for θ in θ, φ in φ]
V = zero(F)
A_mul_B!(V, FastTransforms.plan_analysis(F), F)
UE = fourier2sph(V)

fcoeffs = []
for l = 0:N
    c = zeros(Complex128,2l+1)
    c[l+1] = UO[l+1,1]
    for i = 1:l
        c[i] = UO[i,2(l-i+1)]
        c[end-i+1] = UO[i,2(l-i+1)+1]
    end
    append!(fcoeffs,c)
end
fcoeffs
x,y = 0.8,0.5
z = sqrt(1-x^2-y^2)
θ = acos(z)/π
ϕ = asin(y/sinpi(θ))/π
z = cospi(θ)
y = sin(θ)
f(θ,ϕ)
func_eval(fcoeffs,x,y,z)
sphevaluate(θ*π,ϕ*π,2,-2)
sh_eval(2,-2,x,y,z)
a1 = sphevaluate(θ*π,ϕ*π,2,2)
am1 = sphevaluate(θ*π,ϕ*π,2,-2)
(-1)^2*am1

inds = zeros(Int, 5)
k = 1
for ij in eachindex(UE)
    if abs(UE[ij]) > 1e-7
        inds[k] = ij
        k += 1
    end
end
inds
k
NN = 20
fcoeffs = zeros(Complex64, (NN+1)^2)
fcoeffs[2^2+1] = UO[inds[1]]
fcoeffs[3^2] = UO[inds[2]]
func_eval(fcoeffs,x,y,z)
sphevaluate(θ*π,ϕ*π,2,2)*UO[inds[2]]+sphevaluate(θ*π,ϕ*π,2,-2)*UO[inds[1]]
