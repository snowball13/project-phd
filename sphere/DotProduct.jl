
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
function function2sph(f)
    n = 20
    θ = (0.5:n-0.5)/n * π
    φ = (0:2n-2)*2/(2n-1) * π
    F = [f(θ,φ) for θ in θ, φ in φ]
    V = zero(F)
    A_mul_B!(V, FastTransforms.plan_analysis(F), F)
    M = size(V,2)
    P = eye(Complex128,M)
    for k = 2:4:M
        P[k:k+1,k:k+1] = [im im;
                      1 -1]/sqrt(2)
     end
    for k = 4:4:M
      P[k:k+1,k:k+1] = [im -im;
                    1 1]/sqrt(2)
    end
    return (V |> fourier2sph) * P
end
function gradP1_dot_prodcuct_operators(N)
    gradP1length = 6
    T1 = Vector{BandedBlockBandedMatrix}(gradP1length)
    rows = 1:2:2N+1
    cols = 6:4:2(2N+1)
    l,u = 2,1
    λ,μ = 2(2N+1),2(2N+1)
    for k = 1:gradP1length
        println("k = ", k)
        a = (θ,ϕ)->tangent_basis_eval(1,sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))[Block(k+2)]
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        jj = 1
        for j=1:size(J)[2]
            println("j = ", j)
            if j+2 >= 2(jj+1)^2
                jj += 1
            end
            b = (θ,ϕ)->tangent_basis_eval(jj,sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))[Block(j+2)]
            f = (θ,ϕ)->(a(θ,ϕ).'*b(θ,ϕ))
            U = function2sph(f)
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
    return T1
end

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

# Execute the reccurrance of the dot product to gain the operators for
# f_n(x) = ∇P_n.x for each n = 1,...,N.
function tangent_op_dot_product_operators(N1, N2)
    N = N2
    DT, DTB, DTC = get_clenshaw_matrices(N)
    J_x = Jx(N)
    J_y = Jy(N)
    J_z = Jz(N)
    TP = Vector{Matrix{Complex128}}(2(N1+1)^2)
    view(TP, 1:6) .= T1
    A = (-DTB[1]+DT[1]*clenshaw_matrix_G_operator(1, J_x, J_y, J_z))
    A*T1
    view(TP, 7:16) .= (-DTB[1]+DT[1]*clenshaw_matrix_G_operator(1, J_x, J_y, J_z))*T1
    for n = 2:N1-1
        view(TP, 2(n+1)^2-1:2(n+2)^2-2) .= (
                (-DTB[n]+DT[n]*clenshaw_matrix_G_operator(n, J_x, J_y, J_z))*view(TP, 2n^2-1:2(n+1)^2-2)
                - DTC[n]*view(TP, 2(n-1)^2-1:2n^2-2))
    end
    # TP is a vector of matrices, each matrix being the operator for the dot product
    # a the tangent basis vector. Let u, v be in the tangent space with coefficient
    # vectors u_c, v_c. Then the coeffs of u.v are given by u_c^T*TP^T*v_c.
    return TP
end

# Function to put this all together
function tangent_space_dot_product(u, v)
    M1 = length(u)
    N1 = round(Int, sqrt(M1/2) - 1)
    @assert (M1 > 0 && sqrt(M1/2) - 1 == N1) "invalid length of u"
    M2 = length(v)
    N2 = round(Int, sqrt(M2/2) - 1)
    @assert (M2 > 0 && sqrt(M2/2) - 1 == N2) "invalid length of v"
    TP = tangent_op_dot_product_operators(N1, N2)
    v2 = view(v, 3:M2)
    ret = zeros(Complex128, M1)
    for i=1:M1
        ret += u[i]*TP[i]*v2
    end
    return ret
end



# Test example
N = 2
u = rand(2(N+1)^2)
v = rand(2(N+1)^2)
tangent_space_dot_product(u, v)
TP = tangent_op_dot_product_operators(N)
M1 = length(u)
N1 = round(Int, sqrt(M1/2) - 1)
@assert (M1 > 0 && sqrt(M1/2) - 1 == N1) "invalid length of u"
M2 = length(v)
N2 = round(Int, sqrt(M2/2) - 1)
@assert (M2 > 0 && sqrt(M2/2) - 1 == N2) "invalid length of v"
TP = tangent_op_dot_product_operators(N1, N2)
v2 = view(v, 3:M2)
ret = zeros(Complex128, round(Int,M1/2))
i = 1
TP[i]*v2
ret += u[i]*TP[i]*v2
for i=1:M1
    ret += u[i]*TP[i]*v2
end


#===#

# Slevinsky FT testing
a = (θ,ϕ)->tangent_basis_eval(1,sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))[Block(3)]
b = (θ,ϕ)->tangent_basis_eval(1,sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))[Block(3)]
f = (θ,ϕ)->a(θ,ϕ).'*b(θ,ϕ)
f = (θ,ϕ)->(sin(θ)*cos(ϕ))^2+im
U = function2sph(f)
x,y = 0.8,0.5
z = sqrt(1-x^2-y^2)
θ = acos(z)
ϕ = asin(y/sin(θ))

UN = size(U)[1]-1
fcoeffs = zeros(Complex128,(UN+1)^2)
k = 1
for l = 0:size(U)[1]-1
    c = zeros(Complex128,2l+1)
    c[l+1] = U[l+1,1]
    for i = 1:l
        c[i] = U[i,2(l-i+1)]
        c[end-i+1] = U[i,2(l-i+1)+1]
    end
    view(fcoeffs, k:k+2l) .= c
    k += 2l+1
end
f(θ,ϕ) ≈ func_eval(fcoeffs,x,y,z)





#
# using FastTransforms
# function doubleFeval(G, θ, φ)
#    ret = 0
#    for k = 1:size(G,1), j=1:size(G,2)
#        l = k-1
#        m = (isodd(j) ? 1 : (-1)) * (j ÷ 2 )
#        ret += G[k,j] * exp(im*m*φ)/sqrt(2π) * (iseven(m) ? cos(l*θ) : sin((l+1)*θ))
#    end
#    ret
# end
# F = zeros(10,10)
# m = -1; l = 3;
# j = (m ≥ 0 ? 2m+1 : 2(-m))
# k = l+1-abs(m)
# F[k,j] = 1
# G = sph2fourier(F)
# sphevaluate(0.1,0.2,3,-1)
# doubleFeval(G, 0.1, 0.2) ≈ 0.12490839648307457 - 0.02532018548641539im
#
#


#
# f = (θ,φ) -> sh_eval(0, 0, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 2sh_eval(1, 0, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 3sh_eval(2, 0, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 4sh_eval(1, -1, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 5sh_eval(1, 1, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 6sh_eval(2, 1, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 7sh_eval(2, -2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 8sh_eval(3, -2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 9sh_eval(2, 2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 10sh_eval(3, 2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 11sh_eval(3, -3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 12sh_eval(4, -3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 13sh_eval(3, 3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 14sh_eval(4, 3, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 15sh_eval(4, 4, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ)) +
#                 16sh_eval(6, 2, sin(θ)*cos(φ), sin(θ)sin(φ), cos(θ))
#     n = 20
#     θ = (0.5:n-0.5)/n * π
#     φ = (0:2n-2)*2/(2n-1) * π
#     F = [f(θ,φ) for θ in θ, φ in φ]
#     V = zero(F)
#     A_mul_B!(V, FastTransforms.plan_analysis(F), F)
#     M = size(V,2)
#     P = eye(Complex128,M)
#     for k = 2:4:M
#         P[k:k+1,k:k+1] = [im im;
#                       1 -1]/sqrt(2)
#       end
#       for k = 4:4:M
#           P[k:k+1,k:k+1] = [im -im;
#                         1 1]/sqrt(2)
#         end
#         round.(Int,real((V |> fourier2sph) * P))
