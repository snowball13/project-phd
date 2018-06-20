# Get operator matrix for the dot product of each tangent OP (grad/gradperp Ylm)
# acting on a function in the tangent space. The operator will be a matrix that
# is applied to the coeffs vector of the tangent space function.
# We will use J to represent a matrix corresponding to a dot product.

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
        P[k:k+1,k:k+1] = [im im; 1 -1]/sqrt(2)
     end
    for k = 4:4:M
        P[k:k+1,k:k+1] = [im -im; 1 1]/sqrt(2)
    end
    return (V |> fourier2sph) * P
end
function gradP1_dot_prodcuct_operators(N1, N2)
    # Setup
    gradP1length = 6
    T1 = Vector{BandedBlockBandedMatrix}(gradP1length)
    N = N1 + N2
    rows = 1:2:2N+1
    cols = 6:4:2(2N2+1)
    l,u = 3,1
    λ,μ = 2(2N+1),2(2N+1)
    # There is an operator J for each ∇Y in ∇P_1
    for k = 1:gradP1length
        println("k = ", k)
        a = (θ,ϕ)->tangent_basis_eval(1,sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))[Block(k+2)]
        J = BandedBlockBandedMatrix(0.0im*I, (rows,cols), (l,u), (λ,μ))
        jj = 1
        for j=1:size(J)[2]
            println("j = ", j)
            if j+2 > 2(jj+1)^2
                jj += 1
            end
            b = (θ,ϕ)->tangent_basis_eval(jj,sin(θ)*cos(ϕ),sin(θ)*sin(ϕ),cos(θ))[Block(j+2)]
            f = (θ,ϕ)->(a(θ,ϕ).'*b(θ,ϕ))
            U = function2sph(f)
            for n = jj-1:jj+1
                println("n = ", n)
                c = zeros(Complex128,2n+1)
                c[n+1] = U[n+1,1]
                for i = 1:n
                    c[i] = U[i,2(n-i+1)]
                    c[end-i+1] = U[i,2(n-i+1)+1]
                end
                view(J, n^2+1:(n+1)^2,j) .= c
            end
        end
        println("jj = ", jj)
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
function tangent_op_dot_product_operators(N1, N2, T1)
    N = N1 + N2
    DT, DTB, DTC = get_clenshaw_matrices(N)
    J_x = Jx(N).'
    J_y = Jy(N).'
    J_z = Jz(N).'
    TP = Vector{Matrix{Complex128}}(2(N1+1)^2-2)
    # T1 = gradP1_dot_prodcuct_operators(N1)
    view(TP, 1:6) .= T1
    A = DT[1]*clenshaw_matrix_G_operator(1, J_x, J_y, J_z)
    for ij in eachindex(A)
        A[ij] -= DTB[1][ij]I
    end
    view(TP, 7:16) .= A*T1
    for n = 2:N1-1
        A = (DT[n]*clenshaw_matrix_G_operator(n, J_x, J_y, J_z))
        for ij in eachindex(A)
            A[ij] -= DTB[n][ij]I
        end
        view(TP, 2(n+1)^2-1:2(n+2)^2-2) .= (A*view(TP, 2n^2-1:2(n+1)^2-2)
                                            - DTC[n]*view(TP, 2(n-1)^2-1:2n^2-2))
    end
    # TP is a vector of matrices, each matrix being the operator for the dot product
    # a the tangent basis vector. Let u, v be in the tangent space with coefficient
    # vectors u_c, v_c. Then the coeffs of u.v are given by u_c^T*TP^T*v_c.
    return TP
end

# Function to put this all together
function tangent_space_dot_product(u, v, T1)
    M1 = length(u)
    N1 = round(Int, sqrt(M1/2) - 1)
    @assert (M1 > 0 && sqrt(M1/2) - 1 == N1) "invalid length of u"
    M2 = length(v)
    N2 = round(Int, sqrt(M2/2) - 1)
    @assert (M2 > 0 && sqrt(M2/2) - 1 == N2) "invalid length of v"
    TP = tangent_op_dot_product_operators(N1, N2, T1)
    u2 = view(u, 3:M1)
    v2 = view(v, 3:M2)
    ret = zeros(Complex128, (N1+N2+1)^2)
    for i=1:M1-2
        ret += u2[i]*TP[i]*v2
    end
    return ret
end

#
# # Test example
# x,y = 0.8,0.5
# z = sqrt(1-x^2-y^2)
# N1 = 3
# N2 = 4
# u = rand(2(N1+1)^2)
# v = rand(2(N2+1)^2)
# # T1 = gradP1_dot_prodcuct_operators(N1, N2)
# @time b = tangent_space_dot_product(u, v, T1)
# @test func_eval(b, x, y, z) ≈ tangent_func_eval(u, x, y, z).'*tangent_func_eval(v, x, y, z)
#
# #===#
#
# @time T1 = gradP1_dot_prodcuct_operators(N1, N2)
