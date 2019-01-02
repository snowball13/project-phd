include("/Users/sheehanolver/Documents/SUpervision/project-phd/circle-arc/CircleArc.jl")

using CircleArc, ApproxFun, SO, BlockArrays, Plots

N = 100; h = 0
C = arc2Q(N, h)


n = 30; inv(C[Block.(n:n+1), Block.(n:n+1)])



P = initialise_arc_ops(N, h)

θ = 0.1

j = 11;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P, cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 4E-15
j = 12;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P, cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 4E-8
j = 21;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P, cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 1E-14
j = 22;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P, cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 0.7


using SO

D = arc_derivative_operator_in_Q(N, h, P)


#### There exists an extension

f = Fun(θ -> cos(50cos(θ-0.1)-0.1), Fourier())
    f̃ = func_to_Q_coeffs((x,y) -> f(atan2(y,x)), N, h)
    f̃ = PseudoBlockArray(f̃, [1; fill(2,N)])

ε =1.0
u = (ε*Derivative() + I) \ f
    ũ = func_to_coeffs((x,y) -> u(atan2(y,x)), N, h)
    ũ = PseudoBlockArray(ũ, [1; fill(2,N)])
    ((ε*D + C) \ f̃) - ũ |> norm

norm((ε*D + C) * ũ - f̃)


plot(abs.(f̃[Block.(1:50)]); yscale=:log10)





#### Playground


svdvals(Matrix(ε*D + C))

ε = 0.1
    m = floor(Int,1/sqrt(ε))
    [[zeros(m-1);  1 ; zeros(size(D,2)-m)]' ;
        (ε*D + C)] |> svdvals |> minimum


ε = 0.05
    m = floor(Int,1/sqrt(ε))
    [[zeros(m-1);  1 ; zeros(size(D,2)-m)]' ;
     [zeros(m);  1 ; zeros(size(D,2)-m-1)]' ;
        (ε*D + C)] |> svdvals |> plot
D[Block(30,30)]

m = floor(Int,1/(ε))
    plot(abs.((im*ε*D + C)[m+1:end-1,m+2:end] |> inv |> M -> M[:,1]); yscale=:log10)


f = Fun(θ -> cos(0.01cos(θ)-0.1), Fourier())

plot(abs.(svd(ε*D + C)[end][end,:]); yscale=:log10)



C

D[Block(30,31)]
using Plots
(ε*D + C)
D

C

zeros(m-1);  1 ; zeros(size(D,2)-m-1)]'
(ε*D + C)

Matrix(ε*D + C)[1:10,1:10] |> svdvals |> minimum
L[1:10,1:10] |> Matrix |> svdvals |> minimum
ε = 0.005; N = 1000
    m = floor(Int,1/ε)

    L = (im*ε*Derivative(Chebyshev(-π/2 .. π/2)) + I)

    [[zeros(m-1);  1 ; zeros(N-m)]' ; L[1:N,1:N]] |> svdvals |> minimum

Matrix(0.1D + C)


1.0D + C

f = Fun(θ -> cos(15cos(θ)-0.1), Fourier())
ncoefficients(f)
f̃ = func_to_Q_coeffs((x,y) -> f(atan2(y,x)), N, h)

(ε*D+C) \ f̃
ε =1.0
u = (ε*Derivative() + I) \ f
    ũ = func_to_coeffs((x,y) -> u(atan2(y,x)), N, h)
    ((ε*D + C) \ f̃) - ũ |> norm


(ε*D + C)*ũ - f̃ |> norm


plot(abs.(ũ); yscale=:log10)
    plot!(abs.(f̃); yscale=:log10)
D
C
n = 30; (D + C)[Block(N,N)] \ (D + C)[Block(N,N+1)]
using BlockArrays
A = (D + C)
b = PseudoBlockArray(f̃, [1; fill(2,N)])
ũ = PseudoBlockArray(ũ, [1; fill(2,N)])


A

A*ũ - f̃


A[Block(n,n)]*ũ[Block(n)] - b[Block(n)]


n = 26
    ũ[Block(n)] - (A[Block(n,n)] \ b[Block(n)]) |> norm

 (A[Block(n,n)] \ b[Block(n)])

3


ũ


b[Block(25)]



ũ[2*25-2:2*25-1]
plot(ũ)



plot(abs.((D + C)[1:50,1:50] \ f̃[1:50] ); yscale=:log10)
    plot!(abs.((D + C) \ f̃ ); yscale=:log10)
plot(abs.(ũ ); yscale=:log10)
    plot!(abs.(f̃ ); yscale=:log10)


ε
(ε*D + C) |> svdvals

using Plots
ε = 0.0001
    v = [ldirichlet();
        ε*Derivative(Chebyshev(0 .. π)) + I] \ [0.0; Fun(f, 0..π)]
        plot(abs.(v.coefficients); yscale=:log10)
ε = 0.1
    v = (ε*Derivative(Chebyshev(0 .. π)) + I) \ Fun(θ -> cos(10cos(θ)-0.1), 0..π)
        plot(abs.(v.coefficients); yscale=:log10)

ε = 1.0
    L = (ε*Derivative(Chebyshev(0 .. π)) + I)
svdvals(Matrix(L[1:21,1:21]))


ε = 1.0
    L = (ε*D + C)


L[end-5:end, end-5:end] |> chopm

f̃ = func_to_Q_coeffs((x,y) -> f(atan2(y,x)), N, h)




θ = 0.1
    arc_func_eval(ũ,h, P..., cos(θ), sin(θ))


u(θ)

f(θ) - Q_func_eval(f̃, h, cos(θ), sin(θ))

cos(0.01cos(θ)-0.1)-f(θ)


Q_func_eval(ũ,h, cos(θ), sin(θ))

(ε*D + C)*ũ - f̃

















ncoefficients(f)
