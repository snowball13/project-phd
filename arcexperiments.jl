using CircleArc, ApproxFun, SO, BlockArrays

N = 40; h = 0
    D = arc_derivative_operator_in_Q(N, h)
C = arc2Q(N, h)


n = 30; C[Block.(n:n+1), Block.(n:n+1)]



P = initialise_arc_ops(N, h)

j = 11;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P..., cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 4E-15
j = 12;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P..., cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 4E-8
j = 21;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P..., cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 1E-14
j = 22;c = [zeros(j); 1; zeros(2N-j)]
    arc_func_eval(c, h, P..., cos(θ), sin(θ)) - Q_func_eval(C*c, h, cos(θ), sin(θ)) # 0.7


norm(C*c)

0.001*C

inv(Matrix(0.01D + C))

f = Fun(θ -> cos(0.01cos(θ)-0.1), Fourier())




ε= 1.0
    u = (ε*Derivative() + I) \ f
    ũ = func_to_coeffs((x,y) -> u(atan2(y,x)), N, h)
    ((ε*D + C) \ f̃) - ũ |> norm

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
