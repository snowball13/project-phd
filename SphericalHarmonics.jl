using ApproxFun


m =10
α = β = m
k = 5
J = (κ,α,β,x) -> Fun(Jacobi(β,α), [zeros(k);1])(x)

l,m = 10,5

J(l-m,m,m,0.1)

I : Jacobi(1,1) → Ultraspherical(3/2)

P = (l,m,x) -> (-1)^m*gamma(l+m+1)/(2^m*gamma(l+1))*(1-x^2)^(m/2)*J(l-m,m,m,x)

P(l,m,0.1) - (-21961.951238504218)

Y = (l,m,θ,φ) -> sqrt((2l+1)/(4π) * factorial(1.0l-m)/factorial(1.0l+m)) * exp(im*m*φ) * P(l,m,cos(θ))

Y(l,m,0.1,0.2)
