tol = 1e-8
a, b = 1.0, 1.0;
x, y = 0.4, -0.2; z = [x; y] # Test point
St = (S.family)(S.a+1, S.b+1)
maxop = 150
N = 30
C = increaseparamsoperator(S, N)

a = b = BigFloat(1)

@which HalfDiskFamily()

D = HalfDiskFamily(); S = D(a-1, b-1)
H = (S.H)(S.a, S.b + 30 + 0.5)
S = H
n = 10
N₀ = length(S.a) - 1
n ≤ N₀ + 1 && return S
resize!(S.a, n)
resize!(S.b, n)
resize!(S.ops, n + 1)

# x = Fun(identity, space(w)) # NOTE: space(w) does not "work" sometimes
w, P, β, γ, N₀ = S.weight, S.ops, S.a, S.b, N₀

space(w)
x = Fun(identity, BigFloat(0)..1)
S

2S.params

BigFloat(30.5)

w  = (1-x^2)^(BigFloat(30.5))

@which roots(1-x^2)

f = 1-x^2

((1-x^2)^b)(BigFloat(1)/10) -
    w(BigFloat(1)/10)

f(0.1)

d = domain(f)
c = f.coefficients
vscale = maximum(abs,values(f))
if vscale == 0
    throw(ArgumentError("Tried to take roots of a zero function."))
end

hscale = maximum( [abs(leftendpoint(d)), abs(rightendpoint(d))] )
ht = eps(2000.)*max(hscale, 1)  # TODO: choose tolerance better

# calculate Flaot64 roots
r = Array{BigFloat}(ApproxFun.rootsunit_coeffs(convert(Vector{Float64},c./vscale), Float64(ht)))
# Map roots from [-1,1] to domain of f:
rts = fromcanonical.(Ref(d),r)
fp = differentiate(f)

f(0.1)
1-0.1^2
roots(1-x^2)

rts .-=f.(rts)./fp.(rts)

ApproxFun.extrapolate.(fp,)


fp(1.0)


fp.(rts)

rts .-=f.(rts)./fp.(rts)

r = rts[1]
f(r)/fp(r)

# do Newton 3 time
for _ = 1:3
    rts .-=f.(rts)./fp.(rts)
end


S.weight.coefficients

b = BigFloat(30.5)


(1+x)^b |> domain|>typeof

Fun(JacobiWeight(0,b,domain(x)),BigFloat[1])|> domain|>typeof

w = (1+x)^b*Fun(JacobiWeight(0,b,domain(x)),[2^(-b)])

N₀ = 1
f1 = Fun(1/sqrt(sum(w)),space(x))
P[1] = f1
v = x*P[1]
β[1] = sum(w*v*P[1])
v = v - β[1]*P[1]
γ[1] = sqrt(sum(w*v^2))
P[2] = v/γ[1]



N = length(β)
x = Fun(identity, domain(w))

if N₀ <= 0
    N₀ = 1
    f1 = Fun(1/sqrt(sum(w)),space(x))
    P[1] = f1
    v = x*P[1]
    β[1] = sum(w*v*P[1])
    v = v - β[1]*P[1]
    γ[1] = sqrt(sum(w*v^2))
    P[2] = v/γ[1]
end

for k = N₀+1:8
    @show k
    global β, P, x
    v = x*P[k] - γ[k-1]*P[k-1]
    β[k] = sum(w*v*P[k])
    v = v - β[k]*P[k]
    γ[k] = sqrt(sum(w*v^2))
    P[k+1] = v/γ[k]
end

k = 9
v = x*P[k] - γ[k-1]*P[k-1]
β[k] = sum(w*v*P[k])
v = v - β[k]*P[k]
(sum(w*v^2))



using Plots
w.(xx)
xx = 0:BigFloat(1)/1000:1; plot(xx, abs.((w*v^2).(xx)) .+ 1E-100; yscale=:log10, label="w", legend=:bottomleft)
plot!(v^2, label="v^2")

plot(v^2;ylims=(0,4))

plot(((w*v^2)), label="w*v^2", legend=:bottomright)

plot(w*v^2)



(v^2)(0.1)
v(0.1)^2
w(0.1)

S.ops[:], S.a[:], S.b[:] = lanczos!(S.weight, S.ops, S.a, S.b, N₀)


resizedata!(H,10)


recα(Float64, H, 20)


H

resizedata!(S.family(a,b), 40)




S.family(a,b)
