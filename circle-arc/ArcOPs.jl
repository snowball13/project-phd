using ApproxFun
using BlockBandedMatrices
using BlockArrays


function innerTh(T, U)
    xmin = h
    xmax = 1
    x = Fun(identity, xmin..xmax)
    w = (1./sqrt(1-x^2))
    return sum(w*T*U)
end

function innerUh(T, U)
    xmin = h
    xmax = 1
    x = Fun(identity, xmin..xmax)
    w = sqrt(1-x^2)
    return sum(w*T*U)
end


N = 10
h = 0.5

assert(N > 3)

# Get recurrance relation coefficients for the 1D OPs T_k^h(x), U_k^h(x)
xmin = h
xmax = 1
x = Fun(identity, xmin..xmax)
w = (1./sqrt(1-x^2))
Tkh,α,β = lanczos(w,N)
w = sqrt(1-x^2)
Ukh,γ,δ = lanczos(w,N)

# Construct the operators corresponding to multiplication by x of the vectors T
# and U, given by T = [T_0^h(x),T_1^h,...] and U = [U_0^h(x),U_1^h,...].
JT = Tridiagonal(β[1:end-1],α,β[1:end-1])
JU = Tridiagonal(δ[1:end-1],γ,δ[1:end-1])

# Sanity check
z = 0.9
Tz = [Tkh[k](z) for k=1:length(Tkh)-1]
Uz = [Ukh[k](z) for k=1:length(Ukh)-1]
assert(((JT*Tz)[1:end-1] ≈ z*Tz[1:end-1]) == true)
assert(((JU*Uz)[1:end-1] ≈ z*Uz[1:end-1]) == true)

# Obtain the relationship/conversion coefficients between T_k^h and U_k^h:
#   T_k^h = a_k*U_k^h + b_k*U_(k-1)^h + c_k*U_(k-2)^h
#   y^2*U_(k-1)^h = (1-x^2)*U_(k-1)^h = d_k*U_(k-1)^h + e_k*U_k^h + f_k*U_(k+1)^h
a = zeros(N)
b = copy(a)
c = copy(a)
d = copy(a)
e = copy(a)
f = copy(a)
a[1] = innerUh(Tkh[1], Ukh[1])
a[2] = innerUh(Tkh[2], Ukh[2])
b[2] = innerUh(Tkh[2], Ukh[1])
for k = 3:N
    a[k] = innerUh(Tkh[k], Ukh[k])
    b[k] = innerUh(Tkh[k], Ukh[k-1])
    c[k] = innerUh(Tkh[k], Ukh[k-2])
end
for k=1:N-2
    d[k] = innerUh(Ukh[k], Tkh[k])
    e[k] = innerUh(Ukh[k], Tkh[k+1])
    f[k] = innerUh(Ukh[k], Tkh[k+2])
end
d[N-1] = innerUh(Ukh[N-1], Tkh[N-1])
e[N-1] = innerUh(Ukh[N-1], Tkh[N])
d[N] = innerUh(Ukh[N], Tkh[N])

# Sanity check that the relations are correct
z = 0.9
count = 0
for k = 3:N-1
    if (a[k]*Ukh[k](z) + b[k]*Ukh[k-1](z) + c[k]*Ukh[k-2](z) ≈ Tkh[k](z)) == false
        count += 1
        println(k)
    end
    if (d[k-1]*Tkh[k-1](z) + e[k-1]*Tkh[k](z) + f[k-1]*Tkh[k+1](z) ≈ Ukh[k-1](z)*(1-z^2)) == false
        count += 1
        println(k)
    end
end
assert(count == 0)



# Construct the operators corresponding to multiplication by x and y of the
# vector P given by P = [P_0,P_1,...] where P_k = [T_k^h(x),y*U_k^h(x)].
rows = cols = [1]
append!(rows, [2 for i=2:N])
Jx = BandedBlockBandedMatrix(zeros(sum(rows), sum(cols)), (rows,cols), (1,1), (0,0))
Jy = BandedBlockBandedMatrix(zeros(sum(rows), sum(cols)), (rows,cols), (1,1), (1,1))
view(Jx, Block(1,1)) .= [α[1]]
view(Jx, Block(1,2)) .= [β[1] 0.0]
view(Jx, Block(2,1)) .= [β[1]; 0.0]
for i = 2:N-1
    view(Jx, Block(i,i)) .= [α[i] 0; 0 γ[i-1]]
    subblock = [β[i] 0; 0 δ[i-1]]
    view(Jx, Block(i,i+1)) .= subblock
    view(Jx, Block(i+1,i)) .= subblock
end
view(Jx, Block(N,N)) .= [α[N] 0; 0 γ[N-1]]
view(Jy, Block(1,2)) .= [0 a[1]]
view(Jy, Block(2,1)) .= [0; d[1]]
for i = 2:N-1
    view(Jy, Block(i,i)) .= [0 b[i]; e[i-1] 0]
    view(Jy, Block(i,i+1)) .= [0 a[i]; f[i-1] 0]
    view(Jy, Block(i+1,i)) .= [0 c[i+1]; d[i] 0]
end
view(Jy, Block(N,N)) .= [0 b[N]; e[N-1] 0]

# Sanity check to see if Jx, Jy are correct
x = 0.8
y = sqrt(1-x^2)
PT = [Tkh[i](x) for i=2:N]
PU = [y*Ukh[i-1](x) for i=2:N]
P = append!([Tkh[1](x)], reshape([PT PU].', length(PT)+length(PU)))
assert((x*P[1:end-2] ≈ (Jx*P)[1:end-2]) == true)
assert((y*P[1:end-2] ≈ (Jy*P)[1:end-2]) == true)
