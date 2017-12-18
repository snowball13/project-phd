# Script to obtain a point evaluation of the Nth set of OPs for the unit circle

using ApproxFun


# this code gives some idea about "scope", globals and constants
let
    const x = 5 # the scope of x is only this let block. Because I've put const, f below knows it's an Int
    global f() = x # The "global" indicates that f should be defined outside the let block. It knows x is an Int
end
x  # this is an error because x is not defined
x = 7 # creates a new x with value 7
f()  # returns 5 because f sees the x inside the let block, not the global x

y = 6  # creates a non-constant y
g() = y  # this refers to global y but does not "know" the type of y since it may change
g() # returns the Int 6, but the type is not inferred correctly
y = 7.0
g() # returns the Float64 7.0

@code_warntype f()  # in the console prints out what types are inferred and which are ANY. In the REPL, it will be nicely coloured. Here we see that f knows it returns an Int
@code_warntype g()  # Here we see that g only knows it returns an ANY

"""
Function to obtain a point evaluation of the Nth set of OPs for the unit circle
at the point (x, y) = (cos(theta), sin(theta)).

The nth set of OPs are : [sin(n*theta), cos(n*theta)], n non-negative integer
"""
let
    const     DT_0 = [0 1; 1 0]
    const     DT_n = [2 0 0 0; 0 2 0 0]
    const     B_0 = [0; 0]
    const    B_n = [0 0; 0 0; 0 0; 0 0]
    const    C_1 = [0; 0.5; 0.5; 0]
    const    C_n = [0.5 0; 0 0.5; 0 0.5; -0.5 0]
    const    G_0 = [x; y]
    const    G_n = [x 0; 0 x; y 0; 0 y]


    global function OPeval(N, x, y)

        # Check that x and y are on the unit circle
        delta = 0.001
        @assert (x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta) "the point (x, y) must be on unit circle"

        # Check that N is a non-negative integer
        @assert N >= 0 "the argument N should be a non-negative integer"

        # Define the matrices used in our 3-term relations


        # # We initialise P_(-1) = 0, P_0 = 1, and an empty vector for P_1
        # P_nminus1 = 0
        # P_n = 1
        # P_nplus1 = [0; 0]
        #
        # for n = 0:N-1
        #     if n == 0
        #         P_nplus1 = - DT_0 * (B_0 - G_0) * P_n
        #     elseif n == 1
        #         P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_1 * P_nminus1
        #     else
        #         P_nplus1 = - DT_n * (B_n - G_n) * P_n - DT_n * C_n * P_nminus1
        #     end
        #     P_nminus1 = P_n
        #     P_n = P_nplus1
        # end

        # Declare empty array to store the OP values
        P = Array{Float64}(2*N+1)
        # We initialise P_0 = 1, and treat P_(-1) = 0
        P[1] = 1
        # Begin loop, storing OP values
        for n = 0:N-1
            if n == 0
                P[2*(n+1) : 2*(n+1)+1] = - DT_0 * (B_0 - G_0) * P[1]
            elseif n == 1
                P[2*(n+1) : 2*(n+1)+1] = - DT_n * (B_n - G_n) * P[2*n : 2*n+1] - DT_n * C_1 * P[1]
            else
                P[2*(n+1) : 2*(n+1)+1] = - DT_n * (B_n - G_n) * P[2*n : 2*n+1] - DT_n * C_n * P[2*(n-1) : 2*(n-1)+1]
            end
        end


        # Check if we are correct
        theta = acos(x)
        println(P[2*N : 2*N+1])
        println([sin(N*theta); cos(N*theta)])

        return P[2*N : 2*N+1]

    end
end


"""
Function to obtain a point evaluation of a function f(x,y) where f is input as
the coefficients of its expansion in the basis of the OPs for the circle, i.e.
    f(x, y) = sum(vecdot(f_n, P_n))
where the {P_n} are the OPs on the circle, P_0 = 1 and P_n in R^2 for n>0.

We note the structure of the input f is then a vector of length M=2N+1 where N is
the number of OP sets that are used in the expansion, i.e. n = 0,...,N, and where
f_0=f[1] and f_n=f[2*n : 2*n+1] in R^2 for n>0.

Uses the Clenshaw Algorithm.
"""
function funcEval(f, x, y)

    # Check that x and y are on the unit circle
    delta = 0.001
    @assert (x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta) "the point (x, y) must be on unit circle"

    M = length(f)
    N = (M - 1) / 2
    @assert (N % 2 == 0) "invalid length of f - should be odd number"

    # Define the matrices used in our 3-term relation
    DT_0 = [0 1; 1 0]
    DT_n = [2 0 0 0; 0 2 0 0]
    B_0 = [0; 0]
    B_n = [0 0; 0 0; 0 0; 0 0]
    C_1 = [0; 0.5; 0.5; 0]
    C_n = [0.5 0; 0 0.5; 0 0.5; -0.5 0]
    G_0 = [x; y]
    G_n = [x 0; 0 x; y 0; 0 y]
    alpha = -DT_n*(B_n - G_n)
    beta = -DT_n*C_n

    # Define a zeros vector to store the gammas.
    # Note that we add in gamma_(N+1) = gamma_(N+2) = [0;0]
    gamma = zeros(M+(2*2))
    # Complete the reverse recurrance to gain gamma_1, gamma_2
    for n = M:-2:2
        gamma[n-1 : n] = f[n-1 : n] + alpha*gamma[n+1 : n+2] + *(beta, gamma[n+3 : n+4])
    end

    # Calculate the evaluation of f using gamma_1, gamma_2
    beta = - *(DT_n, C_1)
    P_1 = [y; x]
    return f[1] + vecdot(P_1, gamma[2 : 3]) + vecdot(beta, gamma[4 : 5])

end


A = B = rand(2,2)

A * B # matrix product
v = rand(2)
A * v #  matrix-vector product
*(A, v) # same as above

A .* B  # entrywise product

A .* v   # "broadcast" *
broadcast(*, A, v) # same as previous line

exp.(A)
broadcast(exp, A)


# some simple examples of unit tests
using Base.Test
@test 1 == 2

@test 1 ≈ 1+eps()  atol=1E-5
