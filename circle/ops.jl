# Script to obtain a point evaluation of the Nth set of OPs for the unit circle

using ApproxFun

"""
Function to obtain a point evaluation of the Nth set of OPs for the unit circle
at the point (x, y) = (cos(theta), sin(theta)).

The nth set of OPs are : [sin(n*theta), cos(n*theta)], n non-negative integer
"""
function OPeval(N, x, y)

    # Check that x and y are on the unit circle
    delta = 0.001
    @assert (x^2 + y^2 < 1 + delta &&  x^2 + y^2 > 1 - delta) "the point (x, y) must be on unit circle"

    # Check that N is a non-negative integer
    @assert N >= 0 "the argument N should be a non-negative integer"

    # Define the matrices used in our 3-term relations
    DT_0 = [0 1; 1 0]
    DT_n = [2 0 0 0; 0 2 0 0]
    B_0 = [0; 0]
    B_n = [0 0; 0 0; 0 0; 0 0]
    C_1 = [0; 0.5; 0.5; 0]
    C_n = [0.5 0; 0 0.5; 0 0.5; -0.5 0]
    G_0 = [x; y]
    G_n = [x 0; 0 x; y 0; 0 y]

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
