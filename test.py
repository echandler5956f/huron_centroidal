# # # # # !/home/quant/ros_ws/src/huron_centroidal/.venv/bin/python3

# # # # # from casadi import *

# # # # # # Test problem
# # # # # #
# # # # # #    min x^2 + y^2
# # # # # #    s.t.    x + y - 10 = 0
# # # # # #

# # # # # # Optimization variables
# # # # # x = MX.sym("x")
# # # # # y = MX.sym("y")

# # # # # # Objective
# # # # # f = x*x + y*sin(y)

# # # # # # Constraints
# # # # # g = x+y-10

# # # # # # Create an NLP problem structure
# # # # # nlp = {"x": vertcat(x,y), "f": f, "g": g}

# # # # # mode = "jit"

# # # # # # Pick a compiler
# # # # # compiler = "gcc"    # Linux
# # # # # # compiler = "clang"  # OSX
# # # # # # compiler = "cl.exe" # Windows

# # # # # # Run this script in an environment that recognised the compiler as command.
# # # # # # On Windows, the suggested way is to run this script from a "x64 Native Tools Command Promt for VS" (Requires Visual C++ components or Build Tools for Visual Studio, available from Visual Studio installer. You also need SDK libraries in order to access stdio and math.)

# # # # # flags = ["-O3"] # Linux/OSX

# # # # # for mode in ["jit"]:

# # # # #   if mode=="jit":
# # # # #     # By default, the compiler will be gcc or cl.exe
# # # # #     jit_options = {"flags": flags, "verbose": True, "compiler": compiler}
# # # # #     options = {"jit": False, "compiler": "shell", "jit_options": jit_options, "snopt": {"User integer workspace": 0}}

# # # # #     # Create an NLP solver instance
# # # # #     solver = nlpsol("solver", "snopt", nlp, options)

# # # # #   elif mode=="external":
# # # # #     pass
# # # # #     # # Create an NLP solver instance
# # # # #     # solver = nlpsol("solver", "ipopt", nlp)

# # # # #     # # Generate C code for the NLP functions
# # # # #     # solver.generate_dependencies("nlp.c")

# # # # #     # import subprocess
# # # # #     # # On Windows, use other flags
# # # # #     # cmd_args = [compiler,"-fPIC","-shared"]+flags+["nlp.c","-o","nlp.so"]
# # # # #     # subprocess.run(cmd_args)

# # # # #     # # Create a new NLP solver instance from the compiled code
# # # # #     # solver = nlpsol("solver", "ipopt", "./nlp.so")

# # # # #   arg = {}

# # # # #   arg["lbx"] = -DM.inf()
# # # # #   arg["ubx"] =  DM.inf()
# # # # #   arg["lbg"] =  0
# # # # #   arg["ubg"] =  0
# # # # #   arg["x0"] = 0

# # # # #   # Solve the NLP
# # # # #   res = solver(**arg)

# # # # #   # Print solution
# # # # #   print("-----")
# # # # #   print("objective at solution =", res["f"])
# # # # #   print("primal solution =", res["x"])
# # # # #   print("dual solution (x) =", res["lam_x"])
# # # # #   print("dual solution (g) =", res["lam_g"])


# # # # # import numpy as np
# # # # # import time
# # # # # print("importing casadi...")
# # # # # from casadi import *


# # # # # # number of inputs to evaluate in parallel
# # # # # N = 50


# # # # # # dummy input
# # # # # dummyInput = np.linspace(0.0, 2.0*np.pi, N)


# # # # # # make a dummy function that's moderately expensive to evaluate
# # # # # print("creating dummy function....")
# # # # # x = SX.sym('x')
# # # # # y = x
# # # # # for k in range(100000):
# # # # #     y = sin(y)
# # # # # f0 = Function('f', [x], [y])


# # # # # # evaluate it serially, the old-fasioned way
# # # # # X = MX.sym('x',N)
# # # # # Y = vertcat(*[f0(X[k]) for k in range(N)])
# # # # # fNaiveParallel = Function('fParallel', [X], [Y])

# # # # # print("evaluating naive parallel function...")
# # # # # t0 = time.time()
# # # # # outNaive = fNaiveParallel(dummyInput)
# # # # # t1 = time.time()
# # # # # print("evaluated naive parallel function in %.3f seconds" % (t1 - t0))


# # # # # # evaluate it using new serial map construct
# # # # # fMap = f0.map(N)

# # # # # print("evaluating serial map function...")
# # # # # t0 = time.time()
# # # # # outMap = fMap(dummyInput)
# # # # # t1 = time.time()
# # # # # print("evaluated serial map function in %.3f seconds" % (t1 - t0))
# # # # # # the following has different shaped outputs, so it's commented out
# # # # # #print outNaive == outMap


# # # # # # evaluate it using new parallel map construct
# # # # # fMap = f0.map(N, "thread", 20)

# # # # # print("evaluating parallel map function...")
# # # # # t0 = time.time()
# # # # # outMap = fMap(dummyInput)
# # # # # t1 = time.time()
# # # # # print("evaluated parallel map function in %.3f seconds" % (t1 - t0))
# # # # # # the following has different shaped outputs, so it's commented out
# # # # # #print outNaive == outMap


import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

# Coefficients of the collocation equation
C = np.zeros((d+1,d+1))

# Coefficients of the continuity equation
D = np.zeros(d+1)

# Coefficients of the quadrature function
B = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d+1):
        C[j,r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)

print("C:\n", C)
print("D:\n", D)
print("B:\n", B)

# Time horizon
T = 10.

# Declare model variables
x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')
x = ca.vertcat(x1, x2)
u = ca.SX.sym('u')

# Model equations
xdot = ca.vertcat((1-x2**2)*x1 - x2 + u, x1)

# Objective term
L = x1**2 + x2**2 + u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

# Control discretization
N = 20 # number of control intervals
h = T/N

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []
xp_plot = []

# "Lift" initial conditions
Xk = ca.MX.sym('X0', 2)
w.append(Xk)
lbw.append([0, 1])
ubw.append([0, 1])
w0.append([0, 1])
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k))
    w.append(Uk)
    lbw.append([-1])
    ubw.append([1])
    w0.append([0])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 2)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-0.25, -np.inf])
        ubw.append([np.inf,  np.inf])
        w0.append([0, 0])

    # Loop over collocation points
    Xk_end = D[0]*Xk
    for j in range(1,d+1):
       # Expression for the state derivative at the collocation point
       xp = C[0,j]*Xk
       for r in range(d): xp = xp + C[r+1,j]*Xc[r]

       xp_plot.append(xp)

       # Append collocation equations
       fj, qj = f(Xc[j-1],Uk)
       g.append(h*fj - xp)
       lbg.append([0, 0])
       ubg.append([0, 0])

       # Add contribution to the end state
       Xk_end = Xk_end + D[j]*Xc[j-1];

       # Add contribution to quadrature function
       J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym('X_' + str(k+1), 2)
    w.append(Xk)
    lbw.append([-0.25, -np.inf])
    ubw.append([np.inf,  np.inf])
    w0.append([0, 0])
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end-Xk)
    lbg.append([0, 0])
    ubg.append([0, 0])

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
solver = ca.nlpsol('solver', 'snopt', prob);

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N+1)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[0], '--')
plt.plot(tgrid, x_opt[1], '-')
plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()


# # # from casadi import *
# # # from numpy import *
# # # import matplotlib.pyplot as plt

# # # # Excercise 1, chapter 10 from Larry Biegler's book
# # # print("program started")

# # # # Test with different number of elements
# # # for N in range(1,2):
# # #   print("N = ", N)

# # #   # Degree of interpolating polynomial
# # #   K = 3

# # #   # Legrandre roots
# # # #   tau_root = [0., 0.211325, 0.788675]

# # #   # Radau roots (K=3)
# # #   tau_root = [0, 0.155051, 0.644949, 1]

# # #   # Time
# # #   t = SX.sym("t")

# # #   # Differential equation
# # #   z = SX.sym("z")
# # #   F = Function("dz_dt", [z],[z*z - 2*z + 1])

# # #   z0 = -3

# # #   # Analytic solution
# # #   z_analytic = Function("z_analytic", [t], [(4*t-3)/(3*t+1)])

# # #   # Collocation point
# # #   tau = SX.sym("tau")

# # #   # Step size
# # #   h = 1.0/N

# # #   # Get the coefficients of the continuity and collocation equations
# # #   D = DM.zeros(K+1)
# # #   C = DM.zeros(K+1,K+1)
# # #   for j in range(K+1):
# # #     # Lagrange polynomial
# # #     L = 1
# # #     for k in range(K+1):
# # #       if(k != j):
# # #         L *= (tau-tau_root[k])/(tau_root[j]-tau_root[k])

# # #     # Evaluate at end for coefficients of continuity equation
# # #     lfcn = Function("lfcn", [tau],[L])
# # #     D[j] = lfcn(1.)

# # #     # Differentiate and evaluate at collocation points
# # #     tfcn = Function("tfcn", [tau],[tangent(L,tau)])
# # #     for k in range(K+1): C[j,k] = tfcn(tau_root[k])
# # #   print("C = ", C)
# # #   print("D = ", D)

# # #   # Collocated states
# # #   Z = SX.sym("Z",N,K+1)

# # #   # Construct the NLP
# # #   x = vec(Z.T)
# # #   g = []
# # #   for i in range(N):
# # #     for k in range(1,K+1):
# # #       # Add collocation equations to NLP
# # #       rhs = 0
# # #       for j in range(K+1):
# # #         rhs += Z[i,j]*C[j,k]
# # #       FF = F(Z[i,k])
# # #       g.append(h*FF-rhs)

# # #     # Add continuity equation to NLP
# # #     rhs = 0
# # #     for j in range(K+1):
# # #       rhs += D[j]*Z[i,j]

# # #     if(i<N-1):
# # #       g.append(Z[i+1,0] - rhs)

# # #   g = vertcat(*g)

# # #   print("g = ", g)

# # #   # NLP
# # #   nlp = {'x':x, 'f':x[0]**2, 'g':g}

# # #   ## ----
# # #   ## SOLVE THE NLP
# # #   ## ----

# # #   # NLP solver options
# # #   opts = {"ipopt.tol" : 1e-10}

# # #   # Allocate an NLP solver and buffer
# # #   solver = nlpsol("solver", "ipopt", nlp, opts)
# # #   arg = {}

# # #   # Initial condition
# # #   arg["x0"] = x.nnz() * [0]

# # #   # Bounds on x
# # #   lbx = x.nnz()*[-100]
# # #   ubx = x.nnz()*[100]
# # #   lbx[0] = ubx[0] = z0
# # #   arg["lbx"] = lbx
# # #   arg["ubx"] = ubx

# # #   # Bounds on the constraints
# # #   arg["lbg"] = 0
# # #   arg["ubg"] = 0

# # #   # Solve the problem
# # #   res = solver(**arg)

# # #   ## Print the time points
# # #   t_opt = N*(K+1) * [0]
# # #   for i in range(N):
# # #     for j in range(K+1):
# # #       t_opt[j + (K+1)*i] = h*(i + tau_root[j])

# # #   print("time points: ", t_opt)

# # #   # Print the optimal cost
# # #   print("optimal cost: ", float(res["f"]))

# # #   # Print the optimal solution
# # #   xopt = res["x"].nonzeros()
# # #   print("optimal solution: ", xopt)

# # #   # plot to screen
# # #   plt.plot(t_opt,xopt)

# # # # show the plots
# # # plt.show()


# # from casadi import *
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # -----------------------------------------------------------------------------
# # # Collocation setup
# # # -----------------------------------------------------------------------------
# # nicp = 1        # Number of (intermediate) collocation points per control interval

# # xref = 0.1 # chariot reference

# # l = 1. #- -> crane, + -> pendulum
# # m = 1.
# # M = 1.
# # g = 9.81
# # tf = 5.0
# # nk = 50
# # ndstate = 6
# # nastate = 1
# # ninput = 1

# # # Degree of interpolating polynomial
# # deg = 4
# # # Radau collocation points
# # cp = "radau"
# # # Size of the finite elements
# # h = tf/nk/nicp

# # # Coefficients of the collocation equation
# # C = np.zeros((deg+1,deg+1))
# # # Coefficients of the continuity equation
# # D = np.zeros(deg+1)

# # # Collocation point
# # tau = SX.sym("tau")

# # # All collocation time points
# # tau_root = [0] + collocation_points(deg, cp)

# # T = np.zeros((nk,deg+1))
# # for i in range(nk):
# #     for j in range(deg+1):
# #         T[i][j] = h*(i + tau_root[j])

# # # For all collocation points: eq 10.4 or 10.17 in Biegler's book
# # # Construct Lagrange polynomials to get the polynomial basis at the collocation point
# # for j in range(deg+1):
# #     L = 1
# #     for j2 in range(deg+1):
# #         if j2 != j:
# #             L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])

# #     # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
# #     lfcn = Function('lfcn', [tau],[L])
# #     D[j] = lfcn(1.0)

# #     # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
# #     tfcn = Function('tfcn', [tau],[tangent(L,tau)])
# #     for j2 in range(deg+1):
# #         C[j][j2] = tfcn(tau_root[j2])

# # # -----------------------------------------------------------------------------
# # # Model setup
# # # -----------------------------------------------------------------------------
# # # Declare variables (use scalar graph)
# # t  = SX.sym("t")          # time
# # u  = SX.sym("u")          # control
# # xd  = SX.sym("xd",ndstate)      # differential state
# # xa  = SX.sym("xa",nastate)    # algebraic state
# # xddot  = SX.sym("xdot",ndstate) # differential state time derivative
# # p = SX.sym("p",0,1)      # parameters

# # x = SX.sym("x")
# # y = SX.sym("y")
# # w = SX.sym("w")

# # dx = SX.sym("dx")
# # dy = SX.sym("dy")
# # dw = SX.sym("dw")


# # res = vertcat(xddot[0] - dx,\
# #        xddot[1] - dy,\
# #        xddot[2] - dw,\
# #        m*xddot[3] + (x-w)*xa, \
# #        m*xddot[4] +     y*xa - g*m,\
# #        M*xddot[5] + (w-x)*xa +   u,\
# #        (x-w)*(xddot[3] - xddot[5]) + y*xddot[4] + dy*dy + (dx-dw)*(dx-dw))


# # xd[0] = x
# # xd[1] = y
# # xd[2] = w
# # xd[3] = dx
# # xd[4] = dy
# # xd[5] = dw


# # # System dynamics (implicit formulation)
# # ffcn = Function('ffcn', [t,xddot,xd,xa,u,p],[res])

# # # Objective function
# # MayerTerm = Function('mayer', [t,xd,xa,u,p],[(x-xref)*(x-xref) + (w-xref)*(w-xref) + dx*dx + dy*dy])
# # LagrangeTerm = Function('lagrange', [t,xd,xa,u,p],[(x-xref)*(x-xref) + (w-xref)*(w-xref)])

# # # Control bounds
# # u_min = np.array([-2])
# # u_max = np.array([ 2])
# # u_init = np.array((nk*nicp*(deg+1))*[[0.0]]) # needs to be specified for every time interval (even though it stays constant)

# # # Differential state bounds
# # #Path bounds
# # xD_min =  np.array([-inf, -inf, -inf, -inf, -inf, -inf])
# # xD_max =  np.array([ inf,  inf,  inf,  inf,  inf,  inf])
# # #Initial bounds
# # xDi_min = np.array([ 0.0,  l,  0.0,  0.0,  0.0,  0.0])
# # xDi_max = np.array([ 0.0,  l,  0.0,  0.0,  0.0,  0.0])
# # #Final bounds
# # xDf_min = np.array([-inf, -inf, -inf, -inf, -inf, -inf])
# # xDf_max = np.array([ inf,  inf,  inf,  inf,  inf,  inf])

# # #Initial guess for differential states
# # xD_init = np.array((nk*nicp*(deg+1))*[[ 0.0,  l,  0.0,  0.0,  0.0,  0.0]]) # needs to be specified for every time interval

# # # Algebraic state bounds and initial guess
# # xA_min =  np.array([-inf])
# # xA_max =  np.array([ inf])
# # xAi_min = np.array([-inf])
# # xAi_max = np.array([ inf])
# # xAf_min = np.array([-inf])
# # xAf_max = np.array([ inf])
# # xA_init = np.array((nk*nicp*(deg+1))*[[sign(l)*9.81]])

# # # Parameter bounds and initial guess
# # p_min = np.array([])
# # p_max = np.array([])
# # p_init = np.array([])


# # # -----------------------------------------------------------------------------
# # # Constraints setup
# # # -----------------------------------------------------------------------------
# # # Initial constraint
# # ic_min = np.array([])
# # ic_max = np.array([])
# # ic = SX()
# # #ic.append();       ic_min = append(ic_min, 0.);         ic_max = append(ic_max, 0.)
# # icfcn = Function('icfcn', [t,xd,xa,u,p],[ic])
# # # Path constraint
# # pc_min = np.array([])
# # pc_max = np.array([])
# # pc = SX()
# # #pc.append();       pc_min = append(pc_min, 0.);         pc_max = append(pc_max, 0.)
# # pcfcn = Function('pcfcn', [t,xd,xa,u,p],[pc])
# # # Final constraint
# # fc_min = np.array([])
# # fc_max = np.array([])
# # fc = SX()
# # #fc.append();       fc_min = append(fc_min, 0.);         fc_max = append(fc_max, 0.)
# # fcfcn = Function('fcfcn', [t,xd,xa,u,p],[fc])

# # # -----------------------------------------------------------------------------
# # # NLP setup
# # # -----------------------------------------------------------------------------
# # # Dimensions of the problem
# # nx = xd.nnz() + xa.nnz()  # total number of states        #MODIF
# # ndiff = xd.nnz()           # number of differential states #MODIF
# # nalg = xa.nnz()            # number of algebraic states
# # nu = u.nnz()               # number of controls
# # NP  = p.nnz()              # number of parameters

# # # Total number of variables
# # NXD = nicp*nk*(deg+1)*ndiff # Collocated differential states
# # NXA = nicp*nk*deg*nalg      # Collocated algebraic states
# # NU = nk*nu                  # Parametrized controls
# # NXF = ndiff                 # Final state (only the differential states)
# # NV = NXD+NXA+NU+NXF+NP

# # # NLP variable vector
# # V = MX.sym("V",NV)

# # # All variables with bounds and initial guess
# # vars_lb = np.zeros(NV)
# # vars_ub = np.zeros(NV)
# # vars_init = np.zeros(NV)
# # offset = 0

# # # Get the parameters
# # P = V[offset:offset+NP]
# # vars_init[offset:offset+NP] = p_init
# # vars_lb[offset:offset+NP] = p_min
# # vars_ub[offset:offset+NP] = p_max
# # offset += NP

# # # Get collocated states and parametrized control
# # XD = np.resize(np.array([],dtype=MX),(nk+1,nicp,deg+1)) # NB: same name as above
# # XA = np.resize(np.array([],dtype=MX),(nk,nicp,deg)) # NB: same name as above
# # U = np.resize(np.array([],dtype=MX),nk)
# # for k in range(nk):
# #     # Collocated states
# #     for i in range(nicp):
# #         #
# #         for j in range(deg+1):

# #             # Get the expression for the state vector
# #             XD[k][i][j] = V[offset:offset+ndiff]
# #             if j !=0:
# #                 XA[k][i][j-1] = V[offset+ndiff:offset+ndiff+nalg]
# #             # Add the initial condition
# #             index = (deg+1)*(nicp*k+i) + j
# #             if k==0 and j==0 and i==0:
# #                 vars_init[offset:offset+ndiff] = xD_init[index,:]

# #                 vars_lb[offset:offset+ndiff] = xDi_min
# #                 vars_ub[offset:offset+ndiff] = xDi_max
# #                 offset += ndiff
# #             else:
# #                 if j!=0:
# #                     vars_init[offset:offset+nx] = np.append(xD_init[index,:],xA_init[index,:])

# #                     vars_lb[offset:offset+nx] = np.append(xD_min,xA_min)
# #                     vars_ub[offset:offset+nx] = np.append(xD_max,xA_max)
# #                     offset += nx
# #                 else:
# #                     vars_init[offset:offset+ndiff] = xD_init[index,:]

# #                     vars_lb[offset:offset+ndiff] = xD_min
# #                     vars_ub[offset:offset+ndiff] = xD_max
# #                     offset += ndiff

# #     # Parametrized controls
# #     U[k] = V[offset:offset+nu]
# #     vars_lb[offset:offset+nu] = u_min
# #     vars_ub[offset:offset+nu] = u_max
# #     vars_init[offset:offset+nu] = u_init[index,:]
# #     offset += nu

# # # State at end time
# # XD[nk][0][0] = V[offset:offset+ndiff]
# # vars_lb[offset:offset+ndiff] = xDf_min
# # vars_ub[offset:offset+ndiff] = xDf_max
# # vars_init[offset:offset+ndiff] = xD_init[-1,:]
# # offset += ndiff
# # assert(offset==NV)

# # # Constraint function for the NLP
# # g = []
# # lbg = []
# # ubg = []

# # # Initial constraints
# # ick = icfcn(0., XD[0][0][0], XA[0][0][0], U[0], P)
# # g += [ick]
# # lbg.append(ic_min)
# # ubg.append(ic_max)

# # # For all finite elements
# # for k in range(nk):
# #     for i in range(nicp):
# #         # For all collocation points
# #         for j in range(1,deg+1):
# #             # Get an expression for the state derivative at the collocation point
# #             xp_jk = 0
# #             for j2 in range (deg+1):
# #                 xp_jk += C[j2][j]*XD[k][i][j2]       # get the time derivative of the differential states (eq 10.19b)

# #             # Add collocation equations to the NLP
# #             fk = ffcn(0., xp_jk/h, XD[k][i][j], XA[k][i][j-1], U[k], P)
# #             g += [fk[:ndiff]]                     # impose system dynamics (for the differential states (eq 10.19b))
# #             lbg.append(np.zeros(ndiff)) # equality constraints
# #             ubg.append(np.zeros(ndiff)) # equality constraints
# #             g += [fk[ndiff:]]                               # impose system dynamics (for the algebraic states (eq 10.19b))
# #             lbg.append(np.zeros(nalg)) # equality constraints
# #             ubg.append(np.zeros(nalg)) # equality constraints

# #             #  Evaluate the path constraint function
# #             pck = pcfcn(0., XD[k][i][j], XA[k][i][j-1], U[k], P)

# #             g += [pck]
# #             lbg.append(pc_min)
# #             ubg.append(pc_max)

# #         # Get an expression for the state at the end of the finite element
# #         xf_k = 0
# #         for j in range(deg+1):
# #             xf_k += D[j]*XD[k][i][j]

# #         # Add continuity equation to NLP
# #         if i==nicp-1:
# # #            print "a ", k, i
# #             g += [XD[k+1][0][0] - xf_k]
# #         else:
# # #            print "b ", k, i
# #             g += [XD[k][i+1][0] - xf_k]

# #         lbg.append(np.zeros(ndiff))
# #         ubg.append(np.zeros(ndiff))

# # # Periodicity constraints
# # #   none

# # # Final constraints (Const, dConst, ConstQ)
# # fck = fcfcn(0., XD[k][i][j], XA[k][i][j-1], U[k], P)
# # g += [fck]
# # lbg.append(fc_min)
# # ubg.append(fc_max)

# # # Objective function of the NLP
# # #Implement Mayer term
# # Obj = 0
# # obj = MayerTerm(0., XD[k][i][j], XA[k][i][j-1], U[k], P)
# # Obj += obj

# # # Implement Lagrange term
# # lDotAtTauRoot = C.T
# # lAtOne = D

# # ldInv = np.linalg.inv(lDotAtTauRoot[1:,1:])
# # ld0 = lDotAtTauRoot[1:,0]
# # lagrangeTerm = 0
# # for k in range(nk):
# #     for i in range(nicp):
# #         dQs = h*veccat(*[LagrangeTerm(0., XD[k][i][j], XA[k][i][j-1], U[k], P) \
# #                         for j in range(1,deg+1)])
# #         Qs = mtimes( ldInv, dQs)
# #         m = mtimes( Qs.T, lAtOne[1:])
# #         lagrangeTerm += m

# # Obj += lagrangeTerm

# # # NLP
# # nlp = {'x':V, 'f':Obj, 'g':vertcat(*g)}

# # ## ----
# # ## SOLVE THE NLP
# # ## ----

# # # NLP solver options
# # opts = {}
# # opts["expand"] = True
# # opts["ipopt.max_iter"] = 1000
# # opts["ipopt.tol"] = 1e-4
# # # opts["ipopt.linear_solver"] = 'ma27'

# # # Allocate an NLP solver
# # solver = nlpsol("solver", "ipopt", nlp, opts)
# # arg = {}

# # # Initial condition
# # arg["x0"] = vars_init

# # # Bounds on x
# # arg["lbx"] = vars_lb
# # arg["ubx"] = vars_ub

# # # Bounds on g
# # arg["lbg"] = np.concatenate(lbg)
# # arg["ubg"] = np.concatenate(ubg)

# # # Solve the problem
# # res = solver(**arg)

# # # Print the optimal cost
# # print("optimal cost: ", float(res["f"]))

# # # Retrieve the solution
# # v_opt = np.array(res["x"])


# # ## ----
# # ## RETRIEVE THE SOLUTION
# # ## ----
# # xD_opt = np.resize(np.array([],dtype=MX),(ndiff,(deg+1)*nicp*(nk)+1))
# # xA_opt = np.resize(np.array([],dtype=MX),(nalg,(deg)*nicp*(nk)))
# # u_opt = np.resize(np.array([],dtype=MX),(nu,(deg+1)*nicp*(nk)+1))
# # offset = 0
# # offset2 = 0
# # offset3 = 0
# # offset4 = 0

# # for k in range(nk):
# #     for i in range(nicp):
# #         for j in range(deg+1):
# #             xD_opt[:,offset2] = v_opt[offset:offset+ndiff][:,0]
# #             offset2 += 1
# #             offset += ndiff
# #             if j!=0:
# #                 xA_opt[:,offset4] = v_opt[offset:offset+nalg][:,0]
# #                 offset4 += 1
# #                 offset += nalg
# #     utemp = v_opt[offset:offset+nu][:,0]
# #     for i in range(nicp):
# #         for j in range(deg+1):
# #             u_opt[:,offset3] = utemp
# #             offset3 += 1
# #     #    u_opt += v_opt[offset:offset+nu]
# #     offset += nu

# # xD_opt[:,-1] = v_opt[offset:offset+ndiff][:,0]


# # # The algebraic states are not defined at the first collocation point of the finite elements:
# # # with the polynomials we compute them at that point
# # Da = np.zeros(deg)
# # for j in range(1,deg+1):
# #     # Lagrange polynomials for the algebraic states: exclude the first point
# #     La = 1
# #     for j2 in range(1,deg+1):
# #         if j2 != j:
# #             La *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
# #     lafcn = Function('lafcn', [tau], [La])
# #     Da[j-1] = lafcn(tau_root[0])

# # xA_plt = np.resize(np.array([],dtype=MX),(nalg,(deg+1)*nicp*(nk)+1))
# # offset4=0
# # offset5=0
# # for k in range(nk):
# #     for i in range(nicp):
# #         for j in range(deg+1):
# #             if j!=0:
# #                 xA_plt[:,offset5] = xA_opt[:,offset4]
# #                 offset4 += 1
# #                 offset5 += 1
# #             else:
# #                 xa0 = 0
# #                 for j in range(deg):
# #                     xa0 += Da[j]*xA_opt[:,offset4+j]
# #                 xA_plt[:,offset5] = xa0
# #                 #xA_plt[:,offset5] = xA_opt[:,offset4]
# #                 offset5 += 1

# # xA_plt[:,-1] = xA_plt[:,-2]


# # tg = np.array(tau_root)*h
# # for k in range(nk*nicp):
# #     if k == 0:
# #         tgrid = tg
# #     else:
# #         tgrid = np.append(tgrid,tgrid[-1]+tg)
# # tgrid = np.append(tgrid,tgrid[-1])
# # # Plot the results
# # plt.figure(1)
# # plt.clf()
# # plt.subplot(2,2,1)
# # plt.plot(tgrid,xD_opt[0,:],'--')
# # plt.title("x")
# # plt.grid
# # plt.subplot(2,2,2)
# # plt.plot(tgrid,xD_opt[1,:],'-')
# # plt.title("y")
# # plt.grid
# # plt.subplot(2,2,3)
# # plt.plot(tgrid,xD_opt[2,:],'-.')
# # plt.title("w")
# # plt.grid

# # plt.figure(2)
# # plt.clf()
# # plt.plot(tgrid,u_opt[0,:],'-.')
# # plt.title("Crane, inputs")
# # plt.xlabel('time')


# # plt.figure(3)
# # plt.clf()
# # plt.plot(tgrid,xA_plt[0,:],'-.')
# # plt.title("Crane, lambda")
# # plt.xlabel('time')
# # plt.grid()
# # plt.show()

# import casadi as ca
# from dataclasses import dataclass
# from typing import Dict


# @dataclass(frozen=True)
# class EndEffector:
#     frame_id: int
#     type_6D: bool


# @dataclass
# class Phase:
#     contacts: Dict[EndEffector, bool]
#     fixed_timing: float = None
#     timing_var: ca.SX = None
#     knot_points: int = 1

#     def set_timing_var(self, idx: int):
#         self.timing_var = ca.SX.sym(f'timing_var_{idx}')


# class ContactSequence:
#     def __init__(self):
#         # Each entry in sequence will be a dictionary {end_effector: phase}
#         self.sequence = []
#         self.cumulative_knots = []  # Cumulative knot points for the entire system

#     def add_phase(self, contacts: Dict[EndEffector, Phase]):
#         self.sequence.append(contacts)

#         # Update cumulative knot points
#         max_knot_points_in_phase = max(
#             [phase.knot_points for phase in contacts.values()]
#         )
#         previous_knots = self.cumulative_knots[-1] if self.cumulative_knots else 0
#         self.cumulative_knots.append(previous_knots + max_knot_points_in_phase)

#     def get_phase_from_knot(self, k: int):
#         # Returns the phase (i.e., dictionary {end_effector: phase}) associated with the given knot value k
#         for idx, cum_knot in enumerate(self.cumulative_knots):
#             if k < cum_knot:
#                 return self.sequence[idx]
#         return None  # k exceeds the total number of knot points

#     def get_all_end_effectors(self):
#         end_effectors = set()
#         for phases_dict in self.sequence:
#             end_effectors.update(phases_dict.keys())
#         return list(end_effectors)

#     def get_phase_for_all_end_effectors(self, phase_idx: int):
#         if phase_idx < len(self.sequence):
#             return self.sequence[phase_idx]
#         else:
#             return {}

#     def iterate_over_legs_during_phase(self, phase_idx: int):
#         for end_effector, phase in self.get_phase_for_all_end_effectors(
#             phase_idx
#         ).items():
#             yield end_effector, phase

#     def is_in_contact(self, end_effector: EndEffector, phase_idx: int) -> bool:
#         return (
#             self.sequence[phase_idx]
#             .get(end_effector, Phase(in_contact=False))
#             .in_contact
#         )

#     def get_phase_duration(self, end_effector: EndEffector, phase_idx: int):
#         phase = self.sequence[phase_idx].get(end_effector)
#         if phase and phase.fixed_timing:
#             return phase.fixed_timing
#         elif phase:
#             return phase.timing_var
#         return 0  # Default duration if end_effector is not in the phase

#     def get_frame_id(self, end_effector: EndEffector) -> int:
#         return end_effector.frame_id

#     def get_contact_type(self, end_effector: EndEffector) -> bool:
#         return end_effector.type_6D

#     def get_num_phases(self) -> int:
#         return len(self.sequence)
    


# # Create end effectors
# left_foot = EndEffector(frame_id=1, type_6D=True)
# right_foot = EndEffector(frame_id=2, type_6D=False)
# hand = EndEffector(frame_id=3, type_6D=False)

# # Initialize the contact sequence
# contact_seq = ContactSequence()

# # Add a phase where the left foot and hand are in contact, each with different timings
# phase_1 = {left_foot: Phase(in_contact=True, fixed_timing=0.5, knot_points=5),
#            hand: Phase(in_contact=True, fixed_timing=0.6, knot_points=5)}
# contact_seq.add_phase(phase_1)

# # Add another phase where only the right foot is in contact
# phase_2 = {right_foot: Phase(in_contact=True, fixed_timing=0.7, knot_points=3)}
# contact_seq.add_phase(phase_2)

# # Get all end effectors in the contact sequence
# all_end_effectors = contact_seq.get_all_end_effectors()
# print("All End Effectors:", all_end_effectors)

# # Check if left_foot is in contact during the first phase
# in_contact = contact_seq.is_in_contact(left_foot, 0)
# print(f"Is {left_foot.frame_id} in contact during the first phase? {in_contact}")

# # Get the phase duration for the right_foot during the second phase
# duration = contact_seq.get_phase_duration(right_foot, 1)
# print(f"Duration for {right_foot.frame_id} during the second phase: {duration}")

# # Iterate over legs during a specific phase
# print("\nEnd Effectors in the second phase:")
# for ee, phase in contact_seq.iterate_over_legs_during_phase(1):
#     print(f"End Effector ID: {ee.frame_id}, In Contact: {phase.in_contact}, Duration: {phase.fixed_timing or phase.timing_var}")

# # Get the phase associated with a particular knot value
# k = 6
# associated_phase = contact_seq.get_phase_from_knot(k)
# print(f"\nPhase associated with knot {k}:", associated_phase)



# import casadi as ca
# from dataclasses import dataclass
# from typing import Dict, List


# @dataclass(frozen=True)
# class EndEffector:
#     frame_id: int
#     type_6D: bool


# @dataclass
# class Phase:
#     contacts: Dict[EndEffector, bool]
#     fixed_timing: float = None
#     timing_var: bool = False
#     knot_points: int = 1


# class ContactSequence:
#     def __init__(self):
#         self.sequence = []
#         self.cumulative_knots = []

#     def add_phase(self, phase: Phase):
#         self.sequence.append(phase)

#         previous_knots = self.cumulative_knots[-1] if self.cumulative_knots else 0
#         self.cumulative_knots.append(previous_knots + phase.knot_points)

#     def get_all_end_effectors(self) -> List[EndEffector]:
#         end_effectors = set()
#         for phase in self.sequence:
#             end_effectors.update(phase.contacts.keys())
#         return list(end_effectors)

#     def get_phase(self, phase_idx: int) -> Phase:
#         return self.sequence[phase_idx]

#     def iterate_over_legs_during_phase(self, phase_idx: int):
#         phase = self.get_phase(phase_idx)
#         for end_effector, is_in_contact in phase.contacts.items():
#             yield end_effector, is_in_contact

#     def is_in_contact(self, end_effector: EndEffector, phase_idx: int) -> bool:
#         phase = self.get_phase(phase_idx)
#         return phase.contacts.get(end_effector, False)

#     def get_frame_id(self, end_effector: EndEffector) -> int:
#         return end_effector.frame_id

#     def get_contact_type(self, end_effector: EndEffector) -> bool:
#         return end_effector.type_6D

#     def get_num_phases(self) -> int:
#         return len(self.sequence)

#     def get_phase_from_knot(self, k: int) -> Phase:
#         for idx, cum_knot in enumerate(self.cumulative_knots):
#             if k < cum_knot:
#                 return self.get_phase(idx)
#         return None
    

# # Define two end effectors
# ee_left_foot = EndEffector(frame_id=1, type_6D=True)
# ee_right_foot = EndEffector(frame_id=2, type_6D=False)

# # Define a phase where both end effectors are in contact, with fixed timing
# contacts_phase1 = {
#     ee_left_foot: True,
#     ee_right_foot: True
# }
# phase1 = Phase(contacts=contacts_phase1, fixed_timing=0.5, knot_points=10)

# # Define a phase where only the left foot is in contact, with variable timing
# contacts_phase2 = {
#     ee_left_foot: True,
#     ee_right_foot: False
# }
# phase2 = Phase(contacts=contacts_phase2)

# # Create a contact sequence and add phases
# contact_seq = ContactSequence()
# contact_seq.add_phase(phase1)
# contact_seq.add_phase(phase2)

# # Using the interface
# print(f"Total number of phases: {contact_seq.get_num_phases()}")
# print(f"Frame ID for left foot: {contact_seq.get_frame_id(ee_left_foot)}")
# print(f"Is left foot a 6D contact? {contact_seq.get_contact_type(ee_left_foot)}")

# print("\nEnd effectors in contact during phase 1:")
# for ee, in_contact in contact_seq.iterate_over_legs_during_phase(0):
#     print(f"{contact_seq.get_frame_id(ee)}: {'In contact' if in_contact else 'Not in contact'}")

# print("\nEnd effectors in contact during phase 2:")
# for ee, in_contact in contact_seq.iterate_over_legs_during_phase(1):
#     print(f"{contact_seq.get_frame_id(ee)}: {'In contact' if in_contact else 'Not in contact'}")