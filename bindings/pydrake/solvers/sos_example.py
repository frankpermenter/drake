from __future__ import print_function

import numpy as np
import meshcat
import timeit
from matplotlib import cm
import matplotlib.pyplot as plt

from pydrake.solvers import mathematicalprogram as mp
import pydrake.symbolic as sym 
from pydrake.solvers.mosek import MosekSolver

# constants and geometry 
mg = 10.
l = 0.5
mu = 0.2
f_max = 0.5*mg

def CalcTau(s, c):
    return mg*np.array([s, c, 0])

def Cross2D(u,v):
    assert u.size == 2
    assert v.size == 2
    return u[0]*v[1] - u[1]*v[0]
    
def CalcG(s, c, x1, x2):
    n1 = np.array([1, 0])
    n2 = np.array([-1, 0])
    n3 = np.array([s, c])
    N = np.vstack((n1, n2, n3)).T

    t1 = np.array([0, 1])
    t2 = np.array([0, 1])
    t3 = np.array([-c, s])
    T = np.vstack((t1, t2, t3)).T
    
    r1 = np.array([-l, x1 - l])
    r2 = np.array([l, x2 - l])
    r3 = np.array([-l, -l])
    R = np.vstack((r1, r2, r3))
    
    V = np.hstack((N + mu*T, N - mu*T))
    G = np.vstack((V, np.zeros(6)))
    
    for i in range(3):
        G[2, i] = Cross2D(R[i], G[0:2, i])
        G[2, i+3] = Cross2D(R[i], G[0:2, i+3])
    
    return G / np.sqrt(1+mu**2)

if __name__ == "__main__":
    x1 = 0.6
    x2 = 0.6
    for index in range(1):
        prog = mp.MathematicalProgram()

        s = prog.NewIndeterminates(1, "s")[0]
        c = prog.NewIndeterminates(1, "c")[0]
        f = prog.NewIndeterminates(6, "f")

        n = 1 # total degree of lagrange multiplier polynomials
        monomial_basis = sym.MonomialBasis(np.hstack((f, s, c)), n)

        rho = prog.NewContinuousVariables(1, "rho")[0]
        variables = sym.Variables(np.hstack((f, s, c)))
        p = sym.Polynomial(s - rho, variables)

        lambda_sl, Q_sl = prog.NewSosPolynomial(monomial_basis)  # sin lower bound
        lambda_cl, Q_cl = prog.NewSosPolynomial(monomial_basis)  # cos lower bound
                                  
        p -= lambda_sl * sym.Polynomial(s)
        p -= lambda_cl * sym.Polynomial(c)

        # s^2 + c^2 = 1

        lambda_sc = prog.NewFreePolynomial(variables, 4)
        p += sym.Polynomial(s**2 + c**2 - 1)*lambda_sc

        # force balance constraint
        G = CalcG(s, c, x1, x2)
        tau = CalcTau(s, c)
        Gf = G.dot(f)
        for i in range(3):
            lambda_balance = prog.NewFreePolynomial(variables, 4)
            p += sym.Polynomial(Gf[i] - tau[i])*lambda_balance

        # bounding box constraint on f
        if True:
            for fi in f:
                lambda_fl, Ql = prog.NewSosPolynomial(monomial_basis)  # f >= 0
                p -= lambda_fl * sym.Polynomial(fi)

            if False:
                # upper bound on contact forces
                for i in range(2): 
                    lambda_fu, Qu = prog.NewSosPolynomial(monomial_basis)  
                    p -= lambda_fu * sym.Polynomial(f_max - (f[i] + f[i+3])/np.sqrt(1+mu**2))


        print("Adding SOS constraint")
        start = timeit.timeit()
        prog.AddSosConstraint(p)
        end = timeit.timeit()
        print("Adding SOS constraint took:")
        print(end - start)


        prog.AddLinearCost(-rho)

        #solver = MosekSolver()
        #solver.set_stream_logging(True, "")
        #print("Calling solver")
        #solver.Solve(prog)
	#rho_value = prog.GetSolution(rho)
        #print(rho_value, np.arcsin(rho_value)/np.pi*180)
