"""D3 Density NLBVP

This is the *simplest* model we will consider for modeling spheromaks evolving in the SSX wind tunnel.

Major simplifications fall in two categories

Geometry
--------
We consider a square duct using real Fourier bases (sin/cos) in all directions.

Equations
---------
The equations themselves are those from Schaffner et al (2014), with the following simplifications

* hall term off
* constant eta instead of Spitzer
* no wall recycling term
* no mass diffusion

For this model, rather than kinematic viscosity nu and thermal
diffusivity chi varying with density rho as they should, we are here
holding them *constant*. This dramatically simplifies the form of the
equations in Dedalus.

We use the vector potential, and enforce the Coulomb Gauge, div(A) = 0.

Currently working on solving T/rho unphysicalities.

Dedalus 3 edits made by Alex Skeldon. Direct all queries to askeldo1@swarthmore.edu.

Lane-Emden is the one NLBVP example reference I have so far. Add more if you see them.

CURRENT ISSUE:
    "File "/home/alske/mambaforge/envs/dedalus3/lib/python3.11/site-packages/dedalus/core/solvers.py", line 74, in __init__
    raise ValueError(f"Problem is coupled along distributed dimensions: {tuple(np.where(coupled_nonlocal)[0])}")
ValueError: Problem is coupled along distributed dimensions: (0, 1)"
Will try in serial.

next error:
"AttributeError: 'NonlinearBoundaryValueSolver' object has no attribute 'solve'"
So I made an iterator.

next error:
"/home/alske/mambaforge/envs/dedalus3/lib/python3.11/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py:276: MatrixRankWarning: Matrix is exactly singular
  warn("Matrix is exactly singular", MatrixRankWarning)
/mnt/c/users/alske/Downloads/SSX/SSX-Simulations-GitRepo/SSX_Scenarios/D3_IVP_SSX.py:279: RuntimeWarning: divide by zero encountered in log
  lnrho['g'] = np.log(rho0['g'])
2023-10-29 11:47:36,898 __main__ 0/1 INFO :: Starting loop
/home/alske/mambaforge/envs/dedalus3/lib/python3.11/site-packages/dedalus/core/operators.py:387: RuntimeWarning: divide by zero encountered in power
  np.power(arg0.data, arg1, out.data)"

Singular matrix and lnrho reading a 0. Okay, let's turn off ln formulation to fix second problem.
    
Next error:
"  File "/home/alske/mambaforge/envs/dedalus3/lib/python3.11/site-packages/dedalus/core/subsystems.py", line 159, in field_shape
    comp_shape = tuple(cs.dim for cs in field.tensorsig)
                                        ^^^^^^^^^^^^^^^
AttributeError: 'int' object has no attribute 'tensorsig' "
    
Okay, no idea why that's happening. I'm guessing it's stemming from the singular matrix problem AKA equations are currently ill-posed.
Will probably need to refer to some actual Dedalus resources and/or ask Doc about NLBVP equation formulations compared to IVPs once
we are off the plane. Good start, though.

Ok now that I adjusted the initial rho density guess to the original cos(r) * cos(z) and removed the initial velocity, running
with lnrho on just gives "Killed" so that's not ominous at all.


"""

import numpy as np
import os
import dedalus.public as d3
from dedalus.extras import flow_tools
from mpi4py import MPI


from scipy.special import j0, j1, jn_zeros
from D3_LBVP_SSX import spheromak_pair, zero_modes

def pair_density(xbasis, ybasis, zbasis, coords, parity, distmain, LBVP_A, A_perturb, log_density, a_imp, comm=None):
    #Need to pass parallel distribution in for calculating vector potential here

    # NLBVPs cannot be run in parallel due to Newton iterations, so this must use a serial distributor
    # None causes different issue than comm or mpi.comm_self so that's fun
    distser = d3.Distributor(coords, dtype=np.float64, mesh=None, comm=None) # Comm? Or mpi.comm_self? Or none?
    
    kappa = 0.1
    # try both of these 0.1 see what happens
    mu = 0.005 #Determines Re_k ; 0.05 -> Re_k = 20 (try 0.005?)
    eta = 0.01 # Determines Re_m ; 0.001 -> Re_m = 1000; using smaller Rm of 100 for now since 1000 is a bit high.

    rhoInit = 1 #rho0 is redefined later, and generally has a whole mess associated with it
    gamma = 5./3.

    # eta_sp = 2.7 * 10**(-4)
    # eta_ch = 4.4 * 10**(-3)
    # v0_ch = 2.9 * 10**(-2)

    chi = kappa/rhoInit
    nu = mu/rhoInit
    #diffusivities for heat (kappa -> chi), momentum (viscosity) (mu -> nu), current (eta)
    # life time of currents regulated by resistivity
    # linearization time of temperature goes like e^-t/kappa

    # SSX dimensions
    rad = 1
    length = 10

    # Fields
    v = distser.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
    A = distser.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
    if log_density:
        lnrho = distser.Field(name='lnrho', bases=(xbasis, ybasis, zbasis))
    else:
        rho = distser.Field(name='rho', bases=(xbasis, ybasis, zbasis))
    T = distser.Field(name='T', bases=(xbasis, ybasis, zbasis))
    phi = distser.Field(name='phi', bases=(xbasis, ybasis, zbasis))
    tau_A = distser.Field(name='tau_A')
    # eta = distser.Field(name='T', bases=(xbasis, ybasis, zbasis))
    ex, ey, ez = coords.unit_vector_fields(distser)

    # Coulomb Gauge implies J = -Laplacian(A)
    j = -d3.lap(A)
    J2 = j@j
    if log_density:
        rho = np.exp(lnrho)
    B = d3.curl(A)

    #spitzer and chodra resistivity combination
    # eta = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))

    #Problem
    if log_density:
        Rho = d3.NLBVP([v, A, lnrho, T, phi, tau_A], namespace=locals())
    else:
        Rho = d3.NLBVP([v, A, rho, T, phi, tau_A],  namespace=locals())

    #variable resistivity
    # Rho.add_equation("eta = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))")

    if log_density:
    # Continuity
        Rho.add_equation("div(v) = - v@grad(lnrho)")

    # Not really good model but this would be how you'd express incompressibility
    # Rho.add-equation("div(v) + tau_p = 0")

    # Momentum
        Rho.add_equation("grad(T) - nu*lap(v) = T*grad(lnrho) - v@grad(v) + cross(j,B)/rho")
    else:
    #Non-lnrho formulation equations:
        Rho.add_equation("0 = -rho*div(v) -v@grad(rho)")
        Rho.add_equation("grad(T) - nu*lap(v) = T*grad(rho)/rho - v@grad(v) + cross(j,B)/rho")

    # MHD equations: A
    Rho.add_equation("grad(phi) + eta*j = cross(v,B)")

    #gauge constraints
    Rho.add_equation("div(A) + tau_A = 0")
    Rho.add_equation("integ(phi) = 0")

    # Energy
    Rho.add_equation("(gamma - 1) * chi*lap(T) = - (gamma - 1) * T * div(v) - v@grad(T) + (gamma - 1)*eta*J2")

    x,y,z = distser.local_grids(xbasis,ybasis,zbasis)
    aa = distser.VectorField(coords, name='aa', bases=(xbasis, ybasis, zbasis))

    # Initial condition parameters
    R = rad
    L = R
    rho_min = 0.011
    T0 = 0.1
    delta = 0.1 # The strength of the perturbation. Schaffner et al 2014 (flux-rope plasma) has delta = 0.1.
    r = np.sqrt(x**2+y**2)

    # Spheromak initial condition

    #BEGINNING of In-line vector potential
    if not(LBVP_A):
        handedness = 1
        j1_zero1 = jn_zeros(1,1)[0]
        kr = j1_zero1/R
        kz = np.pi/L
        b0 = 1
        lam = np.sqrt(kr**2 + kz**2)
        theta = np.arctan2(y,x)

        Ar = -b0*kz*j1(kr*r)*np.cos(kz*z)/lam
        At = handedness*b0*j1(kr*r)*np.sin(kz*z)
        Az = b0*j0(kr*r)*np.cos(kz*z)/lam

        #now we need to add a rotated and translated copy
        # since we have angular symmetry, we just need to translate 10 units
        # and reverse the z component (i.e. negative sign)
        Ar2 = -b0*kz*j1(kr*r)*np.cos(kz*(-(z-10)))/lam
        At2 = handedness*b0*j1(kr*r)*np.sin(kz*(-(z-10)))
        Az2 = - b0*j0(kr*r)*np.cos(kz*(-(z-10)))/lam

        #We need to localize these fields so they go to 0 in 1 < z < 10 and r > 1
        #use similar tanh's to initialized density
        # zVecDist = ((-np.tanh(2 *(z - 1.5)) - np.tanh(-2*(z - 8.5)))/2 + 1) # Keeping here in case I decide to switch back to this expression

        #Here's a z-distribution that goes to zero at z = 10 and z = 0, could be useful for vector potential drop-off
        # (want a constant value or close to it at both sides of the boundary)
        zVecDist2 = (-np.tanh(4*(z - 3)) + np.tanh(4*(z - 1)) - np.tanh(-4*(z - 7)) + np.tanh(-4*(z - 9)))/2
        rVecDist = -np.tanh(5*(r - 1))/2 + 0.5

        aa['g'][0] = ((Ar+Ar2)*np.cos(theta) - (At+At2)*np.sin(theta)) * zVecDist2 * rVecDist
        aa['g'][1] = ((Ar+Ar2)*np.sin(theta) + (At+At2)*np.cos(theta)) * zVecDist2 * rVecDist
        aa['g'][2] = (Az+Az2) * zVecDist2 * rVecDist
    else:
        aa = a_imp

    # The vector potential is subject to some perturbation. This distorts all the magnetic field components in the same direction.
    # if A_perturb:
    #     for i in range(3): 
    #         A['g'][i] = aa['g'][i] *(1 + delta*x*np.exp(-z**2) + delta*x*np.exp(-(z-10)**2)) # maybe the exponent here is too steep of an IC?
    # else: 
    #     for i in range(3): 
    #         # A['g'][i] = aa['g'][i]
    #         A['g'][i] = aa['g'][i]
    
    rho0 = distser.Field(name='rho0', bases=(xbasis, ybasis, zbasis))
    rho0['g'] = np.zeros_like(T['g'])

    
    # velocity implicitly initialized to 0 throughout

    wavz = 2

    #Changed from disk density distribution to a donut distribution
    #I'm not sure why Slava's HiFi simulation only had a z-dependent distribution for density, and no radial
    #(which would produce disks instead of donuts since he had cylindrical geometry)

    #Initial density guess
    
    # cos(r) * cos(z) original expression 
    zdist = -np.cos(wavz*4*np.pi*z/length)*(1-rho_min)/2 + 1/2
    rdist = -np.cos(np.pi*r/rad)*(1-rho_min)/2 + 1/2 # main issue with this is cusps at square edges
        
    
    # rdist = -np.cos(2*np.pi*x/rad)*np.cos(2*np.pi*y/rad)*(1-rho_min)/2 + 1/2

    # Tanh formulation
    # zdist = (-np.tanh(2 *(z - 1.5)) - np.tanh(-2*(z - 8.5)))*(1 - rho_min)/2 + 1
    # rdist = (np.tanh(10*(r - 3/10)) + np.tanh(-10*(r - 9/10)))*(1 - rho_min)/2 + rho_min
    
    rho0['g'] = rdist*zdist+rho_min # adding rho_min here to resolve the rho_min product concern with negative density

    #Note that in some configs, the minimum density reads as being *lower* than 0.011 unless dealias = 3/2 (rather than 1) is used.
    # Could this be an argument for using dealiasing? Both go negative in density anyway, though.

    # rdist = np.tanh(40*r+40)*(zdist-rho_min)/2 + np.tanh(40*(1-r))*(zdist-rho_min)/2 + rho_min old tanh disk distribution

    if log_density:
        lnrho['g'] = np.log(rho0['g'])
    else:
        rho['g'] = rho0['g']
    T['g'] = T0 * rho0['g']**(gamma - 1) # np.exp(lnrho['g'])
    ##eta['g'] = eta_sp/(np.sqrt(T['g'])**3 + (eta_ch/np.sqrt(rho0))*(1 - np.exp((-v0_ch)/(3*rho0*np.sqrt(gamma*T['g']))))

    if parity:
        # zero_modes(A,0)
        # zero_modes(v,1)
        # zero_modes(T,0,scalar=True)
        # zero_modes(lnrho,0,scalar=True)
        # zero_modes(phi,1,scalar=True)

        A['c'][0,1::2,0::2,0::2] = 0
        A['c'][1,0::2,1::2,0::2] = 0
        A['c'][2,0::2,0::2,1::2] = 0

        v['c'][0,0::2,1::2,1::2] = 0
        v['c'][1,1::2,0::2,1::2] = 0
        v['c'][2,1::2,1::2,0::2] = 0

        T['c'][1::2,1::2,1::2] = 0
        if log_density:
            lnrho['c'][1::2,1::2,1::2] = 0
        phi['c'][0::2,0::2,0::2] = 0

    # Not sure if I'll need an ncc_cutoff for the solver
    solver = Rho.build_solver()

    #NLBVP's are a little different. Can't just .solve() them.
    #solver.solve()
    tolerance = 1e-10
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    
    if log_density:
        dens = lnrho
    else:
        dens = rho
    return dens
    # return A, T, dens