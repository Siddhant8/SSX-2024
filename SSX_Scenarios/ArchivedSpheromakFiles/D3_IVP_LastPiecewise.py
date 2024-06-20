"""SSX_model_A.py

This is the *simplest* model we will consider for modeling spheromaks evolving in the SSX wind tunnel.

Major simplifications fall in two categories

Geometry
--------
We consider a square duct using parity bases (sin/cos) in all directions. (RealFourier in D3)

Equations
---------
The equations themselves are those from Schaffner et al (2014), with the following simplifications

* hall term off
* constant eta instead of Spitzer
* no wall recycling term
* no mass diffusion

For this first model, rather than kinematic viscosity nu and thermal
diffusivity chi varying with density rho as they should, we are here
holding them *constant*. This dramatically simplifies the form of the
equations in Dedalus.

We use the vector potential, and enforce the Coulomb Gauge, div(A) = 0.

File formerly called D3_SSX_A_2_spheromaks
- when looking for older versions, check both current name and that name.

# This week I am working on solving T/rho unphysicalities.

Dedalus 3 edits made by Alex Skeldon. Direct all queries to askeldo1@swarthmore.edu.

# I will be changing the initializations in this and the LBVP to not have piecewise functions anymore,
# because those are bad. These last piecewise versions will be saved in the archive folder for reference.
"""

import numpy as np
import os
import sys
import dedalus.public as d3
from dedalus.extras import flow_tools


from D3_LBVP_SSX import spheromak_pair, zero_modes

import logging
logger = logging.getLogger(__name__)


# for optimal efficiency: nx should be divisible by mesh[0], ny by mesh[1], and
# nx should be close to ny. Bridges nodes have 128 cores, so mesh[0]*mesh[1]
# should be a multiple of 128.
# nx,ny,nz = 32,32,160 #formerly 32 x 32 x 160? Current plan is 64 x 64 x 320 or 640
nx,ny,nz = 64,64,160
#nx,ny,nz = 128,128,640
r = 1
length = 10

# for 3D runs, you can divide the work up over two dimensions (x and y).
# The product of the two elements of mesh *must* equal the number
# of cores used.
# mesh = [32,32]
#mesh = [32,16]
# mesh = [16,16]
#mesh = [16,8]
mesh = [2,2]
#mesh = None
data_dir = "scratch" #change each time or overwrite

kappa = 0.01
# try both of these 0.1 see what happens
mu = 0.05 #Determines Re_k ; 0.05 -> Re_k = 20 (try 0.005?)
eta1 = 0.01 # Determines Re_m ; 0.001 -> Re_m = 1000
rhoIni = 1 #rho0 is redefined later, and generally has a whole mess associated with it
gamma = 5./3.
eta_sp = 2.7 * 10**(-4)
eta_ch = 4.4 * 10**(-3)
v0_ch = 2.9 * 10**(-2)
chi = kappa/rhoIni
nu = mu/rhoIni
#diffusivities for heat (kappa -> chi), momentum (viscosity) (mu -> nu), current (eta)
# life time of currents regulated by resistivity
# linearization time of temperature goes like e^-t/kappa

# I assume we don't have a reason to use dealias yet
# though I'm not quite sure how to identify 2h waves
# that would warrant it, either.
# dealias=3/2
dealias=1

#Coords, dist, bases
coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64, mesh = mesh)

xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(-r, r), dealias = dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(-r, r), dealias = dealias)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, length), dealias = dealias)

# Fields
t = dist.Field(name='t')
v = dist.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
lnrho = dist.Field(name='lnrho', bases=(xbasis, ybasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
phi = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))
tau_A = dist.Field(name='tau_A')
# eta1 = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
ex, ey, ez = coords.unit_vector_fields(dist)

# Coulomb Gauge implies J = -Laplacian(A)
j = -d3.lap(A)
J2 = j@j
rho = np.exp(lnrho)
B = d3.curl(A)
#spitzer and chodra resistivity combination
# eta1 = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))

# CFL substitutions
Va = B/np.sqrt(rho)
Cs = np.sqrt(gamma*T)
Cs_vec = Cs*ex + Cs*ey + Cs *ez

#Problem
SSX = d3.IVP([v, A, lnrho, T, phi, tau_A], time=t, namespace=locals())

#variable resistivity
# SSX.add_equation("eta1 = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))")

# Continuity
SSX.add_equation("dt(lnrho) + div(v) = - v@grad(lnrho)")

# Not really good model but this would be how you'd express incomp.
# SSX.add-equation("div(v) + tau_p = 0")
# sqrt(sum(|lnrhoD2 - lnrhoD3|^2)) (L2)


# Momentum
SSX.add_equation("dt(v) + grad(T) - nu*lap(v) = T*grad(lnrho) - v@grad(v) + cross(j,B)/rho")

# MHD equations: A
SSX.add_equation("dt(A) + grad(phi) = -eta1*j + cross(v,B)")

#gauge constraints
SSX.add_equation("div(A) + tau_A = 0")
SSX.add_equation("integ(phi) = 0")

# Energy
SSX.add_equation("dt(T) - (gamma - 1) * chi*lap(T) = - (gamma - 1) * T * div(v) - v@grad(T) + (gamma - 1)*eta1*J2")

solver = SSX.build_solver(d3.RK222) # (now 222, formerly 443; try both)

logger.info("Solver built")

# Initial timestep
dt = 1e-4

# Integration parameters
solver.stop_sim_time = 1 #historically 20
solver.stop_wall_time = np.inf #e.g. 60*60*3 would limit runtime to three hours
solver.stop_iteration = np.inf

x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
rho0 = np.zeros_like(lnrho['g'])

# Initial condition parameters
R = r
L = R
# lambda_rho = 0.4 # half-width of z transition region for initial conditions - not used in current z transition
# expression but good to keep in mind (refer to older IVP versions for full sine loop)
# lambda_rho1 = 0.2 #Similar parameter, but used for r-direction transition; historically 0.1, will try to smooth with 0.2
lambda_rho1 = 0.1
rho_min = 0.011
T0 = 0.1
delta = 0.1 # The strength of the perturbation. Schaffner et al 2014 (flux-rope plasma) has delta = 0.1.

# Spheromak initial condition
# The vector potential is subject to some perturbation. This distorts all the magnetic field components in the same direction.
aa = spheromak_pair(xbasis,ybasis,zbasis, coords, dist)
for i in range(3):
   A['g'][i] = aa['g'][i] *(1 + delta*x*np.exp(-z**2) + delta*x*np.exp(-(z-10)**2)) # maybe the exponent here is too steep of an IC?
# for i in range(3):
#     A['g'][i] = aa['g'][i]


# Frame for meta params in D3 with RealFourier
# What is even our theoretical basis for these parities?
# I don't see a particular reason they should be even or odd in each dimension
# Apparently the parity can force zero values at boundaries, as a sort of faux-bc?
# That's what I gleaned from https://groups.google.com/u/1/g/dedalus-users/c/XwHzS_T3zIE/m/WUQlQVIKAgAJ
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
lnrho['c'][1::2,1::2,1::2] = 0
phi['c'][0::2,0::2,0::2] = 0

#initial velocity - use z, or zVal??
max_vel = 0.1
v['g'][2] = -np.tanh(z-2)*max_vel/2 + -np.tanh(z - 8)*max_vel/2
# v['g'][2] = -np.tanh(6*z - 6)*max_vel/2 + -np.tanh(6*z - 54)*max_vel/2


# should always use local grid - never loop over things like this, apparently
for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
        yVal = y[0,j,0]
        for k in range(z.shape[2]):
            zVal = z[0,0,k]
            # v version in for loop - i assume outside as written above is preferable, but not sure if that's doable
            # with rho0, since not a dist.field()
            # v['g'][2] = -np.tanh(6*zVal - 6)*max_vel/2 + -np.tanh(6*zVal - 54)*max_vel/2
            rho0[i][j][k] = -np.tanh(2*zVal-3)*(1-rho_min)/2 -np.tanh(2*(10-zVal)-3)*(1-rho_min)/2 + 1 #density in the z direction with tanh transition
            # rho0[i][j][k] = -np.tanh(6*zVal-6)*(1-rho_min)/2 -np.tanh(6*(10-zVal)-6)*(1-rho_min)/2 + 1 # original steeper transition

            #initial vector potential when using dealias = 3/2:
            # A['g'][0,i,j,k] = aa['g'][0,i,j,k] *(1 + delta*xVal*np.exp(-zVal**2) + delta*xVal*np.exp(-(zVal-10)**2))
            # A['g'][1,i,j,k] = aa['g'][1,i,j,k] *(1 + delta*xVal*np.exp(-zVal**2) + delta*xVal*np.exp(-(zVal-10)**2))
            # A['g'][2,i,j,k] = aa['g'][2,i,j,k] *(1 + delta*xVal*np.exp(-zVal**2) + delta*xVal*np.exp(-(zVal-10)**2))

# enforcing circular cross-section of density

for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
        yVal = y[0,j,0]
        for k in range(z.shape[2]):
            zVal = z[0,0,k]
            rad = np.sqrt(xVal**2 + yVal**2)

##rho0[i][j][k] = np.tanh(40*r+40)*(rho0[i][j][k]-rho_min)/2 + np.tanh(40*(1-r))*(rho0[i][j][k]-rho_min)/2 + rho_min #tanh transition - this line working would require r being identified like how x and y are, which I don't know how to do (if possible)


# sinusodial transition (what z-transition used to be, essentially)
# This appears to be where the negative initial density is coming from, more or less. Note that even with the extra inequalities
# at the end the density still starts negative here. While in the D2 version of this code (also in this folder), it does not.
# This suggests that there is something later in the code that is messing with this, OR that interpolation is somehow different
# in this code than in the D2 code. That would be due to some sort of structural difference between them, the only one of which
# has seemed relevant thus far in my investigations is parity enforcement for the basis (even though parity is off in this
# code right now)

# could always be something else I've forgotten or haven't thought of yet.

# Do note that commenting all lines below here except the one setting the density to a constant of 1 DOES lead to the density minimum just being
# positive one in the timestepping readouts. So that somehow works or is unaffected by whatever is messing with this later.
# commenting these various additional density setting lines in and out and also commenting the sinusoidial transition in and out
# is suggested to get an idea for what effect each combination has. Suggestions as to what the solution to this problem may be
# are MOST WELCOME.
            # rho0[i,j,k] = 1
            # Change this to tanh


            if(rad <= 1 - lambda_rho1):
               rho0[i][j][k] = rho0[i][j][k]
            elif((rad >= 1 - lambda_rho1 and rad <= 1 + lambda_rho1)):
               rho0[i][j][k] = (rho0[i][j][k] + rho_min)/2 + (rho0[i][j][k] - rho_min)/2*np.sin((1-rad) * np.pi/(2*lambda_rho1))
            else:
               rho0[i][j][k] = rho_min
            
            #extra debugging enforcement lines - even with these, it reads some negative initial density
            if rho0[i,j,k] < rho_min:
                rho0[i,j,k] = 1
            if rho0[i][j][k] < rho_min:
                rho0[i][j][k] = 1

#"logical arrays in numpy"

lnrho['g'] = np.log(rho0)
T['g'] = T0 * rho0**(gamma - 1) # np.exp(lnrho['g'])


# Need to put zero mode here after initializing for initial fields to match, maybe?
A['c'][0,1::2,0::2,0::2] = 0
A['c'][1,0::2,1::2,0::2] = 0
A['c'][2,0::2,0::2,1::2] = 0

v['c'][0,0::2,1::2,1::2] = 0
v['c'][1,1::2,0::2,1::2] = 0
v['c'][2,1::2,1::2,0::2] = 0

T['c'][1::2,1::2,1::2] = 0
lnrho['c'][1::2,1::2,1::2] = 0
phi['c'][0::2,0::2,0::2] = 0

##eta1['g'] = eta_sp/(np.sqrt(T['g'])**3 + (eta_ch/np.sqrt(rho0))*(1 - np.exp((-v0_ch)/(3*rho0*np.sqrt(gamma*T['g']))))

# analysis output
wall_dt_checkpoints = 2
output_cadence = 0.1 # This is in simulation time units

fh_mode = 'overwrite'

# load state for restart
# also, does the virtual file work for restarting
# Should just be this line - pretty straightforward.
# not sure if want to switch fh to append when loading states.
# solver.load_state("scratch/checkpoints2/checkpoints2_s1.h5")

#handle data output dirs
# I'm realizing the else statement doesn't necessarily work so well for the Bridges job submitting scheme...
if dist.comm.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # else:
    #     ow = input("this directory already exists. Would you like to overwrite it? (y/n) ")
    #     if ow == 'n':
    #         name = input("what would you like to name the new directory? ('n' to cancel script) ")
    #         if name == 'n':
    #             print("please press ctrl-c.")
    #             quit()
    #         else:
    #             os.mkdir(name)

# wall_dt=wall_dt_checkpoints

# Only look at data from checkpoints - 
checkpoint = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints2'), max_writes=10, sim_dt = output_cadence, mode = fh_mode) #other things big, this generally small (when not doing every iter) # but iter = 1 is the diagnostic term # sim_dt = 0.5*output_cadence
checkpoint.add_tasks(solver.state)


field_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'fields_two'), max_writes = 20, sim_dt = output_cadence, mode = fh_mode)
# trying to just put j for third one yields issues - because j not variable in problem? # sim_dt = output_cadence
field_writes.add_task(v)
field_writes.add_task(B, name = 'B')
field_writes.add_task(d3.curl(B), name='j')
#Supposed to enforce positive rho, but still seeing negative numbers in h5 reader

# These two should be only issues
field_writes.add_task(np.exp(lnrho), name = 'rho')
field_writes.add_task(T)
# field_writes.add_task(eta1)

# Helicity
helicity_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'helicity'), max_writes=20, sim_dt = output_cadence, mode=fh_mode)
helicity_writes.add_task(d3.integ(A@B), name="total_helicity")
helicity_writes.add_task(A@B, name="helicity_at_pos")

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence = 1)
flow.add_property(np.sqrt(v@v) / nu, name = 'Re_k')
flow.add_property(np.sqrt(v@v) / eta1, name = 'Re_m')
flow.add_property(np.sqrt(v@v), name = 'flow_speed')
flow.add_property(np.sqrt(v@v) / np.sqrt(T), name = 'Ma') # Mach number; T going negative?
flow.add_property(np.sqrt(B@B) / np.sqrt(rho), name = 'Al_v')
flow.add_property(T, name = 'temp')
flow.add_property(lnrho, name = 'log density')
flow.add_property(np.exp(lnrho), name = 'density')
flow.add_property(Cs_vec, name = 'Cs_vector')

char_time = 1. # this should be set to a characteristic time in the problem (the alfven crossing time of the tube, for example)
CFL_safety = 0.3
CFL = flow_tools.CFL(solver, initial_dt = dt, cadence = 10, safety = CFL_safety, #cadence 10 or 1, reasons for either (higher dt resolution at merging point - check every 1)
                     max_change = 1.5, min_change = 0.005, max_dt = output_cadence, threshold = 0.05)
CFL.add_velocity(v)
CFL.add_velocity(Va)
CFL.add_velocity(Cs_vec)
#not sure how to turn Cs into a vector; or if that's still something that we ought to be doing
# But I will keep this expression with the unit vector dot prod addition in here fow now

good_solution = True
# Main loop
try:
    logger.info('Starting loop')
    logger_string = 'kappa: {:.3g}, mu: {:.3g}, dt: {:.3g}'.format(kappa, mu, dt) # eta1: {:.3g}, eta1
    logger.info(logger_string)
    while solver.proceed:

        dt = CFL.compute_timestep()
        solver.step(dt)

        # enforce parities for appropriate dynamical variables at each timestep to prevent non-zero buildup
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
        lnrho['c'][1::2,1::2,1::2] = 0
        phi['c'][0::2,0::2,0::2] = 0
            
        if (solver.iteration-1) % 1 == 0:
            logger_string = 'iter: {:d}, t/tb: {:.2e}, dt/tb: {:.2e}, sim_time: {:.4e}, dt: {:.2e}'.format(solver.iteration, solver.sim_time/char_time, dt/char_time, solver.sim_time, dt)
            ##logger_string += 'min_rho: {:.4e}'.format(lnrho['g'].min())
            Re_k_avg = flow.grid_average('Re_k')
            Re_m_avg = flow.grid_average('Re_m')
            v_avg = flow.grid_average('flow_speed')
            Al_v_avg = flow.grid_average('Al_v')
            logger_string += ' Max Re_k = {:.2g}, Avg Re_k = {:.2g}, Max Re_m = {:.2g}, Avg Re_m = {:.2g}, Max vel = {:.2g}, Avg vel = {:.2g}, Max alf vel = {:.2g}, Avg alf vel = {:.2g}, Max Ma = {:.1g}, min log rho = {:.2g}, min rho = {:.2g}, min T = {:.2g}, min Al_v = {:.2g}'.format(flow.max('Re_k'), Re_k_avg, flow.max('Re_m'),Re_m_avg, flow.max('flow_speed'), v_avg, flow.max('Al_v'), Al_v_avg, flow.max('Ma'), flow.min('log density'), flow.min('density'),flow.min('temp'),flow.min('Al_v')) #min test rho = {:.2g}, flow.min('test rho')
            logger.info(logger_string)

            if not np.isfinite(Re_k_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_k_avg))
            if not np.isfinite(Re_m_avg):
                good_solution = False
                logger.info("Terminating run. Trapped on magnetic Reynolds = {}".format(Re_m_avg))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
