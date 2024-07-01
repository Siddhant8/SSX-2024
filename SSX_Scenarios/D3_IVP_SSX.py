"""D3_IVP_SSX.py

This is the *simplest* model we will consider for modeling spheromaks evolving in the SSX wind tunnel.

Major simplifications fall in two categories
lnrho = dist.Field(name = 'lnrho', bases = (xbasis, ybasis,))
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

File formerly called D3_SSX_A_2_spheromaks or similar
- when looking for older versions, check both current name and that name.

Currently working on solving T/rho unphysicalities.

Dedalus 3 edits made by Alex Skeldon and Siddhant Ranka. Direct all queries to askeldo1@swarthmore.edu or sranka1@swarthmore.edu.

To Do: Figure out how to install dedalus3 in jupyter notebook - ask Alex possibly
To Do: Run code with the value of beta I calculated
Why does beta suddenly spike up after 225 iterations to become greater than 1?
Why does the minimum alfvén time become negative after beta comes back down?
Try increasing the resolution even more or making the IC curve less steep
Code crashes when the temperature becomes negative - happens at 844 iterations - check why this happens

"""

import numpy as np
import os
import dedalus.public as d3
from dedalus.extras import flow_tools
from mpi4py import MPI


from scipy.special import j0, j1, jn_zeros
from D3_LBVP_SSX import spheromak_pair, zero_modes
from D3_Rho_NLBVP_SSX import pair_density
import logging
logger = logging.getLogger(__name__)


# Parameters we commonly modify between runs include:
# resolution, dealias, parity enforcement, source of A, if we perturb A, mesh, data_dir name, mu (Re), and eta (Rm)

# for optimal efficiency: nx should be divisible by mesh[0], ny by mesh[1], and
# nx should be close to ny. Bridges nodes have 128 cores, so mesh[0]*mesh[1]
# should be a multiple of 128.
nx,ny,nz = 32,32,160 #formerly 32 x 32 x 160? Current plan is 64 x 64 x 320 or 640
#nx,ny,nz = 64,64,320
#nx,ny,nz = 128,128,640
# nx,ny,nz = 16, 16, 80

dealias = 1

# These control whether to use parity enforcement lines,
# whether to initialize A from the LBVP or from analytical expression here, and whether to perturb A
# Recently added these flags to make it easier to turn those on or off rather than (de)commenting
A_perturb = True

# for 3D runs, you can divide the work up over two dimensions (x and y).
# The product of the two elements of mesh *must* equal the number
# of cores used.
#mesh = [16,16]
#mesh = [8,8]
mesh = [2,2]
# mesh = None
data_dir = "scratch" #change each time or overwrite

kappa = 0.1
# try both of these 0.1 see what happens
mu = 0.05 #Determines Re_k ; 0.05 -> Re_k = 20 (try 0.005?)
eta = 0.01 # Determines Re_m ; 0.001 -> Re_m = 1000; using smaller Rm of 100 for now since 1000 is a bit high.

gamma = 5./3.

B0 = 1

# eta_sp = 2.7 * 10**(-4)
# eta_ch = 4.4 * 10**(-3)
# v0_ch = 2.9 * 10**(-2)

chi = kappa
nu = mu
#diffusivities for heat (kappa -> chi), momentum (viscosity) (mu -> nu), current (eta)
# life time of currents regulated by resistivity
# linearization time of temperature goes like e^-t/kappa

# SSX dimensions
rad = 1
length = 10

offset = 1

#Coords, dist, bases
coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64, mesh = mesh)

xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(-rad, rad), dealias = dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(-rad, rad), dealias = dealias)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, length), dealias = dealias)

# Fields
t = dist.Field(name='t')
v = dist.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
#rho = dist.Field(name='rho', bases=(xbasis, ybasis, zbasis))

T = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
phi = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))
tau_A = dist.Field(name='tau_A')
tau_p = dist.Field(name='tau_p')
# eta = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
ex, ey, ez = coords.unit_vector_fields(dist)

# Coulomb Gauge implies J = -Laplacian(A)
j = -d3.lap(A)
J2 = j@j
B = d3.curl(A)
rho_min = 0.011
#spitzer and chodra resistivity combination
# eta = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))

#lnrho = np.log(rho)
#lnrho_dt = d3.TimeDerivative(lnrho)

#lnrho = dist.Field(name = 'lnrho', bases=(xbasis, ybasis, zbasis))

# CFL substitutions
Va = B/np.sqrt(rho_min) # mu_0 = 1 in our sim
Cs = np.sqrt(gamma*T)
Cs_vec = Cs*ex + Cs*ey + Cs*ez


#Problem
SSX = d3.IVP([v, A, T, phi, tau_A, tau_p], time=t, namespace=locals())

#variable resistivity
# SSX.add_equation("eta = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))")

# Not really good model but this would be how you'd express incompressibility

SSX.add_equation("div(v) + tau_p = 0")

# Continuity
#SSX.add_equation("dt(lnrho) = -div(v) - v@grad(lnrho)")

# Momentum
SSX.add_equation("dt(v) + grad(T) - nu*lap(v) = - v@grad(v) + cross(j,B)/(rho_min)")

# MHD equations: A
SSX.add_equation("dt(A) + grad(phi) + eta*j = cross(v,B)")

#gauge constraints
SSX.add_equation("div(A) + tau_A = 0")
SSX.add_equation("integ(phi) = 0")

# Energy
SSX.add_equation("dt(T) - (gamma - 1) * chi*lap(T) = - (gamma - 1) * T * div(v) - v@grad(T) + (gamma - 1)*eta*J2")

solver = SSX.build_solver(d3.RK222) # (now 222, formerly 443; try both)

logger.info("Solver built")

# Initial timestep
dt = 1e-4

# Integration parameters
solver.stop_sim_time = 20 #historically 20
solver.stop_wall_time = np.inf #e.g. 60*60*3 would limit runtime to three hours
solver.stop_iteration = np.inf

x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
aa = dist.VectorField(coords, name='aa', bases=(xbasis, ybasis, zbasis))

# Initial condition parameters
R = rad
L = R
rho_min = 0.011
T0 = 0.1
delta = 0.1 # The strength of the perturbation. Schaffner et al 2014 (flux-rope plasma) has delta = 0.1.
r = np.sqrt(x**2+y**2)

# Spheromak initial condition
j1_zero1 = jn_zeros(1,1)[0]
kr = j1_zero1/R
kz = np.pi/L

#BEGINNING of In-line vector potential
# if not(LBVP_A):
#     handedness = 1
#     b0 = 1
#     lam = np.sqrt(kr**2 + kz**2)
#     theta = np.arctan2(y,x)

#     Ar = -b0*kz*j1(kr*r)*np.cos(kz*z)/lam
#     At = handedness*b0*j1(kr*r)*np.sin(kz*z)
#     Az = b0*j0(kr*r)*np.cos(kz*z)/lam

#     #now we need to add a rotated and translated copy
#     # since we have angular symmetry, we just need to translate 10 units
#     # and reverse the z component (i.e. negative sign)
#     Ar2 = -b0*kz*j1(kr*r)*np.cos(kz*(-(z-10)))/lam
#     At2 = handedness*b0*j1(kr*r)*np.sin(kz*(-(z-10)))
#     Az2 = - b0*j0(kr*r)*np.cos(kz*(-(z-10)))/lam

#     #We need to localize these fields so they go to 0 in 1 < z < 10 and r > 1
#     #use similar tanh's to initialized density
#     # zVecDist = ((-np.tanh(2 *(z - 1.5)) - np.tanh(-2*(z - 8.5)))/2 + 1) # Keeping here in case I decide to switch back to this expression

#     #Here's a z-distribution that goes to zero at z = 10 and z = 0, could be useful for vector potential drop-off
#     # (want a constant value or close to it at both sides of the boundary)
#     zVecDist2 = (-np.tanh(4*(z - 3)) + np.tanh(4*(z - 1)) - np.tanh(-4*(z - 7)) + np.tanh(-4*(z - 9)))/2
#     rVecDist = -np.tanh(5*(r - 1))/2 + 0.5

#     aa['g'][0] = ((Ar+Ar2)*np.cos(theta) - (At+At2)*np.sin(theta)) * zVecDist2 * rVecDist
#     aa['g'][1] = ((Ar+Ar2)*np.sin(theta) + (At+At2)*np.cos(theta)) * zVecDist2 * rVecDist
#     aa['g'][2] = (Az+Az2) * zVecDist2 * rVecDist

aa = spheromak_pair(xbasis,ybasis,zbasis, coords, dist)

'''''
handedness = 1
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

# The vector potential is subject to some perturbation. This distorts all the magnetic field components in the same direction.
'''
if A_perturb:
    for i in range(3):
        A['g'][i] = aa['g'][i] *(1 + delta*x*np.exp(-z**2) + delta*x*np.exp(-(z-10)**2)) # maybe the exponent here is too steep of an IC?
else:
    for i in range(3):
        # A['g'][i] = aa['g'][i]
        
        # zero vec pot. init.
        A['g'][i] = 0

rho0 = dist.Field(name='rho0', bases=(xbasis, ybasis, zbasis))
rho0['g'] = np.zeros_like(T['g'])




wavz = 2

#First full-time run took a while to move towards each other, might want to increase max_vel, or modify the tanh distribution
max_vel = 0.2
v['g'][2] = -np.tanh(z-2)*max_vel/2 + -np.tanh(z - 8)*max_vel/2
#v['g'][2] = -np.tanh(6*z - 6)*max_vel/2 + -np.tanh(6*z - 54)*max_vel/2 # original steeper transition
#v['g'][2] = -np.sin(wavz*np.pi*z/length)*max_vel


#try making v sine and rdist and zdist tanh at some point

#Changed from disk density distribution to a donut distribution
#I'm not sure why Slava's HiFi simulation only had a z-dependent distribution for density, and no radial
#(which would produce disks instead of donuts since he had cylindrical geometry)

# Original cos(r) * cos(z) density distributions

# main issue with this is cusps at square edges/increasing values at the corners
#consider putting a decaying radial exponential on this to offset larger values at boundaries
#need this decaying radial exponential - -8, -10, and -12 have better results
#don't subtract rdist by an offset
#don't add rdist by offset either
#try removing the 2 in the cos function

#rdist and zdist are what is responsible for showing four plasmas
#rdist = (-np.cos(np.pi*r/rad)*(1-rho_min)/2 + 1/2)*np.exp(-4*r)
#zdist = -np.cos(wavz*4*np.pi*z/length)*(1-rho_min)/2 + 1/2

zdist = (-np.tanh(4*(z - 3)) + np.tanh(4*(z - 1)) - np.tanh(-4*(z - 7)) + np.tanh(-4*(z - 9)))/2
rdist = -np.tanh(5*(r - 1))/2 + 0.5

#zdist = (np.tanh(2*(z-1.5)) - np.tanh(-2*(x-8.5)))*(1-rho_min)/2 + 1
#rdist = (np.tanh(10*(r-3/10)) + np.tanh(-10*(r-9/10))) * (1-rho_min)/2 + rho_min
# other current option is to try the poloidal flux formulation as Doc suggested:
#rdist = r *j1(kr *r)
#zdist = np.sin(kz*z)
#poloidal flux didn't work - ask Doc what is going wrong

# Attempted well-behaving at square boundaries xy distribution
# rdist = -np.cos(2*np.pi*2*x/rad)*np.cos(2*np.pi*y/rad)*(1-rho_min)/2 + 1/2

# Solve the NLBVP for initial density to get smooth ICs that agree with magnetics
totaldist = rdist*zdist+rho_min # adding rho_min here to resolve the rho_min product concern with negative density

# constant density test
#totaldist = 1
#making totaldist = 2 didn't work 

rho0['g'] = totaldist

#Note that in some configs, the minimum density reads as being *lower* than 0.011 unless dealias = 3/2 (rather than 1) is used.
# Could this be an argument for using dealiasing? Both go negative in density anyway, though.

# rdist = np.tanh(40*r+40)*(zdist-rho_min)/2 + np.tanh(40*(1-r))*(zdist-rho_min)/2 + rho_min old tanh disk distribution

#rho['g'] = rho0['g']
#lnrho['g'] = np.log(rho0['g'])

T['g'] = T0 * rho0['g']**(gamma - 1)
##eta['g'] = eta_sp/(np.sqrt(T['g'])**3 + (eta_ch/np.sqrt(rho0))*(1 - np.exp((-v0_ch)/(3*rho0*np.sqrt(gamma*T['g']))))

# Frame for meta params in D3 with RealFourier
# Apparently the parity can force zero values at boundaries, as makeshift approach to BCs.
# That's what I gleaned from https://groups.google.com/u/1/g/dedalus-users/c/XwHzS_T3zIE/m/WUQlQVIKAgAJ
#ignore zero_modes function calling lines for now since I wrote out the 9 manual ones anyway
#Just need to decomment the var['c'] lines here, in the timestepping loop, and in the LBVP to turn on parity enforcement

# analysis output
wall_dt_checkpoints = 2
output_cadence = 0.1 # This is in simulation time units

fh_mode = 'overwrite'

# load state for restart
# Don't forget to switch to append for the mode when you load state!
#solver.load_state("scratch/checkpoints2/checkpoints2_s1.h5")


#handle data output dirs
if dist.comm.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

#other things big, this generally small (when not doing every iter) # but iter = 1 is the diagnostic term # sim_dt = output_cadence
checkpoint = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), max_writes=10, iter = 10, mode = fh_mode)
checkpoint.add_tasks(solver.state)


field_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'fields_derived'), max_writes = 10, iter = 10, mode = fh_mode)
# trying to just put j for third one yields issues - because j not variable in problem? # sim_dt = output_cadence
field_writes.add_task(B, name = 'B')
field_writes.add_task(d3.curl(B), name='j')
field_writes.add_task(T)
field_writes.add_task(d3.cross(j,B), name='f_mag')
# field_writes.add_task(eta)

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), max_writes=10, sim_dt = output_cadence, mode=fh_mode)
timeseries.add_task(d3.integ(A@B), name="total_helicity")
timeseries.add_task(A@B, name="helicity_at_pos")
timeseries.add_task(0.5*d3.integ(v@v),name='Ekin')
timeseries.add_task(0.5*d3.integ(B@B),name='Emag')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence = 1)
flow.add_property(np.sqrt(v@v) / nu, name = 'Re_k')
flow.add_property(np.sqrt(v@v) / eta, name = 'Re_m')
flow.add_property(np.sqrt(v@v), name = 'flow_speed')
flow.add_property(np.sqrt(v@v) / np.sqrt(T), name = 'Ma') # Mach number; T going negative? T goes negative because rho goes negative
#flow.add_property(np.sqrt(B@B) / rho_min, name = 'Al_v')
flow.add_property(np.sqrt(B@B / np.sqrt(rho_min)), name = 'Al_v') # see if this makes it more positively well-behaved
flow.add_property(T, name = 'temp')
#flow.add_property(np.exp(lnrho), name = 'density')
#flow.add_property(lnrho, name = 'log_density')
flow.add_property(0.5*d3.integ(v@v), name='Ekin')
flow.add_property(0.5*d3.integ(B@B),name='E_mag')
flow.add_property(0.5*0.1*(np.sqrt(T)**2)/(0.5*(B@B)), name = 'beta')
flow.add_property(Cs_vec, name = 'Cs_vector')
flow.add_property(B, name='B')

#beta is 0.036 to start when setting 0.5nkT as 1 - follows the beta we want
#beta is 0.037 when code crashes - beta is not an issue unless temperature 

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
    while solver.proceed and good_solution:
        dt = CFL.compute_timestep()
        solver.step(dt)
            
        if (solver.iteration-1) % 1 == 0:
            logger_string = 'iter: {:d}, t/tb: {:.2e}, dt/tb: {:.2e}, sim_time: {:.4e}, dt: {:.2e}'.format(solver.iteration, solver.sim_time/char_time, dt/char_time, solver.sim_time, dt)
            Re_k_avg = flow.grid_average('Re_k')
            Re_m_avg = flow.grid_average('Re_m')
            v_avg = flow.grid_average('flow_speed')
            Al_v_avg = flow.grid_average('Al_v')

            #This seems to be how you spill over lines
            #not sure if there's a way to read out E_mag without applying flow methods to it, but it gives a value anyway

            # version without Re_k and Re_m
#             logger_string += ', Max vel = {:.2g}, Avg vel = {:.2g}, Max alf vel = {:.2g}, Avg alf vel = {:.2g}, \
# Max Ma = {:.1g}, max rho = {:.2g}, min rho = {:.2g}, \
# min T = {:.2g}, min Al_v = {:.2g}, Emag = {:.2g}, '.format(\
# flow.max('flow_speed'), v_avg, flow.max('Al_v'), Al_v_avg, flow.max('Ma'), flow.max('density'), flow.min('density'),\
# flow.min('temp'),flow.min('Al_v'),flow.grid_average('E_mag'))
            
            # version with Re_k and Re_m
            logger_string += ' Max Re_k = {:.2g}, Avg Re_k = {:.2g}, Max Re_m = {:.2g}, \
Avg Re_m = {:.2g}, Max vel = {:.2g}, Avg vel = {:.2g}, Max alf vel = {:.2g}, Avg alf vel = {:.2g}, \
Max Ma = {:.1g}, max T = {:.2g}, \
min T = {:.2g}, min Al_v = {:.2g}, Emag = {:.2g}, Ekin = {:.2g}, B = {:.2g}, beta = {:.2g} '.format(flow.max('Re_k'), Re_k_avg, flow.max('Re_m'), Re_m_avg,\
flow.max('flow_speed'), v_avg, flow.max('Al_v'), Al_v_avg, flow.max('Ma'), flow.max('temp'), \
flow.min('temp'),flow.min('Al_v'),flow.grid_average('E_mag'), flow.grid_average('Ekin'), flow.max('B'), flow.grid_average('beta'))
#can't do dot product of non-coordinate fields
#can't take grid average of temperature - go into why I can't do this
            logger.info(logger_string)

            # if not np.isfinite(Re_k_avg):
            #     good_solution = False
            #     logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_k_avg))
            # if not np.isfinite(Re_m_avg):
            #     good_solution = False
            #     logger.info("Terminating run. Trapped on magnetic Reynolds = {}".format(Re_m_avg))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
