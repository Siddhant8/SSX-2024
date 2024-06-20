"""SSX_model_A.py

This is the *simplest* model we will consider for modelling spheromaks evolving in the SSX wind tunnel.

Major simplificiations fall in two categories

Geometry
--------
We consider a square duct using parity bases (sin/cos) in all directions.

Equations
---------
The equations themselves are those from Schaffner et al (2014), with the following simplifications

* hall term off
* constant eta instead of Spitzer
* no wall recycling term
* no mass diffusion

For this first model, rather than kinematic viscosity nu and thermal
diffusivitiy chi varying with density rho as they should, we are here
holding them *constant*. This dramatically simplifies the form of the
equations in Dedalus.

We use the vector potential, and enforce the Coulomb Gauge, div(A) = 0.

"""

import os
import sys
import time
import numpy as np

import dedalus.public as de
from dedalus.extras import flow_tools

from matplotlib import pyplot
import matplotlib as mpl

from spheromak import spheromak_A

import logging
logger = logging.getLogger(__name__)


# for optimal efficiency: nx should be divisible by mesh[0], ny by mesh[1], and
# nx should be close to ny. Bridges nodes have 28 cores, so mesh[0]*mesh[1]
# should be a multiple of 28.
nx = 28 
ny = 24
nz = 180
r = 1
length = 10

# for 3D runs, you can divide the work up over two dimensions (x and y).
# The product of the two elements of mesh *must* equal the number
# of cores used.
# mesh = None
mesh = [14,12]

kappa = 0.01
mu = 0.05
eta = 0.001
rho0 = 1
gamma = 5./3.

x = de.SinCos('x', nx, interval=(-r, r))
y = de.SinCos('y', ny, interval=(-r, r))
z = de.SinCos('z', nz, interval=(0,length))

domain = de.Domain([x,y,z],grid_dtype='float', mesh=mesh)

SSX = de.IVP(domain, variables=['lnrho','T', 'vx', 'vy', 'vz', 'Ax', 'Ay', 'Az', 'phi'])
##########################################################################################
"""Meta Parameters"""
##########################################################################################
SSX.meta['T','lnrho']['x', 'y', 'z']['parity'] = 1
SSX.meta['phi']['x', 'y', 'z']['parity'] = -1

SSX.meta['vx']['y', 'z']['parity'] =  1
SSX.meta['vx']['x']['parity'] = -1
SSX.meta['vy']['x', 'z']['parity'] = 1
SSX.meta['vy']['y']['parity'] = -1
SSX.meta['vz']['x', 'y']['parity'] = 1
SSX.meta['vz']['z']['parity'] = -1

SSX.meta['Ax']['y', 'z']['parity'] =  -1
SSX.meta['Ax']['x']['parity'] = 1
SSX.meta['Ay']['x', 'z']['parity'] = -1
SSX.meta['Ay']['y']['parity'] = 1
SSX.meta['Az']['x', 'y']['parity'] = -1
SSX.meta['Az']['z']['parity'] = 1
#########################################################################################
"""    Parameters """
#########################################################################################
SSX.parameters['mu'] = mu
SSX.parameters['chi'] = kappa/rho0
SSX.parameters['nu'] = mu/rho0
SSX.parameters['eta'] = eta
SSX.parameters['gamma'] = gamma
#########################################################################################
"""   Subsitutions """
#########################################################################################
SSX.substitutions['divv'] = "dx(vx) + dy(vy) + dz(vz)"
SSX.substitutions['vdotgrad(A)'] = "vx*dx(A) + vy*dy(A) + vz*dz(A)"
SSX.substitutions['Bdotgrad(A)'] = "Bx*dx(A) + By*dy(A) + Bz*dz(A)"
SSX.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A)) + dz(dz(A))"
SSX.substitutions['Bx'] = "dy(Az) - dz(Ay)"
SSX.substitutions['By'] = "dz(Ax) - dx(Az)"
SSX.substitutions['Bz'] = "dx(Ay) - dy(Ax)"

# Coulomb Gauge implies J = -Laplacian(A)
SSX.substitutions['jx'] = "-Lap(Ax)"
SSX.substitutions['jy'] = "-Lap(Ay)"
SSX.substitutions['jz'] = "-Lap(Az)"
SSX.substitutions['J2'] = "jx**2 + jy**2 + jz**2"
SSX.substitutions['rho'] = "exp(lnrho)"
# CFL substitutions
SSX.substitutions['Va_x'] = "Bx/sqrt(rho)"
SSX.substitutions['Va_y'] = "By/sqrt(rho)"
SSX.substitutions['Va_z'] = "Bz/sqrt(rho)"
SSX.substitutions['Cs'] = "sqrt(gamma*T)"
#########################################################################################
"""    Equations    """
#########################################################################################
# Continuity
SSX.add_equation("dt(lnrho) + divv = - vdotgrad(lnrho)")

# Momentum
SSX.add_equation("dt(vx) + dx(T) - nu*Lap(vx) = T*dx(lnrho) - vdotgrad(vx) + (jy*Bz - jz*By)/rho")
SSX.add_equation("dt(vy) + dy(T) - nu*Lap(vy) = T*dy(lnrho) - vdotgrad(vy) + (jz*Bx - jx*Bz)/rho")
SSX.add_equation("dt(vz) + dz(T) - nu*Lap(vz) = T*dz(lnrho) - vdotgrad(vz) + (jx*By - jy*Bx)/rho")

# MHD equations: A
# Resistive Ohm's Law: A - form
SSX.add_equation("dt(Ax) + eta*jx + dx(phi) = vy*Bz - vz*By")
SSX.add_equation("dt(Ay) + eta*jy + dy(phi) = vz*Bx - vx*Bz")
SSX.add_equation("dt(Az) + eta*jz + dz(phi) = vx*By - vy*Bx")
# Coulomb Gauge for A
# Away from walls
SSX.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition="(nx != 0) or (ny != 0) or (nz != 0)")
# Density condition at the origin
# Electric potential at the walls
SSX.add_equation("phi = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")


# Energy
SSX.add_equation("dt(T) - (gamma - 1) * chi*Lap(T) = - (gamma - 1) * T * divv  - vdotgrad(T) + (gamma - 1)*eta*J2")
#########################################################################################
"""   Solver Setup """
#########################################################################################
solver = SSX.build_solver(de.timesteppers.RK443)

# Initial timestep
dt = 1e-4

# Integration parameters
solver.stop_sim_time = 50
solver.stop_wall_time = 60*60*9
solver.stop_iteration = np.inf


# Initial conditions
Ax = solver.state['Ax']
Ay = solver.state['Ay']
Az = solver.state['Az']
lnrho = solver.state['lnrho']
T = solver.state['T']
vz = solver.state['vz']
vy = solver.state['vy']
vx = solver.state['vx']

# Grid Points
x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
fullGrid = x*y*z

# Initial condition parameters
R = r
L = R
lambda_rho = 0.4 # half-width of transition region for initial conditions
lambda_rho1 = 0.1  # Radius of Initial Density 
rho_min = 0.011
T0 = 0.1
# Mati has perturbation term
delta = 0.1 # The strength of the perturbation. PSST 2014 has delta = 0.1 .
######################################################################################
"""  Spheromak Initial Conditions  """
######################################################################################
## Spheromak initial condition
aa_x, aa_y, aa_z = spheromak_A(domain, center=(0,0, 0), R=R, L=L)
Ax['g'] = aa_x*(1 + delta*x*np.exp(-z**2))
Ay['g'] = aa_y*(1 + delta*x*np.exp(-z**2))
Az['g'] = aa_z*(1 + delta*x*np.exp(-z**2))
# Plasma density initializations. Equation 7 within the reference from above.
####################################################################################################################################################################
""" Initial Density definition """
####################################################################################################################################################################

# Z - Dependence
for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
       yVal = y[0,j,0]
       for k in range(z.shape[2]):
           zVal = z[0,0,k]
           fullGrid[i][j][k] = -np.tanh(6*zVal - 6)*(1 - rho_min)/2 + 1 #density in the z direction with tanh transition
# X, Y - Dependence: Circle
for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
       yVal = y[0,j,0]
       for k in range(z.shape[2]):
            zVal = z[0,0,k]
            r = np.sqrt(xVal**2 + yVal**2)
            if(r <= 1 - lambda_rho1):
               fullGrid[i][j][k] = fullGrid[i][j][k]
            elif((r >= 1 - lambda_rho1 and r <= 1 + lambda_rho1)):
               fullGrid[i][j][k] = (fullGrid[i][j][k] + rho_min)/2 + (fullGrid[i][j][k] - rho_min)/2*np.sin((1 - r) * np.pi/(2*lambda_rho1))
                # I wonder where the holes are located, it maybe caused by the above function.
            else:
               fullGrid[i][j][k] = rho_min
####################################################################################################################################################################
""" Previous Density definition """
####################################################################################################################################################################
"""for i in range(x.shape[0]):
	xVal = x[i,0,0]
	for j in range(y.shape[1]):
		yVal = y[0,j,0]
		for k in range(z.shape[2]):
			zVal = z[0,0,k]
			if(0 <= zVal and zVal < ( 1 - lambda_rho) and (np.sqrt(xVal**2 + yVal**2)<R)):
				fullGrid[i][j][k] = 1
			elif((( (1 - lambda_rho) <= zVal) and  zVal<( 1 + lambda_rho)) and (np.sqrt(xVal**2 + yVal**2)<R)):
				fullGrid[i][j][k] = (1 + rho_min)/2 + (1 - rho_min)/2*(np.sin((1 - zVal)*np.pi/(2*lambda_rho)))#rho_min + rho_min*np.cos(zVal*np.pi/(2*lambda_rho)
			else:
				fullGrid[i][j][k] = rho_min"""
    ####################################################################################################################################################################
# Initial Velocity
max_vel = 0.1
vz['g'] = -np.tanh(10*z - 2)*max_vel + max_vel

rho0 = domain.new_field()
rho0['g'] = fullGrid

lnrho['g'] = np.log(rho0['g'])
T['g'] = T0 * rho0['g']**(gamma - 1)


# analysis output
#data_dir = './'+sys.argv[0].split('.py')[0]
wall_dt_checkpoints = 60*55
output_cadence = .10 # This is in simulation time units

'''checkpoint = solver.evaluator.add_file_handler('checkpoints2', max_writes=1, wall_dt=wall_dt_checkpoints, mode='overwrite')
checkpoint.add_system(solver.state, layout='c')'''

field_writes = solver.evaluator.add_file_handler('fields10', max_writes = 500, sim_dt = output_cadence, mode = 'overwrite')
field_writes.add_task('vx')
field_writes.add_task('vy')
field_writes.add_task('vz')
field_writes.add_task('Bx')
field_writes.add_task('By')
field_writes.add_task('Bz')
field_writes.add_task("exp(lnrho)", name = 'rho')
field_writes.add_task('T')

parameter_writes = solver.evaluator.add_file_handler('parameters10', max_writes = 1, mode = 'overwrite')
parameter_writes.add_task('chi')
parameter_writes.add_task('nu')
parameter_writes.add_task('mu')
parameter_writes.add_task('eta')
parameter_writes.add_task('gamma')

load_writes = solver.evaluator.add_file_handler('load_data10', max_writes = 500, sim_dt = output_cadence, mode = 'overwrite')
load_writes.add_task('vx')
load_writes.add_task('vy')
load_writes.add_task('vz')
load_writes.add_task('Ax')
load_writes.add_task('Ay')
load_writes.add_task('Az')
load_writes.add_task('lnrho')
load_writes.add_task('T')
load_writes.add_task('phi')


# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence = 1)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / nu", name = 'Re_k')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / eta", name = 'Re_m')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / sqrt(T)", name = 'Ma_K')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz)/sqrt(Va_x*Va_x + Va_y*Va_y + Va_z*Va_z)", name = 'Ma_A')
flow.add_property("T", name = 'temp')

char_time = 1.0 # this should be set to a characteristic time in the problem (the alfven crossing time of the tube, for example)
CFL_safety = 0.3
CFL = flow_tools.CFL(solver, initial_dt = dt, cadence = 1, safety = CFL_safety,
                     max_change = 1.5, min_change = 0.005, max_dt = output_cadence, threshold = 0.05)
CFL.add_velocities(('vx', 'vy', 'vz'))
CFL.add_velocities(('Va_x', 'Va_y', 'Va_z'))
CFL.add_velocities(( 'Cs', 'Cs', 'Cs'))


good_solution = True
# Main loop
try:
    logger.info('Starting loop')
    logger_string = 'kappa: {:.3g}, mu: {:.3g}, eta: {:.3g}, dt: {:.3g}'.format(kappa, mu, eta, dt)
    logger.info(logger_string)
    start_time = time.time()
    while solver.ok and good_solution:
        dt = CFL.compute_dt()
        solver.step(dt)

        if (solver.iteration - 1) % 1 == 0:
            logger_string = 'iter: {:d}, t/tb: {:.4e}, dt/tb: {:.4e} '.format(solver.iteration, solver.sim_time/char_time, dt/char_time)
           # logger_string += 'min_rho: {:.4e}'.format(lnrho['g'].min())
            Re_k_avg = flow.grid_average('Re_k')
            Re_m_avg = flow.grid_average('Re_m')
            logger_string += ' Max Re_k = {:.4g}, Avg Re_k = {:.4g}, Max Re_m = {:.4g}, Avg Re_m = {:.4g}, Max Ma_K = {:.4g}, Max Ma_A = {:.4g}'.format(flow.max('Re_k'), Re_k_avg, flow.max('Re_m'), Re_m_avg, flow.max('Ma_K'), flow.max('Ma_A'))
            logger.info(logger_string)
            np.clip(lnrho['g'], -4.9, 2, out=lnrho['g'])
            np.clip(T['g'], 0.001, 1000, out=T['g'])
            np.clip(vx['g'], -100, 100, out=vx['g'])
            np.clip(vy['g'], -100, 100, out=vy['g'])
            np.clip(vz['g'], -100, 100, out=vz['g'])
            if not np.isfinite(Re_k_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on kinetic Reynolds = {}".format(Re_k_avg))
            if not np.isfinite(Re_m_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on magnetic Reynolds = {}".format(Re_m_avg))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Iter/sec: {:g}'.format(solver.iteration/(end_time-start_time)))
