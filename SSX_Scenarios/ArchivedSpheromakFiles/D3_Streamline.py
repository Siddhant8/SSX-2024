import numpy as np
import os
import sys
import dedalus.public as d3
from dedalus.extras import flow_tools


from D3_LBVP_SSX import spheromak_pair, parity

import logging
logger = logging.getLogger(__name__)

nx,ny,nz = 32,32,160 #formerly 32 x 32 x 160? Current plan is 64 x 64 x 320 or 640
#nx,ny,nz = 64,64,320
#nx,ny,nz = 128,128,640
r = 1
length = 10

mesh = [2,2]
# mesh = None
data_dir = "scratch" #change each time or overwrite

kappa = 0.01
mu = 0.05 #Determines Re_k ; 0.05 -> Re_k = 20 (try 0.005?)
eta = 0.001 # Determines Re_m ; 0.001 -> Re_m = 1000
rhoIni = 1 #rho0 is redefined later, and generally has a whole mess associated with it
gamma = 5./3.
eta_sp = 2.7 * 10**(-4)
eta_ch = 4.4 * 10**(-3)
v0_ch = 2.9 * 10**(-2)
chi = kappa/rhoIni
nu = mu/rhoIni

dealias = 1

coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64, mesh = mesh)

xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(-r, r), dealias = dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(-r, r), dealias = dealias)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, length), dealias = dealias)

t = dist.Field(name='t')
v = dist.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
lnrho = dist.Field(name='lnrho', bases=(xbasis, ybasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
phi = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))
tau_A = dist.Field(name='tau_A')
# eta1 = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
ex, ey, ez = coords.unit_vector_fields(dist)

j = -d3.lap(A)
J2 = j@j
rho = np.exp(lnrho)
B = d3.curl(A)
#spitzer and chodra resistivity combination
# eta1 = eta_sp/(np.sqrt(T)**3) + (eta_ch/np.sqrt(rho))*(1 - np.exp((-v0_ch*np.sqrt(J2))/(3*rho*np.sqrt(gamma*T))))
eta1 = 0.001

Va = B/np.sqrt(rho)
Cs = np.sqrt(gamma*T)
Cs_vec = Cs*ex + Cs*ey + Cs *ez


SSX = d3.IVP([v, A, lnrho, T, phi, tau_A], time=t, namespace=locals())

# Continuity
SSX.add_equation("dt(lnrho) + div(v) = - v@grad(lnrho)")

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
solver.stop_sim_time = 20 #historically 20
solver.stop_wall_time = np.inf #e.g. 60*60*3 would limit runtime to three hours
solver.stop_iteration = np.inf

x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
rho0 = np.zeros_like(lnrho['g'])

R = r
L = R

lambda_rho1 = 0.1
rho_min = 0.011
T0 = 0.1
delta = 0.1

# Spheromak initial condition
# The vector potential is subject to some perturbation. This distorts all the magnetic field components in the same direction.
aa = spheromak_pair(xbasis,ybasis,zbasis, coords, dist)

# for i in range(3):
#    A['g'][i] = aa['g'][i] *(1 + delta*x*np.exp(-z**2) + delta*x*np.exp(-(z-10)**2)) # maybe the exponent here is too steep of an IC?
for i in range(3):
    A['g'][i] = aa['g'][i] 

max_vel = 0.1

for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
        yVal = y[0,j,0]
        for k in range(z.shape[2]):
            zVal = z[0,0,k]
            # v version in for loop - i assume outside as written above is preferable, but not sure if that's doable
            # with rho0, since not a dist.field()
            # v['g'][2] = -np.tanh(6*zVal - 6)*max_vel/2 + -np.tanh(6*zVal - 54)*max_vel/2
            # rho0[i][j][k] = -np.tanh(2*zVal-3)*(1-rho_min)/2 -np.tanh(2*(10-zVal)-3)*(1-rho_min)/2 + 1 #density in the z direction with tanh transition
            rho0[i][j][k] = -np.tanh(6*zVal-6)*(1-rho_min)/2 -np.tanh(6*(10-zVal)-6)*(1-rho_min)/2 + 1 # original steeper transition


for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
        yVal = y[0,j,0]
        for k in range(z.shape[2]):
            zVal = z[0,0,k]
            rad = np.sqrt(xVal**2 + yVal**2)

            if(rad <= 1 - lambda_rho1):
                rho0[i][j][k] = rho0[i][j][k]
            elif((rad >= 1 - lambda_rho1 and rad <= 1 + lambda_rho1)): # sine arg goes from pi/2 to -pi/2; so this should just generate a curve from rho0 to rho_min
                rho0[i][j][k] = (rho0[i][j][k] + rho_min)/2 + (rho0[i][j][k] - rho_min)*np.sin((1-rad) * np.pi/(2*lambda_rho1))/2
            else:
                rho0[i][j][k] = rho_min

lnrho['g'] = np.log(rho0)
T['g'] = T0 * rho0**(gamma - 1) # np.exp(lnrho['g'])

wall_dt_checkpoints = 2
output_cadence = 0.1 # This is in simulation time units

if dist.comm.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence = 1)
flow.add_property(np.sqrt(v@v) / nu, name = 'Re_k')
flow.add_property(np.sqrt(v@v) / eta, name = 'Re_m')
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

good_solution = True
# Main loop
try:
    logger.info('Starting loop')
    logger_string = 'kappa: {:.3g}, mu: {:.3g}, dt: {:.3g}'.format(kappa, mu, dt) # eta: {:.3g}, eta1
    logger.info(logger_string)
    while solver.proceed:

        dt = CFL.compute_timestep()
        solver.step(dt)

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