"""3D Hartmann flow d3 example
Uses vector potential form of MHD in Coulomb gauge; incompressible Navier-Stokes, no further approximations.
Note that because of the difference between curl in 2D and 3D, this is written in full 3D.

Mainly from J. Oishi, with some edits by Alex Skeldon.
"""
import os
import dedalus.public as d3
from dedalus.extras import flow_tools
import numpy as np

import logging
logger = logging.getLogger(__name__)

Lx = 10.
Ly = 2.
Lz = 1.

nx = 64
ny = 64 #2; or 64 for full 3D
nz = 64

Ha = 20. #20, 10, 1, or 0 (Pouiselle flow)
Re = 1.
Rm = 1.
Pi = 1.
tau = 0.1

stop_time = 10 #5 or 10
data_dir = "scratch" #change each time or overwrite

dealias = 3/2
mesh = [2,2] #[8,8] or [2,2], generally; None to keep non-parallelized

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.float64, mesh = mesh)
ex, ey, ez = coords.unit_vector_fields(dist)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(-Lx, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(-Ly, Ly), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=nz, bounds=(-Lz, Lz), dealias=dealias)

t = dist.Field()
v = dist.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
phi = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))

tau_p = dist.Field(name='tau_p')
tau_phi = dist.Field(name='tau_phi')
tau_v1 = dist.VectorField(coords, name='tau_v1', bases=(xbasis, ybasis))
tau_v2 = dist.VectorField(coords, name='tau_v2', bases=(xbasis, ybasis))
tau_A1 = dist.VectorField(coords, name='tau_A1', bases=(xbasis, ybasis))
tau_A2 = dist.VectorField(coords, name='tau_A2', bases=(xbasis, ybasis))

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_v = d3.grad(v) + ez*lift(tau_v1) # First-order reduction
grad_A = d3.grad(A) + ez*lift(tau_A1) # First-order reduction

# MHD substitutions
J = -d3.lap(A) # Coulomb Gauge + double curl identity
B = d3.curl(A) + ez

# 22 vars, 22 eqns; 12 BCs, 14 taus with Coulomb and incomp.
hartmann = d3.IVP([v, p, tau_p, tau_v1, tau_v2, A, phi, tau_phi, tau_A1, tau_A2], time=t, namespace=locals())

# Pressure gradient ramp in time
Pi_ramp = ex*Pi*(np.exp(-t/tau)-1)

# Navier Stokes
hartmann.add_equation("dt(v) - div(grad_v)/Re + grad(p)  + lift(tau_v2) = -v@grad(v) + Ha**2/(Re*Rm) * cross(J,B) - Pi_ramp")

# div(v) = 0
hartmann.add_equation("trace(grad_v) + tau_p = 0")

# pressure gauge
hartmann.add_equation("integ(p) = 0")

# A
#hartmann.add_equation("dt(Az) - Lap(Az, Az_y)/Rm = vx*By - vy*Bx")
hartmann.add_equation("dt(A) - div(grad_A)/Rm + grad(phi) + lift(tau_A2) = cross(v,B)")

# div(A) = 0
hartmann.add_equation("trace(grad_A) + tau_phi = 0")

# Coulomb gauge
hartmann.add_equation("integ(phi) = 0")

# boundary conditions
hartmann.add_equation("v(z='left') = 0") # no-slip
hartmann.add_equation("ex@A(z='left') = 0")
hartmann.add_equation("ey@A(z='left') = 0")
hartmann.add_equation("phi(z='left') = 0")

hartmann.add_equation("v(z='right') = 0") # no-slip
hartmann.add_equation("ex@A(z='right') = 0")
hartmann.add_equation("ey@A(z='right') = 0")
hartmann.add_equation("phi(z='right') = 0")


# build solver
solver = hartmann.build_solver(d3.RK222)
logger.info("Solver built")

# Integration parameters
solver.stop_sim_time = stop_time
solver.stop_wall_time = 5*24*60.*60
solver.stop_iteration = np.inf
dt = 1e-3

# Initial conditions are zero by default in all fields

# Makes dir if doesn't exist
# interesting note: the first if statement handles the checking issue
# when parallelized across threads.
if dist.comm.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

# Analysis
check = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), wall_dt=3540, max_writes=50)
check.add_tasks(solver.state)

snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1e-1, max_writes=200)
snap.add_task(B(y=0), scales=1)
snap.add_task(A(y=0), scales=1)
snap.add_task(v(y=0), scales=1)

integ = solver.evaluator.add_file_handler(os.path.join(data_dir,'integrals'), sim_dt=0.1)
integ.add_task(d3.Average(v@ex, ('x', 'y')), name='<vx>_xy', scales=1)
integ.add_task(d3.Average(v@ez, ('x', 'y')), name='<vz>_xy', scales=1)
integ.add_task(d3.Average(B@ex, ('x', 'y')), name='<Bx>_xy', scales=1)
integ.add_task(d3.Average(B@ez, ('x', 'y')), name='<Bz>_xy', scales=1)

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), sim_dt=1e-2)
timeseries.add_task(0.5*d3.Integrate(v@v),name='Ekin')
timeseries.add_task(0.5*d3.Integrate(B@B),name='Emag')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property(0.5*v@v, name='Ekin')

try:
    logger.info('Starting loop')
    while solver.proceed:
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max E_kin = %17.12e' %flow.max('Ekin'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
