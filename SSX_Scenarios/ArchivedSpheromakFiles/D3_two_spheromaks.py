import dedalus.public as d3
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt

#See "Turbulence analysis of an experimental flux-rope plasma", D A Schaffner et al, 2014.
###########################################################################################
"""
    This scripts contains the two initializations of the spheromaks. spheromak_A is the main 
    and the script that initializes the two spheromaks. 
        There is spheromak_B which appears to be a 1 spheromak initialization.
        getS1 and getS are the two shape functions for the spheromaks.

    These two files include a lot of lines and functions that did not appear to be relevant
    for SSX Simulations in Summer 2023. They are kept here in case these functions turn out
    to be useful in later SSX work.
"""
###########################################################################################
#Shape function
def getS1(r, z, L, R, zCenter):
    #############################################################################################
    """
        This is a script for the shape function described in equation (9) of the above reference.
    """
    #############################################################################################
    lamJ = .1*L  # Lambda J is a width control variable
    S = np.zeros((r*z).shape)

    r1 = np.copy(r)
    z1 = np.copy(z)
    #################################
    # r - conditions
    #################################
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(r.shape[2]):
                entry = r1[i][j][k]
                if(entry < (R - lamJ)):
                    r1[i][j][k] = 1
                elif(entry <= R and entry >= (R - lamJ)):
                    #r1[i][j][k] = .5*(1-np.cos(np.pi*(R/2-entry)/lamJ))
                    r1[i][j][k] = 0.5*(1 - np.cos(np.pi*(R - entry)/lamJ))
                elif(entry > R):
                    r1[i][j][k] = 0
                else:
                    r1[i][j][k] = 0.5
                    print("r out of bounds!", r[i][j][k])

    #################################
    # z - conditions
    #################################
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
           for k in range(z.shape[2]):
               entry = z1[i][j][k]
               val = -np.abs(entry)
               if(val >= -lamJ and val <= 0):
                   z1[i][j][k] = -0.5*(1 - np.cos(np.pi*entry/lamJ))
               elif(val > -(L - lamJ) and val < -lamJ):
                   z1[i][j][k] = -1
               elif(val >= -L and val <= -(L - lamJ)):
                   #z1[i][j][k] = -.5*(1-np.cos(np.pi*(L/2+entry)/lamJ))  See getS().
                   z1[i][j][k] = -0.5*(1 - np.cos(np.pi*(L + entry)/lamJ))
               elif(val < -L):
                   z1[i][j][k] = 0
               else:
                   print("z out of bounds!", entry)
                   z1[i][j][k] = 1

    S = r1*z1
    #basedir = os.path.dirname(os.path.realpath(__file__))
    #dfile = basedir+'/fields_two/fields_two_s1.h5'
    #data = h5py.File(str(dfile), "r")
    #x = data['scales/x/1.0'][:]
    #y = data['scales/y/1.0'][:]
    #z = data['scales/z/1.0'][:]
    #for i in range (180):
    #    plot_2d(x, y, S[:, :, i], i)
    return S

# Leaving alone for now
def plot_2d(x, y, z, i):
    plt.imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap=plt.cm.hot)
    plt.colorbar()
    plt.savefig('2dplots/fig_'+str(i)+'.png')
    plt.close()

def getS(r, z, L, R, zCenter):
    lamJ = .1*L
    S = np.zeros((r*z).shape)

    r1 = np.copy(r) # This is Sr in PSST 2014
    z1 = np.copy(z) # ... in PSST 2014
    """
        When we are moving through cylindrical space.
    """
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(r.shape[2]):

                entry = r1[i][j][k]
                if(entry < (R - lamJ)): # R is 1, entry is little r. (ref. PSST 2014)
                    r1[i][j][k] = 1
                elif(entry <= R and entry >= (R - lamJ)):
                    #r1[i][j][k] = 0.5*(1 - np.cos(np.pi*(R/2 - entry)/lamJ))  # The function looks different then what is written in PSST. 2014, but it has the same result; ranging from R - lamJ to R.
                    r1[i][j][k] = 0.5*(1 - np.cos(np.pi*(R - entry)/lamJ))
                    # This part is inverted. Needs to be corrected.
                elif(entry > R):
                    r1[i][j][k] = 0
                else:
                    r1[i][j][k] = 0.5
                    print("r out of bounds!", r[i][j][k])

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            for k in range(z.shape[2]):

                entry = z1[i][j][k]
                val = np.abs(entry)
                if(val <= lamJ - zCenter and val >= 0 - zCenter):
                    z1[i][j][k] = 0.5 * (1 - np.cos(np.pi * entry/lamJ))
                elif(val < (L-lamJ) - zCenter and val > lamJ - zCenter):
                    z1[i][j][k] = 1
                elif(val <= L - zCenter and val >= (L - lamJ) - zCenter):
                    #z1[i][j][k] = 0.5*(1 - np.cos(np.pi*(L/2 - entry)/lamJ))
                    z1[i][j][k] = 0.5*(1 - np.cos(np.pi*(L - entry)/lamJ))
                elif(val > L - zCenter):
                    z1[i][j][k] = 0
                else:
                    print("z out of bounds!", entry)
                    z1[i][j][k] = 1

    S = r1 * z1
    return S

def spheromak_A(xbasis,ybasis,zbasis, coords, dist, center=(0,0,0), B0 = 1, R = 1, L = 1):
    """
    This function returns the intial 2X-spheromak vector potential components (x, y, z).
    J0 - Current density
    J1 - Bessel of order 1
    
    
    Solve:
    Laplacian(A) = - J0

    J0 = S(r) lam [ -pi J1(a, r) cos(pi z) rhat + lam*J1(a, r)*sin(pi z)     Eq. (9)
    # B0 should always be 1, but we are leaving it as a parameter for safe keeping.
    """
    r = 1
    length = 10
    #mesh = None #[16,16]
    data_dir = "scratch_init"
    # dist = d3.Distributor(coords, dtype=np.float64, mesh = mesh)
    # ex, ey, ez = coords.unit_vector_fields(dist)

    j1_zero1 = jn_zeros(1,1)[0]
    kr = j1_zero1/R
    kz = np.pi/L

    #Force-free configuration: curl(B) = J = lam*B
    lam = np.sqrt(kr**2 + kz**2)
    J0 = B0 # This should be 1.
    #####################################################################
    """ Setting up the problem in dedalus. """
    #####################################################################
    """ Creating fields/variables """
    # Current density components
    #####################################################################

    #if not a problem variable, is it correct to express our (fixed) J as a vectorfield like normal?
    J = dist.VectorField(coords, name='J', bases=(xbasis, ybasis, zbasis))
    xx, yy, zz = dist.local_grids(xbasis, ybasis, zbasis)
    #####################################################################
    """ Setting cylindrical coordinates """
    #####################################################################
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    theta = np.arctan2(yy,xx)
    z = zz - center[2]
    z1 = zz - 10
    #####################################################################
    """ Creating the two shape functions """
    #####################################################################
    S = getS(r, z, L, R, center[2])
    S1 = getS1(r,z1, L, R, 10)
    #####################################################################
    """ Current density; cylindrical components Eq. (9) """
    # Note they are sums of two separate shape functions. S and S1.
    # S - centered at 0
    # S1 - centered at 10 (The other end of the domain)
    #####################################################################
    J_r = S*lam*(-np.pi*j1(kr*r)*np.cos(z*kz)) + S1*lam*(np.pi*j1(kr*r)*np.cos((-z1)*kz))
    J_t = S*lam*(lam*j1(kr*r)*np.sin(z*kz)) - S1*lam*(-lam*j1(kr*r)*np.sin((-z1)*kz))
    #####################################################################
    """ Initializing the J fields for the dedalus problem. """
    # J0 is set to B0, which should be 1.
    #####################################################################
    
    J['g'][0] = J0*(J_r*np.cos(theta) - J_t*np.sin(theta))
    J['g'][1] = J0*(J_r*np.sin(theta) + J_t*np.cos(theta))
    J['g'][2] = J0*S*lam*(kr*j0(kr*r)*np.sin(z*kz)) + J0*S1*lam*(kr*j0(kr*r)*np.sin((-z1)*kz))

    #####################################################################
    """ Initialize the problem """
    #####################################################################
    A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))

    """ Meta Parameters """
    #####################################################################
    #components of the direct product yet, but we generally want:

    A['c'][0, 1::2, 0::2, 0::2] = 0
    A['c'][1,0::2, 1::2, 0::2] = 0
    A['c'][2, 0::2, 0::2, 1::2] = 0

    # J['c'][0][y,z][0::2] = 0
    # J['c'][0][x][1::2] = 0
    # J['c'][1][x,z][0::2] = 0
    # J['c'][1][y][1::2] = 0
    # J['c'][0][x,y][0::2] = 0
    # J['c'][0][z][1::2] = 0

    #Former meta:
    # problem.meta['Ax']['y', 'z']['parity'] =  -1
    # problem.meta['Ax']['x']['parity'] = 1
    # problem.meta['Ay']['x', 'z']['parity'] = -1
    # problem.meta['Ay']['y']['parity'] = 1
    # problem.meta['Az']['x', 'y']['parity'] = -1
    # problem.meta['Az']['z']['parity'] = 1

    # J0_x.meta['y', 'z']['parity'] = -1
    # J0_x.meta['x']['parity'] = 1
    # J0_y.meta['x', 'z']['parity'] = -1
    # J0_y.meta['y']['parity'] = 1
    # J0_z.meta['x', 'y']['parity'] = -1
    # J0_z.meta['z']['parity'] = 1

    # phi = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
    tau_phi = dist.VectorField(coords, name='tau_phi')
    B = d3.curl(A).evaluate()
    
    #h5py documentation - write B field into it

    problem = d3.LBVP([A, tau_phi],namespace=locals())
    #####################################################################
    """ Force Free Equations/Spheromak """
    #####################################################################

    # lap(A) = -J
    # Need to come up with a good way to check if what this gives is correct. Add a task to this to make an h5 file that saves A or B.
    problem.add_equation("lap(A) + tau_phi =  -J") # + grad(phi)
    problem.add_equation("integ(A) = 0") #check that this gives divA = 0
    # problem.add_equation("div(A) = 0")

    #####################################################################
    """ Building the solver """
    # That is, setting things into play.
    #####################################################################
    solver = problem.build_solver()
    solver.solve()

    #analysis
    # Not sure how to add analysis tasks for a non-IVP. Just want to be able to see B field itself.
    # Current method is to just add B to the problem.
    # snapshots.add_task(d3.curl(A), name='Magnetic field')
    
    return A

# spheromak_B has not been updated yet: Will leave as was until we have a use for it
def spheromak_B(domain, center=(0,0,10), B0 = 1, R=1, L=1):
    """ 
        Returns the intial 1X-spheromak vector potential components.
        Solve
        Laplacian(A) = - J0

        J0 = S(r) l_sph [ -pi J1(a r) cos(pi z) rhat + l_sph*J1(a r)*sin(pi z)

        """

    j1_zero1 = jn_zeros(1,1)[0]
    kr = j1_zero1/R
    kz = np.pi/L

    lam = np.sqrt(kr**2 + kz**2)
    J0 = B0

    problem = d3.LBVP(domain, variables=['Ax', 'Ay', 'Az'])
    # problem.meta['Ax']['y', 'z']['parity'] =  -1
    # problem.meta['Ax']['x']['parity'] = 1
    # problem.meta['Ay']['x', 'z']['parity'] = -1
    # problem.meta['Ay']['y']['parity'] = 1
    # problem.meta['Az']['x', 'y']['parity'] = -1
    # problem.meta['Az']['z']['parity'] = 1

    J0_x = domain.new_field()
    J0_y = domain.new_field()
    J0_z = domain.new_field()
    xx, yy, zz = domain.grids()

    r = np.sqrt((xx-center[0])**2 + (yy-center[1])**2)
    theta = np.arctan2(yy,xx)
    z = zz - center[2]

    S = getS1(r,z, L, R, center[2])

    J_r = S*lam*(np.pi*j1(kr*r)*np.cos((-z)*kz))
    J_t = S*lam*(-lam*j1(kr*r)*np.sin((-z)*kz))

    J0_x['g'] = J0*(J_r*np.cos(theta) - J_t*np.sin(theta))
    J0_y['g'] = J0*(J_r*np.sin(theta) + J_t*np.cos(theta))
    J0_z['g'] = J0*S*lam*(kr*j0(kr*r)*np.sin((-z)*kz))

    # J0_x.meta['y', 'z']['parity'] = -1
    # J0_x.meta['x']['parity'] = 1
    # J0_y.meta['x', 'z']['parity'] = -1
    # J0_y.meta['y']['parity'] = 1
    # J0_z.meta['x', 'y']['parity'] = -1
    # J0_z.meta['z']['parity'] = 1

    problem.add_equation("dx(dx(Ax)) + dy(dy(Ax)) + dz(dz(Ax)) =  -J0_x", condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ax = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    problem.add_equation("dx(dx(Ay)) + dy(dy(Ay)) + dz(dz(Ay)) =  -J0_y", condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ay = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    problem.add_equation("dx(dx(Az)) + dy(dy(Az)) + dz(dz(Az)) =  -J0_z", condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Az = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    # Build solver
    solver = problem.build_solver()
    solver.solve()

    return solver.state['Ax']['g'], solver.state['Ay']['g'], solver.state['Az']['g']

# main function which calls getS,GetS1, and spheromak_A
def spheromak_1(xbasis,ybasis,zbasis, coords, dist):
    aa = spheromak_A(xbasis,ybasis,zbasis, coords, dist)
    #aa_x_2, aa_y_2, aa_z_2 = spheromak_B(coords)
    
    return aa

# Also leaving this function alone until we need it
def spheromak(Bx, By, Bz, domain, center = (0, 0, 0), B0 = 1, R = 1, L = 1):
    """
    Incomplete function
    domain must be a dedalus domain
    Bx, By, Bz must be Dedalus fields
        *** center has been changed from (0, 0, 0) to (0, 0, 0.5)
    """

    # parameters
    xx, yy, zz = domain.grids()

    j1_zero1 = jn_zeros(1,1)[0]
    kr = j1_zero1/R
    kz = np.pi/L

    lam = np.sqrt(kr**2 + kz**2)

    # construct cylindrical coordinates centered on center
    r = np.sqrt((xx- center[0])**2 + (yy- center[1])**2)
    theta = np.arctan2(yy,xx)
    z = zz - center[2]


    # calculate cylindrical fields
    Br = -B0 * kz/kr * j1(kr*r) * np.cos(kz*z)
    Bt = B0 * lam/kr * j1(kr*r) * np.sin(kz*z)

    # convert back to cartesian, place on grid.
    Bx['g'] = Br*np.cos(theta) - Bt*np.sin(theta)
    By['g'] = Br*np.sin(theta) + Bt*np.cos(theta)
    Bz['g'] = B0 * j0(kr*r) * np.sin(kz*z)

# spheromak_1()