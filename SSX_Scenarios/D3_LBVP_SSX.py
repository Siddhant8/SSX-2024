import dedalus.public as d3
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt
from mpi4py import MPI

#See "Turbulence analysis of an experimental flux-rope plasma", D A Schaffner et al, 2014.

# File formerly called "D[2/3]_two_spheromaks"
# - when looking for older versions, check both current name and that name.

# Dedalus 3 edits made by Alex Skeldon.
# Only spheromak_pair is used at this point. The others are here only for reference as of now.
###########################################################################################
"""
    This scripts contains the initialization of the two spheromaks.
    spheromak_pair is the main function, which initializes the two spheromaks. 
"""
###########################################################################################

# Only function being used in this file at the moment.
def spheromak_pair(xbasis,ybasis,zbasis, coords, dist, center=(0,0,0), B0 = 1, R = 1, L = 1, comm=None):
    """
    This function returns the intial 2X-spheromak vector potential components (x, y, z).
    J0 - Current density
    J1 - Bessel of order 1
    
    Solve:
    Laplacian(A) = - J0

    J0 = S(r) lam [ -pi J1(a, r) cos(pi z) rhat + lam*J1(a, r)*sin(pi z)     Eq. (9)
    # B0 should always be 1, but we are leaving it as a parameter for safe keeping.
    """

    j1_zero1 = jn_zeros(1,1)[0]
    kr = j1_zero1/R
    kz = np.pi/L
    lam = np.sqrt(kr**2 + kz**2)
    J0 = B0 # This should be 1.
    #define handedness of Taylor states. Difference between LH and RH should just be sign on the theta component
    hand1 = 1
    hand2 = 1

    # if not a problem variable, is it correct to express our (fixed) J as a vectorfield like normal?
    J = dist.VectorField(coords, name='J', bases=(xbasis, ybasis, zbasis))
    x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

    # Setting cylindrical coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)

    # removed shape function calling approach; doing it in-line instead

    # Sr = 0.5*np.tanh(1*(0.8-r))+0.5 # TODO: Check/plot these, smooth them, lower coeffs
    # Sz1 = (np.tanh(1*(z-1.5))+np.tanh(1*(1.5-z)))/2
    # Sz2 = (np.tanh(1*((10-z)-1.5))+np.tanh(1*(1.5-(10-z))))/2

    # Sr*Sz1* localization multiplier to be toggled
    J_r1 = lam*(-np.pi*j1(kr*r)*np.cos(kz*z))
    J_t1 = hand1*lam*(lam*j1(kr*r)*np.sin(kz*z)) 
    J_z1 = lam*(kr*j0(kr*r)*np.sin(kz*z))

    #For our second spheromak's current densities, we rotate the original and translate to z=10
    #So z-dependence becomes (10-z), and the theta and z components have negatives.
    # Sr*Sz2 localization multiplier can be added or removed
    J_r2 = lam*(-np.pi*j1(kr*r)*np.cos(kz*(10-z)))
    J_t2 = -hand2*lam*(lam*j1(kr*r)*np.sin(kz*(10-z)))
    J_z2 = -lam*(kr*j0(kr*r)*np.sin(kz*(10-z)))

    J_r = J_r1+J_r2
    J_t = J_t1+J_t2
    J_z = J_z1+J_z2

    """ Initializing the J fields for the dedalus problem. """
    # J0 is set to B0, which should be 1.
    
    #Convert to Cartesian coordinates
    J['g'][0] = J0*(J_r*np.cos(theta) - J_t*np.sin(theta))
    J['g'][1] = J0*(J_r*np.sin(theta) + J_t*np.cos(theta))
    J['g'][2] = J0*J_z

    A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))

    #Meta/Parity specifiers
    # e.g. A_i is even in i-basis, odd in other two (see func further below)
    # zero_modes(A,0)
    # zero_modes(J,0)

    # Decomment these six for parity enforcement in triple RealFourier

    # phi field not necessary if integ(A) is correct gauge
    # phi = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
    tau_phi = dist.VectorField(coords, name='tau_phi')

    problem = d3.LBVP([A, tau_phi],namespace=locals())

    # Force Free Equations/Spheromak """

    # lap(A) = -J
    # Need to come up with a good way to check if what this gives is correct.
    problem.add_equation("lap(A) + tau_phi =  -J") # + grad(phi) term for Div(A) case # + tau_phi
    problem.add_equation("integ(A) = 0")
    # problem.add_equation("div(A) = 0") # haven't been able to implement coulomb gauge instead of integ gauge here yet.
    # Need to figure out why it isn't working, since it doesn't seem that integ(A) is an equivalent gauge.

    #Building the solver """
    solver = problem.build_solver()
    solver.solve()
    
    return A

# Not currently used due to manually written out parity enforcement. This function can be ignored.
def zero_modes(initfield, par, scalar=False):
#enforce meta parity parameters on fields - 0 is even/cosine, 1 is odd/sine
# modifies the field passed in to have zeroed odd or even modes acc. to second and third args.
# Works for our 2:1 parity cycles and scalar fields- not general for any combination.
# Does this return the object that's been modified? Or does it return the original? This is an aspect of OOP I'm not sure of.
    if scalar == True:
        initfield['c'][1-par::2,1-par::2,1-par::2] = 0
    else:
        initfield['c'][0, 1-par::2, par::2, par::2] = 0
        initfield['c'][1, par::2, 1-par::2, par::2] = 0
        initfield['c'][2, par::2, par::2, 1-par::2] = 0

    return initfield

# Now that our shapes are made smoothly in-line in spheromak_pair, the two piecewise-generating functions can be ignored.
# And I haven't used plot_2d at all yet, either.
def getS1(r, z, L, R, zCenter):
    # Shape function for spheromak at z = 0
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
                    # r1[i][j][k] = 0.5
                    r1[i][j][k] = 0
                    # I have never seen any of these out of bounds
                    # print statements come up, so I think it's safe
                    # to assume this and similar conditions
                    # Do not occur when we're doing the LBVP
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

def getS(r, z, L, R, zCenter):
    # Shape function for the second spheromak at z = 10
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
                    # What makes this inverted? It's written the same as in PSST 2014 paper. If anything, it would be GetS1 that should be "inverted" if we expect the opposite end to have rotational symmetry with original spheromak.
                elif(entry > R):
                    r1[i][j][k] = 0
                else:
                    # r1[i][j][k] = 0.5
                    r1[i][j][k] = 0 # should have 0 everywhere else due to nature of sinusoidial transition, no?
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
                    #Shouldn't necessarily matter since I haven't seen this
                    # Conditional be reached at all, but it seems to me it ought
                    # to be 0 instead of 1 in such a case?
                    print("z out of bounds!", entry)
                    #z1[i][j][k] = 1
                    z1[i][j][k] = 0

    S = r1 * z1
    return S

def plot_2d(x, y, z, i):
    plt.imshow(z, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap=plt.cm.hot)
    plt.colorbar()
    plt.savefig('2dplots/fig_'+str(i)+'.png')
    plt.close()
