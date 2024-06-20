import dedalus.public as d3
import numpy as np
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt

#See "Turbulence analysis of an experimental flux-rope plasma", D A Schaffner et al, 2014.

# File formerly called "D3_two_spheromaks"
# - when looking for older versions, check both current name and that name.

# Dedalus 3 edits made by Alex Skeldon.
###########################################################################################
"""
    This scripts contains the two initializations of the spheromaks.
    spheromak_pair is the main function, which initializes the two spheromaks. 
    getS1 and getS are the two shape functions for the spheromaks.
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

# Leaving alone for now - isn't invoked anywhere at the moment.
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

# main function which calls getS and GetS1
def spheromak_pair(xbasis,ybasis,zbasis, coords, dist, center=(0,0,0), B0 = 1, R = 1, L = 1):
    """
    This function returns the intial 2X-spheromak vector potential components (x, y, z).
    J0 - Current density
    J1 - Bessel of order 1
    
    Solve:
    Laplacian(A) = - J0

    J0 = S(r) lam [ -pi J1(a, r) cos(pi z) rhat + lam*J1(a, r)*sin(pi z)     Eq. (9)
    # B0 should always be 1, but we are leaving it as a parameter for safe keeping.
    """

    # data_dir = "scratch_init"

    j1_zero1 = jn_zeros(1,1)[0]
    kr = j1_zero1/R
    kz = np.pi/L

    #Force-free configuration: curl(B) = J = lam*B
    lam = np.sqrt(kr**2 + kz**2)
    J0 = B0 # This should be 1.


    # if not a problem variable, is it correct to express our (fixed) J as a vectorfield like normal?
    J = dist.VectorField(coords, name='J', bases=(xbasis, ybasis, zbasis))
    xx, yy, zz = dist.local_grids(xbasis, ybasis, zbasis)

    # Setting cylindrical coordinates
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    theta = np.arctan2(yy,xx)
    z = zz - center[2]
    z1 = zz - 10

    # Creating the two shape functions
    S = getS(r, z, L, R, center[2])
    S1 = getS1(r,z1, L, R, 10)

    """ Current density; cylindrical components Eq. (9) """
    # Note they are sums of two separate shape functions. S and S1.
    # S - centered at 0
    # S1 - centered at 10 (The other end of the domain)
    J_r = S*lam*(-np.pi*j1(kr*r)*np.cos(z*kz)) + S1*lam*(np.pi*j1(kr*r)*np.cos((-z1)*kz))
    J_t = S*lam*(lam*j1(kr*r)*np.sin(z*kz)) - S1*lam*(-lam*j1(kr*r)*np.sin((-z1)*kz))

    """ Initializing the J fields for the dedalus problem. """
    # J0 is set to B0, which should be 1.
    
    J['g'][0] = J0*(J_r*np.cos(theta) - J_t*np.sin(theta))
    J['g'][1] = J0*(J_r*np.sin(theta) + J_t*np.cos(theta))
    J['g'][2] = J0*S*lam*(kr*j0(kr*r)*np.sin(z*kz)) + J0*S1*lam*(kr*j0(kr*r)*np.sin((-z1)*kz))


    A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))

    #Meta/Parity specifiers
    # e.g. A_i is even in i-basis, odd in other two (see func further below)
    # zero_modes(A,0)
    # zero_modes(J,0)

    A['c'][0,1::2,0::2,0::2] = 0
    A['c'][1,0::2,1::2,0::2] = 0
    A['c'][2,0::2,0::2,1::2] = 0

    J['c'][0,1::2,0::2,0::2] = 0
    J['c'][1,0::2,1::2,0::2] = 0
    J['c'][2,0::2,0::2,1::2] = 0

    # phi field not necessary if integ(A) is correct gauge
    # phi = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
    tau_phi = dist.VectorField(coords, name='tau_phi')
    # B = d3.curl(A).evaluate()

    #h5py documentation - write B field into it

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

    #analysis
    # Not sure how to add analysis tasks for a non-IVP. Just want to be able to see B field itself.
    
    return A

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

# There used to be an incomplete "spheromak" function here. Refer back to the stored D2
# Version at some point and see what the point of that func was/if it's still relevant.