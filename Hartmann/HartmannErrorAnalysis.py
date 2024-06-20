import sys
import numpy as np
import pathlib
import h5py
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# import newHmann as hmann
#import csv
import os
#from mpi4py import MPI

# basedir = os.path.dirname(os.path.realpath(__file__))


def chi_squared(y_data, y_fit, y_unc=None):
    '''Given two input arrays of equal length, calculate and
    return the chi_squared value.  Use uncertainties if they
    are given.
    
    INPUTS:
    Y_DATA is an array object representing the actual data points
    Y_FIT is an array object representing the fit values at the same x-locations as the data points
    
    OUTPUTS:
    CHI_S is the chi squared statistic, weighted by uncertainties if they are provided.'''
    
    if y_unc is not None:
        chi_s = np.sum( (y_data-y_fit)**2/y_unc**2 )
    else:
        chi_s = np.sum( (y_data-y_fit)**2)
        
    return chi_s

def hartmann_z(z, Ha, u):
    return (u*(1-(np.cosh(Ha*z)/np.cosh(Ha))))

def hartmann_b_z(z, Ha):
    return (-(z/Ha) + (1/400*(np.sinh(Ha*z)/np.cosh(Ha))))
    
def hartmann_y(y, Ha, u):
   return (u*(1- (np.cosh(Ha*y)/np.cosh(Ha))))	

def poiseuille_flow_fit(y, u):
	return (u*(1- y**2))	
	
# vxxx = hartmann(z, 1, 1)
# plt.semilogy(z, vxxx, 'r.')
# plt.xlabel("z")
# plt.ylabel(r"$<v_x>$")
# plt.savefig('vx_zprof.png')

def plot_fit_z(func, x_array, y_array, i):
    popt, pcov = scipy.optimize.curve_fit(func, x_array, y_array, p0=[20,0.0025])
    #print(popt[0])
    #print(popt[1])
    plt.semilogy(x_array, y_array, 'g.', label = "simulation data")
    #plt.semilogy(x_array, func(x_array, popt[0] , popt[1]), 'r-')
    plt.semilogy(x_array, func(x_array, *popt), 'r-', label = "analytical solution")
    #plt.title("Hartmann profile for conducting walls")
    plt.legend(loc='lower center')
    plt.title(" v_x Velocity Profile")
    plt.xlabel("z")
    plt.ylabel(r"$v_x$")
    chi_square = (chi_squared(y_array, func(x_array, *popt)))
    R_squared = (r2_score(y_array, func(x_array, *popt)))
    Root_mean_squared_error = (np.sqrt(mean_squared_error(y_array, func(x_array, *popt))))
    plt.savefig('vx_curvefit_z'+str(i)+'.png',dpi=2000)
    plt.clf()
    return popt[0], popt[1], chi_square, R_squared, Root_mean_squared_error

def plot_fit_b_z(func, x_array, y_array, i):
    popt, pcov = scipy.optimize.curve_fit(func, x_array, y_array*i, p0 = [20])
    #print(popt[0])
    #print(popt[1])
    plt.plot(x_array, y_array, 'g.', label = "simulation data")
    #plt.plot(x_array, func(x_array, popt[0] , popt[1]), 'r-')
    plt.plot(x_array, func(x_array, *popt), 'r-', label = "analytical solution")
    #plt.title("Induced magnetic field for Hartmann flow")
    plt.legend(loc='lower center')
    plt.title(" B_x Induced Magnetic Field Profile")
    plt.xlabel("z")
    plt.ylabel(r"$B_x$")
    chi_square = (chi_squared(y_array, func(x_array, *popt)))
    R_squared = (r2_score(y_array, func(x_array,*popt)))
    Root_mean_squared_error = (np.sqrt(mean_squared_error(y_array, func(x_array, *popt))))
    plt.savefig('bxx_curvefit_z'+str(i)+'b.png',dpi=2000)
    plt.clf()
    return popt[0], chi_square, R_squared, Root_mean_squared_error
   
    
def hartmann_errors():
        data = h5py.File("./Full3DHartmannRunData/integrals/integrals_s1.h5", "r")
        # y = data['scales/y_hash_f08571fe67493aa5454ad70f13fdb247e5cbe449/'][:]
        z = (data['scales/z_hash_24c956bcc02c8dc6378aaf5ce148dd993d0851b1/'][:]) #changed based on what your h5 file's z label is
        # there's supposedly a better way to handle spatial dimensions (i.e. you don't
        # have to write the hash label) based off mailing list discussion
        # But I don't yet know what it is
        # x   = data['tasks/<vx>_xy'][:]
        bz = data['tasks/<Bx>_xy'][-1,0,0,:]*20
        vxz = data['tasks/<vx>_xy'][-1,0,-1,:]
        ha, u, chi_square, R_squared, Root_mean_squared_error = plot_fit_z(hartmann_z, z, vxz, 1)
        print("ha, u, chi_square, r_square, RMSE are (for flow profile):")
        print(ha)
        print(u)
        print(chi_square)
        print(R_squared)
        print(Root_mean_squared_error)
    
    
        ha, chi_square, R_squared, Root_mean_squared_error = plot_fit_b_z(hartmann_b_z, z, bz, 1)
        print("ha, u, chi_square, r_square, RMSE are (for magnetic field profile): ")
        print(ha)
        print(u)
        print(chi_square)
        print(R_squared)
        print(Root_mean_squared_error)
       
hartmann_errors()
