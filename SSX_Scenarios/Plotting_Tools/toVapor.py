"""
Converts 3D, time varying HDF5 data from Dedalus simulations to VAPOR vdf format

Usage:
    toVapor.py <fileIn> <fileOutName> [<dimensionRatio>]

If no netCDF file is created, you may have to call "source VAPORPATH" yourself. See below.
"""

import h5py
import os
import sys
import numpy as np
import netCDF4 as n4
from docopt import docopt
import matplotlib.pyplot as plt

'''Clearly, this only works on MacOS. To activate VAPOR environment on windows, see
https://www.vapor.ucar.edu/docs/vapor-installation/vapor-windows-binary-installation.
Moreover, the Dedalus path is specific to this Swarthmore Physics department laptop
"Manjit's MacBook Pro". Change these variables as needed.'''

#PYTHONPATH = '/Users/dedalus/dedalus/bin/activate'
#VAPORPATH = '/Applications/VAPOR/VAPOR.app/Contents/MacOS/vapor-setup.sh'
#os.system("source %s" %VAPORPATH)

args = docopt(__doc__)
try:
    hf = h5py.File(args['<fileIn>'], 'r')
except IOError:
    sys.exit("Your input file could not be accessed")

try:
    tasks = list(hf['tasks'])
    shape = hf['tasks/%s' % tasks[0]].shape
except:
    sys.exit("There was an error reading the h5 data")

#get dimension sizes
nx = shape[1]
ny = shape[2]
nz = shape[3]
nt = shape[0]

#create dimension data
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
z = np.linspace(0,1,nz)
t = np.arange(float(nt))

#create new ncdf4 file
dataset = n4.Dataset(args['<fileOutName>']+'.nc', 'w', format='NETCDF4')

#create dimensions
xset = dataset.createDimension('x', nx)
yset = dataset.createDimension('y', ny)
zset = dataset.createDimension('z', nz)
time = dataset.createDimension('t', None)

#create dimensional variables
xs = dataset.createVariable('x', np.float64, ('x',))
ys = dataset.createVariable('y', np.float64, ('y',))
zs = dataset.createVariable('z', np.float64, ('z',))
ts = dataset.createVariable('t', np.float64, ('t',))
print("Time entries: ", nt)
"""
try:
    dataList = []
    for i in range(len(tasks)):
        dataList.append(hf['tasks/%s' % tasks[i]][()]))
except:
    sys.exit("There was an error putting your data into arrays")
"""
#create variables that vary in x,y,z, and t
varList = []
for i in range(len(tasks)):
    varList.append(dataset.createVariable(tasks[i], np.float64, ('t','x','y','z')))

#assign dimensional data
xs[:] = x
ys[:] = y
zs[:] = z
ts[:] = t

#assign all other variables
try:
    for i in range(len(varList)):
        varList[i][:] = hf[f'tasks/%s'% tasks[i]][()]
except:
    sys.exit("There was an error writing the data arrays into netCDF variables")

dataset.close()
print('netCDF file written')
