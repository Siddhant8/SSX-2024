import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly
import plotly.io as pio
pio.renderers.default="png" # disables browser

if(len(sys.argv) != 2):
    print ('usage: plotkh3.py <timestep>')
    sys.exit(0)
else:
    it = int(sys.argv[1])
file = 'fields_two'
# with h5py.File("2Sph_Run4_TandRhobad/scratch/fields_two/fields_two_s1.h5", mode='r') as file:   
with h5py.File('DataFolders/NoParity(Run14)/'+file+'/'+file+'_s1.h5', mode='r') as file:
    S = file['tasks']['rho']
    #t = S.dims[0]['sim_time']
    #t = file['scales']['sim_time']

    #these four are the variables formatted normally
    t = S.dims[0]
    x = S.dims[1][0]
    y = S.dims[2][0]
    z = S.dims[3][0]

    nx = 64
    ny = 64
    nz = 320

    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

    fig = go.Figure(data=go.Isosurface(colorbar=dict(title='Temp'),
                                       x=X.flatten(),
                                       y=Y.flatten(),
                                       z=Z.flatten(),
                                       value=S[it,:,:,:].flatten(),
                                       isomin=0.01,
                                       isomax=0.99,
                                       colorscale='jet',
                                       surface_count=5,
                                       opacity=0.3,
                                       #showscale=False,
                                       caps=dict(x_show=False,y_show=False,z_show=False)))
    
    camera = dict(
        eye=dict(x=-2.0,y=-1.5,z=1.25)
        )

    fig.update_layout(scene_camera=camera,title='isosurfaces',
                      margin=dict(t=0,l=0,b=0),
                      scene=dict(xaxis=dict(title='x'),
                                 yaxis=dict(title='y'),
                                zaxis=dict(title='z (lengthwise)')))
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1,y=1,z=1))
    
    fname = "kh3iso.png"
    print("where we at")
    #fig.show()
    fig.write_image(fname)
