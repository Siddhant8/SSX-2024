import numpy as np
import h5py as h5
import matplotlib.pylab as plt

# https://tri-coplasmagroup.slack.com/archives/CA69SPRV1/p1623774549005100
# A file for h5 data accessing through python
# Posted by Carlos

f = h5.File('fields_two_s1.h5','r')
scales = f['scales'] # Contains the scaling of the simulation
tasks = f['tasks'] # Contains the data
"""for keys in tasks:
    print(tasks[keys])"""
#x = scales['x/1.0']

total_Density = np.zeros(500)
for i in range(0, 500):
  
    total_Density[i] = np.sum(tasks['rho'][i,:,:,:])


# Rate Calculation
density_Rate = np.zeros(498)
rate_Time = np.zeros(498)
sim_Time_Step = scales['sim_time'][1] - scales['sim_time'][0]
for i in range(0, 498):
    density_Rate[i] = (total_Density[i + 1] - total_Density[i])
    rate_Time[i] = (scales['sim_time'][i + 1] + scales['sim_time'][i])/2
    

print(rate_Time)
plt.figure()
plt.title('Change in Density vs time')
plt.ylabel('Delta Density Rate')
plt.xlabel('time')
plt.plot(rate_Time, density_Rate)
plt.show()
plt.close()
