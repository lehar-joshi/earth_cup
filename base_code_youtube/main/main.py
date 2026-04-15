# import python built in libraries
from sys import path
path.append('C:\\Users\\amer8\\OneDrive\\Dokument\\R1\\Code\\python_tools') ###CHANGE THIS TO YOUR OWN PATH

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from planetary_coverage import MetaKernel
plt.style.use('dark_background')

# import personal tools
from OrbitPropagator import OrbitPropagator as OP
from OrbitPropagator import null_perts
import planetary_data as pd
import tools as t
import spice_tools as st

# time parameters
tspan=3600*24*365.0 # 1 days
dt=5000

# central body
cb = pd.earth

date0='2020-03-08'

h=30.0e-3 # km
w=35e-3 # km
A=h*w # km^2

if __name__ == '__main__':

    # initial conditions
    state1=[42095.0,0.81818,28.5,180.0,298.2253,357.857]
    state0=t.tle2coes('geo.txt')

    # null perturbations dictionary
    perts=null_perts()

    # add lunar gravity perturbation
    perts['srp']=True
    perts['A_srp']=A
    perts['CR']=1.0
    mass0=7000.0 # kg

    # create orbit propagator instance
    op0=OP(state0,tspan,1000,coes=True,deg=True,perts=perts,date0=date0,propagator='dopri5',mass0=mass0)
    op0.plot_3d(save_plot=True,title='Geostationary Orbit')
    op0.calculate_coes()
    op0.plot_coes(days=True,save_plot=True,rel=True,title='Geostationary')

    op1=OP(state1,tspan,1000,coes=True,deg=True,perts=perts,date0=date0,propagator='dopri5',mass0=mass0)
    op1.plot_3d(save_plot=True,title='Molniya Orbit')
    op1.calculate_coes()
    op1.plot_coes(days=True,save_plot=True,rel=True,title='Molniya')
