###NOTE: When running this program, make sure in the terminal that you're running it from the folder containing main.py!

# import python built in libraries
from sys import path
path.append('C:\\Users\\amer8\\OneDrive\\Dokument\\R1\\Code\\python_tools') ###Change this to your own path!

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
tspan=3600*24*30.0 # 1 month in seconds
dt=10000 #seconds

# central body
cb = pd.earth

# Date of launch
date0='2027-03-01'

#Total area of spacecraft
s=0.5e-3 # km
A=3*np.sqrt(3)/2*s**2 # km^2

#Distance between each sail center
l=np.sqrt(3)*s # km

if __name__ == '__main__':

    # initial conditions
    state0=[500.0+cb['radius'],0.0,0.0,0.0,7.61,0.0]

    #initial orientation of the area
    n=t.normed([1,0,0])

    #Initial orientation of the side sail relative to center sail
    r_wing=t.normed([0,0,1])*l

    # null perturbations dictionary
    perts=null_perts()

    # add solar pressure radiation
    perts['srp']=True
    perts['A_srp']=A
    perts['CR']=0.9

    # add aerodynamic drag
    perts['aero']=True
    perts['Cd']=2.2
    perts['A']=A # km^2

    mass0=0.250 # kg

    # create orbit propagator instance
    op0=OP(state0, n, tspan,dt,coes=False,deg=True,perts=perts,date0=date0,propagator='dopri5',mass0=mass0, rot=False, r_wing=r_wing, l=l)
    op0.plot_3d(show_plot=True,title='Solar Sail Orbit')#, save_plot=True)
    op0.calculate_coes()
    op0.plot_coes(days=True,show_plot=True,rel=False,title='Solar Sail')
    op0.plot_alts(show_plot=True,hours=True)#, save_plot=True)
    op0.calculate_apoapse_periapse()
    op0.plot_apoapse_periapse(show_plot=True, hours=True)#, save_plot=True)
    #op0.plot_qs(show_plot=True, hours=True)