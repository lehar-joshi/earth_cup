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
tspan=3600*24*365.0 # 1 days
dt=5000

# central body
cb = pd.earth

date0='2027-01-01'

h=30.0e-3 # km
w=35e-3 # km
A=4.3*1e-6 # km^2

if __name__ == '__main__':

    # initial conditions
    state0=[500.0+cb['radius'],0.0,0.0,0.0,7.61,0.0]

    # null perturbations dictionary
    perts=null_perts()

    # add solar pressure radiation
    perts['srp']=1 #Set to 1 for Eq. (4), set to 2 for Eq. (5) in the report
    perts['A_srp']=A
    perts['CR']=1.0

    # add aerodynamic drag
    perts['aero']=True
    perts['Cd']=2.2
    perts['A']=A/2 # km^2

    mass0=0.250 # kg

    # create orbit propagator instance
    op0=OP(state0,tspan,1000,coes=False,deg=True,perts=perts,date0=date0,propagator='dopri5',mass0=mass0)
    op0.plot_3d(show_plot=True,title='Solar Sail Orbit')
    op0.calculate_coes()
    op0.plot_coes(days=True,show_plot=True,rel=False,title='Solar Sail')
    op0.plot_alts(show_plot=True,hours=True)
    op0.calculate_apoapse_periapse()
    op0.plot_apoapse_periapse(show_plot=True, hours=True)
