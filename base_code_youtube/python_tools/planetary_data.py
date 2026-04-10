G_meters=6.67408e-11 # m**3/kg/s**2
G=G_meters*10**-9 # km**3/kg/s**2
from numpy import array,pi

# days to seconds
day2sec=24*3600.0

sun={
    'name':'Sun',
    'mass':1.989e30,
    'mu': 1.981e30*G,
    'radius':695510.0,
    'G1':10.0**8, # kg*km**3/s**2/m**2,
    'spice_file': 'spice_data/de432s.bsp',
    'deorbit_altitude':2*695510.0
}

atm=array([[63.096,2.059e-4],[251.189,5.909e-11],[1000.0,3.561e-15]])
earth={
    'name':'Earth',
    'mass':5.972e24,
    'mu': 5.972e24*G,
    'radius':6378.0,
    'J2':1.082635854e-3,
    'zs':atm[:,0], # km
    'rhos':atm[:,1]*10**8, # kg / km*3,
    'atm_rot_vector':array([0.0,0.0,72.9211e-6]), # rad/s,
    'deorbit_altitude':10.0, #km
    'spice_file': 'spice_data/de432s.bsp'
}

moon={
    'name':'Moon',
    'mass':7.34767309e22,
    'mu': 7.34767309e22*G,
    'radius':1737.1,
    'orbit_T':29*day2sec+12*3600.0+44*60.0+2.8,
    'dist2earth':384400.0,
    'spice_file': 'spice_data/de432s.bsp',
}
moon['orbit_w']=2*pi/moon['orbit_T']