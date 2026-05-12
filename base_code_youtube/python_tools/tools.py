import math as m
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

import planetary_data as pd

d2r=np.pi/180.0
r2d=180.0/np.pi
km2AU=1.4959787e8 # km to AU

def norm(v):
    return np.linalg.norm(v)

def normed(v):
    return np.array(v)/norm(v)

def oproj(u, n):
    n_norm = norm(n)
    if n_norm == 0:
        return u  # Avoid division by zero if n is the zero vector
    proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
    return u - proj_of_u_on_n

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

def plot_n_orbits(rs, labels, cb=pd.earth, show_plot=False, save_plot=False, title='Many orbits', AU=False, ER=False, figsize=(16,8)):
        
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(111,projection='3d')

        # plot trajectory
        n=0
        max_val=0
        for r in rs:
            if AU:
                r/=km2AU
            elif ER:
                r/=pd.earth['radius']
            ax.plot(r[:,0], r[:,1], r[:,2], label=labels[n])
            ax.plot([r[0,0]], [r[0,1]], [r[0,2]], 'wo')


            max__=np.max(r)
            if max__>max_val:
                max_val=max__
            n+=1

        # radius central body
        r_plot=cb['radius']
        if AU:
            r_plot/=km2AU
        elif ER:
            r_plot/=pd.earth['radius']



        # plot central body
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        _x=r_plot*np.cos(_u)*np.sin(_v)
        _y=r_plot*np.sin(_u)*np.sin(_v)
        _z=r_plot*np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='Blues')

        # plot the x,y,z vectors
        l=r_plot*2
        x,y,z=[[0,0,0], [0,0,0], [0,0,0]]
        u,v,w=[[l,0,0], [0,l,0], [0,0,l]]
        ax.quiver(x, y, z, u, v, w, color='k')

        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])

        if AU:
            ax.set_xlabel(['X (AU)'])
            ax.set_ylabel(['Y (AU)'])
            ax.set_zlabel(['Z (AU)'])
        elif ER:
            ax.set_xlabel(['X (ER)'])
            ax.set_ylabel(['Y (ER)'])
            ax.set_zlabel(['Z (ER)'])
        else:
            ax.set_xlabel(['X (km)'])
            ax.set_ylabel(['Y (km)'])
            ax.set_zlabel(['Z (km)'])

        #ax.set_aspect('equal')

        ax.set_title(title)

        plt.legend()

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png', dpi=300)

# convert classical orbital elements to r and v vectors
def coes2rv(coes, deg=False, mu=pd.earth['mu']):
    if deg:
          a,e,i,ta,aop,raan=coes#,date=coes
          i*=d2r
          ta*=d2r
          aop*=d2r
          raan*=d2r
    else:
        a,e,i,ta,aop,raan#,date=coes

    E=ecc_anomaly([ta, e], 'tae')

    r_norm=a*(1-e**2)/(1+e*np.cos(ta))

    # calculate r and v vectors in perifocal frame
    r_perif=r_norm*np.array([m.cos(ta), m.sin(ta),0])
    v_perif=m.sqrt(mu*a)/r_norm*np.array([-m.sin(E), m.cos(E)*m.sqrt(1-e**2),0])

    # rotation matrix from perifocal to ECI
    perif2eci=np.transpose(eci2perif(raan, aop, i))

    # calculate r and v vectors in inertial frames
    r=np.dot(perif2eci, r_perif)
    v=np.dot(perif2eci, v_perif)

    return r,v#,date

def rv2coes(r,v,mu=pd.earth['mu'],degrees=True,print_results=False):
    # norm of position vector
    r_norm=norm(r)

    # specific angular momentum
    h=np.cross(r,v)
    h_norm=norm(h)

    # inclination
    i=m.acos(h[2]/h_norm)

    # eccentricity vector
    e=((norm(v)**2-mu/r_norm)*r-np.dot(r,v)*v)/mu

    # eccentricity scalar
    e_norm=norm(e)

    # node line
    N=np.cross([0,0,1],h)
    N_norm=norm(N)

    # RAAN
    raan=m.acos(N[0]/N_norm)
    if N[1]<0: raan=2*np.pi-raan # quadrant check

    # argument of perigee
    aop=m.acos(np.dot(N,e)/N_norm/e_norm)
    if e[2]<0: aop=2*np.pi-aop # quadrant check

    # true anomaly
    ta=m.acos(np.dot(e,r)/e_norm/r_norm)
    if np.dot(r,v)<0: ta=2*np.pi-ta # quadrant check

    # semi major axis
    a=r_norm*(1+e_norm*m.cos(ta))/(1-e_norm**2)

    if print_results:
        print('a', a)
        print('e', e_norm)
        print('i', i*r2d)
        print('RAAN', raan*r2d)
        print('AOP', aop*r2d)
        print('TA', ta*r2d)

    # convert to degrees if specified
    if degrees: return [a,e_norm,i*r2d,ta*r2d,aop*r2d,raan*r2d]
    else: return [a,e_norm,i,ta,aop,raan]

# inertial to perifocal rotation matrix    
def eci2perif(raan, aop, i):
    row0=[-m.sin(raan)*m.cos(i)*m.sin(aop)+m.cos(raan)*m.cos(aop),m.cos(raan)*m.cos(i)*m.sin(aop)+m.sin(raan)*m.cos(aop),m.sin(i)*m.sin(aop)]
    row1=[-m.sin(raan)*m.cos(i)*m.cos(aop)-m.cos(raan)*m.sin(aop),m.cos(raan)*m.cos(i)*m.cos(aop)-m.sin(raan)*m.sin(aop),m.sin(i)*m.cos(aop)]
    row2=[m.sin(raan)*m.sin(i),-m.cos(raan)*m.sin(i),m.cos(i)]
    return np.array([row0, row1, row2])

# calculate eccentric anomaly (E) 
def ecc_anomaly(arr,method,tol=1e-8):
    if method=='newton':
        # newton's method for iteratively finding E
        Me,e=arr
        if Me<np.pi/2.0: E0=Me+e/2.0
        else: E0=Me-e
        for n in range(200): # arbitrary max number of steps
                ratio=(E0-e*np.sin(E0)-Me)/(1-e*np.cos(E0))
                if abs(ratio)<tol:
                    if n==0: return E0
                    else: return E1
                else:
                    E1=E0-ratio
                    E0=E1
        # did not converge
        return False
    elif method=='tae':
        ta,e=arr
        return 2*m.atan(m.sqrt((1-e)/(1+e))*m.tan(ta/2.0))
    else:
        print('Invalid method for eccentric anomaly')

def tle2coes(tle_filename,mu=pd.earth['mu'], deg=True):
    # read tle file
    with open(tle_filename, 'r') as f:
        lines=f.readlines()
    
    # separate into three lines
    line0=lines[0].strip() # name of satellite
    line1=lines[1].strip().split()
    line2=lines[2].strip().split()

    # epoch (year and day)
    epoch=line1[3]
    year,month,day,hour=calc_epoch(epoch)

    # collect coes

    # inclination
    i=float(line2[2])*d2r # rad
    # right ascension of ascending node
    raan=float(line2[3])*d2r # rad
    # eccentricity
    e_string=line2[4]
    e=float('0.'+e_string) # i know this is janky
    # argument of perigee
    aop=float(line2[5])*d2r # rad
    # mean anomaly
    Me=float(line2[6])*d2r # rad
    # mean motion
    mean_motion=float(line2[7]) # revs/day
    # period
    T=1/mean_motion*24*3600 # seconds
    # semi major axis
    a=(T**2*mu/4.0/np.pi**2)**(1/3.0)

    # calculate eccentric anomaly
    E=ecc_anomaly([Me, e], 'newton')

    # calculate true anomaly
    ta=true_anomaly([E, e])

    if deg: return a,e,i*r2d,ta*r2d,aop*r2d,raan*r2d#,[year,month,day,hour]
    return a,e,i,ta,aop,raan#,[year,month,day,hour]

def calc_epoch(epoch):
    #epoch year
    year=int('20'+epoch[:2])

    epoch=epoch[2:].split('.')

    # day of year
    day_of_year=int(epoch[0])-1

    # decimal hour of day
    hour=float('0.'+epoch[1])*24.0

    # get year-month-day
    date=datetime.date(year,1,1)+datetime.timedelta(day_of_year)

    # extract month and day
    month=float(date.month)
    day=float(date.day)

    return year,month,day,hour

def true_anomaly(arr):
    E,e=arr
    return 2*m.atan(m.sqrt((1+e)/(1-e))*m.tan(E/2.0))

def tle2rv(tle_filename):
    return coes2rv(tle2coes(tle_filename))

# calculate atmospheric density from given altitude
def calc_atmospheric_density(z):
    rhos,zs=find_rho_z(z)
    if rhos[0]==0: return 0.0

    Hi=-(zs[1]-zs[0])/m.log(rhos[1]/rhos[0])

    return rhos[0]*m.exp(-(z-zs[0])/Hi)

# find endpoints of altitude and density surrounding input altitude
def find_rho_z(z,zs=pd.earth['zs'],rhos=pd.earth['rhos']):
    if not 1.0<z<1000.0:
        return [[0.0,0.0],[0.0,0.0]]
    
    # find the two points surrounding the given input altitude
    for n in range(len(rhos)-1):
        if zs[n]<z<zs[n+1]:
            return [[rhos[n],rhos[n+1]],[zs[n],zs[n+1]]]
        
    # if out of range return zeros
    return [[0.0,0.0],[0.0,0.0]]