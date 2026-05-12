from sys import path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice

import planetary_data as pd
import tools as t
import spice_tools as st


# time conversions
hours=3600.0
days=hours*24

def null_perts():
    return {
        'J2':False,
        'aero':False,
        'moon_gravity':False,
        'solar_gravity':False,
        'Cd': 2.2,
        'A': (1e-3)**2/4.0, # km^2,
        'thrust': 0,
        'isp': 0,
        'thrust_direction':0,
        'n_bodies':[],
        'srp':False,
        'custom_thrust_function':False,
        'A_srp':False,
        'CR':False

    }

class OrbitPropagator:
    def __init__(self, state0, n, tspan, dt, r_wing, l, coes=False, deg=True, mass0=0, cb=pd.earth, perts=null_perts(),sc={}, date0='2019-12-3', propagator='lsoda', frame='J2000', rot=False):
        if coes:
            self.r0, self.v0=t.coes2rv(state0, deg=deg, mu=cb['mu'])
        else:
            self.r0=np.array(state0[:3])
            self.v0=np.array(state0[3:])


        self.mass0=mass0
        self.date0=date0
        self.frame=frame
        self.y0=np.hstack((self.r0, self.v0))
        self.tspan = tspan
        self.dt = dt
        self.cb = cb
        self.rot = rot
        self.n = n #Orientation vector
        # total number of steps
        self.n_steps = int(np.ceil(self.tspan / self.dt))+1

        self.r_wing=r_wing
        self.l=l

        # initialize arrays
        self.ys = np.zeros((self.n_steps+1, 7))
        self.ts = np.zeros((self.n_steps+1, 1))
        self.alts = np.zeros((self.n_steps+1))

        # initial conditions
        #self.L0=np.zeros(3)
        self.y0 = self.r0.tolist()+self.v0.tolist()+[self.mass0]#+self.L0.tolist()
        self.alts[0]=t.norm(self.r0)-self.cb['radius']
        self.ts[0] = 0
        self.ys[0] = np.array(self.y0)
        self.step = 0
        self.spice_files_loaded = []

        # initiate solver
        self.solver = ode(self.diffy_q)
        self.solver.set_integrator(propagator)
        self.solver.set_initial_value(self.y0, 0)

        # define perturbations dictionary
        self.perts = perts

        # store stop conditions dictionary
        self.stop_conditions_dict=sc

        # define dictionary to map internal methods
        self.stop_conditions_map={'max_alt':self.check_max_alt, 'min_alt':self.check_min_alt}

        # create stop condition function list with deorbit always checked
        self.stop_condition_functions=[self.check_deorbit]

        # fill in rest of stop conditions
        for key in self.stop_conditions_dict.keys():
            self.stop_condition_functions.append(self.stop_conditions_map[key])

        # check if loading in spice data
        if self.perts['n_bodies'] or self.perts['srp']:

            # load leap seconds kernel
            spice.furnsh('spice_data/latest_leapseconds.TLS')

            # add to list of loaded spice files
            self.spice_files_loaded.append('spice_data/latest_leapseconds.TLS')

            # converts start date to seconds after J2000
            self.start_time=spice.utc2et(self.date0)

            # create timespan array in seconds after J2000
            self.spice_tspan=np.linspace(self.start_time,self.start_time+self.tspan,self.n_steps)

            # if srp get states of sun
            if self.perts['srp']:

                # load spice file for given body
                spice.furnsh(self.cb['spice_file'])

                # add to spice file list
                self.spice_files_loaded.append(self.cb['spice_file'])

                # calculate central body's states throughout entire propagation WRT sun
                self.cb['states']=st.get_ephemeris_data(self.cb['name'],self.spice_tspan,self.frame,'SUN')

        # load kernels for each body
        for body in self.perts['n_bodies']:

            # if spice hasn't already been loaded
            if body['spice_file'] not in self.spice_files_loaded:

                # load spice file for given body
                spice.furnsh(body['spice_file'])

                # add to spice file list
                self.spice_files_loaded.append(body['spice_file'])

            # calculate body states WRT central body
            body['states']=st.get_ephemeris_data(body['name'],self.spice_tspan,self.frame,self.cb['name'])
        
        # check for custom thrust function
        if self.perts['custom_thrust_function']:
            self.thrust_func=self.perts['custom_thrust_function']

        else:
            if self.perts['thrust']:
                self.thrust_func=self.default_thrust_func

        if self.rot:
            #Define quaternion
            theta=0 #rad
            self.q=np.array([np.cos(theta/2), np.sin(theta/2)*(1/np.sqrt(3)), np.sin(theta/2)*(1/np.sqrt(3)), np.sin(theta/2)*(1/np.sqrt(3))])
        
        self.propagate_orbit()
    
    # check if s/c has deorbited
    def check_deorbit(self):
        if self.alts[self.step]<self.cb['deorbit_altitude']:
            print('Spacecraft deorbited after %.1f seconds' % self.ts[self.step][0])
            return False
        return True
    
    # check if maximum altitude exceeded
    def check_max_alt(self):
        if self.alts[self.step]<self.stop_conditions_dict['max_alt']:
            print('Spacecraft reached maximum altitude after %.1f seconds' % self.ts[self.step][0])
            return False
        return True
    
    # check if minimum altitude exceeded
    def check_min_alt(self):
        if self.alts[self.step]<self.stop_conditions_dict['min_alt']:
            print('Spacecraft reached minimum altitude after %.1f seconds' % self.ts[self.step][0])
            return False
        return True

    # function called at each time step to check all stop conditions
    def check_stop_conditions(self):
        # for each stop condition
        for sc in self.stop_condition_functions:

            # if returns False
            if not sc():

                # stop condition reached, return False
                return False
            
        # if no stop condition reached, return True
        return True
    
    def propagate_orbit(self):
        # propagate orbit
        while self.solver.successful() and self.step < self.n_steps and self.check_stop_conditions():
            self.solver.integrate(self.solver.t + self.dt)
            print(self.solver.t, " seconds have passed")
            self.step += 1
            self.ts[self.step] = self.solver.t
            self.ys[self.step] = self.solver.y
            self.alts[self.step]=t.norm(self.solver.y[:3])-self.cb['radius']
        # extract arrays at the step where the propagation stopped
        self.ts=self.ts[:self.step]
        self.rs=self.ys[:self.step,:3]
        self.vs=self.ys[:self.step,3:6]
        self.masses=self.ys[:self.step,-1]
        self.alts=self.alts[:self.step]
        if self.rot:
            self.qs=self.qs[:self.step]

    def diffy_q(self,t_,y):

        """time=[0,0]
        time[0]=time[1]
        time[1]=t_
        Δt=time[1]-time[0]"""

        # unpack state
        rx,ry,rz,vx,vy,vz,mass=y#,Lx,Ly,Lz = y
        r=np.array([rx,ry,rz])
        v=np.array([vx,vy,vz])
        """I = mass*self.l**2
        ωx,ωy,ωz = Lx/I,Ly/I,Lz/I
        ω=np.array([ωx,ωy,ωz])
        Ω=np.array([[ 0, -ωx, -ωy, -ωz], 
                    [ωx,   0,  ωz, -ωy], 
                    [ωy, -ωz,   0,  ωx], 
                    [ωz,  ωy, -ωx,   0]])"""
        dmdt=0

        # norm of the radius vector
        norm_r=np.linalg.norm(r)

        # two body acceleration
        a=-r*self.cb['mu']/norm_r**3

        #Calculate rotation angle
        #rot_propagator=np.cos(t.norm(ω)*(Δt)/2)*np.eye(4)+1/t.norm(ω)*np.sin(t.norm(ω)*(Δt)/2)*Ω if t.norm(ω) != 0 else np.eye(4)
        #print(np.cos(t.norm(ω)*(Δt)/2)*np.eye(4))
        #print(1/t.norm(ω)*np.sin(t.norm(ω)*(Δt)/2)*Ω)
        #print(Lx, Ly, Lz)
        #self.q= t.normed(np.dot(rot_propagator, self.q))
        #θ = 2*np.acos(self.q[0])
        #if t_ > 0:
            #exit()

        #Apply rotation angle to the area and wing vector
        #R = np.array([[np.cos(θ), -np.sin(θ),0],[np.sin(θ), np.cos(θ),0],[0,0,1]])
        #P = np.array([np.cross(ω, self.r_wing), self.r_wing, ω])
        #self.n = t.normed(np.transpose(P) @ R @ P @ self.n)
        #self.r_wing = t.normed(np.transpose(P) @ R @ P @ self.r_wing) * self.l

        # J2 perturbation
        if self.perts['J2']:
            z2=r[2]**2
            r2=norm_r**2
            tx=r[0]/norm_r*(5*z2/r2-1)
            ty=r[1]/norm_r*(5*z2/r2-1)
            tz=r[2]/norm_r*(5*z2/r2-3)

            a_j2=1.5*self.cb['J2']*self.cb['mu']*self.cb['radius']**2/norm_r**4*np.array([tx,ty,tz])
            a+=a_j2

        # aerodynamic drag
        if self.perts['aero']:
            # calculate altitude and air density
            z=norm_r-self.cb['radius']
            rho=t.calc_atmospheric_density(z)

            # calculate motion of s/c with respect to a rotating atmosphere
            v_rel=v-np.cross(self.cb['atm_rot_vector'],r)

            # drag direction
            d=t.normed(-v_rel)

            #Align the surface of the spacecraft perpendicular to the drag direction with minimal rotation
            self.n = t.normed(t.oproj(self.n, d))
            
            #Align the wing of the spacecraft perpendicular to the surface of the spacecraft with minimal rotation
            #self.r_wing = t.normed(t.oproj(self.r_wing, self.n))*self.l

            #Full formula of the drag acceleration
            drag=-v_rel*0.5*rho*t.norm(v_rel)*self.perts['Cd']*self.perts['A']*np.abs(np.dot(self.n, d))/mass

            #Simplifying assumption with the drag being minimized because of the tail
            drag=0

            a+=drag

        # thrust perturbation
        if self.perts['thrust']:
            # thrust vector
            a+=self.perts['thrust_direction']*t.normed(v)*self.perts['thrust']/mass/1000.0 #km/s**2

            # derivative of total mass
            dmdt=-self.perts['thrust']/self.perts['isp']/9.81

        # n body perturbation
        for body in self.perts['n_bodies']:

            # vector pointing from satellite to body
            r_cb2nb=body['states'][self.step,:3]

            # vector poitning from satellite to body
            r_sat2body=r_cb2nb-r

            # nth body acceleration vector
            a+=body['mu']*(r_sat2body/t.norm(r_sat2body)**3 - r_cb2nb/t.norm(r_cb2nb)**3)

        # solar radiation pressure
        if self.perts['srp']:

            # vector pointing from sun to spacecraft
            r_sun2sc=self.cb['states'][self.step,:3]+r

            P=4.53 * 1e-3 # kg/km/s^2
            A=self.perts['A_srp']
            s_hat=-t.normed(r_sun2sc)
            n_hat=self.n
            rho_s=self.perts['CR']
            cos_α=np.dot(n_hat, s_hat)

            #Position check - if the spacecraft is in the shadow of the central body, set SRP acceleration to 0
            if np.dot(self.cb['states'][self.step,:3], r) > 0:
                a_srp=0
            else:
                a_srp=P*A*cos_α*((1-rho_s)*s_hat+2*rho_s*cos_α*n_hat)/mass
            a+=a_srp

        """if self.rot:
            #Calculate torque
            F_SRP = a_srp*mass
            τ = np.cross(self.r_wing, F_SRP)"""


        
        return [vx,vy,vz,a[0],a[1],a[2],dmdt] #, τ[0], τ[1], τ[2]]
    
    def calculate_coes(self,degrees=True, parallel=False):
        print('Calculating COEs...')

        self.coes=np.zeros((self.step, 6))
        self.coes_rel=np.zeros((self.step, 6))

        # fill array
        for n in range(self.step):
            self.coes[n,:]=t.rv2coes(self.rs[n,:],self.vs[n,:],mu=self.cb['mu'],degrees=degrees)
        
        self.coes_rel=self.coes-self.coes[0,:]

    def plot_coes(self,hours=False,days=False,show_plot=False,save_plot=False,title='COEs', figsize=(16,8), rel=True, parallel=False):
        print('Plotting ODEs...')

        # create figure and axes instances
        fig,axs=plt.subplots(nrows=2,ncols=3,figsize=figsize)

        # figure title
        fig.suptitle(title,fontsize=20)

        # x axis
        if hours:
            ts=self.ts/3600.0
            xlabel='Time elapsed (hours)'
        elif days:
            ts=self.ts/3600.0/24.0
            xlabel='Time elapsed (days)'
        else:
            ts=self.ts
            xlabel='Time elapsed (seconds)'

        if rel:
            coes=self.coes_rel
        else:
            coes=self.coes

        # plot true anomaly
        axs[0,0].plot(ts,coes[:,3])
        axs[0,0].set_title('True Anomaly vs. Time')
        axs[0,0].grid(True)
        axs[0,0].set_ylabel('Angle (degrees)')

        # plot semi major axis
        axs[1,0].plot(ts,coes[:,0])
        axs[1,0].set_title('Semi-Major Axis vs. Time')
        axs[1,0].grid(True)
        axs[1,0].set_ylabel('Semi-Major Axis (km)')
        axs[1,0].set_xlabel(xlabel)

        # plot eccentricity
        axs[0,1].plot(ts,coes[:,1])
        axs[0,1].set_title('Eccentricity vs. Time')
        axs[0,1].grid(True)

        # plot argument of periapse
        axs[0,2].plot(ts,coes[:,4])
        axs[0,2].set_title('Argument of Periapse vs. Time')
        axs[0,2].grid(True)

        # plot inclination
        axs[1,1].plot(ts,coes[:,2])
        axs[1,1].set_title('Inclination vs. Time')
        axs[1,1].grid(True)
        axs[1,1].set_ylabel('Angle (degrees)')
        axs[1,1].set_xlabel(xlabel)

        # plot RAAN
        axs[1,2].plot(ts,coes[:,5])
        axs[1,2].set_title('RAAN vs. Time')
        axs[1,2].grid(True)
        axs[1,2].set_xlabel(xlabel)

        # spread out subplots
        plt.subplots_adjust(wspace=0.3)

        if show_plot:
            plt.show()

        if save_plot:
            plt.savefig(title+'.png', dpi=300)

    def calculate_apoapse_periapse(self):
        # define empty arrays
        self.apoapses=self.coes[:,0]*(1+self.coes[:,1])
        self.periapses=self.coes[:,0]*(1-self.coes[:,1])
    
    def plot_apoapse_periapse(self,show_plot=False,save_plot=False,hours=False,days=False,title='Apoapse and Periapse vs. Time',dpi=500):
        #create figure
        plt.figure(figsize=(20,10))

        # x axis
        if hours:
            ts=self.ts/3600.0
            x_unit='Time elapsed (hours)'
        elif days:
            ts=self.ts/3600.0/24.0
            x_unit='Time elapsed (days)'
        else:
            ts=self.ts
            x_unit='Time elapsed (seconds)'

        # plot each
        plt.plot(ts,self.apoapses,'b',label='Apoapse')
        plt.plot(ts,self.periapses,'r',label='Periapse')

        # labels
        plt.xlabel('Time (%s)' % x_unit)
        plt.ylabel('Altitude (km)')

        # other parameters
        plt.grid(True)
        plt.title(title)
        plt.legend()

        if show_plot:
            plt.show()

        if save_plot:
            plt.savefig(title+'.png')

    # plot altitude over time
    def plot_alts(self,show_plot=False,save_plot=False,hours=False,days=False,title='Radial Distance vs. Time',figsize=(16,8),dpi=500):
        # x axis
        if hours:
            ts=self.ts/3600.0
            x_unit='Time elapsed (hours)'
        elif days:
            ts=self.ts/3600.0/24.0
            x_unit='Time elapsed (days)'
        else:
            ts=self.ts
            x_unit='Time elapsed (seconds)'

        plt.figure(figsize=figsize)
        plt.plot(ts,self.alts,'w')
        plt.grid(True)
        plt.xlabel('Time (%s)' % x_unit)
        plt.ylabel('Altitude (km)')
        plt.title(title)

        if show_plot:
            plt.show()

        if save_plot:
            plt.savefig(title+'.png', dpi)

    # plot altitude over time
    def plot_qs(self,show_plot=False,save_plot=False,hours=False,days=False,title='Rotation Angle vs. Time',figsize=(16,8),dpi=500):
        # x axis
        if hours:
            ts=self.ts/3600.0
            x_unit='Time elapsed (hours)'
        elif days:
            ts=self.ts/3600.0/24.0
            x_unit='Time elapsed (days)'
        else:
            ts=self.ts
            x_unit='Time elapsed (seconds)'
        plt.figure(figsize=figsize)
        plt.plot(ts,2*np.acos(self.qs[:,0]),'w')
        plt.grid(True)
        plt.xlabel('Time (%s)' % x_unit)
        plt.ylabel('Angle (rad)')
        plt.title(title)

        if show_plot:
            plt.show()

        if save_plot:
            plt.savefig(title+'.png', dpi)

    def plot_3d(self, show_plot=False, save_plot=False, title='Test Title'):
        fig=plt.figure(figsize=(16,8))
        ax=fig.add_subplot(111,projection='3d')

        # plot central body
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        _x=self.cb['radius']*np.cos(_u)*np.sin(_v)
        _y=self.cb['radius']*np.sin(_u)*np.sin(_v)
        _z=self.cb['radius']*np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='Blues')

        ax.plot(self.rs[:,0], self.rs[:,1], self.rs[:,2], 'w', label='Trajectory', zorder=10)
        ax.plot([self.rs[0,0]], [self.rs[0,1]], [self.rs[0,2]], 'wo', label='Initial Position', zorder=10)

        # plot the x,y,z vectors
        l=self.cb['radius']*2
        x,y,z=[[0,0,0], [0,0,0], [0,0,0]]
        u,v,w=[[l,0,0], [0,l,0], [0,0,l]]
        ax.quiver(x, y, z, u, v, w, color='k')

        # plot the direction of the sun
        #ax.quiver(0,0,0,-0.24235989*l,0.89015157*l,0.3858651*l, color='g')

        # plot the area vector of satellite
        #ax.quiver(self.rs[0,0], self.rs[0,1], self.rs[0,2], l*self.n[0], l*self.n[1], l*self.n[2], color='r')

        max_val=np.max(np.abs(self.rs))

        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])

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