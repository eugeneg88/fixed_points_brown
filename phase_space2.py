#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:58:42 2024

@author: evgeni
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#%%
def ecc(jz, eps):
    y= 9*eps*jz/8
    x2 = np.fabs(jz) * (5 / 3 * (1+y)/(1-y))**0.5 
    e2 = 1 - x2
    if e2<0:
        return 0
    return e2**0.5

def incc(jz, eps):
    rr = jz/(1-ecc(jz,eps)**2)**0.5 
    return np.arccos(rr)*180/np.pi

plt.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 20})
# Define the Hamiltonian function
def hamiltonian(ee,x,H, eps_sa):
    G = (1-ee**2)**0.5
  #  cosi = H/G
    # Example Hamiltonian (modify this according to your specific problem)
    sec_term =   6*G**2 - 3*H**2 + 15*(1-G**2 - H**2/G**2 + H**2)*np.sin(x)**2
  #  sec_term = - ee**2/2 + cosi**2 + 1.5*ee**2*cosi**2 + 2.5*ee**2*(1-cosi**2)*np.cos(2*x)
    katz =  (-8+5*np.sin(x)**2)*G**2 + 5 * np.sin(x)**2 * (H**2/G**2 - 1 - H**2)
    return sec_term - 27 * H * eps_sa / 8 * katz

def plot_energy_levels(eps, inc_deg):
#eps=0.
#jz=np.cos(89*np.pi/180)
# Define the range for x and p
    x_min, x_max = 0, np.pi
    H = np.cos(inc_deg*np.pi/180)
    G_min, G_max = H, 1
    e_min, e_max = 0, (1-G_min**2)**0.5
#    p_min = (1-e_max**2)**0.5
 #   p_max = (1-e_min**2)**0.5
# Generate a grid of x and p values
    num_points = 300
    x_vals = np.linspace(x_min, x_max, num_points)
    e_vals = np.linspace(e_min, e_max, num_points) 
    G_vals = np.linspace(G_min, G_max, num_points) 
#[(1-y**2)**0.5 for y in p_vals]
    x_grid, G_grid = np.meshgrid(x_vals, G_vals)
    x_grid, e_grid = np.meshgrid(x_vals, e_vals)
# Calculate the Hamiltonian values at each point on the grid
    H_values = hamiltonian(e_grid, x_grid, H, eps)

# Plot the phase space
    p = 0.2 + 0.7*ecc(H,eps)
    plt.figure(1)
  #  plt.figure(figsize=(8, 6))
    plt.contourf(x_grid/np.pi, e_grid, H_values, 50, cmap='viridis')
    plt.colorbar(label='$\mathcal{H}$')
#plt.contour(x_grid/np.pi, e_grid, H_values, 20, cmap='Blues')
    plt.contour(x_grid/np.pi, e_grid, H_values, cmap='copper', levels=np.linspace(np.min(H_values)*p+np.max(H_values)*(1-p) ,np.max(H_values)+0.01,21))
    plt.contour(x_grid/np.pi, e_grid, H_values, colors='r', levels=[hamiltonian(0.001,0,H, eps)], linewidths=3)
    plt.scatter(1/2, ecc(H,eps))

#    plt.xlabel(r'$\omega / \pi$')
#    plt.ylabel('eccentricity')
    plt.title(r'$\epsilon_{\rm SA}=$' + str(eps) + '\quad' +r'$j_{\rm z}=$' + "%.2f" % H)
#plt.grid(True)
 #   plt.show()
    
plt.figure(figsize=(15,10))
plt.subplot(231)
plot_energy_levels(eps=0., inc_deg=120)
plt.ylabel('eccentricity')
plt.subplot(232)
plot_energy_levels(eps=0., inc_deg=135)
plt.subplot(233)
plot_energy_levels(eps=0., inc_deg=150)
plt.subplot(234)
plt.ylabel('eccentricity')
plot_energy_levels(eps=0.2, inc_deg=120)
plt.xlabel(r'$\omega / \pi$')
plt.subplot(235)
plot_energy_levels(eps=0.2, inc_deg=135)
plt.xlabel(r'$\omega / \pi$')
plt.subplot(236)
plot_energy_levels(eps=0.2, inc_deg=150)
plt.xlabel(r'$\omega / \pi$')

plt.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.16, hspace=0.18)

#%%
eps_lin=np.linspace(-0.0,0.3,100)

rh = lambda eps: 3**(1/3) * eps **(2/3)

hr = lambda f: f**1.5 / 3 **0.5

plt.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 20})


def plot_efix(inc0):

    fig, ax = plt.subplots(1)
#    zz = [np.cos(inc*np.pi/180) for inc in inc0]
    for i in range(0, len(inc0)):
        ax.plot(eps_lin, [ecc( np.cos(inc0[i]*np.pi/180), x) for x in eps_lin], linewidth=3, label=str(inc0[i]) + r'$^\circ$')
    plt.xlabel(r'$\epsilon_{\rm SA}$')
    plt.ylabel(r'$e_{\rm fix}$')
   # secax = ax.secondary_xaxis('top', functions=(rh, rh))
    #secax.set_xlabel(r'$a_1/r_{\rm H}$')
    plt.legend()
plot_efix([40,45, 50, 60, 143.7])
#plot_efix([135,140, 150, 160])

plt.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.88, wspace=0.16, hspace=0.18)

#%%
plt.subplot(121)
plot_efix( inc0=30)
plt.subplot(122)
plot_efix(inc0=45)
#%%
import matplotlib.pyplot as plt
import numpy as np

# Define the range of epsilon where f(epsilon) > 0.5
epsilon_upper_limit = ((0.5 / 3**(1/3))**(3/2))

# Define primary x-axis values (linear) within the range
x_primary = np.linspace(0, epsilon_upper_limit, 100)

# Define secondary x-axis values using the provided function
def transform_secondary(x):
    return 3**(1/3) * x**(2/3)

# Define the inverse function
def inverse_transform_secondary(x):
    return (x / (3**(1/3)))**(3/2)

# Create y-axis values (you can define this based on your specific function/data)
y = np.sin(x_primary)

# Create figure and primary axes
fig, ax1 = plt.subplots()

# Create secondary axes
ax2 = ax1.twiny()

# Plot the data on the primary axes
ax1.plot(x_primary, y, label='Primary X-axis', color='blue')

# Set the transformation for the secondary axes
ax2.set_xscale('function', functions=(transform_secondary, inverse_transform_secondary))

# Set the tick positions and labels for the secondary axes
# Generate ticks for secondary axis
x_secondary = np.linspace(0, epsilon_upper_limit, 10)  # Adjusted range for x_secondary
x_secondary_transformed = transform_secondary(x_secondary)
ax2.set_xticks(x_secondary_transformed)
ax2.set_xticklabels([f'{val:.2f}' for val in x_secondary])

# Set labels and title
ax1.set_xlabel('Primary X-axis')
ax2.set_xlabel('Secondary X-axis')
ax1.set_ylabel('Y-axis')
plt.title('Plot with Two X-axes')

# Add legend
ax1.legend()

# Show plot
plt.show()
#%%
import rebound

sim = rebound.Simulation()

# Add the Sun
sim.add(m=1.0)  # Sun's mass is set to 1.0

# Add the other giant planets: Saturn, Uranus, and Neptune
#sim.add(m=0.0002857, a=9.537, e=0.05415060, inc=0.93, Omega=1.983, omega=1.080, M=5.554)  # Saturn
#sim.add(m=0.00004365, a=19.191, e=0.04716771, inc=1.0, Omega=0.772, omega=0.257, M=5.149)  # Uranus
#sim.add(m=0.00005149, a=30.069, e=0.00858587, inc=0.769, Omega=1.077, omega=1.770, M=6.116)  # Neptune

# Add Jupiter
jupiter_mass = 1.0/1047.56  # Jupiter's mass relative to the Sun
a_j = 5.203
eps=0.1

inc0 = np.pi / 180 * 60
HH = np.cos(inc0) #* (1 - e0**2)**0.5
e0 = ecc(HH, eps)

sim.add(m=jupiter_mass, a=a_j)  # Jupiter

# Add Jupiter's four largest moons: Io, Europa, Ganymede, and Callisto
# Define masses of the Galilean satellites relative to Jupiter's mass
mass_io = 8.93e-5 * jupiter_mass
mass_europa = 4.8e-5 * jupiter_mass
mass_ganymede = 1.48e-4 * jupiter_mass
mass_callisto = 1.08e-4 * jupiter_mass

# Add the moons with their respective masses
#sim.add(primary=sim.particles[4], m=mass_io, a=0.00282, e=0.0041, inc=0.050, Omega=-0.235, omega=0.613, M=1.523)  # Io
#sim.add(primary=sim.particles[4], m=mass_europa, a=0.00449, e=0.0094, inc=0.471, Omega=0.001, omega=0.008, M=2.488)  # Europa
#sim.add(primary=sim.particles[4], m=mass_ganymede, a=0.00754, e=0.0013, inc=0.204, Omega=-0.067, omega=0.050, M=3.699)  # Ganymede
#sim.add(primary=sim.particles[4], m=mass_callisto, a=0.01258, e=0.0074, inc=0.205, Omega=-0.016, omega=0.202, M=4.760)  # Callisto
def get_a1(eps, m1, m2, m3, a2, e2):
    k = (m3**2 / (m1+m2) * (m1+m2+m3) )
    b2 = a2 * (1-e2**2)**0.5
    
    return k**(-1/3) * b2 * eps**(2/3)

def get_eps(m1,m2,m3,a1,a2,e2):
    b2 = a2 * (1-e2**2)**0.5
    return (a1/b2)**1.5 * m3 / (m1+m2) **0.5 / (m1+m2+m3)

m_test=3e-12
a1 = get_a1(eps, m_test, jupiter_mass, 1, a_j, 0)
# Add Pasiphae with JPL orbital elements
sim.add(m=3.0e-12, a=a1, e=e0, inc=np.arccos(HH/(1-(e0)**2)**0.5), Omega=np.pi / 4, omega= np.pi / 180  *90, M=np.pi / 180  *1, primary=sim.particles[1])

# Set the integrator
sim.integrator = "ias15"  # Wisdom-Holman symplectic integrator is efficient for this kind of simulation

# Integrate the system for a certain number of years
times = np.linspace(0, 4000, 4000)
args_jupiter = np.zeros(len(times))
args_pasiphae = np.zeros(len(times))
nodes_jupiter = np.zeros(len(times))
nodes_pasiphae = np.zeros(len(times))

ecc_jupiter = np.zeros(len(times))
ecc_pasiphae = np.zeros(len(times))

for i, t in enumerate(times):
    sim.integrate(t)
    #orb0 = sim.calculate_orbits(primary=sim.particles[0])
   # orb4 = sim.particles[4].calculate_orbits(primary=sim.particles[4])
    args_jupiter[i] = sim.particles[1].calculate_orbit(primary=sim.particles[0]).omega  # Argument of pericentre of Jupiter
    args_pasiphae[i] = sim.particles[2].calculate_orbit(primary=sim.particles[1]).omega  # Argument of pericentre of Pasiphae
    nodes_jupiter[i] = sim.particles[1].calculate_orbit(primary=sim.particles[0]).Omega  # Argument of pericentre of Jupiter
    nodes_pasiphae[i] = sim.particles[2].calculate_orbit(primary=sim.particles[1]).Omega  # Argument of pericentre of Pasiphae
    ecc_jupiter[i] = sim.particles[1].calculate_orbit(primary=sim.particles[0]).e  # Argument of pericentre of Jupiter
    ecc_pasiphae[i] = sim.particles[2].calculate_orbit(primary = sim.particles[1]).e
    
plot_energy_levels(eps, inc0 * 180 / np.pi)

plt.scatter(args_pasiphae/np.pi, ecc_pasiphae, color='grey', alpha=0.1)
plt.xlim([0,1])
plt.ylabel('eccentricity')
plt.xlabel(r'$\omega / \pi$')
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.88, wspace=0.16, hspace=0.18)

#%%
plt.figure(2)
plt.plot(times, args_pasiphae/np.pi, label=r'$\omega_p$')
plt.plot(times, ecc_pasiphae, label=r'$\omega_p$')
#%%
eps=0.16

inc0 = np.pi / 180 * 60
HH = np.cos(inc0) #* (1 - e0**2)**0.5
e0 = ecc(HH, eps)
E_fix = hamiltonian(e0, np.pi/2, HH, eps)
E_sep = hamiltonian(0.001, 0, HH, eps)

de = 1*16/9*0.6**0.5*e0*np.fabs(HH)
dH_de = 18*e0**2 + 5 * HH**2 * 2*e0 / (1-e0**2)**2

fluc = dH_de * de

print (E_fix - E_sep, E_sep, e0, HH, 1200*eps**3* e0**2*HH)
#%%
import rebound

def get_a1(eps, m1, m2, m3, a2, e2):
    k = (m3**2 / (m1+m2) * (m1+m2+m3) )
    b2 = a2 * (1-e2**2)**0.5
    
    return k**(-1/3) * b2 * eps**(2/3)

def run_system(eps, HH, planet):
    sim = rebound.Simulation()

# Add the Sun
    sim.add(m=1.0)  # Sun's mass is set to 1.0
    if planet == 'jupiter':
        jupiter_mass = 1.0/1047.56  # Jupiter's mass relative to the Sun
        a_j = 5.203

 #   inc0 = np.pi / 180 * inc_deg
  #  HH = np.cos(inc0) #* (1 - e0**2)**0.5
        e0 = min(ecc(HH, eps), 0.999)

    sim.add(m=jupiter_mass, a=a_j)  # Jupiter

    m_test=3e-12
    a1 = get_a1(eps, m_test, jupiter_mass, 1, a_j, 0)
# Add Pasiphae with JPL orbital elements
    if eps == 0.175 or eps == 0.185:
        print (eps, e0, HH)
        
    sim.add(m=3.0e-12, a=a1, e=e0, inc=np.arccos(HH/(1-(e0)**2)**0.5), Omega=np.pi / 4, omega= np.pi / 180  *90, M=np.pi / 180  *1, primary=sim.particles[1])
 
# Set the integrator
    sim.integrator = "ias15"  # Wisdom-Holman symplectic integrator is efficient for this kind of simulation
# Integrate the system for a certain number of years
    p_out = sim.particles[1].calculate_orbit(primary=sim.particles[0]).P
    times = np.linspace(0, 10 * p_out / eps, 5000)
 
#    sim.status()
 #   print(sim.particles[2].calculate_orbit(primary=sim.particles[1]).e, 0)

    stat = 0
    for i, t in enumerate(times):
        sim.integrate(t)
 #       print(sim.particles[2].calculate_orbit(primary=sim.particles[1]).e, t)
        if  sim.particles[2].calculate_orbit(primary=sim.particles[1]).e>=1.1:
            stat = 2
            return stat
   #     print ( np.sin(sim.particles[2].calculate_orbit(primary=sim.particles[1]).omega))
 
        elif  np.sin(sim.particles[2].calculate_orbit(primary=sim.particles[1]).omega) <= -0.2:
 #           print ('ok')
            stat = 1
            if  sim.particles[2].calculate_orbit(primary=sim.particles[1]).e>=1.1:
                stat = 2
                
    #        return stat
  #  sim.status()
    return stat
 
run_system(0.26, -0.1, 'jupiter')    
 #%%
import itertools
import multiprocessing # create a process pool that uses all cpus

epsilons_rp22 = np.linspace(0.171, 0.179, 305)
incs_rp22 = np.linspace(0.43, 0.48, 330) 
res_rp22 =np.zeros([len(epsilons_rp22), len(incs_rp22)])

paramlist = list(itertools.product(epsilons_rp22, incs_rp22))

#A function which will process a tuple of parameters
def func(params):
  eps = params[0]
  inc = params[1]
  return run_system(eps, inc, 'jupiter')

pool = multiprocessing.Pool()

#Distribute the parameter sets evenly across the cores
res_rp22 = pool.map(func,paramlist)
#%%
np.save('res_pro2.npy', res_rp22)
#%%
plt.pcolor(incs_rp22, epsilons_rp22, np.array(res_rp22).reshape([len(epsilons_rp22), len(incs_rp22)]), cmap='Set1')#'nipy_spectral')
#for i in range(0, len(epsilons_all)):
#    for j in range(0, len(incs_all)):
#        res_all[i][j] = run_system(epsilons_all[i], incs_all[j], 'jupiter')
#%%
#incs_p = np.linspace(40,85, 46)
#epsilons_p = np.linspace(0.03,0.3, 55)
res22 = np.array(res_rr)
res24 = res22.reshape([len(epsilons_rr), len(incs_rr)])
#%%
res22zz = np.array(res_allzz)
res24zzz = res22zzz.reshape([len(epsilons_allzzz), len(incs_allzzz)])
#%%
plt.rc('text', usetex=True)
plt.figure(figsize=(10,6))
plt.pcolor(incs_all, epsilons_all, res24, cmap='Blues_r')#'nipy_spectral')
#plt.plot(incs, [0.03 + 0.4*np.sin((ii-40)*np.pi/180) for ii in incs], color='k', linewidth=3)
plt.text(0.55, 0.036, 'Circulating', rotation=-65, color='white', size=36, font="serif", usetex=False)
#plt.text(145, 0.15, 'Librating', color='white')
#plt.text(120, 0.28, 'Unstable', color = 'black')

#plt.text(42, 0.15, 'Circulating', color='white')
plt.text(-0.6, 0.15, 'Librating', rotation=0, color='white', size=40, font="serif", usetex=False)
plt.text(0.2, 0.24, 'Unstable', color = 'black', size=40, font='serif', usetex=False)

plt.ylabel(r'$\epsilon_{\rm SA}$')
plt.xlabel(r'$j_z$')
plt.subplots_adjust(left=0.1, bottom=0.12, right=0.97, top=0.96, wspace=0.16, hspace=0.18)

#plt.colorbar()
#%%
plot_energy_levels(eps, inc0 * 180 / np.pi)

plt.scatter(args_pasiphae/np.pi, ecc_pasiphae, color='grey', alpha=0.1)
plt.xlim([0,1])
plt.ylabel('eccentricity')
plt.xlabel(r'$\omega / \pi$')
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.94, top=0.88, wspace=0.16, hspace=0.18)


#%%
plt.plot(times, args_pasiphae)
#%%
import matplotlib.pyplot as plt

print(plt.rcParams["font.sans-serif"][0])
print(plt.rcParams["font.monospace"][0])
plt.show()
#%%
from tkinter import *
from tkinter import font
root = Tk()
list_fonts = list(font.families())
for i in list_fonts:
    print(i)
root.mainloop()


#%%
def inc_in(eps,jz):
    cosi = (3/5)**0.25 *jz**0.5 *  ( (1 - 9 * eps * jz / 8)/(1 + 8 * eps * jz / 8))**0.5 
    return np.arccos(cosi) * 180 / np.pi

