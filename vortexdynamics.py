import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider
from time import time
import itertools as it
from matplotlib import patches
import scipy as sc
from scipy import interpolate

# --- Initializations ---
R = 1. # Radius of container
Nvorts = 10
circs = np.ones(Nvorts); circs[:int(Nvorts/2.)] *= -1.0
nt = 10000 # Temporal resolution
r = []#np.zeros((nt,Nvorts,2))
v = []#np.zeros_like(r)
dt = 0.001
t = np.linspace(0, int(nt*dt), nt)
r.append(R*(np.random.random((Nvorts,2))- 1/2.))
core = 0.05 #vortex core radius
prox = 0.02 #proximity
gamma = 0.1 #dissipation param

def rot(theta, side, ind):
    temp = np.eye(2*side)
    temp[ind[0],ind[0]] = np.cos(theta[ind[0], ind[0]]); temp[ind[0],ind[1]+side] = -np.sin(theta[ind[0], ind[1]])
    temp[ind[1]+side,ind[0]] = np.sin(theta[ind[1], ind[0]]); temp[ind[1]+side,ind[1]+side] = np.cos(theta[ind[1], ind[1]])
    return temp

t1 = time()
for i in range(nt-1):
    pairedVorts = np.array(list(it.product(r[-1], repeat=2))).reshape((Nvorts,Nvorts,2,2))
    vortDiffs = pairedVorts[:,:,0] - pairedVorts[:,:,1]
    numerators = np.cross([0,0,1], vortDiffs)[:,:,:-1]
    distsSq = np.linalg.norm(vortDiffs, axis=2)**2
    # Removal of vortex pairs of opposite signs due to annihilation
    annih = (np.sqrt(distsSq) < 2*prox) & (np.outer(circs,circs) < 0.0)
    annih = list(set(np.ravel(np.where(annih==True))))
    # Method of Images
    images = r[-1]*R*R/np.linalg.norm(r[-1], axis=1)[:,None]**2
    imDiffs = r[-1] - images
    imNums = np.cross([0,0,1], imDiffs)[:,:-1]
    imDists = np.linalg.norm(imDiffs, axis=1)
    # dissipation
    #gamma * (np.einsum('i,ijk->jk', 1./np.abs(circs), np.multiply(np.outer(circs,circs), (vortDiffs/distsSq[:,:,None] + core**2)))  - \
    temp = 1./(2*np.pi)*(np.einsum('i,ijk->jk', circs, \
              numerators/(distsSq[:,:,None] + core**2)) + \
                   circs[:,None] * imNums/imDists[:,None]**2)
    r[-1] = np.delete(r[-1], annih, axis=0)
    temp = np.delete(temp, annih, axis=0)
    circs = np.delete(circs, annih)
    v.append(temp)
    Nvorts -= len(annih)
    # Euler integrator
    r.append(r[-1] + dt*v[-1])
#    # Symplectic integrator(try with "static" omg)
#    circsTemp = np.array(list(it.product(circs, repeat=2))).reshape((Nvorts,Nvorts,2))
#    omega = 1./(distsSq+core**2)*(circsTemp[:,:,0] + circsTemp[:,:,1])
#    #CentVort = np.einsum('ijk,->', pairedVorts, circsTemp)
#    coords = np.concatenate((r[-1][:,0], r[-1][:,1]))
#    temp1 = np.vstack((np.zeros(Nvorts-1), np.arange(Nvorts-1)+1)).T
#    temp2 = np.vstack((np.arange(Nvorts-2)+1, np.arange(Nvorts-2)+2)).T
#    multOrder = np.concatenate((temp1, temp2, temp2[::-1], temp1[::-1]))
#    multOrder = np.array(multOrder, dtype=int)
#    Phi = np.linalg.multi_dot([rot(omega*dt/.2, Nvorts, k) for k in multOrder])
#    r.append(Phi.dot(coords).reshape((2, Nvorts)).T)

print(time()-t1)

v = np.array(v)
r = np.array(r)

# Constants of motion
energy = [sum(vel*vel) for vel in v]
#enstrophy = np.sum(circs**2)
#
#print("#####")
print("mean energy: ", np.mean(energy))
#print("std energy: ", np.std(energy))
#print("#####")

fig, ax = plt.subplots(figsize=(8,8))
vorts, = plt.plot([],[], "bo")
ax.set_xlim(-1.1*R, 1.1*R)
ax.set_ylim(-1.1*R, 1.1*R)
ax.set_aspect('equal')
boundary = patches.Circle((0,0), R, color='g', fill=False)
ax.add_artist(boundary)

#fig2 = plt.figure(figsize=(8,8))
#plt.plot(energy)

time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def update(i):
    time_text.set_text(time_template%(t[i]))
    vorts.set_data(r[i][:,0], r[i][:,1])
    return vorts, time_text
    
anim = animation.FuncAnimation(fig, update, blit=False, interval=10,\
                              repeat=True)

pause = False
def onPress(event):
    if event.key != 'space':
        pass
    global pause
    if pause:
        anim.event_source.stop()
    if not pause:
        anim.event_source.start()
    pause ^= True

fig.canvas.mpl_connect('key_press_event', onPress)

plt.show()
