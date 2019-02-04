import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
import itertools as it
from matplotlib import patches
import scipy as sc

#TODO
# 1. Symplectic k=4
# 2. Method of Images
# 3. Vortex core interactions
# 4. Balance +/- circulation

# --- Initializations ---
R = 3. # Radius of container
Nvorts = 20
circs = np.random.choice([1.,-1.], size=Nvorts)
print(sum(circs>0), sum(circs<0))
nt = 1000 # Temporal resolution
nx = 100 # Spatial resolution
r = np.zeros((nt,Nvorts,2))
v = np.zeros_like(r)
dt = 0.01
t = np.linspace(0, 10, nt)
r[0] = np.sqrt(2)*R*(np.random.random((Nvorts,2))- 1/2.)
core = 0.05

# Distance from edge at which method of images kicks in
eps = 0.1
delta = 0.001

a = np.matrix([[1./4, 1./4 - np.sqrt(3)/6],
               [1./4 + np.sqrt(3)/6, 1./4]])

#w = np.zeros(())

t1 = time()
for i in range(nt-1):
    # Vectorized equations of motion
    pairedVorts = np.array(list(it.product(r[i], repeat=2))).reshape((Nvorts,Nvorts,2,2))
    vortDiffs = pairedVorts[:,:,0] - pairedVorts[:,:,1]
    numerators = np.cross([0,0,1], vortDiffs)[:,:,:-1]
    distsSq = np.linalg.norm(vortDiffs, axis=2)**2
    distsSq[distsSq < core] = np.inf
    matr = np.einsum('i,ijk->jk', circs, numerators/distsSq[:,:,None])
    # Method of Images
    nearEdge = np.ones(Nvorts, dtype=bool)#(R - np.linalg.norm(r[i], axis=1) < eps)
    images = r[i]*R*R/np.linalg.norm(r[i], axis=1)[:,None]**2
    imDiffs = r[i] - images
    imDiffs[np.logical_not(nearEdge)] = 0.0
    imNums = np.cross([0,0,1], imDiffs)[:,:-1]
    imDists = np.linalg.norm(imDiffs, axis=1)
    #imDists[imDists == 0.0] = np.inf
    v[i] = 1/(2*np.pi) * (matr - circs[:,None]*imNums/(imDists[:,None]+delta)**2    )
    # Euler integrator
    #r[i+1] = r[i] + dt*v[i+1]
    # Symplectic integrator (k=4)
    k0 = dt*

print(time()-t1)
energy = np.sum(np.linalg.norm(v, axis=2)**2, axis=1)

fig, ax = plt.subplots(figsize=(8,8))
vorts, = plt.plot([],[],"bo")
ax.set_xlim(-1.1*R, 1.1*R)
ax.set_ylim(-1.1*R, 1.1*R)
ax.set_aspect('equal')
boundary = patches.Circle((0,0), R, color='r', fill=False)
ax.add_artist(boundary)

#fig2 = plt.figure(figsize=(8,8))
#plt.plot(energy)

def update(i):
    plt.title(i)
    vorts.set_data(r[i,:,0], r[i,:,1])
    return vorts,
    
ani = animation.FuncAnimation(fig, update, blit=False, interval=1)
#ani.save("vortices.mp4")
plt.show()