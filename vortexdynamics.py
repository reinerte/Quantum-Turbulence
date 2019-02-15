import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider
from time import time
import itertools as it
from matplotlib import patches
import scipy as sc
from scipy import interpolate

#TODO
# Make Method of Images reliable
# Color code vortices
# Add dissipation

# --- Initializations ---
R = 10 # Radius of container
Nvorts = 40
circs = np.ones(Nvorts); circs[:int(Nvorts/2.)] *= -1.0
nt = 10000 # Temporal resolution
r = np.zeros((nt,Nvorts,2))
v = np.zeros_like(r)
dt = 0.1
t = np.linspace(0, 100, nt)
r[0] = R*(np.random.random((Nvorts,2))- 1/2.)
core = 0.5 #vortex core radius
edge = 0.4 #boundary regime
gamma = 0.1 #dissipation param

a = np.matrix([[1./4, 1./4 - np.sqrt(3)/6],
               [1./4 + np.sqrt(3)/6, 1./4]])

def vel(p):
    pairedVorts = np.array(list(it.product(p, repeat=2))).reshape((Nvorts,Nvorts,2,2))
    vortDiffs = pairedVorts[:,:,0] - pairedVorts[:,:,1]
    numerators = np.cross([0,0,1], vortDiffs)[:,:,:-1]
    distsSq = np.linalg.norm(vortDiffs, axis=2)**2
    # Method of Images
    images = p*R*R/np.linalg.norm(p, axis=1)[:,None]**2
    imDiffs = p - images
    imNums = np.cross([0,0,1], imDiffs)[:,:-1]
    imDists = np.linalg.norm(imDiffs, axis=1)
    # dissipation
    #gamma * (np.einsum('i,ijk->jk', 1./np.abs(circs), np.multiply(np.outer(circs,circs), (vortDiffs/distsSq[:,:,None] + core**2)))  - \
    return 1./(2*np.pi)*(np.einsum('i,ijk->jk', circs, \
              numerators/(distsSq[:,:,None] + core**2) - \
                   circs[:,None] * imNums/imDists[:,None]**2))

t1 = time()
for i in range(nt-1):
    # Euler integrator
    v[i] = vel(r[i])
    r[i+1] = r[i] + dt*v[i]
    # Symplectic integrator (k=4)
#    w = np.array([vel(r[i]), vel(r[i])])
#    for j in range(20):
#        aw1 = a[0,0] * w[0] + a[0,1] * w[1]
#        aw2 = a[1,0] * w[0] + a[1,1] * w[1]
#        w = np.array([vel(r[i] + dt*aw1), \
#                       vel(r[i] + dt*aw2)])
#    r[i+1] = r[i] + dt/2.*(w[0] + w[1])

print(time()-t1)

# Constants of motion
energy = np.sum(np.linalg.norm(v[:-1], axis=2)**2, axis=1)
Q = r[:,:,0].dot(circs)
P = r[:,:,1].dot(circs)
QQPP = Q*Q + P*P
enstrophy = np.sum(circs**2)

print("#####")
print("mean energy: ", np.mean(energy))
print("std energy: ", np.std(energy))
print("-----")
print("mean Q: ", np.mean(Q))
print("std Q: ", np.std(Q))
print("-----")
print("mean P: ", np.mean(P))
print("std P: ", np.std(P))
print("-----")
print("mean QQPP: ", np.mean(QQPP))
print("std QQPP: ", np.std(QQPP))
print("#####")

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
    vorts.set_data(r[i,:,0], r[i,:,1])
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

#ani.save("vortices.mp4")
plt.show()
