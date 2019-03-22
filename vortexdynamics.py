import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider
from time import time
import itertools as it
from matplotlib import patches
import scipy as sc
import scipy.signal as sgn
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

# --- Initializations ---
R = 2. # Radius of container
Nvorts = 4
circs = np.ones(Nvorts); circs[:int(Nvorts/2.)] *= -1.0
nt = 10000 # Temporal resolution
r = []#np.zeros((nt,Nvorts,2))
v = []#np.zeros_like(r)
dt = 0.01
t = np.linspace(0, int(nt*dt), nt)
#r.append(R*(np.random.random((Nvorts,2))- 1/2.))
r.append(np.array([[-0.2, 0.0], [-0.3, 0.0], [0.2, 0.0], [0.3, 0.0]]))
#r.append(np.array([[0.1, 0.0], [-0.1, 0.0]]))
core = 0.05 #vortex core radius
prox = 0.0 #proximity
gamma = 0.1 #dissipation param

theta = np.zeros(Nvorts)

#Impurity/stirring potential
rimp = []
rimp.append(np.array([0.0, 0.0]))
vimp = []
vimp.append(np.array([0.0, 0.0]))
a = 1.0  #amplitude
sg = 0.1 #width
m = 10000.  #mass

rwf = np.outer(np.linspace(0, R, 100), np.ones(Nvorts))
phi = np.linspace(0, 2*np.pi, 100)

# Meshgrid
RWF, PHI = np.meshgrid(rwf[:,0], phi)
X = RWF*np.cos(PHI); x = rwf[:,0]*np.cos(phi)
Y = RWF*np.sin(PHI); y = rwf[:,0]*np.sin(phi)

RWF4 = np.stack([RWF for i in range(Nvorts)], axis=0)
wfunc = lambda pos, size: np.sqrt((RWF4-pos)**2/((RWF4-pos)**2 + size**2))
Pwf = []
F = []

MeshTemp = np.ones((Nvorts,100*100))
X_, Y_ = X-rimp[-1][0], Y-rimp[-1][1]
GradVimp = -2*a/sg**2*np.array([X_,Y_])*np.exp(-1./sg**2*(X_**2 + Y_**2))[None,:,:]

#Stir pot
omgs = 20.0
Rs = 0.5
rstir = Rs*np.array([np.cos(omgs*t), np.sin(omgs*t)]).T

t1 = time()
for i in range(nt-1):
    pairedVorts = np.array(list(it.product(r[-1], repeat=2))).reshape((Nvorts,Nvorts,2,2))
    vortDiffs = pairedVorts[:,:,0] - pairedVorts[:,:,1]
    numerators = np.cross([0,0,1], vortDiffs)[:,:,:-1]
    distsSq = np.linalg.norm(vortDiffs, axis=2)**2
    # Gaussian Impurity
    impDists = rimp[-1] - r[-1]
    GradV = -2*a/sg**2*impDists*np.exp(-1./sg**2*np.linalg.norm(impDists, axis=1)**2)[:,None]
    SympGradVimp = np.cross([0,0,1], GradV)[:,:-1]
    # Gaussian stirring potential
    GradVstir = -2*a/sg**2*(r[-1]-rstir[i])*np.exp(-1./sg**2*np.linalg.norm(r[-1]-rstir[i], axis=1)**2)[:,None]
    SympGradVstir = np.cross([0,0,1], GradVstir)[:,:-1]
    # Removal of vortex pairs of opposite signs due to annihilation
    annih = (np.sqrt(distsSq) < 2*prox) & (np.outer(circs,circs) < 0.0)
    annih = list(set(np.ravel(np.where(annih==True))))
    # Method of Images
    images = r[-1]*R*R/np.linalg.norm(r[-1], axis=1)[:,None]**2
    imDiffs = r[-1] - images
    imNums = np.cross([0,0,1], imDiffs)[:,:-1]
    imDists = np.linalg.norm(imDiffs, axis=1)
    # dissipation
    diss = 0.0#gamma * np.einsum('i,ijk->jk', 1./np.abs(circs), np.multiply(np.outer(circs,circs), (vortDiffs/distsSq[:,:,None] + core**2)))  - \
    temp = 1./(2*np.pi)*(np.einsum('i,ijk->jk', circs, \
              numerators/(distsSq[:,:,None] + core**2)) + \
                   circs[:,None] * imNums/imDists[:,None]**2 + diss)
    #temp += SympGradVimp
    temp += SympGradVstir
    r[-1] = np.delete(r[-1], annih, axis=0)
    temp = np.delete(temp, annih, axis=0)
    circs = np.delete(circs, annih)
    v.append(temp)
    Nvorts -= len(annih)
    # Euler integrator
    r.append(r[-1] + dt*v[-1])
    # Wavefunction
    theta += np.einsum('ij, ij -> i', v[-1], v[-1])*dt
    phase = np.exp(1j*theta*circs)
    rlens = np.linalg.norm(r[-1], axis=1)
    rMeshes = (MeshTemp*rlens[:,None]).reshape((Nvorts,100,100))
    pMeshes = (MeshTemp*phase[:,None]).reshape((Nvorts,100,100))
    wfunc = np.sqrt((RWF4-rMeshes)**2/((RWF4-rMeshes)**2 + core**2))*pMeshes
    wf = np.sum(wfunc, axis=0)
    Pwf.append(np.sqrt(np.real(np.einsum('ij, ij -> ij', wf.conj(), wf))))
    # Convolution + interpolation
    Fx = -sgn.fftconvolve(Pwf[-1]**2, GradVimp[0], 'same')
    Fy = -sgn.fftconvolve(Pwf[-1]**2, GradVimp[1], 'same')
    funcx = interpolate.RectBivariateSpline(rwf[:,0], phi, Fx)
    funcy = interpolate.RectBivariateSpline(rwf[:,1], phi, Fy)
    func = lambda x,y: list(funcx(x,y)[0]) + list(funcy(x,y)[0])
    # Particle update
    aimp = 1./m*np.array(func(rimp[-1][0], rimp[-1][1]))
    vimp.append(vimp[-1] + dt*aimp)
    rimp.append(rimp[-1] + dt*vimp[-1])

#    aimp = -
#    vimp = v[i] + dt*aimp
#    rimp[i+1] = rimp[i] + dt*vimp
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

# Wavefunction
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

fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.set_aspect('equal')
plt.plot(r[:,:,0], r[:,:,1])
ax2.set_xlim(-1.1*R, 1.1*R)
ax2.set_ylim(-1.1*R, 1.1*R)
ax2.set_aspect('equal')
boundary = patches.Circle((0,0), R, color='g', fill=False)
ax2.add_artist(boundary)

fig2.show()
