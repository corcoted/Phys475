# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %% [markdown] id="KdLipIaHUMf0"
# # Numerical solution to the 1-dimensional Time Independent Schroedinger Equation
# Based on the paper "Matrix Numerov method for solving Schroedinger's equation" by Mohandas Pillai, Joshua Goglio, and Thad G. Walker, _American Journal of Physics_ **80** (11), 1017 (2012).  [doi:10.1119/1.4748813](http://dx.doi.org/10.1119/1.4748813)
#

# %% id="Jx8eB0JjUMf1"
# import some needed libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# %matplotlib inline

# %% id="yL7rkZKfUMf2"
autoscale = False # set this equal to true to use Pillai's recommended step sizes

# %% id="DPGSodktUMf2"
# values of constants
# harmonic oscillator
hbar = 1.0
mass = 1.0 # changing the mass will also change the energy scale
omega = 1.0
# sin**2
L = np.pi/8.0
V0 = 10.0

# %% id="Ztjpz58oUMf3"
# bounds (These are overwritten if autoscale=True)
xmin = -5.0 # lower bound of position
xmax = 5.0 # upper bound of position
n = 100 # number of steps (may be overwritten if autoscale == True)
dx = (xmax-xmin)/(n-1)


# %% deletable=true editable=true id="inleHviRUMf3"
# the function V is the potential energy function
def V(x):
    # make sure there is no division by zero
    # this also needs to be a "vectorizable" function
    # uncomment one of the examples below, or write your own.
    return 0.5*mass*omega**2*x*x # harmonic oscillator
    #return np.array([np.piecewise(x,[x<=-4,x>-4 and x<4,x>=4],[V0,V0*np.sin(x*L)**2,V0]) for x in x]) # sin**2 well
    #return V0*np.array([np.piecewise(x,[np.abs(x)>=3,np.abs(x)<3],[1,0]) for x in x]) # finite square well
    #return V0*np.array([np.piecewise(x,[np.abs(x)>=3,np.abs(x)<3],[1e10,0]) for x in x]) # ~infinite square well
    #return V0/5*np.abs(x) # triangle well



# %% id="6K4uqFHXUMf4"
# apply autoscale
if (autoscale):
    #Emax is the maximum energy for which to check for eigenvalues
    Emax = 20.0
    #The next lines make some reasonable choices for the position grid size and spacing
    xt = opt.brentq(lambda x: V(x)-Emax ,0,5*Emax) #classical turning point
    dx = 1.0/np.sqrt(2*Emax) #step size
    # bounds and number of steps
    n = np.int(0.5+2*(xt/dx + 4.0*np.pi)) #number of steps
    xmin = -dx*(n+1)/2
    xmax = dx*(n+1)/2

# %% colab={"base_uri": "https://localhost:8080/"} id="E2WIlgozUMf4" outputId="0e7795ad-9cb2-4bbb-abe3-8666bb0aa50f"
#show the limits and number of steps
xmin, xmax, n

# %% id="CG_UIOMXUMf5"
#define the x coordinates
x = np.linspace(xmin,xmax,n)

# %% id="5abw45WjUMf6"
#define the numerov matrices
B = np.array((np.eye(n,k=-1)+10.0*np.eye(n,k=0)+np.eye(n,k=1))/12.0)
A = np.array((np.eye(n,k=-1)-2.0*np.eye(n,k=0)+np.eye(n,k=1))/(dx**2))

# %% deletable=true editable=true id="QyrmXQCxUMf7"
#calculate kinetic energy operator using Numerov's approximation
KE = -0.5*hbar**2/mass*np.linalg.inv(B)@A

# %% id="mtA5geFXUMf7"
#calculate hamiltonian operator approximation
H = KE + np.diag(V(x))

# %% id="_i-j1lyRUMf8"
#Calculate eigenvalues and eigenvectors of H
energies, wavefunctions = np.linalg.eigh(H) # "wavefunctions" is a matrix with one eigenvector in each column.

# %% colab={"base_uri": "https://localhost:8080/"} id="q4IMVz7bUMf8" outputId="6dd9cd08-f439-4974-a062-d3cd2478a15c"
#display the lowest five energies
energies[0:5]

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="shAr0qKgUMf8" outputId="a1460968-e5c8-47c7-9c0a-ec0700133154"
# plot the potential and a few wavefunctions
number = [0,1,2,3,4] #which wavefunctions to plot, starting counting from zero
zoom = -3.0 # zoom factor for wavefunctions to make them more visible
plt.plot(x,V(x),'-k',label="V(x)") # plot the potential
for num in number:
    plt.plot(x,zoom*wavefunctions[:,num]+energies[num],label=num) #plot the num-th wavefunction
plt.legend();
plt.xlabel("x");
plt.ylabel("Energy");
plt.ylim(-1,12); # set limits of vertical axis for plot
