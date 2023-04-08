# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python (scipy)
#     language: python
#     name: scipy
# ---

# %% [markdown]
# # Numerical solution to the 1-dimensional Time Independent Schroedinger Equation
# Based on the paper "Matrix Numerov method for solving Schroedinger's equation" by Mohandas Pillai, Joshua Goglio, and Thad G. Walker, _American Journal of Physics_ **80** (11), 1017 (2012).  [doi:10.1119/1.4748813](http://dx.doi.org/10.1119/1.4748813)
#
# ## Variational method
# Using the Numerov approximation for the kinetic energy operator to calculate the expectation value of energy of a trial wavefunction.  From here, the energy may be minimized to find a variational solution.  Plot the ground state of this solution and the "true" solution, and compare their energies.
#

# %%
# import some needed libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# %matplotlib inline

# %%
#import timeit # for timing the execution

# %%
autoscale = False # set this equal to true to use Pillai's recommended step sizes

# %% [markdown]
# This example is the finite square well.
#
# Using units such that &hbar;, *m*, and *a* are one.

# %%
# values of constants
hbar = 1.0
mass = 1.0 # changing the mass will also change the energy scale
halfwidth = 1.0 # half width of the well
V0 = 18.0*np.sqrt(hbar/mass)/halfwidth**2


# %%
# bounds (These are overwritten if autoscale=True)
xmin = -10.*halfwidth # lower bound of position
xmax = 10*halfwidth # upper bound of position
n = 1000 # number of steps (may be overwritten if autoscale == True)
dx = (xmax-xmin)/(n-1)


# %%
# the function V is the potential energy function
def V(x):
    # make sure there is no division by zero
    # this also needs to be a "vectorizable" function
    # uncomment one of the examples below, or write your own.
    #return 0.5*mass*omega**2*x*x # harmonic oscillator
    return np.piecewise(x, [np.abs(x)<= halfwidth, np.abs(x)>halfwidth],[0.0,V0])
    


# %%
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

# %%
xmin, xmax, n #show the limits and number of steps

# %%
#define the x coordinates
x = np.linspace(xmin,xmax,n)

# %%
#define the numerov matrices
B = np.matrix((np.eye(n,k=-1)+10.0*np.eye(n,k=0)+np.eye(n,k=1))/12.0)
A = np.matrix((np.eye(n,k=-1)-2.0*np.eye(n,k=0)+np.eye(n,k=1))/(dx**2))

# %%
#calculate kinetic energy operator using Numerov's approximation
KE = -0.5*hbar**2/mass*B.I*A

# %%
#calculate hamiltonian operator approximation
H = KE + np.diag(V(x))

# %%
#Calculate eigenvalues and eigenvectors of H
energies, wavefunctions = np.linalg.eigh(H) # "wavefunctions" is a matrix with one eigenvector in each column.

# %%
true_gs_energy = energies[0]


# %%
#trial wavefunction
def psi(a=1.0):
    '''Wavefunction as a function of position x.
    The array "a" stores any parameters.'''
    # Note, normalization is not necessary
    #psi_unnorm = np.exp(-x**2*a) # gaussian
    #psi_unnorm = np.piecewise(x,[np.abs(x*a)<1.0,np.abs(x*a)>=1.0],[lambda x:1.0+np.cos(a*np.pi*x),0.0])# cosine**2 piece
    psi_unnorm = np.piecewise(x,[np.abs(x*a)<1.0,np.abs(x*a)>=1.0],[lambda x:np.cos(a*np.pi*x/2.0),0.0])# cosine piece
    norm = 1.0/np.sqrt(np.dot(psi_unnorm,psi_unnorm))
    return norm*psi_unnorm


# %%
def ev(psi):
    '''Calculate expectation value of energy'''
    return np.dot(np.dot(psi,H),psi)[0,0]


# %%
# minimize expectation value with respect to a
result=opt.minimize_scalar(lambda a: ev(psi(a)))


# %%
print(f"The value of a is {result.x}, with an energy of {result.fun}, compared to the true gs energy of {true_gs_energy}")

# %%
zoom = 3.0 # zoom factor for wavefunctions to make them more visible
plt.plot(x,V(x),'-k',label="V(x)") # plot the potential
plt.plot(x,zoom*np.sign(wavefunctions[n//2,0])*wavefunctions[:,0]+energies[0],label="true") #plot the num-th wavefunction
plt.plot(x,zoom*psi(result.x)+result.fun,label="variational") # plot the variational wavefunction
plt.ylim(-1,3); # set limits of vertical axis for plot
plt.xlim(-3,3); # set limits of horizontal axis for plot
plt.legend();
plt.xlabel("x");
plt.ylabel("Energy");

# %%
