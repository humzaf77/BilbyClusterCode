import numpy as np

def chi(f, eps, f0, gamma):
    w  = 2*np.pi*f
    w0 = 2*np.pi*f0
    G  = 2*np.pi*gamma
    return eps * w0**2 / (w0**2 - w**2 - 1j*G*w)

def propagate(hf, f, eps, f0, gamma, D_sec):
    w = 2*np.pi*f
    return hf * np.exp(1j * w * D_sec * chi(f, eps, f0, gamma))