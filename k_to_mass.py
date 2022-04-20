import numpy as np 
from cosmo import cosmology
cosmo=cosmology()

def k_to_m(cosmo, k, fm=1.):


    rho = fm*((cosmo.Om0/cosmo.eof**3) + (cosmo.Or0/cosmo.eof**4))*cosmo.rhoc

    return (4./3)*np.pi*np.power(cosmo.eof*(2*np.pi/k),3)*rho
