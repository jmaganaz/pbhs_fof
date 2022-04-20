import numpy as np
import mpmath
import math

#import mpld3
from tqdm import tqdm_notebook as tqdm

from math import log10
from math import log
from math import pi
from math import sqrt
from math import exp

from scipy.special import erf

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

from cosmo import cosmology
from constants import km_to_Mpc, G,c

from constants import A0,k0,A1f

    
def dndm_spiky_broken_log(u, a, cosmo, a_fct = 1e-25, M_star = 1e-1, kpiv=10,ns = 0.9649,nb=2, Mc=1, A1=A1f, A2=1e20,epsilon=1e-2,fm = 1., iscomoving = True, dndMlog = False):
    u = np.asarray([u]) if np.isscalar(u) else np.asarray(u)
    #M = np.power(10.,u)
    M = np.power(10.,u)
    
    H0 = cosmo.h*100*km_to_Mpc # In s**-1
    
    cc=(2.*H0*np.sqrt(cosmo.Or0)*np.pi*np.pi*c*fm)/G #B2^2
    
    Chc=np.power(cc,1./2.)
    
    rho_pbh = (cosmo.Odm0 * cosmo.rhoc)
    
    Mpiv=np.power(Chc/kpiv,2.0)
    
    #print(Mpiv)
    
    rA= A2/A1
    
    fA = (rA*Chc**3*Mc**(-3./2.))
     
    ns3 = ns + 3.
    
    nbm1= nb-1.0
    
    nb3 = nb + 3.0
    
    nsb = ns-nb
    
    ns3d = -ns3/3.0
    
    nb1d = -nbm1/2.0
    
    nb2d = -nb3/2.0

    nb9 = -(nb +9.)/3.0
    
    Mcn = 1.0/Mc
    
    x1mpiv = 2.*nsb*np.power(Chc,nb3)*np.power(Mpiv,nb2d)
    
    x1mpivms = x1mpiv*M_star**2.
    
    x1mpivm = x1mpiv*M**2.
    
    x1ms = ns3*np.power(Chc,nb3)*np.power(M_star,nb1d)
    
    x1m = ns3*np.power(Chc,nb3)*np.power(M,nb1d)

    x2 = (ns3*nb1d)*np.power(Chc,nb3)*np.power(M,nb2d)
                                              
    x3 = ns3*nb3*np.power(Chc,nb-ns)/np.power(Mpiv,-nsb/2.)

    argms= (np.log(Chc*Mc**(-1./2.))+(3.*epsilon**2)-np.log(Chc*M_star**(-1./2.)))/(np.sqrt(2.)*epsilon)
    
    argm= (np.log(Chc*Mc**(-1./2.))+(3.*epsilon**2)-np.log(Chc*M**(-1./2.)))/(np.sqrt(2.)*epsilon)
    
    em= np.exp(-argm**2)
    
    erfms= erf(argms)
    
    erfm= erf(argm)
    
    eeps= np.exp((9./2.)*epsilon**2)
    
    s3m= ((fA*eeps*x3)/(2.*epsilon*np.sqrt(2.*np.pi)*M**2))*em
    
    s4ms= (x3*fA/2.)*eeps*(1.0-erfms)
    
    s4m= (x3*fA/2.)*eeps*(1.0-erfm)
    
    nums= np.power(x1mpivms + x1ms + s4ms, 1./2)
    
    #print('nums',nums)
    
    num= np.power(x1mpivm + x1m + s4m, 1./2)
    
    nu= nums/num
                              
    n1=(1.0/(np.sqrt(2.*np.pi)))*rho_pbh
    
    n2= nums/num**3.
    
    n3= x1mpiv-x2+s3m

    n4=np.exp(-0.5*nu**2)

    if dndMlog == False:
        res= n1*n2*n3*n4

    elif dndMlog == True:

        res= np.log(10)*M*n1*n2*n3*n4
    
    
    res = np.where(np.isnan(res), 0, res)
    if iscomoving == False:

        res = np.float_power(a,-3.) * res
        
    
    return res
    
def dndm_spiky_broken_log_spi(u, a, cosmo, a_fct = 1e-25, M_star = 1e-1, kpiv=10,ns = 1.,nb=2, Mc=1, A1=1, A2=1,epsilon=1e-2,fm = 1., iscomoving = True, dndMlog = False):
    u = np.asarray([u]) if np.isscalar(u) else np.asarray(u)
    #M = np.power(10.,u)
    M = np.power(10.,u)
    
    H0 = cosmo.h*100*km_to_Mpc # In s**-1
    
    cc=(2.*H0*np.sqrt(cosmo.Or0)*np.pi*np.pi*c*fm)/G #B2^2
    
    Chc=np.power(cc,1./2.)
    
    rho_pbh = (cosmo.Odm0 * cosmo.rhoc)
    
    Mpiv=np.power(Chc/kpiv,2.0)
    
    #print(Mpiv)
    
    rA= A2/A1
    
    fA = (rA*Chc**3*Mc**(-3./2.))
     
    ns3 = ns + 3.
    
    nbm1= nb-1.0
    
    nb3 = nb + 3.0
    
    nsb = ns-nb
    
    ns3d = -ns3/3.0
    
    nb1d = -nbm1/2.0
    
    nb2d = -nb3/2.0

    nb9 = -(nb +9.)/3.0
    
    Mcn = 1.0/Mc
    
    x1mpiv = 2.*nsb*np.power(Chc,nb3)*np.power(Mpiv,nb2d)
    
    x1mpivms = x1mpiv*M_star**2.
    
    x1mpivm = x1mpiv*M**2.
    
    x1ms = ns3*np.power(Chc,nb3)*np.power(M_star,nb1d)
    
    x1m = ns3*np.power(Chc,nb3)*np.power(M,nb1d)

    x2 = (ns3*nb1d)*np.power(Chc,nb3)*np.power(M,nb2d)
                                              
    x3 = ns3*nb3*np.power(Chc,nb-ns)/np.power(Mpiv,-nsb/2.)

    argms= (np.log(Chc*Mc**(-1./2.))+(3.*epsilon**2)-np.log(Chc*M_star**(-1./2.)))/(np.sqrt(2.)*epsilon)
    
    argm= (np.log(Chc*Mc**(-1./2.))+(3.*epsilon**2)-np.log(Chc*M**(-1./2.)))/(np.sqrt(2.)*epsilon)
    
    em= np.exp(-argm**2)
    
    erfms= erf(argms)
    
    erfm= erf(argm)
    
    eeps= np.exp((9./2.)*epsilon**2)
    
    s3m= ((fA*eeps*x3)/(2.*epsilon*np.sqrt(2.*np.pi)*M**2))*em
    
    s4ms= (x3*fA/2.)*eeps*(1.0-erfms)
    
    s4m= (x3*fA/2.)*eeps*(1.0-erfm)
    
    nums= np.power(x1mpivms + x1ms + s4ms, 1./2)
    
    #print('nums',nums)
    
    num= np.power(x1mpivm + x1m + s4m, 1./2)
    
    nu= nums/num
                              
    n1=(1.0/(np.sqrt(2.*np.pi)))*rho_pbh
    
    n2= nums/num**3.
    
    n3= s3m

    n4=np.exp(-0.5*nu**2)

    
    if dndMlog == False:
        res= n1*n2*n3*n4

    elif dndMlog == True:

        res= np.log(10)*M*n1*n2*n3*n4
    
    #if np.isnan(res)==True:
     #   res=0
    res = np.where(np.isnan(res), 0, res)
    
    if iscomoving == False:

        res = np.float_power(a,-3.) * res
    
    return res


def dndm_spiky_broken_log_std(u, a, cosmo, a_fct = 1e-25, M_star = 1e-1, kpiv=10,ns = 1.,nb=2, Mc=1, A1=1, A2=1,epsilon=1e-2,fm = 1., iscomoving = True, dndMlog = False):
    u = np.asarray([u]) if np.isscalar(u) else np.asarray(u)
    #M = np.power(10.,u)
    M = np.power(10.,u)
    
    H0 = cosmo.h*100*km_to_Mpc # In s**-1
    
    cc=(2.*H0*np.sqrt(cosmo.Or0)*np.pi*np.pi*c*fm)/G #B2^2
    
    Chc=np.power(cc,1./2.)
    
    rho_pbh = (cosmo.Odm0 * cosmo.rhoc)
    
    Mpiv=np.power(Chc/kpiv,2.0)
    
    #print(Mpiv)
    
    rA= A2/A1
    
    fA = (rA*Chc**3*Mc**(-3./2.))
     
    ns3 = ns + 3.
    
    nbm1= nb-1.0
    
    nb3 = nb + 3.0
    
    nsb = ns-nb
    
    ns3d = -ns3/3.0
    
    nb1d = -nbm1/2.0
    
    nb2d = -nb3/2.0

    nb9 = -(nb +9.)/3.0
    
    Mcn = 1.0/Mc
    
    x1mpiv = 2.*nsb*np.power(Chc,nb3)*np.power(Mpiv,nb2d)
    
    x1mpivms = x1mpiv*M_star**2.
    
    x1mpivm = x1mpiv*M**2.
    
    x1ms = ns3*np.power(Chc,nb3)*np.power(M_star,nb1d)
    
    x1m = ns3*np.power(Chc,nb3)*np.power(M,nb1d)

    x2 = (ns3*nb1d)*np.power(Chc,nb3)*np.power(M,nb2d)
                                              
    x3 = ns3*nb3*np.power(Chc,nb-ns)/np.power(Mpiv,-nsb/2.)

    argms= (np.log(Chc*Mc**(-1./2.))+(3.*epsilon**2)-np.log(Chc*M_star**(-1./2.)))/(np.sqrt(2.)*epsilon)
    
    argm= (np.log(Chc*Mc**(-1./2.))+(3.*epsilon**2)-np.log(Chc*M**(-1./2.)))/(np.sqrt(2.)*epsilon)
    
    em= np.exp(-argm**2)
    
    erfms= erf(argms)
    
    erfm= erf(argm)
    
    eeps= np.exp((9./2.)*epsilon**2)
    
    s3m= ((fA*eeps*x3)/(2.*epsilon*np.sqrt(2.*np.pi)*M**2))*em
    
    s4ms= (x3*fA/2.)*eeps*(1.0-erfms)
    
    s4m= (x3*fA/2.)*eeps*(1.0-erfm)
    
    nums= np.power(x1mpivms + x1ms + s4ms, 1./2)
    
    #print('nums',nums)
    
    num= np.power(x1mpivm + x1m + s4m, 1./2)
    
    nu= nums/num
                              
    n1=(1.0/(np.sqrt(2.*np.pi)))*rho_pbh
    
    n2= nums/num**3.
    
    n3= x1mpiv-x2

    n4=np.exp(-0.5*nu**2)

    
    if dndMlog == False:
        res= n1*n2*n3*n4

    elif dndMlog == True:

        res= np.log(10)*M*n1*n2*n3*n4

    #if np.isnan(res)==True:
     #   res=0
    res = np.where(np.isnan(res), 0, res)
    if iscomoving == False:

        res = np.float_power(a,-3.) * res
    
    return res

