import numpy as np

import math

from tqdm import tqdm_notebook as tqdm

from math import log10
from math import log
from math import pi
from math import sqrt
from math import exp

from scipy.special import erf


from cosmo import cosmology
from constants import km_to_Mpc, G,c


from constants import km_to_Mpc, G,c
from constants import A0,k0,A1f


def dndm_spiky_broken_log(u, a, cosmo, a_fct = 2.0485109441035904e-26, M_star = 1e-1, kpiv=10,ns = 0.9649,nb=2, Mc=1, A1=A1f, A2=1,epsilon=1e-2,fm = 1., iscomoving = True, dndMlog = False):
    u = np.asarray([u]) if np.isscalar(u) else np.asarray(u)
    M = np.power(10.,u)
    
    
    rho = ((cosmo.Om0/a_fct**3) + (cosmo.Or0/a_fct**4))*cosmo.rhoc

    Cfct=a_fct*np.power((32. * (np.pi**4.) *rho*fm/3.0),1./3.)
    
    
    
    rho_pbh = (cosmo.Odm0 * cosmo.rhoc)
    
    
    Mpiv=np.power(Cfct/kpiv,3.0)
   
    
    rA= A2/A1
    
    fA = (rA*(Cfct**3)*Mc**(-1))
     
   
    ns3 = ns + 3.
    
    nb3 = nb + 3.0
    
    nsb = ns-nb
    
    ns3d = -ns3/3.0
    
    nb3d = -nb3/3.0

    nb9 = -(nb +9.)/3.0
    
    Mcn = 1.0/Mc
    
    x1mpiv = -nsb*np.power(Cfct,nb3)*np.power(Mpiv,nb3d)
    
    x1ms = ns3*np.power(Cfct,nb3)*np.power(M_star,nb3d)
    
    x1m = ns3*np.power(Cfct,nb3)*np.power(M,nb3d)

    x2 = (nb3*ns3)*np.power(Cfct,nb3)*np.power(M,nb9)
                                              
    x3 = ns3*nb3*np.power(Cfct,-ns+nb)/np.power(Mpiv,-nsb/3)

    argms= (np.log(Cfct*Mc**(-1./3.))+(3.*epsilon**2)-np.log(Cfct*M_star**(-1./3.)))/(np.sqrt(2.)*epsilon)
    
    argm= (np.log(Cfct*Mc**(-1./3.))+(3.*epsilon**2)-np.log(Cfct*M**(-1./3.)))/(np.sqrt(2.)*epsilon)
    
    em= np.exp(-argm**2)
    
    erfms= erf(argms)
    
    erfm= erf(argm)
    
    eeps= np.exp((9./2.)*epsilon**2)
    
    s3m= ((fA*eeps*x3)/(epsilon*np.sqrt(2.*np.pi)*M**2))*em
    
    s4ms= (x3*fA/2.)*eeps*(1.0-erfms)
    
    s4m= (x3*fA/2.)*eeps*(1.0-erfm)
    
    nums= np.power(x1mpiv + x1ms + s4ms, 1./2)
    
    
    num= np.power(x1mpiv + x1m + s4m, 1./2)
    
    nu= nums/num
                              
    n1=(1.0/(3.0*np.sqrt(2.*np.pi)))*rho_pbh
    
    n2= nums/num**3.
    
    n3= x2+s3m

    n4=np.exp(-0.5*nu**2)

    
    if dndMlog == False:
        res= n1*n2*n3*n4

    elif dndMlog == True:

        res= np.log(10)*M*n1*n2*n3*n4
    
    if iscomoving == False:

        res = np.float_power(a,-3.) * res
    
    return res



def dndm_spiky_broken_log_std(u, a, cosmo, a_fct = 1e-25, M_star = 1e-1, kpiv=10,ns = 1.,nb=2, Mc=1, A1=A1f, A2=1,epsilon=1e-2,fm = 1., iscomoving = True, dndMlog = False):
    u = np.asarray([u]) if np.isscalar(u) else np.asarray(u)
    M = np.power(10.,u)
    
    
    rho = ((cosmo.Om0/a_fct**3) + (cosmo.Or0/a_fct**4))*cosmo.rhoc

    Cfct=a_fct*np.power((32. * (np.pi**4.) *rho*fm/3.0),1./3.)
    
    
    rho_pbh = (cosmo.Odm0 * cosmo.rhoc)
    
   
    
    Mpiv=np.power(Cfct/kpiv,3.0)
    
    rA= A2/A1
    
    fA = (rA*Cfct**3*Mc**(-1))
     
   
    ns3 = ns + 3.
    
    nb3 = nb + 3.0
    
    nsb = ns-nb
    
    ns3d = -ns3/3.0
    
    nb3d = -nb3/3.0

    nb9 = -(nb +9.)/3.0
    
    Mcn = 1.0/Mc
    
    x1mpiv = -nsb*np.power(Cfct,nb3)*np.power(Mpiv,nb3d)
    
    x1ms = ns3*np.power(Cfct,nb3)*np.power(M_star,nb3d)
    
    x1m = ns3*np.power(Cfct,nb3)*np.power(M,nb3d)

    x2 = (nb3*ns3)*np.power(Cfct,nb3)*np.power(M,nb9)
                                              
    x3 = ns3*nb3*np.power(Cfct,-ns+nb)/np.power(Mpiv,-nsb/3)

    argms= (np.log(Cfct*Mc**(-1./3.))+(3.*epsilon**2)-np.log(Cfct*M_star**(-1./3.)))/(np.sqrt(2.)*epsilon)
    
    argm= (np.log(Cfct*Mc**(-1./3.))+(3.*epsilon**2)-np.log(Cfct*M**(-1./3.)))/(np.sqrt(2.)*epsilon)
    
    em= np.exp(-argm**2)
    
    erfms= erf(argms)
    
    erfm= erf(argm)
    
    eeps= np.exp((9./2.)*epsilon**2)
    
    s3m= ((fA*eeps*x3)/(epsilon*np.sqrt(2.*np.pi)*M**2))*em
    
    s4ms= (x3*fA/2.)*eeps*(1.0-erfms)
    
    s4m= (x3*fA/2.)*eeps*(1.0-erfm)
    
    nums= np.power(x1mpiv + x1ms + s4ms, 1./2)
   
    
    num= np.power(x1mpiv + x1m + s4m, 1./2)
    
    nu= nums/num
                              
    n1=(1.0/(3.0*np.sqrt(2.*np.pi)))*rho_pbh
    
    n2= nums/num**3.
    
    n3= x2

    n4=np.exp(-0.5*nu**2)

    
    
    if dndMlog == False:
        res= n1*n2*n3*n4

    elif dndMlog == True:

        res= np.log(10)*M*n1*n2*n3*n4
    
    if iscomoving == False:

        res = np.float_power(a,-3.) * res
    
    return res
    
    
def dndm_spiky_broken_log_spi(u, a, cosmo, a_fct = 1e-25, M_star = 1e-1, kpiv=10,ns = 1.,nb=2, Mc=1, A1=A1f, A2=1,epsilon=1e-2,fm = 1., iscomoving = True, dndMlog = False):
    u = np.asarray([u]) if np.isscalar(u) else np.asarray(u)
    M = np.power(10.,u)
    
    
    rho = ((cosmo.Om0/a_fct**3) + (cosmo.Or0/a_fct**4))*cosmo.rhoc

    Cfct=a_fct*np.power((32. * (np.pi**4.) *rho*fm/3.0),1./3.)
    
  
    
    rho_pbh = (cosmo.Odm0 * cosmo.rhoc)
    
    
    Mpiv=np.power(Cfct/kpiv,3.0)
    
    rA= A2/A1
    
    fA = (rA*(Cfct**3)*Mc**(-1))
     
    
   
    ns3 = ns + 3.
    
    nb3 = nb + 3.0
    
    nsb = ns-nb
    
    ns3d = -ns3/3.0
    
    nb3d = -nb3/3.0

    nb9 = -(nb +9.)/3.0
    
    Mcn = 1.0/Mc
    
    x1mpiv = -nsb*np.power(Cfct,nb3)*np.power(Mpiv,nb3d)
    
    x1ms = ns3*np.power(Cfct,nb3)*np.power(M_star,nb3d)
    
    x1m = ns3*np.power(Cfct,nb3)*np.power(M,nb3d)

    x2 = (nb3*ns3)*np.power(Cfct,nb3)*np.power(M,nb9)
                                              
    x3 = ns3*nb3*np.power(Cfct,-ns+nb)/np.power(Mpiv,-nsb/3)

    argms= (np.log(Cfct*Mc**(-1./3.))+(3.*epsilon**2)-np.log(Cfct*M_star**(-1./3.)))/(np.sqrt(2.)*epsilon)
    
    argm= (np.log(Cfct*Mc**(-1./3.))+(3.*epsilon**2)-np.log(Cfct*M**(-1./3.)))/(np.sqrt(2.)*epsilon)
    
    em= np.exp(-argm**2)
    
    erfms= erf(argms)
    
    erfm= erf(argm)
    
    eeps= np.exp((9./2.)*epsilon**2)
    
    s3m= em*((fA*eeps*x3)/(epsilon*np.sqrt(2.*np.pi)*M**2))
    
    s4ms= (x3*fA/2.)*eeps*(1.0-erfms)
    
    s4m= (x3*fA/2.)*eeps*(1.0-erfm)
    
    nums= np.power(x1mpiv + x1ms + s4ms, 1./2)
    
    
    num= np.power(x1mpiv + x1m + s4m, 1./2)
    
    nu= nums/num
                              
    n1=(1.0/(3.0*np.sqrt(2.*np.pi)))*rho_pbh
    
    n2= nums/num**3.
    
    n3= s3m

    n4=np.exp(-0.5*nu**2)

    
    
    if dndMlog == False:
        res= n1*n2*n3*n4

    elif dndMlog == True:

        res= np.log(10)*M*n1*n2*n3*n4
    
    if iscomoving == False:

        res = np.float_power(a,-3.) * res
    
    return res
