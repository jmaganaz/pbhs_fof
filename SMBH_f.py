import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

import SMBH_MF


def init_AGN_MF(cosmo):

    global N_AGN_of_MC, Mc_values

    SMBH_MF.gen_interp(cosmo)
    Mc_values = np.linspace(7,10.2,50)
    N_AGN_of_MC = np.vectorize(SMBH_MF.N_SMBH)(lo=Mc_values)   

def init_dMdHI(cosmo):

    global mmin_halo, mmax_halo, dMdHI

    mhalo=[]
    dndmhalo=[]
    with open('Halo_data/dndmhalo.dat') as fdat:
        for line in fdat:
            cols = [float(x) for x in line.split()]
            mf2,ff2 = cols[0], cols[5]
            mhalo.append(mf2*cosmo.h)
            dndmhalo.append(ff2/cosmo.h**4)

    mmin_halo=min(mhalo)
    mmax_halo=max(mhalo)

    #print('Mmin_Halo = %e  ;  Mmax_Halo = %e'%(mmin_halo,mmax_halo))

    dndmhalo_interp = interp1d(mhalo, dndmhalo) # funcion de masa de Halos

    def dMdHI(M):
        '''
        funcion de masa de Halos
        '''
        if M < mmin_halo or M > mmax_halo:
            return 0
        else:
            return dndmhalo_interp(M)

    return dMdHI

def N_Halos(lo=7,hi=20):  
    
    '''
    Cumulative number density of DM halos
    
    receives the lower limit as log(Mh)
    
    '''
    
    def g_m_integrand(u):
        
        M = np.power(10.,u)
        
        f1 = dMdHI(M)
        
        f2= M*np.log(10)
        
        result = f1*f2
        
        return result
    
    if np.isnan(lo):
        
        return 0
    
    numerator=quad(g_m_integrand,lo,hi)[0]
    
    return numerator


def V_h(cosmo,M_h):
    
    '''
    Halo volume in Mpc^3 as funciton of the mass of the halo
    
    '''
    
    rho = cosmo.rhoc * cosmo.Odm0
    
    return M_h/rho

def MhofMc(MF,Mc):

    '''
    Halo Mass as function of the central PBH mass.
    '''

    #(cosmo,a=1.,Ms=1e-1, nb=2.,An=1.,fm=1.,M1ph=1.,Mc=8):

    
    rho = MF.cosmo.rhoc * MF.cosmo.Odm0
    
    Npbhc = MF.number_density_trapz(Mc,MF.M1ph[0.0])
    
    #nden_HC(cosmo,1., Ms, nb=nb, An=An,fm = fm,lo = Mc, hi = M1ph)
    
    #print("Npbhc =",Npbhc)
    
    try: 
        
        Mh = rho/Npbhc
        
        if Mh > mmax_halo or Mc > MF.M1ph[0.0]:
            
            # si Mh > mmax_halo -> no existe un halo para un pbh de masa Mc -> no existe acumulativa
            
            Mh = np.nan
            
    except ZeroDivisionError:
        
        Mh = np.nan
        
    return Mh

def f(MF):
    
    '''
    receives a Mass function
    '''

    Mh = np.vectorize(MhofMc)(MF,Mc_values)

    NPBHC = np.vectorize(N_Halos)(np.log10(Mh))
    
    fmin = np.nanmin(N_AGN_of_MC/NPBHC)
        
    if np.isinf(fmin) == True:
        
        # Si es infinito no hay restriccion
        
        fmin = 1e11
        #fmin = np.nan
        
    return fmin
    
