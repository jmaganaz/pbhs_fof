import numpy as np 
from scipy.integrate import quad
from scipy.optimize import root
from scipy.interpolate import interp1d
from constants import km_to_Mpc, G,c
from evaporation import M_ev

class cosmology:

    '''
    Cosmology Class.

    It contains the usual cosmological parameters aswell as some functions and quantities
    that are useful in the context of primordial black holes.
    '''
    
    def __init__(self, Om0 = 0.315, Odm0 = 0.264, Or0 = 9.237e-5, h = 0.6736, printing = True):

        self.Om0 = Om0
        self.Odm0 = Odm0
        self.Or0 = Or0
        self.h = h
        
        self.rhoc = self.__rhocrit_f()
        self.Meq = self.horizon_mass(3.2e-4,iscomoving=False) # Horizon Mass at z_eq

        self.Mev = self.__init__Mev() # Mev as function of z
        self.eof = self.a_eof()

        if printing == True:

            print('Cosmology defined with:\nOm0 = %.3f  ;  Odm0 = %.3f  ;  Or0 = %.3e  ;  h = %.4f'%(self.Om0,self.Odm0,self.Or0,self.h))

    def __init__Mev(self):

        '''
        It creates an interpolation of the Evaporated mass as a function of the redshift M_ev(z)
        '''

        a_array = np.logspace(-30,0,100)

        z_array = 1/a_array - 1
        
        age_array = np.vectorize(self.age)(a_array)

        Mev_z = np.vectorize(M_ev)(age_array, Solar=True)

        function = interp1d(z_array,Mev_z)

        return function


    def __rhocrit_f(self):

        '''

        Caluculo de Rho Critico como funcion de H0  

        Rho critico hoy.

        [rho_c] = Solar Mass  Mpc**-3

        '''

        H = 100 * self.h

        H0 = H*km_to_Mpc # In s**-1

        rho_crit = 3*H0**2 / (8*np.pi*G) #Solar Mass Mpc**-3

        return rho_crit

    def rho_tot(self,a=1.):

        '''
        Total energy density of the Universe in units of Solar Mass per Mpc^3 
        
        '''

        rho = (self.Om0 * np.power(a,-3) + self.Or0 * np.power(a,-4) + (1. - self.Om0 - self. Or0)) * self.rhoc

        return rho

    def Hubble(self,a=1.):

        '''
        Returns the Hubble parameter at a in units of s^-1.
        
        '''

        H0 = 100 * self.h

        H = H0 * km_to_Mpc * np.sqrt( self.Or0 * (a**(-4)) + self.Om0 * (a**(-3)) + (1. - self.Om0 - self.Or0) ) # Hubble Parameter at a scale factor (a) in (s^-1)

        return H

    def a_eof(self):

        '''
        Returns the scale factor corresponding to the end of inflation for this cosmology
        '''

        def eq(loga):
    
            a = np.power(10.,loga)
    
            return np.log10(self.age(a)) + 32.

        return np.power(10.,root(eq,x0=-25)['x'][0])

    
    def age(self, a, Years = False):


        '''
        Computes the Age of the Universe, assuming a flat Universe.

        Input:  - Scale factor a to compute the age of the Universe at that epoch.
                - Components defining the cosmology of the Universe. Or0, Om0, H0
                - Can transform the age in years by selecting Years = True

        Returns: The age of the Universe in Seconds (or Years) on a scale factor a using the defined cosmology.
        
        '''
        H_0 = self.h * 100

        Ol0 = 1. - self.Om0 - self.Or0
        
        
        #km=1/(3.0856)*10**(-19)
        
        Units = (1/(H_0*km_to_Mpc)) # Age of the Universe in Seconds
        
        if Years == True:

            j = 1/(3.154e7) #Years in a second
            
            Units = (j/(H_0*km_to_Mpc)) # Age of the Universe in Years
            
        def integrand(u):
        
            return 1./np.sqrt(self.Or0/(u**2) + self.Om0/u + Ol0*u**2)
        
        try:
            integral = quad(integrand,0,a)[0]
        
        except ZeroDivisionError:
            
            print('The scale factor a cannot be equal to zero\n')
            
            print('It will be replaced by 10**(-40)')
            
            integral = quad(integrand,0,1e-40)[0]
            
            
        return Units*integral

    def R_h(self,a,iscomoving = True):

        '''
        Computes the complete horizon radius in Mpc at a certain scale factor a. Chi

        By default is the comoving horizon radius but can be the physical one by setting "iscomoving = False" 
        
        '''


        def integrand(a):

            return 1./((a**2)*self.Hubble(a))

        Chi = quad(integrand,0,a)[0] * c

        if iscomoving == False:

            Chi = a * Chi

        return Chi

    def V_h(self, a, iscomoving = True):

        '''
        Computes the complete horizon Volume in Mpc^3 at a certain scale factor a.

        By default is the comoving horizon Volume but can be the physical one by setting "iscomoving = False" 
        
        '''

        result = (4*np.pi/3) * self.R_h(a,iscomoving)**3

        return result


    def horizon_mass(self, a, iscomoving = True):

        '''
        This function computes the Maximun Mass that can be contained in a Hubble Radius as funtion of the scale factor "a".
        '''

        rhohor = (self.Or0/(a**4) + self.Om0/(a**3)) * self.rhoc

        if iscomoving == True:

            result = rhohor * (4*np.pi/3) * self.R_h(a)**3

        else:

            result = rhohor * (4*np.pi/3) * self.R_h(a,iscomoving)**3

        return result
