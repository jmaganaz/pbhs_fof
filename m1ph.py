import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from tqdm import tqdm_notebook as tqdm

def M1ph(a, MF, progress_bar = False):

    '''
    Computes the mass where the cumulative number density of the mass function equals 
    one per horizon volume.

    Compares the inverse of the comoving volume with the cumulative number density from a comoving mass function.
    '''

    cosmo = MF.cosmo # We take the cosmology from the Mass Function.

    V_h = cosmo.V_h(a,False)

    hi_mass = np.log10(MF.M_star*1e10)
    #loM_vals = np.arange(-50,hi_mass,0.001)
    
    #loM_vals = np.arange(-30,hi_mass,0.1)

    #Nden_values = np.zeros((len(loM_vals)))

    if progress_bar == False:

        #for i in range(len(loM_vals)): # Compute the cumulative Mass Function
        
        loM_vals,Nden_values = MF.number_density(-50,hi_mass,a)[0],MF.An*MF.number_density(-50,hi_mass,a)[1]
        #print('lowm',loM_vals)
            
        #if np.isinf(Nden_values.any()):
         #   print('inf value')

                # This condition is included to avoid problems where we integrate zero
            
          #  Nden_values[i] = np.nan

    else:

        #for i in tqdm(range(len(loM_vals))): # Compute the cumulative Mass Function
        
            loM_vals,Nden_values = np.log10(MF.number_density(-30,hi_mass,a))

            #if np.isinf(Nden_values[i]):

                # This condition is included to avoid problems where we integrate zero
            
             #   Nden_values[i] = np.nan

    # Perform an interpolation of n(>M) (Cumulative MF as a function of M).

    interpolation = interp1d(np.log10(Nden_values),loM_vals,fill_value = "extrapolate")

    result = interpolation(np.log10(1./V_h))

    return float(result)


def generate(MF, progress_bar = False):

    '''
    Computes the M1ph for the redshifts values specified in the key for each entry.
    '''

    M1phdict = { 0 : M1ph(1., MF, progress_bar), 
                1 : M1ph(0.5, MF, progress_bar),
                450 : M1ph(1/451, MF, progress_bar),
                1160.72 : M1ph(1/1161, MF, progress_bar),
                1e10 : M1ph(1/(1+1e10), MF, progress_bar)}
    #print(M1phdict)
    return M1phdict

def M1ph_function(MF, progress_bar = False):

    a_array = np.logspace(-30,0,50)

    z_array = 1/a_array - 1

    M1ph_array = np.zeros(len(a_array))

    for i,a in tqdm(enumerate(a_array)):

        M1ph_array[i] = M1ph(a,MF)

    return interp1d(z_array,M1ph_array)    
