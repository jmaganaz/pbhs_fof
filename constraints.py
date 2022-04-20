import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from corrections import CM,Cz


class g:

    '''
    This class generates an object with the charasteristics of a particular g(M) at a particular redshift.
    Defining its particular function for the constrained mass for each formation scenario FCT and HC.
    '''
    
    def __init__(self,File,cosmo):

        Path = "gM/%s"%File

        self.z = self.__read_z__(Path)
        
        self.DATA = np.loadtxt(Path).transpose()
        
        self.g_int = interp1d(self.DATA[0],self.DATA[1],bounds_error=False,fill_value=0.)
        
        self.mmin = np.nanmin(self.DATA[0])
        
        self.mmax = np.nanmax(self.DATA[0])
        
        self.gmax = np.nanmax(self.DATA[1])
        
        self.cosmo = cosmo
        
        self.a = 1/(1+self.z)
        
        self.Mev = cosmo.Mev(self.a)

    def __read_z__(self,File):

        with open(File) as f:
    
            z_const = float(f.readline()[4:])

        return z_const

    def __repr__(self):

        string = 'g defined at z = %r'%self.z
        return string
        
    def __call__(self,M):
            
        return self.g_int(M)
    
    def plot(self):
        
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        
        ax.loglog(self.DATA[0],self.DATA[1])
        
        ax.set_xlim((self.mmin,self.mmax))
        
        ax.set_xlabel(r'$M$ [$M_\odot$]')
        ax.set_ylabel(r'$g(M)$')
        
        ax.grid(alpha=0.2)
        
        plt.show()
    
    def avg(self,MF):

        '''
        Computes the average output function <g(M)> (Eq. 71) weighted by the mass function. 
        '''

        M1ph = MF.M1ph[self.z]
        Mev = self.Mev

        lo = np.nanmax([np.log10(self.mmin),np.log10(Mev)])
        hi = np.nanmin([np.log10(self.mmax),M1ph])

        if hi < lo:

            # If both, numerator and denominator, are zero, then we do not have PBHs
            # to constraint, hence, this does not make sense. 

            return np.nan

        
        
        vec_M=np.linspace(lo,hi,10000)
        y0=MF.dndM([vec_M,MF.M_star, MF.nb,MF.Mc,MF.A2])
        gM4=self(10**vec_M)
        
        y1 = np.asarray(y0)*np.asarray(np.power(10,vec_M)*np.log(10))*np.asarray(gM4)
        Mint=trapz(y1,vec_M)
        num=Mint
        den=MF.number_density_trapz(lo,hi)

        try:

            result = num/den
            #print(result)
            
            if np.isinf(result):

                # This could happen if the denominator is very small compared to the numerator.
                # Then, we choose an arbitrary large number.
                
                #print('Is infinity')
                
                result = 1e100
            
            elif num == 0. or result < 1e-200:

                # If this is too small, we fix a small number that can be plotted in a log scale plot
            
                result = 1e-200
            
        except ZeroDivisionError:
            
            if num == 0:

                # If both, numerator and denominator, are zero, then we do not have PBHs
                # to constraint, hence, this does not make sense. 
                
                #print('ZeroDivError num = 0')
                
                result = np.nan # 0/0
            
            else:

                # If the denominator is zero, then it is equivalent to have <g(M)> = infty.
                
                #print('ZeroDiv infty')
                
                result = 1e100

        return result


class f:

    def __init__(self, MF, g):

        self.g = g
        self.MF = MF
        self.z = g.z

        self.CM = CM(self.MF,self.g)
        self.Cz = Cz(self.MF,self.z)
        self.gavg = g.avg(self.MF)
        #print(self.gavg)
        self.value = self.compute()

    def compute(self):

        '''
        It computes the value of the allowed fraction f for a particular mass function
        and observation given by g. This follows the expression given by Eq. (75).
        '''
        
        return ( self.CM * self.Cz ) / self.gavg


def list_g(removedisputed=False):

    path = getcwd() + "/gM"
    
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    
    if removedisputed==True:
        filenames.remove('gm_subaru.dat')
        filenames.remove('gm_grb.dat')
        filenames.remove('gm_wdwarfs.dat')
        filenames.remove('gm_ns.txt')
        filenames.remove('gm_cmb_acc.dat')
        filenames.remove('gm_ligovirgo.dat')
    #print(filenames)
    return filenames
