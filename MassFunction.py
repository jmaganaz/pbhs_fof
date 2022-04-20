import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from m1ph import generate
from constraints import f, g, list_g
from corrections import Cz,CM
from scipy import integrate,interpolate
from HC_lognormal import dndm_spiky_broken_log_std as dndmstd
from HC_lognormal import dndm_spiky_broken_log as dndm
from HC_lognormal import dndm_spiky_broken_log_spi as dndmspi
import SMBH_f

from constants import km_to_Mpc, G,c, A1f,ns_pl

class MassFunction:

    '''

    Mass Function initialized with a callable function that depends only on log(M)

    '''

    def __init__(self,func,cosmo, M_star,nb, Mc,A2=None,**kwargs):
		
        #self.logm=logm
   

        
        self.M_star=M_star
        
        self.nb= nb
        
        self.Mc=Mc
        
        self.dndM = func
        self.dndlogM = self.__dndmlog
        
        self.dndM2 = self.__dndmf

    
        self.Mmax = 10. #np.nanmin([np.nanmax(data_x),data_x[np.where(data_y==0.)[0][0]]]) 

        self.cosmo = cosmo

        self.Mev = cosmo.Mev
        
        self.eps=1e-2 #width of the peaks

        self.M1ph = None

        self.f = None
        
        self.An=1.0
        
       
        
        
        if A2 is None:
            self.A2=self.find_A2_max_restricted()[1]
            #self.A2max=self.A2[1]
        else: 
            self.A2=A2		
        
        
        self.An_iterado()
    
    def __dndmlog(self,logM):

        
        return 10**(logM) * np.log(10) * self.dndM(logM,self.M_star, self.nb)
        
    def __dndmf(self,logM,a, A2c):

        return np.float_power(a,-3.)*self.dndM([logM,self.M_star, self.nb,self.Mc,A2c])

    def __call__(self,logM,M_star, nb,Mc,dlogM = False,**kwargs):
        

        if dlogM == True:

            return self.dndlogM(logM, M_star, nb,Mc, a=1e-15, iscomoving=False)

        else:
            
            return self.dndM(logM, M_star, nb,Mc)


    def plot(self,savefig = '',lox=1e-20,loy=1e-20):
        vec_M=np.linspace(np.log10(self.cosmo.Mev(0)),np.log10(1e3*self.M_star),10000)
        y0=dndm(vec_M, 1, self.cosmo, self.cosmo.eof, self.M_star, 10, ns_pl,self.nb, self.Mc, A1f,self.A2,self.eps,1, True, False)
        plt.figure()
        plt.plot(np.power(10,vec_M),y0)
        plt.loglog()
        plt.xlabel(r"$\log_{10}(M/M_\odot)$")
        plt.ylabel(r"$\frac{dn}{dM}$")
        plt.xlim(lox,1e3*self.M_star)
        plt.ylim(loy,1e2*np.max(y0))
        if savefig != '':
            plt.savefig(savefig)
        plt.show()
        
    def compute_M1ph(self, progress_bar = False):

        ''' 
        This function computes the log10(M1ph) value for the relevant redshifts presented in 
        the paper z = {0 , 1 , 450 , 1e10}.

        Returns a dictionary.
       
        '''
        if self.M1ph is not None:

            return self.M1ph

        elif self.M1ph is None:

         #   self.compute_M1ph()
            self.M1ph = generate(self, progress_bar)
            return self.M1ph

    def mass_density_std(self,lo,hi,a=1,A2c=None):
        if A2c is None:
            A2c=self.A2
        

        '''
        Returns the mass density of the PBHs. This is the integral from 'lo' to 'hi' of
        M times the PBH mass function.    
        
        '''

        if hi < lo:

            return 0.
            
       
        vec_M=np.linspace(lo,hi,10000)
        y0=dndmstd(vec_M, a, self.cosmo, self.cosmo.eof, self.M_star, 10, ns_pl,self.nb, self.Mc, A1f, A2c,self.eps,1, True, False)
        y1 = np.asarray(y0)*np.asarray(np.power(10,vec_M)*np.log(10))*np.asarray(np.power(10,vec_M))
        Mint=integrate.trapz(y1,vec_M)
        return Mint
        
        
    def mass_density_spi(self,lo,hi,a=1,A2c=None):
        if A2c is None:
            A2c=self.A2
        

        '''
        Returns the mass density of the PBHs. This is the integral from 'lo' to 'hi' of
        M times the PBH mass function.    
        
        '''

        if hi < lo:

            return 0.
            
       
        vec_M=np.linspace(lo,hi,10000)
        y0=dndmspi(vec_M, a, self.cosmo, self.cosmo.eof, self.M_star, 10, ns_pl,self.nb, self.Mc, A1f, A2c,self.eps,1, True, False)
        y1 = np.asarray(y0)*np.asarray(np.power(10,vec_M)*np.log(10))*np.asarray(np.power(10,vec_M))
        Mint=integrate.trapz(y1,vec_M)
        return Mint
        
    def mass_density_std_tot(self,lo,hi,a=1,A2c=None):
        if A2c is None:
            A2c=self.A2
        int1=self.mass_density_std(lo=lo,hi=np.log10(self.Mc)-6.*self.eps,a=a,A2c=A2c)
        int2=self.mass_density_std(lo=np.log10(self.Mc)+15*self.eps,hi=hi,a=a,A2c=A2c)
        int3=self.mass_density_std(lo=np.log10(self.Mc)-6*self.eps,hi=np.log10(self.Mc)+15*self.eps,a=a,A2c=A2c)
        return int1+int2+int3

    def mass_density_TOT(self,lo,hi,a=1,A2c=None):
        if A2c is None:
            A2c=self.A2
        massden_spi=self.mass_density_spi(lo=np.log10(self.Mc)-6*self.eps,hi=np.log10(self.Mc)+15*self.eps,a=a,A2c=A2c)
        massden_std=self.mass_density_std_tot(lo=lo,hi=hi,a=a,A2c=A2c)
        return (massden_spi+massden_std)
        
        
    def number_density(self,lo,hi,a):

        '''
        Returns the cumulative number density of the PBHs. This is the integral from 'low' to 'high' of
        the PBH mass function.    
        '''        
        x0=x1=np.linspace(lo,hi,60000)
       
        y0=self.dndM2(x1,a=a,A2c=self.A2)
       
        
        y1 = np.asarray(y0)*np.asarray(np.power(10,x1)*np.log(10))
        x1 = np.flip(np.array(x1))
        y1 = np.flip(np.array(y1))
        Nt=np.flip(np.abs(integrate.cumtrapz(y1,x1)))
        Nt=np.append(Nt,0)
        return x0,self.An*Nt
        
    def number_density_trapz(self,lo,hi):

        '''
        Returns the number density of the PBHs. This is the integral from 'low' to 'high' of
        the PBH mass function.    
        '''        
        x1=np.linspace(lo,hi,10000)
        y0=self.dndM([x1,self.M_star, self.nb,self.Mc,self.A2])

        y1 = np.asarray(y0)*np.asarray(np.power(10,x1)*np.log(10))
        Mint=integrate.trapz(y1,x1)
        return self.An*Mint
        
   
    def find_A2_max_restricted(self,lo = -20, hi = 20):
        A2v=np.logspace(10,200,800)
        fraction=[]
        A22=[]
        
        rhodm=self.cosmo.rhoc*self.cosmo.Odm0
        for i in range(len(A2v)):
            x=self.mass_density_spi(lo=np.log10(self.Mc)-6.*self.eps,hi=np.log10(self.Mc)+15.*self.eps,a=1,A2c=A2v[i])
            y=self.mass_density_std_tot(lo,hi,a=1,A2c=A2v[i])
            mass_den_tot=x+y
            fraction.append(x/(x+y)) #test
            A22.append(A2v[i]) #test
            error=abs(mass_den_tot-rhodm)/rhodm
        
            if error<0.01:
                fraction.append(x/(x+y))
                A22.append(A2v[i])
            else:
                break
    
        self.A2=a2max=A22[-1]
        return A22,A22[-1],interpolate.interp1d(A22,fraction) 

    def AN(self,lo=-19,hi=-7):
        x=self.compute_M1ph()
        #print('x---------------',x[0])
        if x[0]<lo:

            An=np.nan

        else:
			
            num=self.mass_density_TOT(-60,np.log10(1e3*self.M_star),a=1,A2c=self.A2)
            den=self.mass_density_TOT(lo,x[0],a=1,A2c=self.A2)
        
        try:
            An=num/den
    
        except ZeroDivisionError:
            print('zero division')
            An=np.nan
        self.An=An
        return An


    def An_iterado(self):
        A_old=1.0
        hi_inicial=self.compute_M1ph()[0]
    
	
        An_new=self.AN(-19,hi_inicial)
	
	
        for i in range(25):
		
            if np.isnan(An_new):
			
                return An_new, hi_inicial
                break
		
            error=abs(An_new-A_old)/A_old
		
            if error<1e-3:
                #print('stop in An=%d An=%e M1ph=%f'%(i,An_new,hi_inicial))
                return An_new
                break
            hi_new=self.compute_M1ph()[0]
          
            An_new=self.AN(-19,hi_new)
            #print('An_new=%e'%An_new)
            hi_inicial=hi_new
            A_old=An_new
            self.An=A_old

       
    def Mean_Mass(self,lo,hi,A2file):

        '''
        Returns the Mean Mass for the PBH's distribution. This is the Mass density divided
        by the number density.
        '''

        numerator = self.mass_density_TOT(lo,hi,1,A2c=A2file)
        
        denominator = self.number_density_trapz(lo,hi)

        try:

            result = numerator/denominator

        except ZeroDivisionError:

            result = np.nan

        return result 

    def Cz(self,z):

        return Cz(self,z)

    def CM(self,gM):

        return CM(self,gM)

    def compute_f(self, removedisputed=False):

        if self.f is not None:

            return self.f

        elif self.M1ph is None:

            self.compute_M1ph()

        filenames = list_g(removedisputed)

        f_values = []

        for File in filenames:

            gM = g(File,self.cosmo)
            

            f_values.append(f(self,gM).value)

        ## En este punto hay que calcular el constraint de SMBH

        SMBH_f.init_dMdHI(self.cosmo)
        SMBH_f.init_AGN_MF(self.cosmo)
        f_smbh = SMBH_f.f(self)

        #print("f_SMBH = %e"%f_smbh)
        f_values.append(f_smbh)

        self.f = np.nanmin(f_values)

        return self.f
