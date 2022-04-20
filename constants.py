'''
This script contains the values for different physical constant and their convertion 
factors to the desired units

'''

G_cgs = 6.674*10**(-8) #cm^3/(g*s^2)
c_kms = 299792 # km/s
hbar_cgs = 1.0546e-27 # in cgs
c_cgs = 3e10 # in cm per second



km_to_Mpc = 3.241*10**(-20)
cm_to_Mpc = 3.241*10**(-25)
g_to_Ms = 5.03e-34

G = G_cgs*(cm_to_Mpc**3)/(g_to_Ms) # In Mpc**3 M_s**-1 s**-2
c = c_kms * km_to_Mpc # Light speed in Mpc/s


# Parameters of the Power Spectrum as measured by Planck 2018

A0 = (2.10521e-9)
k0 = 0.05 #Mpc**-1

ns_pl=0.9649
A1=A1f=A0/(k0**ns_pl)
