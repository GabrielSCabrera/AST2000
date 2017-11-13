import classes as cl
import functions as fx
import numpy as np
from numpy import linalg as LA

P= cl.Planet('jevelan')
S= cl.Solar_System()
b= cl.Planet('bertela')
c= cl.Planet('sarplo')
d= cl.Planet('poppengo')

'''print P.apoapsis
print P.sun.temperature
print P.sun.radius
flux= 5.670e-8*(P.sun.temperature**4)
print 'Flux at sun= ',flux
A_sun= 4.*np.pi*(P.sun.radius*1e3)**2
print 'Solar surface area: ', A_sun
L= A_sun*flux
print 'Luminosity: ',L
A= 4.*np.pi*((1.496e11*P.apoapsis)**2)
print 'A= ',A
flux_planet= L/A
print 'Flux at planet= ',flux_planet
A_panel= 40./(0.12*flux_planet)
print 'A_panel= ',A_panel,'m**2'
print 'Surface temperature: ',P.temperature
print S.number_of_planets
print b.plot(numerical= True)'''

t_ap= b.get_time_from_angle(b.psi)
t_pe= b.get_time_from_angle(b.psi-np.pi)
print b.get_area(t0= t_ap-0.1, dt= 0.2)
print b.get_area(t0= t_pe-0.1, dt= 0.2)

print b.get_arc_length(t0= t_ap-0.1, dt= 0.2)
print b.get_arc_length(t0= t_pe-0.1, dt= 0.2)

print b.get_mean_velocity(t0= t_ap-0.1, dt= 0.2)
print b.get_mean_velocity(t0= t_pe-0.1, dt= 0.2)

print LA.norm(b.get_velocity_from_time(t= t_ap))
print LA.norm(b.get_velocity_from_time(t= t_pe))

print b.T**2./(b.a**3)
print P.T**2./(P.a**3)
print c.T**2./(c.a**3)
print d.T**2./(d.a**3)

print b.sun.mass
print 1./b.sun.mass
print b.G, 4*np.pi**2
