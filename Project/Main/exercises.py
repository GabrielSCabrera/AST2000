import classes as C

S= C.Solar_System()

'''Part 2.2'''

#Planet List
print S.get_ordered_list(names= True)

#2
max_e= S.get_max('e')
print('Planet with Maximum Eccentricity: Planet %s'%(max_e))

#3
S.planets['bertela'].plot(analytical= True, numerical= True)

#5
print S.planets['bertela'].get_area(t0= 0, dt= 0.1)
print S.planets['bertela'].get_area(t0= 1, dt= 0.1)
print S.planets['bertela'].get_area(t0= 2, dt= 0.1)
