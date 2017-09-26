from classes import Planet

a = Planet(name = 'Bertela', dt = 1e-4)
print a
print a.get_area(t0 = 0.4, dt = 0.2)-a.get_area(dt = 0.2)
print a.get_arc_length(t0 = 0.3, dt = 0.2), a.get_arc_length(dt = 0.2)
print a.get_mean_velocity(t0 = 0.3, dt = 0.2), a.get_mean_velocity(dt = 0.2)
b = Planet(name = 'sarplo', dt = 1e-4)
print a.T**2./a.a**3., b.T**2./b.a**3.
