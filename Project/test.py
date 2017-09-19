import matplotlib.pyplot as plt
import numpy as np
from classes import Planet
import functions as fx
import os

steps = 60

sarplo = Planet(name = 'Sarplo')
f = np.linspace(0,2*np.pi,steps)
r = sarplo.get_data_from_angle(f)['position']

plt.plot(r[0],r[1])
plt.plot(0,0,'oy',ms=10)
plt.show()

'''We must create a series of images for our gif'''

axes = [1.2*np.min(r[0]),1.2*np.max(r[0]),1.2*np.min(r[1]),1.2*np.max(r[1])]
legend = ['Planet','Sun']
for i in range(0, steps):
    plt.axis(axes)
    plt.plot(r[0,i],r[1,i],'ob',ms=5)
    plt.plot(0,0,'oy',ms=10)
    plt.title("Orbital Visualization")
    plt.legend(legend)
    plt.savefig("orbit%02d.png"%(i))
    plt.figure()

'''The following will combine all images named frame*.png,
where the * represents any number, into a gif animation'''

os.system('convert -delay 6 orbit*.png orbit.gif')

'''The following will remove the individual images for neatness'''

for i in range(0,steps):
	filename = "orbit%02d.png"%(i)
	os.remove(str(filename))
