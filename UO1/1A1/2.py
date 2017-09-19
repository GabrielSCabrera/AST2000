import numpy as np

class gaussianFunction(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return (1./(np.sqrt(2.*np.pi)*self.sigma))*np.exp(-0.5*(((x - self.mu)/self.sigma))**2.)

def integrateFunctionVectorized(f, a, b, dt = 2e-3):
    a = float(a)
    b = float(b)
    xStep = dt*f(np.linspace(a, b, (b-a)/dt))
    return np.sum(xStep)

f = gaussianFunction(mu = 29., sigma = 2.)
print integrateFunctionVectorized(f = f, a = 27., b = 31.)
