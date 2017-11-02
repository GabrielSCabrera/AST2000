from ast2000solarsystem_27_v4 import AST2000SolarSystem as A2000

class Scripts(object):

    def __init__(self, seed = None):
        if seed == None:
            seed = 45355
        self.mySolarSystem = A2000(seed)

    def get_lambda_deg_from_ref_stars(self):
        self.mySolarSystem.get_ref_stars()
