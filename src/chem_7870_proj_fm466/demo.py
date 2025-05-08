import numpy as np
import main

class semicond:
    def __init__(self, bandgap, permittivity, eaffinity):
        self.bandgap = bandgap
        self.permittivity = permittivity
        self.eaffinity = eaffinity

class insulator:
    def __init__(self, permittivity, thickness):
        self.permittivity = permittivity
        self.thickness = thickness

class density:
    def __init__(self, donor, acceptor, valence_states, cond_states):
        self.donor = donor
        self.acceptor = acceptor
        self.valence_states = valence_states
        self.cond_states = cond_states

class param:
    def __init__(self, vmin, vmax, temp, work_func_metal):
        self.vmin = vmin
        self.vmax = vmax
        self.temp = temp
        self.work_func_metal = work_func_metal


# Gold tip and semiconductor surface, short distance
si = semicond(1.12, 11.68, 4.05)
vac = insulator(1, 2e-9)
example = density(1e16, 1e17, 9.84e18, 2.78e19)
au = param(-3, 3, 298, 5.1)
main.plot_cap_vs_V(si, vac, example, au)


# Gold tip and semiconductor surface, long distance
vac = insulator(1, 200e-9)
main.plot_cap_vs_V(si, vac, example, au)

# Applied voltage band bending
main.plot_band_bending(si, vac, example, au, 1.5)
