import numpy as np
from chem_7870_proj_fm466 import main

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

def test_capacitance_values_valid():
    
    si = semicond(1.12, 11.68, 4.05)
    vac = insulator(1, 2e-9)
    example = density(1e16, 1e17, 9.84e18, 2.78e19)
    au = param(-3, 3, 298, 5.1)
    
    V_app, capacitance = main.plot_cap_vs_V(si, vac, example, au)
    
    assert np.isnan(np.sum(capacitance)) == False
    
def test_band_bending():

    si = semicond(1.12, 11.68, 4.05)
    vac = insulator(1, 2e-9)
    example = density(1e16, 1e17, 9.84e18, 2.78e19)
    au = param(-3, 3, 298, 5.1)

    calculated = _calc_properties(example, si, vac, au)
    depth, potential, potential_surf = main._calc_potential(example, si, au, calculated, 1)
    # the order of the parameters seems to get scrambled when using pytest, so this fails
    
    assert potential[0] - potential[-1] !==0
    
def test_surface_potential_consistent_with_potential_array():

    si = semicond(1.12, 11.68, 4.05)
    vac = insulator(1, 2e-9)
    example = density(1e16, 1e17, 9.84e18, 2.78e19)
    au = param(-3, 3, 298, 5.1)
    
    calculated = _calc_properties(example, si, vac, au)
    depth, potential, potential_surf = main._calc_potential(example, si, au, calculated, 1)
    # the order of the parameters seems to get scrambled when using pytest, so this fails
    
    assert  np.allclose(potential_surf, potential[0], atol=1e-2)
