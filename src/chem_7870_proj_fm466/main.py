import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root
from scipy.optimize import fsolve

# Properties of the semiconductor
class semicond:
    def __init__(self, bandgap, permittivity, eaffinity):
        self.bandgap = bandgap # in eV
        self.permittivity = permittivity # Relative permittivity
        self.eaffinity = eaffinity # Electron affinity, eV

# Properties of the insulator
class insulator:
    def __init__(self, permittivity, thickness):
        self.permittivity = permittivity # Relative permittivity
        self.thickness = thickness # in meters

# Densities
class density:
    def __init__(self, donor, acceptor, valence_states, cond_states):
        self.donor = donor # Density of donors in the semiconductor
        self.acceptor = acceptor # Density of acceptors
        self.valence_states = valence_states # Density of states in the valence band
        self.cond_states = cond_states # DOS in the conduction band

# Experiment setup parameters
class param:
    def __init__(self, vmin, vmax, temp, work_func_metal):
        self.vmin = vmin # Minimum applied voltage across the system
        self.vmax = vmax # Maximum applied voltage
        self.temp = temp # Temperature
        self.work_func_metal = work_func_metal # Metal surface work function
        

def _calc_properties(density, semicond, insulator, param):
    
    """
    Calculate additional constants of the system based on the user-specified
    properties of the semiconductor, insulator, and metal-insulator-semiconductor
    system

    Parameters
    ----------
    density : class

    semicond : class
    
    insulator : class

    param : class

    Returns
    -------
    class
        The collection of constants calculated
    """
    
    qe = 1.60217733e-19
    k = 1.380658e-23
    e0 = 8.854187817e-12
    kr = k/qe
    B = qe/(k*param.temp)
    
    # Intrinsic carier density
    n_intrinsic = np.sqrt(density.cond_states*density.valence_states) * np.exp(-semicond.bandgap / (2*kr*param.temp))

    E_valence = 0
    E_cond = semicond.bandgap
    E_intrinsic = kr*param.temp*np.log(n_intrinsic / density.cond_states) + E_cond

    space_func = lambda E_fermi: (density.donor - density.acceptor) + density.valence_states * np.exp((E_valence - E_fermi)*B) - density.cond_states * np.exp((-E_cond + E_fermi)*B)
    E_fermi = (root(space_func, (E_cond + E_valence) / 2)).x
    
    work_func_semicond = semicond.eaffinity + (E_cond - E_fermi)    
    # Flat band potential
    V_flat_band = param.work_func_metal - work_func_semicond
    
    # Capacitance per cm^2
    aerial_cap_ins = (insulator.permittivity * e0) / (insulator.thickness*100)
    
    # electron and hole densities in the bulk far from the surface
    n_elec_eq = density.cond_states * np.exp((-E_cond - E_fermi)*B)
    n_hole_eq = density.valence_states * np.exp((E_valence - E_fermi)*B)

    class result:
        def __init__(self, E_valence, E_cond, E_intrinsic, E_fermi, V_flat_band, aerial_cap_ins, n_elec_eq, n_hole_eq):
            self.E_valence = E_valence
            self.E_cond = E_cond
            self.E_intrinsic = E_intrinsic
            self.E_fermi = E_fermi
            self.V_flat_band = V_flat_band
            self.aerial_cap_ins = aerial_cap_ins
            self.n_elec_eq = n_elec_eq
            self.n_hole_eq = n_hole_eq

    return result(E_valence, E_cond, E_intrinsic, E_fermi, V_flat_band, aerial_cap_ins, n_elec_eq, n_hole_eq)

    
def _calc_potential(density, semicond, param, calculated, V_app):
    
    """
    Calculate the potential leading up to the semiconductor surface according 
    to band-bending behavior and the associated depth for each value of the potential

    Parameters
    ----------
    density : class

    semicond : class
    
    param : class
    
    calculated: class
        The result of _calc_properties()

    V_app: float
        Voltage applied to the system

    Returns
    -------
    depth: np.ndarray
        Depth from the semiconductor surface
        
    potential: np.ndarray
        Potential at each depth
        
    potential_surf: float
        Potential at the surface of the semiconductor
        
    """
    
    qe = 1.60217733e-19
    k = 1.380658e-23
    e0 = 8.854187817e-12
    B = qe/(k*param.temp)
    
    F = lambda potential_surf: potential_surf*(density.acceptor - density.donor) + (calculated.n_hole_eq / B)* \
        (np.exp(-potential_surf*B) - 1) + (calculated.n_elec_eq / B)*(np.exp(potential_surf*B) - 1)
    eqn_potential_surf = lambda potential_surf: calculated.V_flat_band + potential_surf - V_app - \
        (1/calculated.aerial_cap_ins)*(-np.sign(potential_surf)*np.sqrt(2*qe*e0*semicond.permittivity*F(potential_surf)))
    Efield = lambda potential: 1/(np.sign(potential) * np.sqrt((2*qe)/(semicond.permittivity * e0) * F(potential)))

    # solve surface potential for the applied voltage
    potential_surf = fsolve(eqn_potential_surf, V_app)
    
    # Calculate band bending in region near the surface at finer intervals than bulk
    potential_shallow = np.linspace(potential_surf, potential_surf/2, 50)
    potential_deep = np.logspace(np.log10(np.abs(potential_surf)/2), np.log10(np.abs(potential_surf)/1000), 150)

    if potential_surf < 0:
        potential_deep = -potential_deep

    potential = np.concatenate((potential_shallow, potential_deep[1:]))
    depth = np.zeros(len(potential))
    
    # integrate and find the corresponding depth for each potential
    for i in range(len(potential)):
        depth_now, err = quad(Efield, potential[i], potential_surf)
        depth[i] = depth_now
        i = i+1

    return depth, potential, potential_surf
    
    
def plot_cap_vs_V(semicond, insulator, density, param):
    
    """
    Calculate capacitance at voltages between vmin and vmax as specified in param
    and plot the capacitance of the system as a function of applied voltage

    Parameters
    ----------
    density : class

    semicond : class
    
    insulator : class

    param : class
    
    Returns
    -------
    V_app : np.ndarray
        Sweep of applied voltages
        
    capacitance : np.ndarray
        Capacitance at each value of V_app

    """

    qe = 1.60217733e-19
    k = 1.380658e-23
    e0 = 8.854187817e-12
    B = qe/(k*param.temp)
    
    V_app = np.linspace(param.vmin, param.vmax, endpoint=True)
    capacitance = np.zeros(len(V_app))

    work_func_metal_o = param.work_func_metal

    for i in range(len(V_app)):
        param.work_func_metal = work_func_metal_o - V_app[i]
        calculated = _calc_properties(density, semicond, insulator, param)

        depth, potential, potential_surf = _calc_potential(density, semicond, param, calculated, V_app[i])

        len_debye = np.sqrt((semicond.permittivity*e0) / (qe*calculated.n_hole_eq*B))
        F = ((np.exp(-B*potential_surf) + (B*potential_surf) - 1) + (calculated.n_elec_eq / calculated.n_hole_eq)*(np.exp(B*potential_surf) - (B*potential_surf) - 1))**0.5
        cap_depletion = ((semicond.permittivity / (np.sqrt(2)*len_debye))*(1 - np.exp(-B*potential_surf) + (calculated.n_elec_eq / calculated.n_hole_eq)*(np.exp(B*potential_surf) - 1))/F)
        cap_ins = insulator.permittivity / (insulator.thickness*100)

        capacitance[i] = - (cap_ins * cap_depletion) / (cap_ins + cap_depletion)
        
        i = i+1
        
    plt.plot(V_app, capacitance)
    plt.xlabel('Applied voltage')
    plt.ylabel('Capacitance [F/cm$^2$]')
    plt.show()
    
    return V_app, capacitance
    
    
def plot_band_bending(semicond, insulator, density, param, V_app):
    
    """
    Plot the band bending behavior for a particular applied voltage

    Parameters
    ----------
    density : class

    semicond : class
    
    insulator : class

    param : class
    
    V_app : float

    """
    
    param.work_func_metal = param.work_func_metal - V_app
    calculated = _calc_properties(density, semicond, insulator, param)
    depth, potential, potential_surf = _calc_potential(density, semicond, param, calculated, V_app)

    # convert to nm
    depth = depth * 1e7

    plt.plot(depth, calculated.E_fermi*np.ones(len(depth)), label='$E_f$')
    plt.plot(depth, calculated.E_cond - potential, label='$E_c$')
    plt.plot(depth, calculated.E_intrinsic - potential, label='$E_i$')
    plt.plot(depth, calculated.E_valence - potential, label='$E_v$')
        
    plt.xlabel('Depth [nm]')
    plt.ylabel('Band bending relative to bulk [eV]')
    plt.legend()
    plt.show()
