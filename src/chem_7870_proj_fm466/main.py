import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

"""
Variables
---------

V_ts : np.ndarray
    range of applied voltages between tip and surface
    positive when metal is positive biased WRT semiconductor surface

df : np.ndarray
    dependent variable, plotted
    
z_ts : np.ndarray
    range of tip-surface distances

e0 : float
    permittivity of free space
    8.8541878188 F*m^(-1) [C^2*kg^(−1)*m^(−3)*s^2]
    
er: float
    relative permittivity of the material

sigma : float
    surface charge density [C*m^(-2)]

E_gap : float
    band gap of the semiconductor, given by the longest "plateau" plotted
    
Ef_metal : float
    Fermi level of the metal tip

Ef_sc : float
    Fermi level of the semiconductor surface

"""

def calculate_log_posterior(X, Y, m_vals, b_vals, sigma):

	posterior_matrix = np.zeros(len(m_values), len(b_values))
	for i, m in enumerate(m_vals):
		for j, b in enumerate(b_vals):
			log_prior = calculate_log_prior()
			log_likelihood = calculate_log_likelihood(X, Y, m, b, sigma)
			log_posterior_matrix[i, j] = log_prior + log_likelihood
	log_evidence = calculate_log_evidence()

	log_posterior = log_prior + log_likelihood - log_evidence
	return log_posterior

def calculate_log_prior():
	return 0

def calculate_log_likelihood(Y:np.ndarray, X:np.ndarray, m:float, b:float, sigma:float) -> float:
	'''
	Calculates the log-likelihood for Bayesian Linear Regression
	'''

	N = len(Y)
	log_likelihood = -N / 2 * np.log(2* np.pi * sigma**2) - (1 / (2 * sigma**2) * np.sum((Y - (m * X + b))**2))
	return log_likelihood
def calculate_log_evidence():
	raise NotImplementedError
