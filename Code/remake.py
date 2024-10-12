import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def black_scholes(S, K, T, r, sigma, type):
    if type == 'Put':
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    if type == 'Call':
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def objective_function(sigma, S, K, T, r, market_price, type):
    return (black_scholes(S, K, T, r, sigma, type) - market_price) ** 2

# Parameters
S = 502.04
K = 451.0
T = 0.0822
r = 0.05
market_price = 70.91

# Initial guess for sigma
initial_guess = 0.2

# Minimize the objective function to find implied volatility
result = minimize(objective_function, initial_guess, args=(S, K, T, r, market_price, "Call"), bounds=[(0.001, None)])

implied_volatility = result.x[0]
print("Implied Volatility:", implied_volatility)