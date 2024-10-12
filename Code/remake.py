import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import norm
from scipy.optimize import minimize

def ctime(exp_series):
    today = date(2024, 2, 12)  # Reference date

    def calculate_days(exp):
        month, day, year = map(int, exp.split('/'))
        expire = date(year, month, day)
        return (expire - today).days  # Return days until expiration

    return exp_series.apply(calculate_days)

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    elif option_type == 'Call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def objective_function(sigma, S, K, T, r, market_price, option_type):
    return (black_scholes(S, K, T, r, sigma, option_type) - market_price) ** 2

# Load the DataFrame
df = pd.read_csv('../Data/options.csv', dtype={'exp': 'string', 'strike': 'float32', 'bid': 'float32', 'ask': 'float32', 'underlying': 'float32'})

# Calculate days to expiration and midpoint price
df['days_to_exp'] = ctime(df['exp'])
df['mid_price'] = (df['bid'] + df['ask']) / 2

# Parameters for Black-Scholes
r = 0.05  # Risk-free rate
implied_volatilities = []

# Calculate implied volatility for each option
for index, row in df.iterrows():
    S = row['underlying']
    K = row['strike']
    T = row['days_to_exp'] / 365  # Convert days to years
    market_price = row['mid_price']
    type = row['right']
    
    # Initial guess for sigma
    initial_guess = 0.2

    # Minimize the objective function to find implied volatility
    result = minimize(objective_function, initial_guess, args=(S, K, T, r, market_price, type), bounds=[(0.001, None)])

    if result.success:
        print(f'S: {S}, K: {K}, T: {T}, r: {r}, market_price: {market_price}, type: {type}, IV: {result.x[0]}')
        implied_volatilities.append(result.x[0])
    else:
        implied_volatilities.append(np.nan)  # Append NaN if optimization fails

df['implied_volatility'] = implied_volatilities
print(df[['exp', 'strike', 'mid_price', 'implied_volatility']])
