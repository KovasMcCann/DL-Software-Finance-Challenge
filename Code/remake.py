import torch
import pandas as pd
from datetime import date
from scipy.stats import norm
from scipy.optimize import minimize

def ctime(exp_series):
    today = pd.to_datetime(date(2024, 2, 12))
    expire_dates = pd.to_datetime(exp_series, format='%m/%d/%Y')
    days_to_expiration = (expire_dates - today).dt.days
    return days_to_expiration

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    
    normal_dist = torch.distributions.Normal(0, 1)
    
    if option_type == 'Put':
        return K * torch.exp(-r * T) * normal_dist.cdf(-d2) - S * normal_dist.cdf(-d1)
    elif option_type == 'Call':
        return S * normal_dist.cdf(d1) - K * torch.exp(-r * T) * normal_dist.cdf(d2)
    else:
        raise ValueError("Invalid option type. Use 'Call' or 'Put'.")

def objective_function(sigma, S, K, T, r, market_price, option_type):
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    return (bs_price.item() - market_price) ** 2  # Convert to Python float

def calculate_implied_volatility(row):
    S = torch.tensor(row['underlying'])  # Default is CPU
    K = torch.tensor(row['strike'])
    T = torch.tensor(row['days_to_exp'] / 365)  # Convert days to years
    market_price = torch.tensor(row['mid_price'])
    option_type = row['right']
    
    initial_guess = torch.tensor(0.2, requires_grad=True)

    # Optimization using scipy
    result = minimize(
        lambda sigma: objective_function(torch.tensor(sigma), S, K, T, r, market_price, option_type),
        initial_guess.item(),
        bounds=[(0.001, None)]
    )

    return result.x[0] if result.success else float('nan')  # Return implied volatility or NaN

# Load the DataFrame
df = pd.read_csv('../Data/options.csv', dtype={'exp': 'string', 'strike': 'float32', 'bid': 'float32', 'ask': 'float32', 'underlying': 'float32'})

# Calculate days to expiration and midpoint price
df['days_to_exp'] = ctime(df['exp'])
df['mid_price'] = (df['bid'] + df['ask']) / 2

# Parameters for Black-Scholes
r = torch.tensor(0.05)  # Risk-free rate

# Calculate implied volatility for each option
implied_volatilities = []
for index, row in df.iterrows():
    implied_volatility = calculate_implied_volatility(row)
    implied_volatilities.append(implied_volatility)

df['implied_volatility'] = implied_volatilities
print(df[['exp', 'strike', 'mid_price', 'implied_volatility']])