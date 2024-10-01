###################################################
# Challenge: Calculate IV of options              #
# Runtime:                                        #
# Creator: Kovas McCann (KovasMcCann@outlook.com) #
# Github: https://github.com/KovasMcCann          #
###################################################

from datetime import date
import torch
import pandas as pd

class BS:
    def __init__(self, underlying, strike, risk, exp, callPrice=None):
        global device
        # Convert inputs to tensors
        self.underlyingPrice = torch.tensor(underlying, dtype=torch.float32, device=device)
        self.strikePrice = torch.tensor(strike, dtype=torch.float32, device=device)
        self.interestRate = torch.tensor(risk / 100, dtype=torch.float32, device=device)
        self.daysToExpiration = torch.tensor(exp, dtype=torch.float32, device=device)

        if callPrice is not None:
            self.callPrice = torch.tensor(callPrice, dtype=torch.float32)
            self.impliedVolatilityCall = self.calculate(self.callPrice)
        else:
            self.callPrice = None

    def call_price(self, volatility):
        S = self.underlyingPrice
        K = self.strikePrice
        r = self.interestRate
        T = self.daysToExpiration / 365  # Convert days to years
        d1 = (torch.log(S / K) + (r + 0.5 * volatility ** 2) * T) / (volatility * torch.sqrt(T))
        d2 = d1 - volatility * torch.sqrt(T)
        return S * self.normal_cdf(d1) - K * torch.exp(-r * T) * self.normal_cdf(d2)

    def vega(self, volatility):
        S = self.underlyingPrice
        K = self.strikePrice
        r = self.interestRate
        T = self.daysToExpiration / 365  # Convert days to years
        d1 = (torch.log(S / K) + (r + 0.5 * volatility ** 2) * T) / (volatility * torch.sqrt(T))
        return S * self.normal_pdf(d1) * torch.sqrt(T)

    def normal_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

    def normal_pdf(self, x):
        return (1.0 / torch.sqrt(2 * torch.tensor(3.141592653589793))) * torch.exp(-0.5 * x ** 2)

    def calculate(self, price, initial_vol=0.2, tol=1e-5, max_iter=1000):
        # Initialize volatility as a tensor with the same size as the number of options
        volatility = torch.full(price.shape, initial_vol, dtype=torch.float32)  # Shape matches number of options

        for _ in range(max_iter):
            estimated_price = self.call_price(volatility)
            vega = self.vega(volatility)

            price_diff = estimated_price - price  # This will have the same shape as price

            # Check for convergence
            if torch.all(torch.abs(price_diff) < tol):
                return volatility  # Convergence achieved

            # Update volatility using Newton's method
            volatility -= price_diff / vega  # Broadcasting happens here

        return volatility

def ctime(exp_series):
    """Convert a Series of expiration dates to days until expiration."""
    today = date(2024, 2, 12)  # Reference date

    def calculate_days(exp):
        month, day, year = map(int, exp.split('/'))
        expire = date(year, month, day)
        return (expire - today).days  # Return days until expiration

    return exp_series.apply(calculate_days)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'running on {device}')

# Load the DataFrame
df = pd.read_csv('options.csv', dtype={'exp': 'string', 'strike': 'float32', 'bid': 'float32', 'ask': 'float32', 'underlying': 'float32'})

# Calculate days to expiration and midpoint price
df['days_to_exp'] = ctime(df['exp'])
df['mid_price'] = (df['bid'] + df['ask']) / 2

# Prepare tensors for vectorized operations
underlying_tensor = df['underlying'].values
strike_tensor = df['strike'].values
risk = 0.05  # Set the risk rate

# Initialize the Black-Scholes model and calculate implied volatilities
bs_model = BS(underlying_tensor, strike_tensor, risk, df['days_to_exp'].values, callPrice=df['mid_price'].values)
implied_volatility = bs_model.impliedVolatilityCall

# Add the implied volatility to the DataFrame
df['implied_volatility'] = implied_volatility

results_df = pd.DataFrame(df[['exp', 'strike', 'bid', 'ask', 'underlying', 'implied_volatility']], columns=['exp', 'strike', 'bid', 'ask', 'underlying', 'implied_volatility'])
results_df.to_csv('out.csv', index=False)

print(df[['exp', 'strike', 'mid_price', 'implied_volatility']])
