import torch
from torch import log, exp

def impliedVolatility(className, args, callPrice=None, putPrice=None, high=1.0, low=0.0):
    '''Returns the estimated implied volatility'''
    
    # Create an instance of the class with the highest volatility to start
    if callPrice is not None:
        target = callPrice
        instance = className(*args, volatility=high, performance=True)
        restimate = instance.callPrice  
        if restimate < target:
            return high
        if args[0] > args[1] + callPrice:
            return 0.001            
    elif putPrice is not None:
        target = putPrice
        instance = className(*args, volatility=high, performance=True)
        restimate = instance.putPrice
        if restimate < target:
            return high
        if args[1] > args[0] + putPrice:
            return 0.001            

    decimals = len(str(target).split('.')[1])  # Count decimals
    for i in range(10000):  # To avoid infinite loops
        mid = (high + low) / 2
        if mid < 0.00001:
            mid = 0.00001
            
        # Create an instance with the current volatility
        if callPrice is not None:
            estimate = className(*args, volatility=mid, performance=True).callPrice
        elif putPrice is not None:
            estimate = className(*args, volatility=mid, performance=True).putPrice

        print(f"Iteration {i}: mid={mid:.5f}, estimate={estimate:.5f}, target={target:.5f}, high={high:.5f}, low={low:.5f}")

        if round(estimate, decimals) == target: 
            break
        elif estimate > target: 
            high = mid
        else: 
            low = mid
            
    return mid

class BS:
    '''Black-Scholes for pricing European options.'''
    
    def __init__(self, exp, right, strike, price, underlying, risk=0.05, volatility=0.2):
        self.device = 'cpu'
        self.underlyingPrice = torch.tensor(underlying, dtype=torch.float32, device=self.device)
        self.strikePrice = torch.tensor(strike, dtype=torch.float32, device=self.device)
        self.interestRate = torch.tensor(risk / 100, dtype=torch.float32, device=self.device)
        self.daysToExpiration = torch.tensor(exp, dtype=torch.float32, device=self.device)
        self.volatility = volatility  # Set initial volatility

        # Prepare arguments for implied volatility calculation
        self.args = [self.underlyingPrice.item(), self.strikePrice.item(), self.interestRate.item() * 100, self.daysToExpiration.item() * 365]

        # Initialize prices based on option type
        self.callPrice = round(float(price), 6) if right == "Call" else None
        self.putPrice = round(float(price), 6) if right == "Put" else None
        self.impliedVolatility = self.calculate_implied_volatility(right)

    def calculate_implied_volatility(self, right):
        return impliedVolatility(self.__class__, self.args, callPrice=self.callPrice) if right == "Call" else impliedVolatility(self.__class__, self.args, putPrice=self.putPrice)

    def _calculate_options(self):
        self._a_ = self.volatility * self.daysToExpiration**0.5
        self._d1_ = (log(self.underlyingPrice / self.strikePrice) + 
                     (self.interestRate + (self.volatility**2) / 2) * 
                     self.daysToExpiration) / self._a_
        self._d2_ = self._d1_ - self._a_
        
        self.callPrice, self.putPrice = self._price()

    def _price(self):
        '''Returns the option price: [Call price, Put price]'''
        if self.strikePrice.item() == 0:
            raise ZeroDivisionError('The strike price cannot be zero')

        # Create a normal distribution
        normal_dist = torch.distributions.Normal(0, 1)

        # Calculate d1 and d2 with current volatility
        self._a_ = self.volatility * self.daysToExpiration**0.5
        self._d1_ = (log(self.underlyingPrice / self.strikePrice) + 
                     (self.interestRate + (self.volatility**2) / 2) * 
                     self.daysToExpiration) / self._a_
        self._d2_ = self._d1_ - self._a_

        # Calculate call and put prices using the CDF from the normal distribution
        call = (self.underlyingPrice * normal_dist.cdf(self._d1_) - 
                self.strikePrice * exp(-self.interestRate * self.daysToExpiration) * normal_dist.cdf(self._d2_))
    
        put = (self.strikePrice * exp(-self.interestRate * self.daysToExpiration) * 
               normal_dist.cdf(-self._d2_) - 
               self.underlyingPrice * normal_dist.cdf(-self._d1_))
    
        return call.item(), put.item()

# Example usage
c = BS(30, "Call", 451.0, 71.67, 502.0411)
print(f"Implied Volatility: {c.impliedVolatility:.6f}")
