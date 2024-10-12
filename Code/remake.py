import torch

device = 'cpu'
dtype = torch.float32

def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=dtype))))

class calculate:
    @staticmethod
    def put(S, K, T, volatility):
        # Creating a tensor with the input data
        Data = torch.tensor([S, K, T, volatility], dtype=dtype, device=device)

        d1 = (torch.log(Data[0] / Data[1]) + (0.5 * Data[3] ** 2) * Data[2]) / (Data[3] * torch.sqrt(Data[2]))
        d2 = d1 - Data[3] * torch.sqrt(Data[2])
    
        # Return the put price calculation (using CDF of normal distribution)
        return Data[1] * torch.exp(-0.5 * Data[2]) * normal_cdf(-d2) - Data[0] * normal_cdf(-d1)

    @staticmethod
    def call(S, K, T, volatility):
        # Creating a tensor with the input data
        Data = torch.tensor([S, K, T, volatility], dtype=dtype, device=device)

        d1 = (torch.log(Data[0] / Data[1]) + (0.05 * Data[3] ** 2) * Data[2]) / (Data[3] * torch.sqrt(Data[2]))
        d2 = d1 - Data[3] * torch.sqrt(Data[2])
    
        # Return the call price calculation (using CDF of normal distribution)
        return Data[0] * normal_cdf(d1) - Data[1] * torch.exp(-0.05 * Data[2]) * normal_cdf(d2)

if __name__ == '__main__':
    # Define the variables
    T = 0.0822  # Time to expiration in years (adjust as needed)
    R = 'Call'  # Right (not used in calculations)
    K = 451.0   # Strike price
    price = (70.76 + 71.06) / 2  # Market price of the call option
    #IV = 0.7174  # Implied volatility
    IV = 1.7672  # Implied volatility
    S = 502.0  # Underlying asset price
    print(price)

    # Calculate the call option price
    call_price = calculate.put(S, K, T, IV)
    print(f"Call Option Price: {call_price.item():.4f}")
