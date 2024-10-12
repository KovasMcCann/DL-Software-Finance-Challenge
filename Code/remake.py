import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

class caclulate:
    def put(S, K, r, T, volatility):
        d1 = (torch.log(S / K) + (r + 0.5 * volatility ** 2) * T) / (volatility * torch.sqrt(T))
        d2 = d1 - volatility * torch.sqrt(T)
        return K * torch.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)

    def call(S, K, r, T, volatility):
        S_tensor = torch.tensor(S, dtype=torch.float32, device=device)
        K_tensor = torch.tensor(K, dtype=torch.float32, device=device)
        r_tensor = torch.tensor(r, dtype=torch.float32, device=device)
        T_tensor = torch.tensor(T, dtype=torch.float32, device=device)
        volatility_tensor = torch.tensor(volatility, dtype=torch.float32, device=device)

        d1 = (torch.log(S_tensor / K_tensor) + (r_tensor + 0.5 * volatility_tensor ** 2) * T_tensor) / (volatility_tensor * torch.sqrt(T_tensor))
        d2 = d1 - volatility_tensor * torch.sqrt(T_tensor)
    
        # Return the call price calculation (using CDF of normal distribution)
        return S_tensor * normal_cdf(d1) - K_tensor * torch.exp(-r_tensor * T_tensor) * normal_cdf(d2)


def newton(S, K, R, T, price):
    # Define the variables
    r = torch.tensor(0.01, dtype=torch.float32, device=device)  # Risk-free rate
    x0 = torch.tensor(0.5, dtype=torch.float32, device=device)  # Initial guess
    epsilon = torch.tensor(0.0001, dtype=torch.float32, device=device)  # Error tolerance
    max_iter = 10000  # Maximum number of iterations

    # Define the function and its derivative
    if R == 'Call':
        f = lambda x: caclulate.call(S, K, r, T, x) - price
    else:
        f = lambda x: caclulate.put(S, K, r, T, x) - price

    f_prime = lambda x: (f(x + epsilon) - f(x)) / epsilon

    # Implement the Newton-Raphson method
    for i in range(max_iter):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < epsilon:
            break
        x0 = x1

    print(f'Implied volatility: {x1:.4f}')

if __name__ == '__main__':
    # Define the variables
    T = 5  # Time to expiration in years
    R = 'Call' #Right
    K = 451.0  # Strike price
    price = (70.76 + 71.06) / 2  # Market price of the call option
    S = 502.0411  # Underlying asset price

    newton(S, K, R, T, price)