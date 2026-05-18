import pandas as pd
import numpy as np
from scipy.stats import norm

def historical_var(returns, confidence=0.99):
    """Calculate VaR using Historical Simulation"""
    return np.percentile(returns, 100 * (1-confidence))

def parametric_var(returns, confidence=0.99, portfolio_value=10000000):
    """Calculate VaR using Variance-Covariance method"""
    mu = np.mean(returns)
    sigma = np.std(returns)
    var_percent = mu - sigma * norm.ppf(confidence)
    return var_percent * portfolio_value

def monte_carlo_var(returns, n_simulations=10000, confidence=0.99):
    """Monte Carlo VaR with 10,000 iterations"""
    mu, sigma = np.mean(returns), np.std(returns)
    sim_returns = np.random.normal(mu, sigma, n_simulations)
    return np.percentile(sim_returns, 100 * (1-confidence))

def expected_shortfall(returns, confidence=0.99):
    """Calculate Expected Shortfall/CVaR"""
    var = historical_var(returns, confidence)
    return returns[returns <= var].mean()

# Example usage with NSE 50 data
if __name__ == "__main__":
    # Replace with your NSE 50 returns data
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    print(f"Historical 1-day 99% VaR: {historical_var(returns):.2%}")
    print(f"Parametric VaR on ₹10M: ₹{parametric_var(returns):,.0f}")
    print(f"Monte Carlo VaR: {monte_carlo_var(returns):.2%}")
    print(f"Expected Shortfall: {expected_shortfall(returns):.2%}")
