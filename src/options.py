import numpy as np
import itertools
from scipy import stats

class BasketOption:
    '''Functions for a European basket call option.'''
    
    def __init__(self, weights, prices, vol, corr, strike, time, rate):
        '''           
        Parameters
        ----------
        weights : ndarray
            Floats representing weights of the underlying assets in the basket. 
            Should sum to 1, be 1-D, and be of length equal to the length of prices.
        prices : ndarray
            Floats representing the asset prices at time zero. Should be 1-D, and same
            length as prices.
        vol : float
            The volatility of the assets. N.B. the Levy formula assumes homogeneous asset
            volatility.
        corr : ndarray
            Correlation matrix of the assets. Should be of shape (n,n), where n is the
            number of assets.
        strike : float
            Strike price.
        time : float
            Time to maturity.
        rate : float
            Riskless interest rate.
        '''
        self.weights = weights
        self.prices = prices
        self.vol = vol
        self.corr = corr
        self.strike = strike
        self.time = time
        self.rate = rate
    
        if not len(weights) == len(prices) == len(corr):
            raise ValueError('Number of weights, prices, corr rows should be equal')
            
        if abs(1-sum(weights))>0.01:
            raise ValueError('The weights must cumulatively sum to 1.0')
            
    def get_levy_price(self):
        """
        Use the Levy formula to approximate the price of a European basket call option.
        """
    
        discount = np.exp(-self.rate*self.time)
    
        # First moment of T-forward prices (also the basket T-forward price)
        m1 = np.sum(self.weights * self.prices * discount)

        # Second moment of T-forward prices
        w_ij, f_ij = [list(map(lambda x: np.product(x), list(itertools.product(q, q)))) 
                          for q in [self.weights, self.prices * discount]]
        m2 = np.sum(np.array(w_ij) * np.array(f_ij)
                    * np.exp(self.corr.flatten() * self.vol**2 * self.time))
    
        vol_basket = ( self.time**(-1) * np.log(m2 / m1**2) )**(0.5)
    
        # Parameters of the price formula
        d1 = np.log(m1 / self.strike)/(vol_basket * self.time**(0.5))\
                + (vol_basket * self.time**(0.5))/2
        d2 = d1 - vol_basket * self.time**(0.5)

        # Levy formula for basket call option price
        self.levy_price = discount * (m1 * stats.norm.cdf(d1) - self.strike * stats.norm.cdf(d2))

        return self.levy_price