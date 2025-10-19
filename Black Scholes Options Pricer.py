import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def CallPrice(S,K,risk,T,sigma):
    d_1 = (np.log(S/K) + (risk + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return S * norm.cdf(d_1) - K* np.exp(-risk*T)*norm.cdf(d_2)

def PutPrice(S,K,risk,T,sigma):
    d_1 = (np.log(S/K) + (risk + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return K * np.exp(-risk*T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

low = float(input('Input the lowest stock price value'))

high = float(input('Input the highest stock price value'))

S = np.arange(low,high,0.1)

#S= float(input('Input the stock price'))
K = float(input('Input the strike price'))
risk = float(input('Input the risk free interest'))
T = float(input('Input the exercise price'))
sigma = float(input('Input the Volatility'))

#print (CallPrice(S,K,risk,T,sigma))
#print (PutPrice(S,K,risk,T,sigma))
calls = [CallPrice(s, K, T, risk, sigma) for s in S]
puts = [PutPrice(s, K, T, risk, sigma) for s in S]
plt.plot(S, calls, label='Call Value')
plt.plot(S, puts, label='Put Value')
plt.xlabel('$S_0$')
plt.ylabel(' Value')
plt.legend()
