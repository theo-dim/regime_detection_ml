#%%

# Import Packages:
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import PolynomialFeatures

#%%

# SPY also captures volatility well. BBUS too, but there is not enough data. Mainly, they are both US ETFs and are not suitable for the argument I want to make.
# https://www.ishares.com/us/products/239623/ishares-msci-eafe-etf
# is composed of large- and mid-cap developed market equities, excluding the US and Canada.

etf = yf.download('EFA', start='2000-01-01', end='2020-09-24')
print(etf)
etf = etf[['Adj Close', 'Volume']] #Using adjusted close (transformed into daily returns) and volume. What other feature of the set can describe volatility?
print(etf)

#%%

# Initiate HMM instance
hmm_1 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)

#%%

# Transform adjusted close prices to daily returns;
# Remove first adjusted close entry becomes null after the transformation;
# A is our arbitrary Borel set (measurable):
A = np.column_stack([etf['Adj Close'].pct_change()[1:], etf['Volume'][1:]])
print(A[:10])

#%%

# Fit model to set A:
hmm_1.fit(A)

#%%

# Predict states through the HMM instantiation above:
hmm_states = hmm_1.predict(A)
print(hmm_states)

#%%

# Filter states:
hmm_state_0 = hmm_states == 0
hmm_state_1 = hmm_states == 1

#%%

# Extract column for convenience:
close_price = etf['Adj Close'][1:]
print(close_price[:10])

#%%

# Plot price data:
default = 'black'
fig = plt.figure(figsize=(25,15))
# Data:
close_price_line = fig.add_subplot(111)
plt.plot_date(close_price.index,close_price, marker='*', ms=5, color='black', alpha=0.3, label='EFA Adj. Close')
# Aesthetic adjustments:
close_price_line.set_title('Adjusted Closing Price, EFA', fontsize=30, color=default)
close_price_line.set_xlabel("Date", fontsize=20, color=default)
close_price_line.set_ylabel("Closing Price", fontsize=20, color=default)
close_price_line.tick_params(axis='both', labelsize=15, colors=default)
close_price_line.legend(loc='upper left', fontsize=15)
# Generate and save graph:
plt.savefig('adj_close_graph.png')
plt.show()

#%%

var_hidden_state_0 = np.cov(close_price[hmm_state_0])
print(var_hidden_state_0)
var_hidden_state_1 = np.cov(close_price[hmm_state_1])
print(var_hidden_state_1)

# Plot states:
default = 'black'
fig = plt.figure(figsize=(25,15))
# Data:
markov_states = fig.add_subplot(111)
plt.plot_date(close_price.index[hmm_state_0],close_price[hmm_state_0], marker='*', ms=5, alpha=0.4, label='Hidden State 0')
plt.plot_date(close_price.index[hmm_state_1],close_price[hmm_state_1], marker='*', ms=5, alpha=0.4, label='Hidden State 1')
# Aesthetic adjustments:
markov_states.set_title('Hidden Markov States', fontsize=30, color=default)
markov_states.set_xlabel("Date", fontsize=20, color=default)
markov_states.set_ylabel("Closing Price", fontsize=20, color=default)
markov_states.tick_params(axis='both', labelsize=15, colors=default)
markov_states.legend(loc='upper left', fontsize=15)
# Generate and save graph:
plt.savefig('hmm_graph.png')
plt.show()

#%%

data = etf['Adj Close'].tolist()

#%%

# Creating monthly partitions/windows that the algorithm uses to predict the current regime.
out = []
for i in range(len(data)):
    train_window = np.array(data[i - 30:i])
    try:
        out.append(train_window / train_window[0])
    except:
        out.append(np.ones(30))

#%%

A = np.array(out)
print(A)

#%%

# Using the Radial Basis Function kernel with default nu value
svm_1 = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)

#%%

svm_1.fit(PolynomialFeatures(degree=3).fit_transform(A))
print(svm_1)
hmm_states = svm_1.predict(PolynomialFeatures(degree=3).fit_transform(A))
hmm_states_rw = pd.Series(hmm_states).rolling(window=10).mean().fillna(1)
print(hmm_states_rw)

#%%

# Filter HSVM states:
hmm_states_rw[hmm_states_rw >= 0] = 1
hmm_states_rw[hmm_states_rw < 0] = 0
hmm_states = hmm_states_rw
hmm_states.index = etf.index
hmm_state_0_cluster = hmm_states == 0
hmm_state_1_cluster = hmm_states == 1

#%%

# Plot Hidden SVM states:
default = 'black'
fig = plt.figure(figsize=(25,15))
# Data:
hidden_svm_states = fig.add_subplot(111)
plt.plot_date(etf.index[hmm_state_0_cluster],etf['Adj Close'][hmm_state_0_cluster], marker='*', ms=5, alpha=0.4, color='C1', label='Cluster 0')
plt.plot_date(etf.index[hmm_state_1_cluster],etf['Adj Close'][hmm_state_1_cluster], marker='*', ms=5, alpha=0.4, color='C0',label='Cluster 1')
# Aesthetic adjustments:
hidden_svm_states.set_title('SVM Clusters', fontsize=30, color=default)
hidden_svm_states.set_xlabel("Date", fontsize=20, color=default)
hidden_svm_states.set_ylabel("Closing Price", fontsize=20, color=default)
hidden_svm_states.tick_params(axis='both', labelsize=15, colors=default)
hidden_svm_states.legend(loc='upper left', fontsize=15)
# Generate and save graph:
plt.savefig('hmm_svm_graph.png')
plt.show()
