#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('./sampleData.csv')
returns = data.pct_change()

crra = 5

alpha = returns.dropna().mean()
sigma = returns.dropna().cov()
inv_sigma = np.linalg.inv(sigma)
myOnes = np.ones(34)

gamma = (alpha.T @ inv_sigma @ myOnes - crra)/(myOnes.T @ inv_sigma @ myOnes)
w = inv_sigma @ (alpha - myOnes*gamma) / crra

plt.plot(w)
plt.title("Simple weights")
plt.show()
plt.close()

print("Sum of weights is " + str(sum(w)))
print("Sum of absolute weights is " + str(sum(abs(w))))