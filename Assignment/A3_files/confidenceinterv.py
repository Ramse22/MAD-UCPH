#!/usr/bin/env python
# coding: utf-8
#
# # Confidence Intervals
#
# Jonas Peters, 20.11.2018
# Modified by Kim Steenstrup Pedersen, 23.11.2020

# You will have to modify the code in the places marked with TODO
# Notice that the code is constructed such that for small number of experiments
# (nexp) the code also makes a plot of the confidence interval from each experiment


import scipy.stats
import matplotlib.pyplot as plt
import numpy as np


# Fix the random generator seed
np.random.seed(111)


# Ground truth values
mu = 3.7
sigma = 2

# Number of samples in each experiment
n = 9

# Confidence level
gamma = 0.99  # 99 %

# Number of experiments to carry out
nexp = 10000


counter = 0
counter_c = 0
for i in range(nexp):
    x = np.random.normal(
        mu, sigma, n
    )  # simulates n realizations from a Gaussian with mean mu and var sigma^2
    sig = np.sqrt(np.var(x, ddof=1))
    fac1 = scipy.stats.norm.ppf(
        (1 - gamma) / 2, 0, 1
    )  # computes the 0.5% quantile of a Gaussian, roughly -2.576
    fac2 = scipy.stats.norm.ppf(
        (1 - gamma) / 2 + gamma, 0, 1
    )  # computes the 99.5% quantile of a Gaussian, roughly 2.576
    fac3 = scipy.stats.t.ppf(
        (1 - gamma) / 2, n - 1, 0, 1
    )  # 0.5% quantile for c) with student
    fac4 = scipy.stats.t.ppf(
        (1 - gamma) / 2 + gamma, n - 1, 0, 1
    )  # 99.5% quantile for c) with student
    xmean = np.mean(x)  # Sample mean
    a = xmean - fac2 * sig / np.sqrt(n)
    b = xmean - fac1 * sig / np.sqrt(n)
    ac = xmean - fac4 * sig / np.sqrt(n)  # for c)
    bc = xmean - fac3 * sig / np.sqrt(n)  # for c)

    # b) plotting and counting code
    if (a <= mu) & (mu <= b):
        if nexp < 1000:
            plt.figure(1)
            plt.plot((a, b), (i, i), "k-")
    else:
        counter = counter + 1
        if nexp < 1000:
            plt.figure(1)
            plt.plot((a, b), (i, i), "r-")

    # c) plotting and counting code
    if (ac <= mu) & (mu <= bc):
        if nexp < 1000:
            plt.figure(2)
            plt.plot((ac, bc), (i, i), "k-")
    else:
        counter_c = counter_c + 1
        if nexp < 1000:
            plt.figure(2)
            plt.plot((ac, bc), (i, i), "r-")


# Number of times the correct mu and confidence interval is not matching
print(str(100.0 * gamma) + "%-confidence interval:")
print(
    "b) Not matching in "
    + str(counter)
    + " (out of "
    + str(nexp)
    + ") experiments, "
    + str(100.0 * counter / nexp)
    + "%"
)
print(
    "c) Not matching in "
    + str(counter_c)
    + " (out of "
    + str(nexp)
    + ") experiments, "
    + str(100.0 * counter_c / nexp)
    + "%"
)


if nexp < 1000:
    plt.figure(1)
    plt.plot((mu, mu), (0, nexp), "b-")
    plt.xlabel("$\hat{\mu}$")
    plt.ylabel("Number of experiments")
    plt.title("b) The correct $\mu$ value in blue")

    plt.figure(2)
    plt.plot((mu, mu), (0, nexp), "b-")
    plt.xlabel("$\hat{\mu}$")
    plt.ylabel("Number of experiments")
    plt.title("c) The correct $\mu$ value in blue")
    plt.show()


# Exercise 4 question c):


def compute_T(k):
    D = np.array([1.0, 0.5, -0.5, 1.5, 0.5])
    D = np.concatenate([D for i in range(k)], axis=0)
    return np.mean(D) / np.sqrt(np.var(D, ddof=1) / D.size)


T = [compute_T(k) for k in range(1, 10)]
t = [scipy.stats.t.ppf(1 - 0.05, 5 * k - 1) for k in range(1, 10)]
k = range(1, 10)


plt.plot(k, T, "r")
plt.plot(k, t, "b")
plt.xlabel("number of duplicates of D")
plt.ylabel("T (red) and critical value (blue)")
plt.title(
    "computed T and critical value as a function of the number of duplication of D"
)
plt.show()
