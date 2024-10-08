# KL Divergence - Kullback-Leibler Divergence

KL divergence is a measure of how one probability distribution diverges from second, probability distribution.

Mathematically, for two probability distributions $P(x)$ and $Q(x)$, the KL divergence from $Q$ to $P$ is defined as:

$D_{KL}(p || q) = Σ p(x) * log\left(\frac{p(x)}{q(x)}\right)$ for discrete distributions 

$D_{KL}(p || q) = ∫ p(x) * log\left(\frac{p(x)}{q(x)}\right) dx$ for continuous distributions  

where the sum/integral is over all possible events $x$. And $p(x)$ and $q(x)$ are the probability density functions of distributions $P(x)$ and $Q(x)$ respectively.

## Intuition

KL divergence is a measure of how one probability distribution diverges from another. It is a measure of the information lost when $Q$ is used to approximate $P$.

## Properties

- KL divergence is not symmetric: $D_{KL}(P || Q) ≠ D_{KL}(Q || P)$
- KL divergence is always non-negative: $D_{KL}(P || Q) ≥ 0$
- KL divergence is 0 if and only if $P$ and $Q$ are the same distribution


