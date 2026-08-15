# FICO-Bucket

A likelihood-based exercise for converting continuous FICO scores into five credit-risk categories.

> 09/2024
> JPMorgan Chase Forage Quantitative Research

# FICO Score Bucketing

## Project overview

Some credit models require categorical rather than continuous inputs. This notebook groups borrower FICO scores into buckets chosen to separate the observed default behaviour in the supplied loan dataset.

For a proposed set of boundaries, each bucket is assigned its empirical default probability. The code then evaluates the Bernoulli log-likelihood of the observed default outcomes and searches for boundaries with a higher likelihood.

## Why bucket FICO scores?

A raw FICO score is continuous, while some risk models and reporting systems operate with a smaller number of ordered credit categories. Useful boundaries should group borrowers with similar observed default behaviour while retaining differences in risk between groups.

The goal is therefore not simply to split the numerical range into equal widths. It is to find intervals whose within-bucket default rates explain the observed default labels more effectively.

## Data

The notebook uses the same 10,000-record loan dataset as the default-prediction task. Only two columns are required for the optimisation:

- `fico_score`, containing observed scores from 408 to 850; and
- `default`, the binary outcome used to estimate each bucket's probability of default.

## Likelihood calculation

For bucket $i$, the code records:

- $n_i$: the number of borrowers in the bucket;
- $k_i$: the number of those borrowers who defaulted; and
- $p_i=k_i/n_i$: the empirical default probability.

The score for a proposed set of boundaries is the sum of the Bernoulli log-likelihood contributions:

$$
\ell = \sum_i \left[k_i\log(p_i) + (n_i-k_i)\log(1-p_i)\right].
$$

Boundaries with a larger log-likelihood provide a better fit to the observed default outcomes under this simplified model.

## Method

The implementation contains two main functions:

- `calculate_ll(fico_scores, defaults, bucket_boundaries)` calculates the combined binomial log-likelihood across the buckets.
- `optimize(fico_scores, defaults, num_buckets)` starts from evenly spaced boundaries and compares them with 100 randomly sampled boundary sets.

`optimize(...)` follows these steps:

1. initialise six evenly spaced boundaries for five buckets;
2. calculate their log-likelihood;
3. sample four internal boundaries uniformly between the minimum and maximum FICO scores;
4. sort the proposed boundaries and add the observed minimum and maximum;
5. retain the proposal if its log-likelihood is higher; and
6. repeat the random search 100 times.

The selected boundaries are passed to `pandas.cut` to assign each borrower to one of five ordered categories. The saved notebook run produced the following example boundaries:

```text
408.00, 521.65, 571.00, 636.77, 688.89, 850.00
```

Because the search is random and no seed is fixed, rerunning the notebook can produce different boundaries.

The example output labels the resulting intervals as `Very Poor`, `Poor`, `Average`, `Good` and `Very Good`. These names describe the ordering used in the exercise; they are not official FICO rating categories.

## Tools used

- pandas for loading the data and assigning categories with `cut`; and
- NumPy for likelihood calculations, random boundary generation and rounding.

## Repository contents

```text
.
├── Loan Data.csv    # Borrower and default data supplied with the simulation
├── task4.ipynb      # Likelihood calculation and boundary search
└── README.md
```

## Scope and limitations

This implementation uses a small stochastic search rather than an exhaustive or dynamic-programming solution. Results are therefore not guaranteed to be globally optimal or reproducible. Empty buckets and buckets with empirical default probabilities of exactly zero or one also require more careful numerical treatment than this exercise provides. A production scorecard would additionally require minimum bucket sizes, monotonicity checks, stability testing and out-of-sample validation.
