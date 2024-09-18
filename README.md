# FICO-Bucket

09/2024
JPMorgan Chase Forage Quantitative Research

# The Task

In this task, you are asked to assist the risk team in evaluating the FICO scores of borrowers in the mortgage portfolio. The team uses FICO scores as an indicator of a customer's likelihood of defaulting on their mortgage. However, since their machine learning model requires categorical data, the continuous FICO scores need to be grouped into meaningful "buckets." Your goal is to find the optimal boundaries for these buckets based on minimizing error or maximizing the log-likelihood of default rates within each bucket.

### Data

The dataset contains customer information, including their income, outstanding loans, years of employment, FICO scores, and whether or not they have defaulted. The key task is to categorize these FICO scores into ranges that best represent their risk of defaulting on a loan.

### Project Goals

- **Quantization**: Group continuous FICO scores into categories or "buckets" using optimization techniques.
- **Optimize bucket boundaries**: The task explores different approaches to finding the best bucket boundaries, such as minimizing mean squared error (MSE) or maximizing log-likelihood.
  
The optimal bucket boundaries will allow the team to assess the risk of default more accurately using categorical FICO scores.

### Steps Involved

1. **Data Preprocessing**: Prepare the dataset by loading and cleaning the relevant data fields, including FICO scores and default status.
2. **Bucket Optimization**: Use optimization techniques such as log-likelihood maximization to determine the boundaries that best represent the risk associated with different FICO score ranges.
3. **Categorization**: Assign each FICO score into a bucket based on the optimized boundaries.

# The Code

Key components of the code include:

- `calculate_ll(fico_scores, defaults, boundaries)`: A function that calculates the log-likelihood for the current bucket boundaries.
- `optimize(fico_scores, defaults, num_buckets)`: A function that finds the optimal bucket boundaries by maximizing the log-likelihood function through random sampling and iteration.

