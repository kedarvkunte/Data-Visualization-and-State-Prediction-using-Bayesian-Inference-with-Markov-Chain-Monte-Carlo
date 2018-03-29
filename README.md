# Data Visualization and State Prediction using Bayesian Inference with Markov Chain Monte Carlo (MCMC)
In this project, the state of the object is predicted using Bayesian Inference Markov Chain Monte Carlo (MCMC) algorithm. For Data visualization, I used Matplotlib to correctly observe the Convergence to Stationary Distribution by plotting Histogram and Bar Charts. Successfully achieved low Mean Squared Error of 0.0003 for various test cases involving prior-posterior probability.

## Requirements
* Python 3.5 or higher
* Numpy
* Scipy
* Matplotlib
* Random
* Secrets

## Datasets
In this project, object state sequences are provided with various lengths. The task is to predict the probability of next state given the probabilty of the earlier states.
This task is achieved by calculating prior-posterior probability.

## Data Visualization
The best way to look that you are heading in the correct direction is to perform Data Visualization. Matplotlib is one of the greatest modules, which is helpful in visualizing data with Bar Charts, Histogram etc.

Here is the visualization for the convergence to steady state distribution:
![Convergence to Stationary Distribution](https://github.com/kedarvkunte/Data-Visualization-and-State-Prediction-using-Bayesian-Inference-with-Markov-Chain-Monte-Carlo/blob/master/Plots/Plot%20of%20B_mean%20vs%20Iterations%20for%201000000%20iterations.png)

There is other way to visualize as well as shown in the following figure.
![Convergence to Stationary Distribution other way round](https://github.com/kedarvkunte/Data-Visualization-and-State-Prediction-using-Bayesian-Inference-with-Markov-Chain-Monte-Carlo/blob/master/Plots/Plot%20of%20alpha%20as%20function%20of%20iterations.png)

The algorithm for Metropolis Hastings is as follows:
```
def MCMC(J, B, alpha, iterations):
  for i = 1 to iterations:
    mean_alpha = 0
    mean_J = 0
    (new_alpha, new_J) = proposed_function(alpha, J)
    ratio_of_accept = Probability(new_alpha, new_J,B)/Probability(alpha,J,B)
    if uniform_random_number_between_0_and_1 <= ratio_of_accept:
      alpha = new_alpha
      J = new_J
    mean_alpha += alpha
    mean_J += J
return (mean_J/iterations,mean_alpha/iterations)
```
Here J can be considered jar states and B can be considered as a ball states. Alpha can be considered as the probability to select the same jar at the subsequent level and draw another ball from it.

Here are some of the bar charts and histograms, I got

|                           |                           |                           |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![Histogram 1](https://github.com/kedarvkunte/Data-Visualization-and-State-Prediction-using-Bayesian-Inference-with-Markov-Chain-Monte-Carlo/blob/master/Plots/Histogram%201.png)|![Histogram 2](https://github.com/kedarvkunte/Data-Visualization-and-State-Prediction-using-Bayesian-Inference-with-Markov-Chain-Monte-Carlo/blob/master/Plots/Histogram%202.png)|![Bar Chart 1](https://github.com/kedarvkunte/Data-Visualization-and-State-Prediction-using-Bayesian-Inference-with-Markov-Chain-Monte-Carlo/blob/master/Plots/Bar%20Chart%201.png)|
    
## Error Analysis

In this case, I used Kaggle Prediction to obtain **mean squared error of 0.00032.** The error can be further improved by improving the number of iterations greater than 1000000.




