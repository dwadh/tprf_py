# tprf_py
Python implementation of the Three Pass Regression Filter [1] introduced in the paper by Bryan Kelly and Seth Pruitt:

This code replicates the simulation results but can be modified to work with other datasets. The implementation is based on my understanding of the paper, open to comments and suggestions. For comparison during the simulation, the code for Principal Components Regression (PCR) [2], LASSO version of the ‘‘targeted predictors’’ approach (PCLAS) [3] is also included.

The main file for execution is runner.py.
tprf_code contains the code for the three pass regression filter, PCR, PCLAS.
data_gen.py contains the code for generating the simualtion data for the Monte Carlo simulations specified in the paper.

## References
[1] Kelly, B., and Pruitt S.. 2015. The three-pass regression filter: A new approach to forecasting using many predictors. Journal of Econometrics 186:294–316.
[2] James H. Stock & Mark W. Watson (2012) Generalized Shrinkage Methods for Forecasting Using Many Predictors, Journal of Business & Economic Statistics, 30:4, 481-493, DOI: 10.1080/07350015.2012.715956
[3] Bai, J., & Ng, S. (2008). Forecasting economic time series using targeted predictors. Journal of Econometrics, 146(2), 304-317. doi:10.1016/j.jeconom.2008.08.010
