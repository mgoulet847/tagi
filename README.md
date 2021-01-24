# tagi
Tractable Approximate Gaussian Inference (TAGI) is a method used in Bayesian neural networks developped by Goulet et al. (2020). This package is its R implementation. From TAGI, the package enables inference of the weights and baises posterior distributions, treats uncertainty in all layers and uses a single observation at a time for online inference of model parameters. It supports many activation functions and can solve regression problems.

# Installation

```{r}
install.packages("devtools")
devtools::install_github("mgoulet847/tagi")
```

# Examples
- Toy 1D Example: read ```vignette("ToyExample")```
- Boston Housing Regression Problem: read ```vignette("BostonHousing")```
- Insurance Regression Problem: read ```vignette("MedicalCost")```

Note: If the vignettes were not installed, you can run the following code. It will take approximately 15 minutes to run.
```{r}
devtools::install(build_vignettes = TRUE)
```
