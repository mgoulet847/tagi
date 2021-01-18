#' Medical Cost of 1,338 insureds.
#'
#' A dataset containing the medical costs ("charges") and other attributes of
#' 1,338 insureds.
#'
#' @format A data frame with 1,338 rows and 10 variables:
#' \describe{
#'   \item{age}{age of the insured}
#'   \item{sex}{gender of the insured, binary (if female)}
#'   \item{BMI}{Body Mass Index of the insured}
#'   \item{children}{number of children covered as dependents}
#'   \item{smoker}{smoking status, binary (if the insured smokes)}
#'   \item{region: northeast}{binary (if the insured lives in that region)}
#'   \item{region: southeast}{binary (if the insured lives in that region)}
#'   \item{region: southwest}{binary (if the insured lives in that region)}
#'   \item{region: northwest}{binary (if the insured lives in that region)}
#'   \item{charges}{medical costs, in US dollars)}
#' }
#'
#' @details The original dataset contains 7 variables, but one-hot encoding was
#' used on the "region" categorical variable.
#' @source \url{https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv}
"MedicalCost"

#' Price of 506 Boston houses.
#'
#' This dataset was originally from the StatLib archive. It contains the price
#' and other attributes of 506 Boston houses.
#'
#' @format A data frame with 506 rows and 14 variables:
#' \describe{
#'   \item{CRIM }{per capita crime rate by town}
#'   \item{ZN}{proportion of residential land zoned for lots over 25,000 sq.ft.}
#'   \item{INDUS}{proportion of non-retail business acres per town}
#'   \item{CHAS}{Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)}
#'   \item{NOX}{nitric oxides concentration (parts per 10 million)}
#'   \item{RM}{average number of rooms per dwelling}
#'   \item{AGE}{proportion of owner-occupied units built prior to 1940}
#'   \item{DIS}{weighted distances to five Boston employment centres}
#'   \item{RAD}{index of accessibility to radial highways}
#'   \item{TAX}{full-value property-tax rate per $10,000}
#'   \item{PTRATIO}{pupil-teacher ratio by town}
#'   \item{B}{1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town}
#'   \item{LSTAT}{% lower status of the population}
#'   \item{MEDV}{median value of owner-occupied homes in $1000's}
#' }
#'
#' @details The dataset from the TAGI repository was used for comparison purposes,
#' but the original dataset was published by Harrison, D. and Rubinfeld, D.L.
#'
#' @references http://lib.stat.cmu.edu/datasets/boston
#' @references Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand
#' for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.
#'
#' @source \url{https://github.com/CivML-PolyMtl/TAGI/blob/master/BostonHousing/data/BostonHousing.mat}
"BH"

#' Inputs used in validation part for 1D toy problem
#'
#' The orignal dataset represents a 1D regression problem from (Hernández-Lobato & Adams, 2015):
#' \eqn{y = x^{3} + \epsilon} where \eqn{\epsilon \sim \mathcal{N}(0,9)} and \eqn{x \in [\,-4,4]\,}. In this dataset, \eqn{x} and \eqn{y} are normalized.
#'
#' @format A data frame with 20 rows and 1 variable \eqn{x}
#'
#' @details The dataset generated with the seed from the TAGI repository was used for comparison purposes.
#'
#' @references Hernández-Lobato, J. M., & Adams, R. "Probabilistic
#' backpropagation for scalable learning of bayesian neural networks."
#' International Conference on Machine Learning. 2015.
#'
#' @source \url{https://github.com/CivML-PolyMtl/TAGI/blob/master/ToyExample/ToyExample_1D.m}
"ToyExample.x_val"

#' Inputs used in training part for 1D toy problem
#'
#' The orignal dataset represents a 1D regression problem from (Hernández-Lobato & Adams, 2015):
#' \eqn{y = x^{3} + \epsilon} where \eqn{\epsilon \sim \mathcal{N}(0,9)} and \eqn{x \in [\,-4,4]\,}. In this dataset, \eqn{x} and \eqn{y} are normalized.
#'
#' @format A data frame with 20 rows and 1 variable \eqn{x}
#'
#' @details The dataset generated with the seed from the TAGI repository was used for comparison purposes.
#'
#' @references Hernández-Lobato, J. M., & Adams, R. "Probabilistic
#' backpropagation for scalable learning of bayesian neural networks."
#' International Conference on Machine Learning. 2015.
#'
#' @source \url{https://github.com/CivML-PolyMtl/TAGI/blob/master/ToyExample/ToyExample_1D.m}
"ToyExample.x_obs"

#' Responses used in validation part for 1D toy problem
#'
#' The orignal dataset represents a 1D regression problem from (Hernández-Lobato & Adams, 2015):
#' \eqn{y = x^{3} + \epsilon} where \eqn{\epsilon \sim \mathcal{N}(0,9)} and \eqn{x \in [\,-4,4]\,}. In this dataset, \eqn{x} and \eqn{y} are normalized.
#'
#' @format A data frame with 20 rows and 1 variable \eqn{y}
#'
#' @details The dataset generated with the seed from the TAGI repository was used for comparison purposes.
#'
#' @references Hernández-Lobato, J. M., & Adams, R. "Probabilistic
#' backpropagation for scalable learning of bayesian neural networks."
#' International Conference on Machine Learning. 2015.
#'
#' @source \url{https://github.com/CivML-PolyMtl/TAGI/blob/master/ToyExample/ToyExample_1D.m}
"ToyExample.y_val"

#' Responses used in training part for 1D toy problem
#'
#' The orignal dataset represents a 1D regression problem from (Hernández-Lobato & Adams, 2015):
#' \eqn{y = x^{3} + \epsilon} where \eqn{\epsilon \sim \mathcal{N}(0,9)} and \eqn{x \in [\,-4,4]\,}. In this dataset, \eqn{x} and \eqn{y} are normalized.
#'
#' @format A data frame with 20 rows and 1 variable \eqn{y}
#'
#' @details The dataset generated with the seed from the TAGI repository was used for comparison purposes.
#'
#' @references Hernández-Lobato, J. M., & Adams, R. "Probabilistic
#' backpropagation for scalable learning of bayesian neural networks."
#' International Conference on Machine Learning. 2015.
#'
#' @source \url{https://github.com/CivML-PolyMtl/TAGI/blob/master/ToyExample/ToyExample_1D.m}
"ToyExample.y_obs"
