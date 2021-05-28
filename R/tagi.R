#' One iteration of the Tractable Approximate Gaussian Inference (TAGI)
#'
#' This function goes through one learning iteration of the neural network model
#' using TAGI.
#'
#' @param NN List that contains the structure of the neural network
#' @param mp List that contains mean vectors of the parameters for each layer \eqn{\mu_{\theta}}
#' @param Sp List that contains covariance matrices of the \eqn{\Sigma_{\theta}}
#' parameters for each layer
#' @param x Set of input data
#' @param y Set of corresponding responses
#' @return A list that contains:
#' @return - \code{mp}: List that contains the updated mean vectors of the parameters
#' for each layer \eqn{\mu_{\theta}}
#' @return - \code{Sp}: List that contains the updated covariance matrices of the
#' parameters for each layer \eqn{\Sigma_{\theta}}
#' @return - \code{zn}: Predicted responses
#' @return - \code{Szn}: Variance vector of predicted responses
#' @export
network <- function(NN, mp, Sp, x, y){
  # Initialization
  numObs = nrow(x)
  numCovariates = NN$nx
  zn = matrix(0, numObs, NN$ny)
  Szn = matrix(0, numObs, NN$ny)

  # Loop
  loop = 0

  for (i in seq(from = 1, to = numObs, by = NN$batchSize)){
    loop = loop + 1
    idxBatch = i:(i + NN$batchSize - 1)
    xloop = matrix(t(x[idxBatch,]), nrow = length(idxBatch) * numCovariates, ncol = 1)

    # Training
    if (NN$trainMode == 1){
      yloop = matrix(t(y[idxBatch,]), nrow = length(idxBatch) * NN$ny, ncol = 1)
      out_feedForward = feedForward(NN, xloop, mp, Sp)
      mz = out_feedForward[[1]]
      Sz = out_feedForward[[2]]
      Czw = out_feedForward[[3]]
      Czb = out_feedForward[[4]]
      Czz = out_feedForward[[5]]
      out_feedBackward = feedBackward(NN, mp, Sp, mz, Sz, Czw, Czb, Czz, yloop)
      mp = out_feedBackward[[1]]
      Sp = out_feedBackward[[2]]
      zn[idxBatch,] = t(matrix(mz[[length(mz)]], nrow = NN$ny, ncol = length(idxBatch)))
      Szn[idxBatch,] = t(matrix(Sz[[length(Sz)]], nrow = NN$ny, ncol = length(idxBatch)))
    }

    # Testing
    else {
      out_feedForward = feedForward(NN, xloop, mp, Sp)
      mz = out_feedForward[[1]]
      Sz = out_feedForward[[2]]
      zn[idxBatch,] = t(matrix(mz, nrow = NN$ny, ncol = length(idxBatch)))
      Szn[idxBatch,] = t(matrix(Sz, nrow = NN$ny, ncol = length(idxBatch)))
    }
  }
  outputs <- list(mp, Sp, zn, Szn)
  return(outputs)
}

#' Forward Uncertainty Propagation
#'
#' This function feeds the neural network forward from input data to
#' responses.
#'
#' @param NN List that contains the structure of the neural network
#' @param mp List that contains mean vectors of the parameters for each layer \eqn{\mu_{\theta}}
#' @param Sp List that contains covariance matrices of the
#' parameters for each layer \eqn{\Sigma_{\theta}}
#' @param x Set of input data
#' @return A list that contains:
#' @return - \code{mz}: List that contains the mean vectors of the units
#' for each layer \eqn{\mu_{Z}}
#' @return - \code{Sz}: List that contains the covariance matrices of the
#' units for each layer \eqn{\Sigma_{Z}}
#' @return - \code{Czw}: List that contains the covariance matrices between
#' units and weights for each layer \eqn{\Sigma_{ZW}}
#' @return - \code{Czb}: List that contains the covariance matrices between
#' units and biases for each layer \eqn{\Sigma_{ZB}}
#' @return - \code{Czz}: List that contains the covariance matrices between
#' previous and current units for each layer \eqn{\Sigma_{ZZ^{+}}}
#' @export
feedForward <- function(NN, x, mp, Sp){

  # Initialization
  numLayers = length(NN$nodes)
  hiddenLayerActFunIdx = activationFunIndex(NN$hiddenLayerActivation)

  # Activation unit
  ma = matrix(list(), nrow = numLayers, ncol = 1)
  ma[[1,1]] = x
  Sa = matrix(list(), nrow = numLayers, ncol = 1)

  # Hidden states
  mz = matrix(list(), nrow = numLayers, ncol = 1)
  Czw = matrix(list(), nrow = numLayers, ncol = 1)
  Czb = matrix(list(), nrow = numLayers, ncol = 1)
  Czz = matrix(list(), nrow = numLayers, ncol = 1)
  Sz = matrix(list(), nrow = numLayers, ncol = 1)
  J = matrix(list(), nrow = numLayers, ncol = 1)

  # Hidden Layers
  for (j in 2:numLayers){
    mz[[j,1]] = meanMz(mp[[j-1,1]], ma[[j-1,1]], NN$idxFmwa[j-1,], NN$idxFmwab[[j-1,1]])
    # Covariance for z^(j)
    Sz[[j,1]] = covarianceSz(mp[[j-1,1]], ma[[j-1,1]], Sp[[j-1,1]], Sa[[j-1,1]],NN$idxFmwa[j-1,], NN$idxFmwab[[j-1,1]])
    if (NN$trainMode == 1){
      # Covariance between z^(j) and w^(j-1)
      out = covarianceCzp(ma[[j-1,1]], Sp[[j-1,1]], NN$idxFCwwa[j-1,], NN$idxFCb[[j-1,1]])
      Czb[[j,1]] = out[[1]]
      Czw[[j,1]] = out[[2]]
      # Covariance between z^(j+1) and z^(j)
      if (!(is.null(Sz[[j-1,1]]))){
        Czz[[j,1]] = covarianceCzz(mp[[j-1,1]], Sz[[j-1,1]], J[[j-1,1]], NN$idxFCzwa[j-1,])
      }
    }

    # Activation
    if (j < numLayers){
      out_act = meanA(mz[[j,1]], mz[[j,1]], hiddenLayerActFunIdx)
      ma[[j,1]] = out_act[[1]]
      J[[j,1]] = out_act[[2]]
      Sa[[j,1]] = covarianceSa(J[[j,1]], Sz[[j,1]])
    }
  }

  # Outputs
  if (!(NN$outputActivation == "linear")){
    ouputLayerActFunIdx = activationFunIndex(NN$outputActivation)
    out_feedForward = meanA(mz[[numLayers,1]], mz[[numLayers,1]], ouputLayerActFunIdx)
    mz[[numLayers,1]] = out_feedForward[[1]]
    J[[numLayers,1]] = out_feedForward[[2]]
    Sz[[numLayers,1]] = covarianceSa(J[[numLayers,1]], Sz[[numLayers,1]])
  }
  if (NN$trainMode == 0){
    mz = mz[[numLayers,1]]
    Sz = Sz[[numLayers,1]]
  }
  outputs <- list(mz, Sz, Czw, Czb, Czz)
  return(outputs)
}

#' Forward Uncertainty Propagation for Derivative Calculation
#'
#' This function feeds the neural network forward from input data to
#' responses and considers components required for derivative calculations.
#'
#' @param NN List that contains the structure of the neural network
#' @param theta List of parameters
#' @param states List of states
#' @return A list that contains:
#' @return - \code{states}: List that contains states
#' @return - \code{mda}: Mean vector of activation units' derivative
#' @return - \code{Sda}: Covariance matrix of activation units' derivative
#' @return - \code{mdda}: Mean vector of activation units' second derivative
#' @return - \code{Sdda}: Covariance matrix of activation units' second derivative
#' @export
feedForwardPass <- function(NN, theta, states){

  # Initialization
  out_extractParameters <- extractParameters(theta)
  mw = out_extractParameters[[1]]
  Sw = out_extractParameters[[2]]
  mb = out_extractParameters[[3]]
  Sb = out_extractParameters[[4]]
  out_extractStates <- extractStates(states)
  mz = out_extractStates[[1]]
  Sz = out_extractStates[[2]]
  ma = out_extractStates[[3]]
  Sa = out_extractStates[[4]]
  J = out_extractStates[[5]]
  mdxs = out_extractStates[[6]]
  Sdxs = out_extractStates[[7]]
  mxs = out_extractStates[[8]]
  Sxs = out_extractStates[[9]]
  numLayers = length(NN$nodes)
  actFunIdx = NN$actFunIdx
  actBound = NN$actBound
  B = NN$batchSize
  rB = NN$repBatchSize
  nodes = NN$nodes
  numParamsPerLayer_2 = NN$numParamsPerLayer_2

  # Derivative
  mda = matrix(list(), nrow = numLayers, ncol = 1)
  Sda = matrix(list(), nrow = numLayers, ncol = 1)
  mdda = matrix(list(), nrow = numLayers, ncol = 1)
  Sdda = matrix(list(), nrow = numLayers, ncol = 1)
  mda[[1,1]] = rep(1, nrow(mz[[1,1]]))
  Sda[[1,1]] = rep(0, nrow(Sz[[1,1]]))
  mdda[[1,1]] = rep(0, nrow(mz[[1,1]]))
  Sdda[[1,1]] = rep(0, nrow(Sz[[1,1]]))

  # Hidden Layers
  for (j in 2:numLayers){
    idxw = (numParamsPerLayer_2[1, j-1]+1):numParamsPerLayer_2[1, j]
    idxb = (numParamsPerLayer_2[2, j-1]+1):numParamsPerLayer_2[2, j]
    # Max pooling
    if (NN$layer[j] == NN$layerEncoder$fc){
      if ((B == 1) & (rB == 1)){
        out_fcMeanVarB1 <- fcMeanVarB1(mw[idxw], Sw[idxw], mb[idxb], Sb[idxb], ma[[j-1,1]], Sa[[j-1,1]], nodes[j-1], nodes[j])
        mz[[j,1]] = out_fcMeanVarB1[[1]]
        Sz[[j,1]] = out_fcMeanVarB1[[2]]
      } else {
        out_fcMeanVar <- fcMeanVar(mz[[j,1]], Sz[[j,1]], mw[idxw], Sw[idxw], mb[idxb], Sb[idxb], ma[[j-1,1]], Sa[[j-1,1]], nodes[j-1], nodes[j], B, rB)
        mz[[j,1]] = out_fcMeanVar[[1]]
        Sz[[j,1]] = out_fcMeanVar[[2]]
      }
    }

    # Activation
    if (actFunIdx[j] != 0){
      out_meanVar = meanVar(mz[[j,1]], mz[[j,1]], Sz[[j,1]], actFunIdx[j])
      ma[[j,1]] = out_meanVar[[1]]
      Sa[[j,1]] = out_meanVar[[2]]
      J[[j,1]] = out_meanVar[[3]]
    } else {
      ma[[j,1]] = mz[[j,1]]
      Sa[[j,1]] = Sz[[j,1]]
      J[[j,1]]= matrix(1, nrow(mz[[j,1]]), ncol(mz[[j,1]]))
    }

    # Derivative for FC
    if ((NN$collectDev > 0) & (actFunIdx[j] != 0)){
      out_meanVarDev = meanVarDev(mz[[j,1]], Sz[[j,1]], actFunIdx[j], actBound[j])
      mda[[j,1]] = out_meanVarDev[[1]]
      Sda[[j,1]] = out_meanVarDev[[2]]
      mdda[[j,1]] = out_meanVarDev[[3]]
      Sdda[[j,1]] = out_meanVarDev[[4]]
    }
  }
  states <- compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs)
  outputs <- list(states, mda, Sda, mdda, Sdda)
  return(outputs)
}

#' Derivative Calculation
#'
#' This function does derivative calculations.
#'
#' @param NN List that contains the structure of the neural network
#' @param theta List of parameters
#' @param states List of states
#' @param mda Mean vector of activation units' derivative
#' @param Sda Covariance matrix of activation units' derivative
#' @param mdda Mean vector of activation units' second derivative
#' @param Sdda Covariance matrix of activation units' second derivative
#' @param dlayer Layer from which derivatives will be in respect to
#' @return A list that contains:
#' @return - \code{mdg}: Mean vector of derivative of the output layer
#' @return - \code{Sdg}: Variance of derivative of the output layer
#' @return - \code{Cdgz}: Covariance of derivative of the output layer
#' @return - \code{mddg}: Mean vector of second derivative of the output layer
#' @export
derivative <- function(NN, theta, states, mda, Sda, mdda, Sdda, dlayer){

  # Initialization
  out_extractParameters <- extractParameters(theta)
  mw = out_extractParameters[[1]]
  Sw = out_extractParameters[[2]]
  out_extractStates <- extractStates(states)
  Sz = out_extractStates[[2]]
  ma = out_extractStates[[3]]
  Sa = out_extractStates[[4]]
  J = out_extractStates[[5]]
  numLayers = length(NN$nodes)
  actFunIdx = NN$actFunIdx
  actBound = NN$actBound
  B = NN$batchSize
  rB = NN$repBatchSize
  nodes = c(NN$nodes, NN$nodes[length(NN$nodes)])
  layer = NN$layer
  numParamsPerLayer_2 = NN$numParamsPerLayer_2

  # Derivative
  mdg = createDevCellarray(nodes, numLayers, B, rB)
  Sdg = mdg
  Cdgz = mdg
  mdge = mdg
  mddg = mdg
  Sddg = mdg

  for (j in (numLayers-1):dlayer){
    idxw = (numParamsPerLayer_2[1, j]+1):numParamsPerLayer_2[1, j+1]
    if (NN$layer[j+1] == NN$layerEncoder$fc){
      if ((NN$collectDev > 0) & (j == (numLayers-1))){
        out_fcMeanVarDnode <- fcMeanVarDnode(mw[idxw], Sw[idxw], mda[[j,1]], Sda[[j,1]], nodes[j], nodes[j+1], B)
        mdgk = out_fcMeanVarDnode[[1]]
        Sdgk = out_fcMeanVarDnode[[2]]
        out_fcCovaz <- fcCovaz(J[[j+1,1]], J[[j,1]], Sz[[j,1]], mw[idxw], nodes[j], nodes[j+1], B)
        Caizi = out_fcCovaz[[1]]
        Caozi = out_fcCovaz[[2]]
        out_fcCovdz <- fcCovdz(ma[[j+1,1]], ma[[j,1]], Caizi, Caozi, actFunIdx[j+1], actFunIdx[j], nodes[j], nodes[j+1], B)
        Cdozi = out_fcCovdz[[1]]
        Cdizi = out_fcCovdz[[2]]
        mdg[[j,1]] = matrix(rowSums(mdgk), nrow(mdgk), 1)
        Sdg[[j,1]] = matrix(rowSums(Sdgk), nrow(Sdgk), 1)
        mdge[[j,1]] = mdgk
        Cdgk = covdx(0, mw[idxw], rep(0, NN$ny*B), 0, rep(1, NN$ny*B), Cdozi, Cdizi, nodes[j], nodes[j+1], 1, B)
        Cdgz[[j,1]] = matrix(rowSums(Cdgk), nrow(Cdgk), 1)

        # For second and higher order derivative
        if (NN$collectDev > 1){
          out_fcMeanVarDnode <- fcMeanVarDnode(mw[idxw], Sw[idxw], mdda[[j,1]], Sdda[[j,1]], nodes[j], nodes[j+1], B)
          mddgk = out_fcMeanVarDnode[[1]]
          Sddgk = out_fcMeanVarDnode[[2]]
          mddg[[j,1]] = matrix(rowSums(mddgk), nrow(mddgk), 1)
          Sddg[[j,1]] = matrix(rowSums(Sddgk), nrow(Sddgk), 1)

          # Matrix of combination types (1 when second order derivative is used, >1 thereafter, 0 when not still used)
          # Read from RIGHT to LEFT (column number corresponds to layer number)
          # (e.g. in 2nd order derivative, only one product wd is of second order, then the other products are first order and multiplied with wd products of their layer.
          # Terms before product' 2nd order derivative are of first order and NOT multiplied with wd products of their layer)
          combinations_matrix = apply(upper.tri(diag(numLayers-1), diag = TRUE),1,cumsum)
          mddg_combinations = matrix(list(createDevCellarray(nodes, numLayers, B, rB)), nrow = nrow(combinations_matrix), ncol = 1)

          for (i in 1:nrow(combinations_matrix)){
            # Cases which start with first order wd
            if (combinations_matrix[i,j] == 0){
              mddg_combinations[[i]][[j]] = mdg[[j,1]]
            }
            # Cases which start with second order wd
            else if (combinations_matrix[i,j] == 1){
              mddg_combinations[[i]][[j]] = mddg[[j,1]]
            }
          }
        }
      } else if ((NN$collectDev > 0) & (j < (numLayers-1))){
        out_fcDerivative <- fcDerivative(mw[idxw], Sw[idxw], mw[idxwo], J[[j+1,1]], J[[j,1]],
                                      ma[[j+1,1]], Sa[[j+1,1]], ma[[j,1]], Sa[[j,1]],
                                      Sz[[j,1]], mda[[j,1]], Sda[[j,1]],
                                      mdg[[j+1,1]], mdge[[j+1,1]], Sdg[[j+1,1]], mdg[[j+2,1]],
                                      actFunIdx[j+1], actFunIdx[j], nodes[j], nodes[j+1], nodes[j+2], B)
        mdgk = out_fcDerivative[[1]]
        Sdgk = out_fcDerivative[[2]]
        Cdgzk = out_fcDerivative[[3]]
        mdg[[j,1]] = matrix(rowSums(mdgk), nrow(mdgk), 1)
        Sdg[[j,1]] = matrix(rowSums(Sdgk), nrow(Sdgk), 1)
        mdge[[j,1]] = mdgk
        Cdgz[[j,1]] = matrix(rowSums(Cdgzk), nrow(Cdgzk), 1)

        # For second and higher order derivative
        if (NN$collectDev > 1){

          Caow = out_fcDerivative[[4]]
          Caoai = out_fcDerivative[[5]]
          Cdow = out_fcDerivative[[6]]
          Cdodi = out_fcDerivative[[7]]
          Cdowdi = out_fcDerivative[[8]]

          # First order derivative of current layer (wd)
          out_fcMeanVarDnode <- fcMeanVarDnode(mw[idxw], Sw[idxw], mda[[j,1]], Sda[[j,1]], nodes[j], nodes[j+1], B)
          mpdi = out_fcMeanVarDnode[[1]]

          # First order derivative of next layer (wd)
          out_fcMeanVarDnode <- fcMeanVarDnode(mw[idxwo], Sw[idxwo], mda[[j+1,1]], Sda[[j+1,1]], nodes[j+1], nodes[j+2], B)
          mpdo = out_fcMeanVarDnode[[1]]

          # Second order derivative of current layer (wdd)
          out_fcMeanVarDnode <- fcMeanVarDnode(mw[idxw], Sw[idxw], mdda[[j,1]], Sdda[[j,1]], nodes[j], nodes[j+1], B)
          mpddi = out_fcMeanVarDnode[[1]]

          for (i in 1:nrow(combinations_matrix)){
            # Case where to multiply first order wd to previous first order wd
            if (combinations_matrix[i,j] == 0){
              mddg_combinations[[i]][[j]] = mdg[[j,1]]
            }
            # Case where to multiply second order wdd to previous first order wd
            else if (combinations_matrix[i,j] == 1){
              mddgk = fcDerivative2(mw[idxw], mw[idxwo], ma[[j+1,1]], ma[[j,1]], mda[[j,1]],
                                                          mdda[[j,1]], mpddi, mddg_combinations[[i]][[j+1]], mddg_combinations[[i]][[j+2]], Caoai, Cdow,
                                                          Cdodi, actFunIdx[j], nodes[j], nodes[j+1], nodes[j+2], B)
              mddg_combinations[[i]][[j]] = matrix(rowSums(mddgk), nrow(mddgk), 1)
            }
            # Case where to multiply first order wd*wd to second order wdd
            else if (combinations_matrix[i,j] == 2){
              mddg_combinations[[i]][[j]] = fcDerivative3(mw[idxw], Sw[idxw], mw[idxwo], ma[[j+1,1]], ma[[j,1]], mda[[j+1,1]],
                                    mda[[j,1]], Sda[[j,1]], mpdi, mddg_combinations[[i]][[j+1]], mddg_combinations[[i]][[j+2]], Caow, Caoai, Cdow,
                                    Cdodi, actFunIdx[j+1], actFunIdx[j], nodes[j], nodes[j+1], nodes[j+2], B, j == dlayer)
            }
            # Case where to multiply first order wd*wd to previous terms' product wd*wd (first one)
            else if (combinations_matrix[i,j] == 3){
              mddg_combinations[[i]][[j]] = fcDerivative4(mw[idxw], Sw[idxw], mw[idxwo], ma[[j+1,1]], ma[[j,1]], mda[[j+1,1]],
                                                          mda[[j,1]], Sda[[j,1]], mpdo, mpdi, mddg_combinations[[i]][[j+1]], mddg_combinations[[i]][[j+2]], Cdowdi,
                                                          actFunIdx[j+1], actFunIdx[j], nodes[j], nodes[j+1], nodes[j+2], B, j == dlayer)
            }
            # Case where to multiply first order wd*wd to previous terms' product wd*wd (not first one)
            else if (combinations_matrix[i,j] > 3){
              mddg_combinations[[i]][[j]] = fcDerivative5(mw[idxw], Sw[idxw], mw[idxwo], ma[[j+1,1]], ma[[j,1]], mda[[j+1,1]],
                                                          mda[[j,1]], Sda[[j,1]], mpdo, mpdi, mddg_combinations[[i]][[j+1]], mddg_combinations[[i]][[j+2]], Cdowdi,
                                                          actFunIdx[j+1], actFunIdx[j], nodes[j], nodes[j+1], nodes[j+2], B, j == dlayer)
            }
          }
          # Only sum combinations that have a second order derivative so far to have g'' with respect to current layer j
          mddg[[j,1]] = matrix(0, nrow(mddg_combinations[[1]][[j]]), 1)
          for (i in 1:nrow(combinations_matrix)){
            if (combinations_matrix[i,j] > 0){
              mddg[[j,1]] = mddg[[j,1]] + matrix(rowSums(mddg_combinations[[i]][[j]]), nrow(mddg_combinations[[i]][[j]]), 1)
            }
          }
        }
      }
    }
    idxwo = idxw
  }
  outputs <- list(mdg, Sdg, Cdgz, mddg)
  return(outputs)
}

#' Backpropagation
#'
#' This function feeds the neural network backward from responses to input data.
#'
#' @param NN List that contains the structure of the neural network
#' @param mp List that contains mean vectors of the parameters for each layer \eqn{\mu_{\theta}}
#' @param Sp List that contains covariance matrices of the
#' parameters for each layer \eqn{\Sigma_{\theta}}
#' @param mz List that contains the mean vectors of the units
#' for each layer \eqn{\mu_{Z}}
#' @param Sz List that contains the covariance matrices of the
#' units for each layer \eqn{\Sigma_{Z}}
#' @param Czw List that contains the covariance matrices between
#' units and weights for each layer \eqn{\Sigma_{ZW}}
#' @param Czb List that contains the covariance matrices between
#' units and biases for each layer \eqn{\Sigma_{ZB}}
#' @param Czz List that contains the covariance matrices between units of the
#' previous and current layers for each layer \eqn{\Sigma_{ZZ^{+}}}
#' @seealso \code{\link{backwardHiddenStateUpdate}},
#' \code{\link{backwardParameterUpdate}}, \code{\link{forwardHiddenStateUpdate}}
#' @param y A vector or a matrix of responses
#' @return A list that contains:
#' @return - List that contains the updated mean vectors of the parameters
#' for each layer
#' @return - List that contains the updated covariance matrices of the
#' parameters for each layer
#' @export
feedBackward <- function(NN, mp, Sp, mz, Sz, Czw, Czb, Czz, y){
  # Initialization
  numLayers = length(NN$nodes)
  mpUd = matrix(list(), nrow = numLayers - 1, ncol = 1)
  SpUd = matrix(list(), nrow = numLayers - 1, ncol = 1)
  mzUd = matrix(list(), nrow = numLayers - 1, ncol = 1)
  SzUd = matrix(list(), nrow = numLayers - 1, ncol = 1)
  lHL = numLayers - 1
  if (NN$ny == length(NN$sv)){
    sv = t(NN$sv)
  } else {sv = matrix(NN$sv, nrow = NN$ny, ncol = 1)}

  # Update hidden states for the last hidden layer
  R = matrix(sv^2, nrow = NN$batchSize, ncol = 1)
  Szv = Sz[[lHL + 1]] + R
  out_forwardHiddenStateUpdate = forwardHiddenStateUpdate(mz[[lHL+1]], Sz[[lHL+1]], mz[[lHL+1]], Szv, Sz[[lHL+1]], y)
  mzUd[[lHL+1]] = out_forwardHiddenStateUpdate[[1]]
  SzUd[[lHL+1]] = out_forwardHiddenStateUpdate[[2]]

  for (k in (numLayers-1):1){
    # Update parameters
    Czp = buildCzp(Czw[[k+1]], Czb[[k+1]], NN$nodes[k+1], NN$nodes[k], NN$batchSize)
    out_backwardParameterUpdate = backwardParameterUpdate(mp[[k]], Sp[[k]], mz[[k+1]], Sz[[k+1]], SzUd[[k+1]], Czp, mzUd[[k+1]], NN$idxSzpUd[[k]])
    mpUd[[k]] = out_backwardParameterUpdate[[1]]
    SpUd[[k]] = out_backwardParameterUpdate[[2]]
    # Update hidden states
    if (k > 1){
      Czzloop = buildCzz(Czz[[k+1]], NN$nodes[k+1], NN$nodes[k], NN$batchSize)
      out_backwardHiddenStateUpdate = backwardHiddenStateUpdate(mz[[k]], Sz[[k]], mz[[k+1]], Sz[[k+1]], SzUd[[k+1]], Czzloop, mzUd[[k+1]], NN$idxSzzUd[[k]])
      mzUd[[k]] = out_backwardHiddenStateUpdate[[1]]
      SzUd[[k]] = out_backwardHiddenStateUpdate[[2]]
    }
  }
  outputs <- list(mpUd, SpUd)
  return(outputs)
}

#' Backpropagation (States' Deltas)
#'
#' This function calculates states' deltas.
#'
#' @param NN List that contains the structure of the neural network
#' @param theta List of parameters
#' @param states List of states
#' @param y A vector or a matrix of responses
#' @param Sy Variance of responses
#' @param udIdx TBD
#' @return A list that contains:
#' @return - Delta of mean vector of units given \eqn{y} \eqn{\mu_{Z}|y}} at all layers
#' @return - Delta of covariance matrix of units given \eqn{y} \eqn{\Sigma_{Z}|y}} at all layers
#' @export
hiddenStateBackwardPass <- function(NN, theta, states, y, Sy, udIdx){

  # Initialization
  out_extractParameters <- extractParameters(theta)
  mw = out_extractParameters[[1]]
  out_extractStates <- extractStates(states)
  mz = out_extractStates[[1]]
  Sz = out_extractStates[[2]]
  ma = out_extractStates[[3]]
  Sa = out_extractStates[[4]]
  J = out_extractStates[[5]]
  mdxs = out_extractStates[[6]]
  Sdxs = out_extractStates[[7]]
  Sxs = out_extractStates[[9]]
  numLayers = length(NN$nodes)
  B = NN$batchSize
  rB = NN$repBatchSize
  nodes = NN$nodes
  layer = NN$layer
  lHL = numLayers - 1
  numParamsPerLayer_2 = NN$numParamsPerLayer_2

  deltaM = matrix(list(), nrow = numLayers, ncol = 1)
  deltaS = matrix(list(), nrow = numLayers, ncol = 1)

  # Update hidden states for the last hidden layer
  if (NN$lastLayerUpdate == 1){
    if (is.null(Sy)){
      R = NN$sv^2
    } else {
      R = NN$sv^2 + Sy
    }
    if (is.null(udIdx)){
      Szv = Sa[[length(Sa),1]] + R
      out_forwardHiddenStateUpdate = forwardHiddenStateUpdate(0, 0, ma[[lHL+1,1]], Szv, J[[lHL+1,1]]*Sz[[lHL+1,1]], y)
      deltaMz = out_forwardHiddenStateUpdate[[1]]
      deltaSz = out_forwardHiddenStateUpdate[[2]]
    } else {
      mzf = ma[[length(ma),1]][udIdx]
      Szf = J[[lHL+1,1]][udIdx] * Sz[[lHL+1,1]][udIdx]
      ys = y
      Szv = Sa[[length(Sa),1]][udIdx] + R
      deltaMz = matrix(0, nrow(mz[[lHL+1,1]]), ncol(mz[[lHL+1,1]]))
      deltaSz = matrix(0, nrow(Sz[[lHL+1,1]]), ncol(Sz[[lHL+1,1]]))
      out_forwardHiddenStateUpdate = forwardHiddenStateUpdate(0, 0, mzf, Szv, Szf, ys)
      deltaMz[udIdx] = out_forwardHiddenStateUpdate[[1]]
      deltaSz[udIdx] = out_forwardHiddenStateUpdate[[2]]
    }
  } else {
    deltaMz = y
    deltaSz = Sy
  }

  for (j in (numLayers-1):1){
    if (is.null(mdxs[[j+1,1]])){
      nSz = Sz[[j+1,1]]
    } else {
      nSz = Sdxs[[j+1,1]]
    }
    if (is.null(mdxs[[j,1]])){
      cSz = Sz[[j,1]]
    } else {
      cSz = Sdxs[[j,1]]
    }

    cSxs = Sxs[[j,1]]
    idxw = (numParamsPerLayer_2[1, j]+1):numParamsPerLayer_2[1, j+1]

    # Innovation vector
    out_innovationVector = innovationVector(nSz, deltaMz, deltaSz)
    deltaM[[j+1,1]] = out_innovationVector[[1]]
    deltaS[[j+1,1]] = out_innovationVector[[2]]

    # Max pooling
    if (layer[j+1] == NN$layerEncoder$fc){
      if ((j > 1)|(NN$convariateEstm == 1)){
        if ((B == 1) & (rB == 1)){
          out_fcHiddenStateBackwardPassB1 <- fcHiddenStateBackwardPassB1(cSz, cSxs, J[[j,1]], mw[idxw], deltaM[[j+1,1]], deltaS[[j+1,1]], nodes[j], nodes[j+1])
          deltaMz = out_fcHiddenStateBackwardPassB1[[1]]
          deltaSz = out_fcHiddenStateBackwardPassB1[[2]]
        } else {
          out_fcHiddenStateBackwardPass <- fcHiddenStateBackwardPass(cSz, cSxs, J[[j,1]], mw[idxw], deltaM[[j+1,1]], deltaS[[j+1,1]], nodes[j], nodes[j+1], B, rB)
          deltaMz = out_fcHiddenStateBackwardPass[[1]]
          deltaSz = out_fcHiddenStateBackwardPass[[2]]
        }
      }
    }
  }
  outputs <- list(deltaM, deltaS)
  return(outputs)
}

#' Backpropagation (Parameters' Deltas)
#'
#' This function calculates parameter's deltas.
#'
#' @param NN List that contains the structure of the neural network
#' @param theta List of parameters
#' @param states List of states
#' @param deltaM Delta of mean vector of units given \eqn{y} \eqn{\mu_{Z}|y} at all layers
#' @param deltaS Delta of covariance matrix of units given \eqn{y} \eqn{\Sigma_{Z}|y} at all layers
#' @return Parameters' deltas (mean and covariance for each)
#' @export
parameterBackwardPass <- function(NN, theta, states, deltaM, deltaS){

  # Initialization
  out_extractParameters <- extractParameters(theta)
  mw = out_extractParameters[[1]]
  Sw = out_extractParameters[[2]]
  mb = out_extractParameters[[3]]
  Sb = out_extractParameters[[4]]
  mwx = out_extractParameters[[5]]
  Swx = out_extractParameters[[6]]
  mbx = out_extractParameters[[7]]
  Sbx = out_extractParameters[[8]]
  out_extractStates <- extractStates(states)
  ma = out_extractStates[[3]]
  numLayers = length(NN$nodes)
  B = NN$batchSize
  rB = NN$repBatchSize
  nodes = NN$nodes
  layer = NN$layer
  numParamsPerLayer_2 = NN$numParamsPerLayer_2

  deltaMw = matrix(mw, ncol = 1)
  deltaSw = matrix(Sw, ncol = 1)
  deltaMb = matrix(mb, ncol = 1)
  deltaSb = matrix(Sb, ncol = 1)
  deltaMwx = matrix(mwx, ncol = 1)
  deltaSwx = matrix(Swx, ncol = 1)
  deltaMbx = matrix(mbx, ncol = 1)
  deltaSbx = matrix(Sbx, ncol = 1)

  for (j in (numLayers-1):1){
    idxw = (numParamsPerLayer_2[1, j]+1):numParamsPerLayer_2[1, j+1]
    idxb = (numParamsPerLayer_2[2, j]+1):numParamsPerLayer_2[2, j+1]

    # Convolutional
    if (layer[j+1] == NN$layerEncoder$fc){
      if ((j > 1)|(NN$convariateEstm == 1)){
        if ((B == 1) & (rB == 1)){
          out_fcParameterBackwardPassB1 <- fcParameterBackwardPassB1(Sw[idxw], Sb[idxb], ma[[j,1]], deltaM[[j+1,1]], deltaS[[j+1,1]], nodes[j], nodes[j+1])
          deltaMw[idxw] = out_fcParameterBackwardPassB1[[1]]
          deltaSw[idxw] = out_fcParameterBackwardPassB1[[2]]
          deltaMb[idxb] = out_fcParameterBackwardPassB1[[3]]
          deltaSb[idxb] = out_fcParameterBackwardPassB1[[4]]
        } else {
          out_fcParameterBackwardPass <- fcParameterBackwardPass(deltaMw[idxw], deltaSw[idxw], deltaMb[idxb], deltaSb[idxb], Sw[idxw], Sb[idxb], ma[[j,1]], deltaM[[j+1,1]], deltaS[[j+1,1]], nodes[j], nodes[j+1], B, rB)
          deltaMw[idxw] = out_fcParameterBackwardPass[[1]]
          deltaSw[idxw] = out_fcParameterBackwardPass[[2]]
          deltaMb[idxb] = out_fcParameterBackwardPass[[3]]
          deltaSb[idxb] = out_fcParameterBackwardPass[[4]]
        }
      }
    }
  }
  deltaTheta = compressParameters(deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx)
  return(deltaTheta)
}

#' Backpropagation (States' Deltas) for Fully Connected Layers (Many Observations)
#'
#' This function calculates units' deltas at a given layer when using more than one
#' observation at the time.
#'
#' @param Sz Covariance of the units from current layer
#' @param Sxs TBD
#' @param J Jacobian of current layer
#' @param mw Mean vector of the weights for the current layer
#' @param deltaM Delta of mean vector of the next layer units given \eqn{y} \eqn{\mu_{Z}|y}
#' @param deltaS Delta of covariance matrix of the next layer units given \eqn{y} \eqn{\Sigma_{Z}|y}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @param rB Number of times batch size is repeated
#' @return - Delta of mean vector of the layer units given \eqn{y} \eqn{\mu_{Z}|y}}
#' @return - Delta of covariance matrix of the layer units given \eqn{y} \eqn{\Sigma_{Z}|y}}
#' @return - TBD
#' @return - TBD
#' @export
fcHiddenStateBackwardPass <- function(Sz, Sxs, J, mw, deltaM, deltaS, ni, no, B, rB){
  deltaMz = Sz
  deltaSz = Sz
  deltaMzx = Sxs
  deltaSzx = Sxs
  mw = matrix(rep(t(matrix(mw, ni, no)),B), nrow = ni*B, byrow = TRUE)

  if (is.null(Sxs)){
    for (t in 1:rB){
      Czz = J[,t]*Sz[,t]*mw
      deltaMzloop = t(matrix(matrix(rep(t(matrix(deltaM[,t], no, B)), ni), ncol = B, byrow = TRUE), no, ni*B))
      deltaSzloop = t(matrix(matrix(rep(t(matrix(deltaS[,t], no, B)), ni), ncol = B, byrow = TRUE), no, ni*B))
      deltaMzloop = Czz*deltaMzloop
      deltaSzloop = Czz*deltaSzloop*Czz
      deltaMz[,t] = matrix(rowSums(deltaMzloop), ncol=1)
      deltaSz[,t] = matrix(rowSums(deltaSzloop), ncol=1)
      deltaMzx[,t] = NULL
      deltaSzx[,t] = NULL
    }
  } else {
    for (t in 1:rB){
      Czz = J[,t]*Sz[,t]*mw
      Czx = J[,t]*Sxs[,t]*mw
      deltaMloop = t(matrix(matrix(rep(t(matrix(deltaM[,t], no, B)), ni), ncol = B, byrow = TRUE), no, ni*B))
      deltaSloop = t(matrix(matrix(rep(t(matrix(deltaS[,t], no, B)), ni), ncol = B, byrow = TRUE), no, ni*B))
      deltaMzloop = Czz*deltaMloop
      deltaSzloop = Czz*deltaSloop*Czz
      deltaMxsloop = Czx*deltaMloop
      deltaSxsloop = Czx*deltaSloop*Czx
      deltaMz[,t] = matrix(rowSums(deltaMzloop), nrow(deltaMzloop), 1)
      deltaSz[,t] = matrix(rowSums(deltaSzloop), nrow(deltaSzloop), 1)
      deltaMzx[,t] = matrix(rowSums(deltaMxsloop), nrow(deltaMxsloop), 1)
      deltaSzx[,t] = matrix(rowSums(deltaSxsloop), nrow(deltaSxsloop), 1)
    }
  }
  outputs <- list(deltaMz, deltaSz, deltaMzx, deltaSzx)
  return(outputs)
}


#' Backpropagation (States' Deltas) for Fully Connected Layers (One Observation)
#'
#' This function calculates units' deltas at a given layer when using one observation at the time.
#'
#' @param Sz Covariance of the units from current layer
#' @param Sxs TBD
#' @param J Jacobian of current layer
#' @param mw Mean vector of the weights for the current layer
#' @param deltaM Delta of mean vector of the next layer units given \eqn{y} \eqn{\mu_{Z}|y}
#' @param deltaS Delta of covariance matrix of the next layer units given \eqn{y} \eqn{\Sigma_{Z}|y}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @return - Delta of mean vector of the layer units given \eqn{y} \eqn{\mu_{Z}|y}}
#' @return - Delta of covariance matrix of the layer units given \eqn{y} \eqn{\Sigma_{Z}|y}}
#' @return - TBD
#' @return - TBD
#' @export
fcHiddenStateBackwardPassB1 <- function(Sz, Sxs, J, mw, deltaM, deltaS, ni, no){
  mw = matrix(mw, ni, no)
  deltaMzx = Sxs
  deltaSzx = Sxs

  if (is.null(Sxs)){
    deltaMloop = matrix(rep(deltaM, ni), nrow = ni, byrow = TRUE)
    deltaSloop = matrix(rep(deltaS, ni), nrow = ni, byrow = TRUE)

    Caz = J*Sz
    Caz = matrix(Caz, nrow(mw), ncol(mw))
    Cwa = mw*Caz
    deltaMzloop = Cwa*deltaMloop
    deltaSzloop = (Cwa^2)*deltaSloop

    deltaMz = matrix(rowSums(deltaMzloop), nrow(deltaMzloop), 1)
    deltaSz = matrix(rowSums(deltaSzloop), nrow(deltaSzloop), 1)
    deltaMzx = NULL
    deltaSzx = NULL
  } else {
    deltaMloop = matrix(rep(deltaM, ni), nrow = ni, byrow = TRUE)
    deltaSloop = matrix(rep(deltaS, ni), nrow = ni, byrow = TRUE)

    Caz = J*Sz
    Caxs = J*Sxs
    Caz = matrix(Caz, nrow(mw), ncol(mw))
    Caxs = matrix(Caz, nrow(mw), ncol(mw))

    out_vectorized4delta <- vectorized4delta(mw, Caz, Caxs, deltaMloop, deltaSloop)
    deltaMzloop = out_vectorized4delta[[1]]
    deltaSzloop = out_vectorized4delta[[2]]
    deltaMzsloop = out_vectorized4delta[[3]]
    deltaSzsloop = out_vectorized4delta[[4]]

    deltaMz = matrix(rowSums(deltaMzloop), nrow(deltaMzloop), 1)
    deltaSz = matrix(rowSums(deltaSzloop), nrow(deltaSzloop), 1)
    deltaMzx = matrix(rowSums(deltaMzsloop), nrow(deltaMzsloop), 1)
    deltaSzx = matrix(rowSums(deltaSzsloop), nrow(deltaSzsloop), 1)
  }
  outputs <- list(deltaMz, deltaSz, deltaMzx, deltaSzx)
  return(outputs)
}

#' Derivatives for Fully Connected layers
#'
#' This function calculates mean and variance of derivatives and covariance of derivative and input layers.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mwo Mean vector of the weights for the next layer
#' @param Jo Jacobian of next layer
#' @param J Jacobian of current layer
#' @param mao Mean vector of the activation units from next layer
#' @param Sao Covariance of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param Sai Covariance of the activation units from current layer
#' @param Szi Covariance of the units from current layer
#' @param mdai Mean vector of the activation units' derivative from current layer
#' @param Sdai Covariance of the activation units' derivative from current layer
#' @param mdgo Mean vector of derivatives in next layer
#' @param mdgoe Mean derivatives at each node in next layer
#' @param Sdgo Variance of derivatives in next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @return Mean vector of the derivatives
#' @return Covariance matrix of the derivatives
#' @return Covariance matrix of derivative and input layers
#' @return Covariance between activation units and weights
#' @return Covariance between activation units from current and next layers
#' @return Covariance between derivatives and weights
#' @return Covariance between derivatives from current and next layers
#' @return Covariance between derivatives from next layer and weights times derivatives from current layer
#' @export
fcDerivative <- function(mw, Sw, mwo, Jo, J, mao, Sao, mai, Sai, Szi, mdai, Sdai, mdgo, mdgoe, Sdgo, mdgo2, acto, acti, ni, no, no2, B){
  out_fcMeanVarDnode <- fcMeanVarDnode(mw, Sw, mdai, Sdai, ni, no, B)
  mpdi = out_fcMeanVarDnode[[1]]
  Spdi = out_fcMeanVarDnode[[2]]
  out_fcCovawaa <- fcCovawaa(mw, Sw, Jo, mai, Sai, ni, no, B)
  Caow = out_fcCovawaa[[1]]
  Caoai = out_fcCovawaa[[2]]
  out_fcCovdwddd <- fcCovdwddd(mao, Sao, mai, Sai, Caow, Caoai, acto, acti, ni, no, B)
  Cdow = out_fcCovdwddd[[1]]
  Cdodi = out_fcCovdwddd[[2]]
  Cdowdi = fcCovdwd(mdai, mw, Cdow, Cdodi, ni, no, B)
  Cdgodgi = fcCovDlayer(mdgo2, mwo, Cdowdi, ni, no, no2, B)
  out_fcMeanVarDlayer <- fcMeanVarDlayer(mpdi, Spdi, mdgo, mdgoe, Sdgo, Cdgodgi, ni, no, no2, B)
  mdgi = out_fcMeanVarDlayer[[1]]
  Sdgi = out_fcMeanVarDlayer[[2]]

  out_fcCovaz <- fcCovaz(Jo, J, Szi, mw, ni, no, B)
  Caizi = out_fcCovaz[[1]]
  Caozi = out_fcCovaz[[2]]
  out_fcCovdz <- fcCovdz(mao, mai, Caizi, Caozi, acto, acti, ni, no, B)
  Cdozi = out_fcCovdz[[1]]
  Cdizi = out_fcCovdz[[2]]
  Cdx = covdx(mwo, mw, mdgo2, mpdi, mdgoe, Cdozi, Cdizi, ni, no, no2, B)

  outputs <- list(mdgi, Sdgi, Cdx, Caow, Caoai, Cdow, Cdodi, Cdowdi)
  return(outputs)
}

#' Second Derivatives for Fully Connected layers
#'
#' This function calculates mean of product of derivatives, when new product term involves second order derivatives (wdd).
#'
#' @param mw Mean vector of the weights for the current layer
#' @param mwo Mean vector of the weights for the next layer
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param mdai Mean vector of the activation units' derivative from current layer
#' @param mddai Mean vector of the activation units' second derivative from current layer
#' @param mpddi Mean vector of the second order derivative product wdd of current layer
#' @param mdgo Mean vector of derivatives in next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param Caoai Covariance between activation units from current and next layers
#' @param Cdow Covariance between derivatives and weights
#' @param Cdodi Covariance between derivatives from current and next layers
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @return Mean vector of the derivatives
#' @export
fcDerivative2 <- function(mw, mwo, mao, mai, mdai, mddai, mpddi, mdgo, mdgo2, Caoai, Cdow, Cdodi, acti, ni, no, no2, B){
  out_fcCovdaddd <- fcCovdaddd(mao, mai, mdai, Caoai, Cdodi, acti, ni, no, B)
  Cdoai = out_fcCovdaddd[[1]]
  Cdoddi = out_fcCovdaddd[[2]]
  Cdowddi = fcCovdwd(mddai, mw, Cdow, Cdoddi, ni, no, B)
  Cdgoddgi = fcCovDlayer(mdgo2, mwo, Cdowddi, ni, no, no2, B)
  out_fcMeanVarDlayer <- fcMeanVarDlayer(mpddi, matrix(0, nrow(mpddi), ncol(mpddi)), mdgo, matrix(0, nrow(mpddi), ncol(mpddi)), matrix(0, nrow(mpddi), ncol(mpddi)), Cdgoddgi, ni, no, no2, B)
  mddgi = out_fcMeanVarDlayer[[1]]

  return(mddgi)
}

#' Products of First Order Derivatives Multiplied to Second Order Derivative for Fully Connected layers
#'
#' This function calculates mean of product of derivatives, when new product term involves product
#' of two first order derivatives (wd*wd) from the same layer multiplied to second order derivatives (wdd) from next layer.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mwo Mean vector of the weights for the next layer
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param mdao Mean vector of the activation units' derivative from next layer
#' @param mdai Mean vector of the activation units' derivative from current layer
#' @param Sdai Covariance of the activation units' derivative from current layer
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mdgo Mean vector of derivatives in next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param Caow Covariance between activation units and weights
#' @param Caoai Covariance between activation units from current and next layers
#' @param Cdow Covariance between derivatives and weights
#' @param Cdodi Covariance between derivatives from current and next layers
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @param dlayer TRUE if layer from which derivatives will be in respect to
#' @return Mean matrix of products of derivatives
#' @export
fcDerivative3 <- function(mw, Sw, mwo, mao, mai, mdao, mdai, Sdai, mpdi, mdgo, mdgo2, Caow, Caoai, Cdow, Cdodi, acto, acti, ni, no, no2, B, dlayer){

  out_fcCovaddddddw <- fcCovaddddddw(mao, mai, mdao, Caoai, Cdodi, Caow, Cdow, acto, acti, ni, no, B)
  Cddodi = out_fcCovaddddddw[[1]]
  Cddow = out_fcCovaddddddw[[2]]
  Cddowdi = fcCovdwd(mdai, mw, Cddow, Cddodi, ni, no, B)
  Cddgodgik = fcCovDlayer(mdgo2, mwo, Cddowdi, ni, no, no2, B)
  Cddgodgik = rowSums(matrix(Cddgodgik, B*ni*no, no2))
  Cddgodgik = matrix(Cddgodgik, B*ni, no)

  if (dlayer == FALSE){
    # Combination of products of first order derivative of current layer (wd)*(wd) (iterations on nodes from same layer (with weights pointing to same next layer node))
    mpdi2n <- fcCombinaisonDnode(mpdi, mw, Sw, mdai, Sdai, ni, no, B)

    Cwdowdiwdi <- fcCovwdowdiwdi(mpdi, Cddgodgik, ni, no, B)
    mddgi <- fcMeanDlayer2row(mpdi, mpdi2n, mdgo, Cwdowdiwdi, ni, no, no2, B)
  } else {
    # (wd)^2 only
    mpdi2wn <- fcCombinaisonDweightNode(mpdi, mw, Sw, mdai, Sdai, ni, no, B)
    Cwdowdi2 <- fcCovwdowdi2(mpdi, Cddgodgik)
    mdgo = t(matrix(matrix(rep(t(matrix(mdgo, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))

    mddgi = mdgo*mpdi2wn + Cwdowdi2
    mddgi = matrix(rowSums(mddgi), nrow(mddgi), 1)
  }


  return(mddgi)
}

#' Products of First Order Derivatives Multiplied to Products of First Order Derivatives for Fully Connected layers
#'
#' This function calculates mean of product of derivatives, when new product term involves product
#' of two first order derivatives (wd*wd) from the same layer multiplied to product
#' of two first order derivatives (wd*wd) from next layer, when second next layer is second order derivatives (wdd).
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mwo Mean vector of the weights for the next layer
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param mdao Mean vector of the activation units' derivative from next layer
#' @param mdai Mean vector of the activation units' derivative from current layer
#' @param Sdai Covariance of the activation units' derivative from current layer
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mdgo Mean vector of derivatives in next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param Cdowdi Covariance between derivatives from next layer and weights times derivatives from current layer
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @param dlayer TRUE if layer from which derivatives will be in respect to
#' @return Mean matrix of products of derivatives
#' @export
fcDerivative4 <- function(mw, Sw, mwo, mao, mai, mdao, mdai, Sdai, mpdo, mpdi, mdgo, mdgo2, Cdowdi, acto, acti, ni, no, no2, B, dlayer){

  # Combination of products of first order derivative of current layer (wd)*(wd) (iterations on weights on the same node)
  mpdi2w <- fcCombinaisonDweight(mpdi, mw, Sw, mdai, Sdai, ni, no, B)
  Cdgodgi <- fcCovDlayer(mdgo2, mwo, Cdowdi, ni, no, no2, B)

  if (dlayer == FALSE){
    # Combination of products of first order derivative of current layer (wd)*(wd) (iterations on nodes from same layer (with weights pointing to same next layer node))
    mpdi2n <- fcCombinaisonDnode(mpdi, mw, Sw, mdai, Sdai, ni, no, B)
    # All possible combinations
    mpdi2wnAll <- fcCombinaisonDweightNodeAll(mpdi, mpdi2n, mpdi2w, ni, no, B)

    Cwdowdowdiwdi <- fcCwdowdowdiwdi(mpdi, mpdo, Cdgodgi, ni, no, no2, B)

    mdgo = array(matrix(rep(rep(matrix(mdgo, ncol = no, byrow = TRUE), times = ni), each = ni), B*ni, no*no), c(B*ni, no, no))

    mdgoA = array(0, c(B*ni, no, ni*no))
    for (b in 0:(no-1)){
      for (i in (b*ni+1):(b*ni+ni)){
        mdgoA[,,i] = mdgo[,,b+1]
      }
    }
    md = mpdi2wnAll * mdgoA
    mddgi = md + Cwdowdowdiwdi

    # Come back to Bni x ni matrix
    mddgi = matrix(apply(mddgi, 3, rowSums), B*ni*ni, no)
    mddgi = matrix(rowSums(mddgi), B*ni, ni)
  } else {
    Cwdowdowwdi2 <- fcCwdowdowwdi2(mpdi, mpdo, Cdgodgi, ni, no, no2, B)
    mddgi <- fcMeanDlayer2array(mpdi2w, mdgo, Cwdowdowwdi2, ni, no, B)
    mddgi = matrix(rowSums(mddgi), nrow(mddgi), 1)
  }

  return(mddgi)
}
#' Products of First Order Derivatives Multiplied to Products of First Order Derivatives (Not Only Last Layer) for Fully Connected layers
#'
#' This function calculates mean of product of derivatives, when new product term involves product
#' of two first order derivatives (wd*wd) from the same layer multiplied to product
#' of two first order derivatives (wd*wd) from next and second next layers.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mwo Mean vector of the weights for the next layer
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param mdao Mean vector of the activation units' derivative from next layer
#' @param mdai Mean vector of the activation units' derivative from current layer
#' @param Sdai Covariance of the activation units' derivative from current layer
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mdgo Mean vector of derivatives in next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param Cdowdi Covariance between derivatives from next layer and weights times derivatives from current layer
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @param dlayer TRUE if layer from which derivatives will be in respect to
#' @return Mean matrix of products of derivatives
#' @export
fcDerivative5 <- function(mw, Sw, mwo, mao, mai, mdao, mdai, Sdai, mpdo, mpdi, mdgo, mdgo2, Cdowdi, acto, acti, ni, no, no2, B, dlayer){

  # Combination of products of first order derivative of current layer (wd)*(wd) (iterations on weights on the same node)
  mpdi2w <- fcCombinaisonDweight(mpdi, mw, Sw, mdai, Sdai, ni, no, B)
  Cdgodgi <- fcCovDlayer(matrix(1, B*no2, 1), mwo, Cdowdi, ni, no, no2, B)

  if (dlayer == FALSE){
    # Combination of products of first order derivative of current layer (wd)*(wd) (iterations on nodes from same layer (with weights pointing to same next layer node))
    mpdi2n <- fcCombinaisonDnode(mpdi, mw, Sw, mdai, Sdai, ni, no, B)
    # All possible combinations
    mpdi2wnAll <- fcCombinaisonDweightNodeAll(mpdi, mpdi2n, mpdi2w, ni, no, B)

    Cwdowdowdiwdi <- fcCwdowdowdiwdi_4hl(mpdi, mpdo, mdgo2, Cdgodgi, ni, no, no2, B)

    mdgo = array(matrix(rep(rep(matrix(mdgo, ncol = no, byrow = TRUE), times = ni), each = ni), B*ni, no*no), c(B*ni, no, no))

    mdgoA = array(0, c(B*ni, no, ni*no))
    for (b in 0:(no-1)){
      for (i in (b*ni+1):(b*ni+ni)){
        mdgoA[,,i] = mdgo[,,b+1]
      }
    }
    md = mpdi2wnAll * mdgoA
    mddgi = md + Cwdowdowdiwdi

    # Come back to Bni x ni matrix
    mddgi = matrix(apply(mddgi, 3, rowSums), B*ni*ni, no)
    mddgi = matrix(rowSums(mddgi), B*ni, ni)

  } else {
    Cwdowdowwdi2 <- fcCwdowdowwdi2_3hl(mpdi, mpdo, mdgo2, Cdgodgi, ni, no, no2, B)
    mddgi <- fcMeanDlayer2array(mpdi2w, mdgo, Cwdowdowwdi2, ni, no, B)
    mddgi = matrix(rowSums(mddgi), nrow(mddgi), 1)
  }

  return(mddgi)
}


#' Mean and Covariance of Derivatives
#'
#' This function calculates the mean vector and the covariance matrix for derivatives.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mda Mean vector of the activation units' derivative from current layer
#' @param Sda Covariance of the activation units' derivative from current layer
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Mean vector of the derivatives
#' @return Covariance matrix of the derivatives
#' @export
fcMeanVarDnode <- function(mw, Sw, mda, Sda, ni, no, B){
  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  Sw = matrix(rep(t(matrix(Sw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  mda = matrix(mda, nrow(mw), ncol(mw))
  Sda = matrix(Sda, nrow(Sw), ncol(Sw))
  md = mw*mda
  Sd = Sw*Sda + Sw*(mda^2) + Sda*(mw^2)

  outputs <- list(md, Sd)
  return(outputs)
}

#' Covariance between Activation Units and Weights
#'
#' This function calculates covariance between activation units and weights and
#' covariance between activation units from consecutive layers.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param Jo Jacobian of next layer
#' @param mai Mean vector of the activation units from current layer
#' @param Sai Covariance of the activation units from current layer
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between activation units and weights
#' @return Covariance between activation units from current and next layers
#' @export
fcCovawaa <- function(mw, Sw, Jo, mai, Sai, ni, no, B){
  Joloop = t(matrix(matrix(rep(t(matrix(Jo, no, B)), ni), nrow =no*ni, ncol = B, byrow = TRUE), no, ni*B))
  Sw = matrix(rep(t(matrix(Sw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  mai = matrix(mai, nrow(Sw), ncol(Sw))
  Caw = Sw*mai*Joloop

  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  Sai = matrix(Sai, nrow(mw), ncol(mw))
  Caa = mw*Sai*Joloop

  outputs <- list(Caw, Caa)
  return(outputs)
}

#' Covariance between Derivatives and Weights
#'
#' This function calculates covariance between derivatives and weights and
#' covariance between derivatives from consecutive layers.
#'
#' @param mao Mean vector of the activation units from next layer
#' @param Sao Covariance of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param Sai Covariance of the activation units from current layer
#' @param Caow Covariance between activation units and weights
#' @param Caoai Covariance between activation units from current and next layers
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between derivatives and weights
#' @return Covariance between derivatives from current and next layers
#' @export
fcCovdwddd <- function(mao, Sao, mai, Sai, Caow, Caoai, acto, acti, ni, no, B){
  mao = t(matrix(matrix(rep(t(matrix(mao, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  Sao = t(matrix(matrix(rep(t(matrix(Sao, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  mai = matrix(mai, nrow(mao), ncol(mao))
  Sai = matrix(Sai, nrow(Sao), ncol(Sao))

  if (acti == 1){ # tanh
    Cdodi = 2*Caoai^2 + 4*mao*Caoai*mai
  } else if (acti == 2){ # sigmoid
    Cdodi = Caoai - 2*Caoai*mai - 2*mao*Caoai + 2*Caoai^2 + 4*mao*Caoai*mai
  } else if (acti == 4){ # relu
    Cdodi = matrix(0, nrow(mao), ncol(mao))
  } else {
    Cdodi = matrix(0, nrow(mao), ncol(mao))
  }

  if (acto == 1){ # tanh
    Cdow = -2*mao*Caow
  } else if (acto == 2){ # sigmoid
    Cdow = Caow*(1-2*mao)
  } else if (acto == 4){ # relu
    Cdow = matrix(0, nrow(mao), ncol(mao))
  } else {
    Cdow = matrix(0, nrow(mao), ncol(mao))
  }

  outputs <- list(Cdow, Cdodi)
  return(outputs)
}

#' Covariance between Second and First Orders Derivatives from Consecutive Layers
#'
#' This function calculates covariance between activation units and first order derivatives and
#' covariance between second and first orders derivatives from consecutive layers.
#'
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param mdai Mean vector of the activation units' derivative from current layer
#' @param Caoai Covariance between activation units from current and next layers
#' @param Cdodi Covariance between derivatives from current and next layers
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between first order derivatives from next layer and activation units from current layer
#' @return Covariance between second and first orders derivatives from consecutive layers
#' @export
fcCovdaddd <- function(mao, mai, mdai, Caoai, Cdodi, acti, ni, no, B){
  mao = t(matrix(matrix(rep(t(matrix(mao, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  mai = matrix(mai, nrow(mao), ncol(mao))
  mdai = matrix(mdai, nrow(mao), ncol(mao))

  if (acti == 1){ # tanh
    Cdoai = - 2*Caoai*mao
    Cdoddi = - 2*Cdodi*mai - 2*Cdoai*mdai
  } else if (acti == 2){ # sigmoid
    Cdoai = Caoai * (1 - 2*mao)
    Cdoddi = Cdodi - 2*Cdodi*mai - 2*Cdoai*mdai
  } else if (acti == 4){ # relu
    Cdoai = matrix(0, nrow(mao), ncol(mao))
    Cdoddi = matrix(0, nrow(mao), ncol(mao))
  } else {
    Cdoai = matrix(0, nrow(mao), ncol(mao))
    Cdoddi = matrix(0, nrow(mao), ncol(mao))
  }

  outputs <- list(Cdoai, Cdoddi)
  return(outputs)
}

#' Covariance between First and Second Orders Derivatives from Consecutive Layers
#'
#' This function calculates covariance between weights and second order derivatives and
#' covariance between first and second orders derivatives from consecutive layers.
#'
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param mdao Mean vector of the activation units' derivative from next layer
#' @param Caoai Covariance between activation units from current and next layers
#' @param Cdodi Covariance between derivatives from current and next layers
#' @param Caow Covariance between activation units and weights
#' @param Cdow Covariance between derivatives and weights
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between first and second orders derivatives from consecutive layers
#' @return Covariance between second order derivatives from next layer and weights
#' @export
fcCovaddddddw <- function(mao, mai, mdao, Caoai, Cdodi, Caow, Cdow, acto, acti, ni, no, B){
  mao = t(matrix(matrix(rep(t(matrix(mao, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  mdao = t(matrix(matrix(rep(t(matrix(mdao, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  mai = matrix(mai, nrow(mao), ncol(mao))

  if (acti == 1){ # tanh
    Caodi = - 2*Caoai*mai
    Cddodi = - 2*Cdodi*mao - 2*Caodi*mdao
  } else if (acti == 2){ # sigmoid
    Caodi = Caoai * (1 - 2*mai)
    Cddodi = Cdodi - 2*Cdodi*mao - 2*Caodi*mdao
  } else if (acti == 4){ # relu
    Caodi = matrix(0, nrow(mao), ncol(mao))
    Cddodi = matrix(0, nrow(mao), ncol(mao))
  } else {
    Caodi = matrix(0, nrow(mao), ncol(mao))
    Cddodi = matrix(0, nrow(mao), ncol(mao))
  }

  if (acto == 1){ # tanh
    Cddow = - 2*Caow*mdao - 2*mao*Cdow
  } else if (acto == 2){ # sigmoid
    Cddow = Cdow - 2*Caow*mdao - 2*mao*Cdow
  } else if (acto == 4){ # relu
    Cddow = matrix(0, nrow(mao), ncol(mao))
  } else {
    Cddow = matrix(0, nrow(mao), ncol(mao))
  }

  outputs <- list(Cddodi, Cddow)
  return(outputs)
}

#' Covariance between Derivatives and Weights*Derivatives
#'
#' This function calculates covariance between derivatives and weights and
#' covariance between derivatives from consecutive layers.
#'
#' @param md Mean vector of derivatives
#' @param mw Mean vector of the weights for the current layer
#' @param Cdow Covariance between derivatives and weights
#' @param Cdodi Covariance between derivatives from current and next layers
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between derivatives and weights times derivatives
#' @export
fcCovdwd <- function(md, mw, Cdow, Cdodi, ni, no, B){
  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  md = matrix(md, nrow(mw), ncol(mw))
  Cdowdi = Cdow*md + Cdodi*mw
  return(Cdowdi)
}

#' Covariance between Products of Derivatives and Weights
#'
#' This function calculates covariance between products of derivatives and
#' weights from consecutive layers.
#'
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param mwo Mean vector of the weights for the next layer
#' @param Cdowdi Covariance between derivatives and weights times derivatives
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @return Covariance between weights times derivatives from consecutive layers
#' @export
fcCovDlayer <- function(mdgo2, mwo, Cdowdi, ni, no, no2, B){
  mdgo2 = matrix(matrix(rep(mdgo2, no), nrow = nrow(t(mdgo2))*no, byrow = TRUE), no*no2, B)
  m = t(matrix(matrix(rep(t(mdgo2*mwo), ni), nrow = nrow(mdgo2*mwo)*ni, byrow = TRUE), no*no2, B*ni))
  Cdowdi = matrix(rep(Cdowdi, no2), nrow = nrow(Cdowdi))
  Cdgodgi = Cdowdi*m
  return(Cdgodgi)
}

#' Mean and Variance of Weights times Derivatives Products Terms
#'
#' This function calculates mean and variance of weights times derivatives products terms.
#'
#' @param mx Mean vector of inputs
#' @param Sx Variance of inputs
#' @param mye Mean derivatives at each node in next layer
#' @param my Mean vector of outputs
#' @param Sy Variance of outputs
#' @param Cxy Covariance between inputs and outputs
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @return Mean of weights times derivatives products terms
#' @return Covariance between weights times derivatives products terms
#' @export
fcMeanVarDlayer <- function(mx, Sx, my, mye, Sy, Cxy, ni, no, no2, B){
  my = t(matrix(matrix(rep(t(matrix(my, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  Sy = t(matrix(matrix(rep(t(matrix(Sy, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  mye = array(aperm(array(t(mye), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))
  mye = t(matrix(mye[, rep(1:ncol(mye), each = ni),], no*no2,B*ni))

  md = mx*my
  Sd = Sx*Sy + Sy*(mx^2)

  SxS = matrix(rep(Sx, no2), nrow = nrow(Sx), ncol = ncol(Sx)*no2)
  Sd2 = rowSums(matrix(SxS*(mye^2), B*ni*no, no2))
  Sd2 = matrix(Sd2, B*ni, no)

  Sd = Sd + Sd2

  Cxym = rowSums(matrix(Cxy, B*ni*no, no2))
  Cxym = matrix(Cxym, B*ni, no)
  md = md + Cxym

  CxyS1 = rowSums(matrix(Cxy^2, B*ni*no, no2))
  CxyS1 = matrix(CxyS1, B*ni, no)
  CxyS2 = rowSums(matrix(2*Cxy*mye, B*ni*no, no2))
  CxyS2 = matrix(CxyS2, B*ni, no)
  CxyS2 = CxyS2*mx
  Sd = Sd + CxyS1 + CxyS2

  outputs <- list(md, Sd)
  return(outputs)
}

#' Mean of Weights times Derivatives Products Terms Squared (wdo x (wdi*wdi))
#'
#' This function calculates mean of weights times derivatives products terms when
#' adding two of those products from current layer to already calculated expectation
#' that ended with one such product of next layer (i.e. wdo x (wdi*wdi)). Mean terms
#' are in array format. Once added, rows need to be
#' summed to aggregate expectations by node*node combinations of current layer.
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mpdi2 Mean array of combination of products of first order derivatives
#' @param mdgo Mean vector of derivatives in next layer
#' @param Cwdowdiwdi Covariance cov(wdo,(wdi*wdi)) of weights times derivatives products terms when there is one product in next layer and two in current
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @return Mean of weights times derivatives products terms
#' @export
fcMeanDlayer2row <- function(mpdi, mpdi2, mdgo, Cwdowdiwdi, ni, no, no2, B){
  mdgo = array(matrix(rep(rep(matrix(mdgo, ncol = no, byrow = TRUE), times = ni), each = ni), B*ni, no), c(B*ni, no, ni))
  md = mpdi2 * mdgo

  md = md + Cwdowdiwdi

  # Rearrange to get (B*ni x ni) matrix (expectation of each node from current layer multiplied by each other)
  # Sum each weight combination related to the same node
  md2 = matrix(apply(md, 3, rowSums), B*ni, ni)

  # Rearrange to have "chunk" of batches (consecutive rows for same batch' nodes)
  if (B>1){
    md3 = matrix(md2[1,],nrow=1)
    for (b in 1:B){
      for (i in 1:ni){
        md3 = rbind(md3, matrix(md2[b + i*B - B,], nrow = 1))
      }
    }
    md2 = md3[2:nrow(md3),]
  }

  return(md2)
}

#' Mean of Weights times Derivatives Products Terms ((wdo*wdo) x (wwdi^2))
#'
#' This function calculates mean of weights times derivatives products terms when
#' adding two of those products from current layer to already calculated expectation
#' that ended with one such product of next layer (i.e. (wdo*wdo) x (wwdi^2)). Mean terms
#' are in array format. Once added, rows need to be
#' summed to aggregate expectations by node*weight combinations of current layer.
#'
#' @param mpdi2w Combination of products of first order derivative of current layer (wd)*(wd) (iterations on weights on the same node)
#' @param mdgo Mean of derivatives in next layer
#' @param Cwdowdowwdi2 Covariance cov(wdowdo,wdiwdi) of weights times derivatives products terms, where the di terms are the same
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Mean of weights times derivatives products terms
#' @export
fcMeanDlayer2array <- function(mpdi2w, mdgo, Cwdowdowwdi2, ni, no, B){
  mdgoA = array(0, c(B*ni, no, no))
  for (k in 1:no){
    for (b in 0:(B-1)){
      mdgoA[(b*ni+1):(b*ni+ni), ,k] = matrix(rep(mdgo[k + no*b,], ni), nrow = ni, byrow = TRUE)
    }
  }

  md = mpdi2w * mdgoA

  md = md + Cwdowdowwdi2

  # Rearrange to get usual (B*ni x no) matrix
  md = matrix(apply(md, 3, rowSums), B*ni, no)

  return(md)
}

#' Covariance between Next Layer Product and Current Layer Multiplied Products
#'
#' This function calculates covariance cov(wdo,wdi*wdi) of weights times derivatives
#' products terms when there are one product in next layer and two in current.
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param Cdgoddgik Covariance between weights times derivatives from consecutive layers
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance cov(wdo,wdi*wdi) of weights times derivatives products terms when there is one product in next layer and two in current
#' @export
fcCovwdowdiwdi <- function(mpdi, Cdgoddgik, ni, no, B){
  Cwdowdiwdi = array(0, c(ni*B, no, ni))
  for (b in 0:(B-1)){
    for (k in 1:ni){
      for (i in (b*ni+1):(b*ni+ni)){
        Cwdowdiwdi[i,,k] = mpdi[i,] * Cdgoddgik[k+b*ni,] + mpdi[k+b*ni,] * Cdgoddgik[i,]
      }
    }
  }
  return(Cwdowdiwdi)
}


#' Covariance between Next Layer Product and Current Layer Squared Product
#'
#' This function calculates covariance cov(wdo,(wdi)^2) of weights times derivatives
#' products terms when there are one product in next layer and two squared in current.
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param Cdgoddgik Covariance between weights times derivatives from consecutive layers
#' @return Covariance cov(wdo,(wdi)^2) of weights times derivatives products terms when there is one product in next layer and two squared in current
#' @export
fcCovwdowdi2 <- function(mpdi, Cdgoddgik){
  Cwdowdi2 = 2*mpdi*Cdgoddgik
  return(Cwdowdi2)
}

#' Covariance between Products in (Same) Next and Current Layers
#'
#' This function calculates covariance cov(wdo^2,wdi*wdi) of weights times derivatives
#' products terms when there are two products in both next and current layers.
#' The product fom next layer is the same squared.
#'
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param Cwdowdiwdi Covariance cov(wdo,wdi*wdi) of weights times derivatives products terms when there are one product in next layer and two in current
#' @return Covariance cov(wdo^2,wdi*wdi) of weights times derivatives products terms when there are two products in both next and current layers
#' @export
fcCovwdo2wdiwdi <- function(mpdo, Cwdowdiwdi){
  Cwdo2wdiwdi = 2*mpdo*Cwdowdiwdi

  return(Cwdo2wdiwdi)
}

#' Covariance between Next Layer Multiplied Products and Current Layer Multiplied Products (Same Derivative)
#'
#' This function calculates covariance cov(wdowdo,wdiwdi) where the di terms are the same, when next second layer involves only a product term (wddo2).
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param Cdgodgi Covariance between weights times derivatives from consecutive layers
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in second next layer
#' @param B Batch size
#' @return Covariance cov(wdowdo,wdiwdi) where the di terms are the same
#' @export
fcCwdowdowwdi2 <- function(mpdi, mpdo, Cdgodgi, ni, no, no2, B){
  # mpdo = array(aperm(array(t(mpdo), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))
  # mpdo = t(matrix(mpdo[, rep(1:ncol(mpdo), each = ni),], no*no2,B*ni))
  #
  # Cwdowdowwdi2 = array(0, c(ni*B, no*no2, no))
  # for (b in 0:(no2-1)){
  #   for (k in 1:no){
  #     for (j in (b*no+1):(b*no+no)){
  #       if (j == (b*no+k)){
  #         Cwdowdowwdi2[,j,k] = 4 * mpdo[,j] * mpdi[,k] * Cdgodgi[,j]
  #       } else {
  #         Cwdowdowwdi2[,j,k] = mpdo[,j] * mpdi[,j-b*no] * Cdgodgi[,(b*no+k)] + mpdo[,(b*no+k)] * mpdi[,k] * Cdgodgi[,j]
  #       }
  #
  #     }
  #   }
  # }
  #
  # # Sum covariances together to come back (B*ni x no x no) array. Iterations for sum are next layer weights that change.
  # sum = array(matrix(Cwdowdowwdi2, nrow = B*ni*no), c(B*ni*no, no2, no))
  # Cwdowdowwdi2 = array(apply(sum, 3, rowSums), c(B*ni, no, no))



  mpdo = array(aperm(array(t(mpdo), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))

  # Replicate Cdgodgi matrix for each dimension of the array
  CdgodgiA_moving = array(Cdgodgi, c(B*ni, no*no2, no))
  mpdiA_moving = array(mpdi, c(B*ni, no*no2, no))
  mpdoA_moving = array(t(matrix(mpdo[, rep(1:ncol(mpdo), each = ni),], no*no2,B*ni)), c(B*ni,no*no2, no))
  # Prepare "fixed" elements for iterations
  CdgodgiA_temp = matrix(Cdgodgi[,1], ncol = 1)
  for (j in 1:no){
    for (i in 1:no2){
      CdgodgiA_temp = cbind(CdgodgiA_temp, matrix(Cdgodgi[,j + i*no - no], ncol = 1))
    }
  }
  CdgodgiA_fixed = array(matrix(rep(t(CdgodgiA_temp[,-1]), each = no), nrow=B*ni, byrow = TRUE), c(B*ni,no*no2, no))
  mpdiA_fixed = array(matrix(rep(t(mpdi), each = no*no2), nrow=B*ni, byrow = TRUE), c(B*ni,no*no2, no))
  mpdoM = matrix(mpdo, ncol = B)
  testA = matrix(mpdoM[1,], nrow = 1)
  for (j in 1:no){
    for (i in 1:no2){
      testA = rbind(testA, matrix(mpdoM[j + i*no - no,], nrow = 1))
    }
  }
  mpdoA = array(testA[-1,], c(no*no2,1,B))
  mpdoA_temp = matrix(mpdoA[, rep(1:ncol(mpdoA), each = ni*no),], no*no2,B*no*ni)
  mpdoA_fixed = array(aperm(array(matrix(t(mpdoA_temp), nrow=B), c(no,B*ni, no*no2)), c(2,1,3)), c(B*ni, no*no2, no))
  # Multiplier (multiply by 2 when start at same node and arrive at same node, no matter the weights)
  multiplier = array(1, c(B*ni, no*no2, no))
  for (j in 1:no){
    for (b in 0:(no2-1)){
      multiplier[,b*no+j,j] = 2*multiplier[,b*no+j,j]
    }
  }

  test = multiplier * (CdgodgiA_fixed * CdgodgiA_moving + mpdoA_moving * mpdiA_moving * CdgodgiA_fixed + mpdoA_fixed * mpdiA_fixed * CdgodgiA_moving)

  # Sum covariances together to come back (B*ni x no x no) array. Iterations for sum are next layer weights that change.
  sum = array(matrix(test, nrow = B*ni*no), c(B*ni*no, no2, no))
  Cwdowdowwdi2 = array(apply(sum, 3, rowSums), c(B*ni, no, no))

  return(Cwdowdowwdi2)

}

#' Covariance between Next Layer Multiplied Products and Current Layer Multiplied Products (Same Derivative, Minimum 3 Hidden Layers)
#'
#' This function calculates covariance cov(wdowdo,wdiwdi) where the di terms are the same when next second layer involves multiplied terms (wdo2wdo2).
#' It is used when there are at least 3 hidden layers.
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param Cdgodgi Covariance between weights times derivatives from consecutive layers
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in second next layer
#' @param B Batch size
#' @return Covariance cov(wdowdo,wdiwdi) where the di terms are the same
#' @export
fcCwdowdowwdi2_3hl <- function(mpdi, mpdo, mdgo2, Cdgodgi, ni, no, no2, B){
  mpdo = array(aperm(array(t(mpdo), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))

  # Replicate Cdgodgi matrix for each dimension of the array
  CdgodgiA_moving = array(Cdgodgi, c(B*ni, no*no2, no*no2))
  mpdiA_moving = array(mpdi, c(B*ni, no*no2, no*no2))
  mpdoA_moving = array(t(matrix(mpdo[, rep(1:ncol(mpdo), each = ni),], no*no2,B*ni)), c(B*ni,no*no2, no*no2))
  # Prepare "fixed" elements for iterations
  CdgodgiA_fixed = array(matrix(rep(t(Cdgodgi), each = no*no2), nrow=B*ni, byrow = TRUE), c(B*ni,no*no2, no*no2))
  mpdiA_fixed = array(matrix(rep(t(mpdi), each = no*no2), nrow=B*ni, byrow = TRUE), c(B*ni,no*no2, no*no2))
  mpdoA_temp = matrix(mpdo[, rep(1:ncol(mpdo), each = ni*no*no2),], no*no2,B*no*ni*no2)
  mpdoA_fixed = aperm(array(matrix(t(mpdoA_temp), nrow=B), c(no*no2,B*ni, no*no2)), c(2,1,3))
  # Prepare mdgo2
  mdgo2_temp1 = matrix(rep(mdgo2, each=no), ncol = no2)
  mdgo2_temp2= array(array(t(mdgo2_temp1), c(no*no2,no2,B)), c(no*no2*no2,1,B))
  mdgo2_temp3 = matrix(mdgo2_temp2[, rep(1:ncol(mdgo2_temp2), each = ni*no),], no*no2*no2,B*no*ni)
  mdgo2_temp4=array(matrix(t(mdgo2_temp3), nrow=B), c(no,B*ni, no*no2*no2 ))
  mdgo2A = array(aperm(mdgo2_temp4, c(2,1,3)), c(B*ni, no*no2, no*no2))
  # Multiplier (multiply by 2 when start at same node and arrive at same node, no matter the weights)
  multiplier = array(1, c(ni*B, no*no2, no))
  for (j in 1:no){
    for (b in 0:(no2-1)){
      multiplier[,b*no+j,j] = 2*multiplier[,b*no+j,j]
    }
  }
  multiplier = array(multiplier, c(ni*B, no*no2, no*no2))

  Cwdowdowwdi2 = multiplier * mdgo2A * (CdgodgiA_fixed * CdgodgiA_moving + mpdoA_moving * mpdiA_moving * CdgodgiA_fixed + mpdoA_fixed * mpdiA_fixed * CdgodgiA_moving)

  # Sum covariances together to come back (B*ni x no x no) array. Iterations for sum are next layer weights that change + across array dimensions (each (no)th array)
  sum = array(matrix(Cwdowdowwdi2, nrow = B*ni*no), c(B*ni*no, no2, no*no2))
  Cwdowdowwdi2 = matrix(apply(sum, 3, rowSums), nrow = B*ni*no*no)
  Cwdowdowwdi2 = array(rowSums(Cwdowdowwdi2), c(B*ni, no, no))

  return(Cwdowdowwdi2)

}

#' Covariance between Next Layer Multiplied Products and Current Layer Multiplied Products (Minimum 3 Hidden Layers)
#'
#' This function calculates covariance cov(wdowdo,wdiwdi) where all terms can be different.
#' It is used when there are at least 3 hidden layers.
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param Cdgodgi Covariance between weights times derivatives from consecutive layers
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in second next layer
#' @param B Batch size
#' @return Covariance cov(wdowdo,wdiwdi) where all terms can be different
#' @export
fcCwdowdowdiwdi <- function(mpdi, mpdo, Cdgodgi, ni, no, no2, B){
  # mpdo_original = mpdo
  # mpdi2 = matrix(rep(mpdi, no2), nrow = B*ni)
  # mpdo2 = array(aperm(array(t(mpdo), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))
  # mpdo2 = t(matrix(mpdo2[, rep(1:ncol(mpdo2), each = ni),], no*no2,B*ni))
  #
  # Cwdowdowdiwdi = array(0, c(ni*B, no*no2, no*ni))
  # seq = c(mpdi)
  # for (k in 1:(no*ni)){
  #   i = as.numeric(which(mpdi == seq[k], arr.ind = TRUE)[,"row"])
  #   j = as.numeric(which(mpdi == seq[k], arr.ind = TRUE)[,"col"])
  #   for (b in 0:(no2-1)){
  #     for (c in (b*no+1):(b*no+no)){
  #       if (c == (b*no+j)){
  #         # When same weight*node from next layer (i.e. (wdo)^2), coming from same current layer combination or not
  #         Cwdowdowdiwdi[,c,k] = 2*(mpdo[j,b+1] * mpdi[i,j] * Cdgodgi[,c] + mpdo2[,c] * mpdi2[,c] * Cdgodgi[i,j+b*no])
  #       } else {
  #         Cwdowdowdiwdi[,c,k] = mpdo[j,b+1] * mpdi[i,j] * Cdgodgi[,c] + mpdo2[,c] * mpdi2[,c] * Cdgodgi[i,j+b*no]
  #       }
  #     }
  #   }
  # }
  #
  #

  mpdo = array(aperm(array(t(mpdo), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))

  # Prepare "moving" elements for iterations
  CdgodgiA_moving = array(Cdgodgi, c(B*ni, no*no2, no*ni)) # Replicate Cdgodgi matrix for each dimension of the array
  mpdiA_moving = array(mpdi, c(B*ni, no*no2, no*ni))
  mpdoA_moving = array(t(matrix(mpdo[, rep(1:ncol(mpdo), each = ni),], no*no2,B*ni)), c(B*ni,no*no2, no*ni))
  # Prepare "fixed" elements for iterations
  CdgodgiA_temp = matrix(Cdgodgi[,1], ncol = 1)
  for (j in 1:no){
    for (i in 1:no2){
      CdgodgiA_temp = cbind(CdgodgiA_temp, matrix(Cdgodgi[,j + i*no - no], ncol = 1))
    }
  }
  CdgodgiA_temp2 = apply(array(CdgodgiA_temp[,-1], c(B*ni, no2, no)), 2, rbind)
  CdgodgiA_fixed = array(rep(t(CdgodgiA_temp2), each = ni*no), c(B*ni, no*no2, ni*no)) # If B = 1
  mpdiA_fixed = array(rep(mpdi, each=ni*no*no2), c(B*ni,no*no2, no*ni)) # If B = 1
  mpdoM = matrix(mpdo, ncol = B)
  testA = matrix(mpdoM[1,], nrow = 1)
  for (j in 1:no){
    for (i in 1:no2){
      testA = rbind(testA, matrix(mpdoM[j + i*no - no,], nrow = 1))
    }
  }
  mpdoA = array(testA[-1,], c(no*no2,1,B))
  mpdoA_temp = matrix(mpdoA[, rep(1:ncol(mpdoA), each = ni*no),], no*no2,B*no*ni)
  mpdoA_temp2 = array(aperm(array(matrix(t(mpdoA_temp), nrow=B), c(no,B*ni, no*no2)), c(2,1,3)), c(B*ni, no*no2, no))
  mpdoA_fixed = array(0, c(B*ni, no*no2, ni*no))
  for (i in 1:no){
    for (j in 1:ni){
      mpdoA_fixed[,,j + i*ni - ni] = mpdoA_temp2[,,i]
    }
  }
  # Multiplier (multiply by 2 when arrive at same node, no matter the weights)
  multiplier = array(1, c(ni*B, no*no2, no*ni))
  for (i in 1:no){
    for (j in 1:ni){
      for (b in 0:(no2-1)){
        multiplier[,b*no+i,j + i*ni - ni] = matrix(2, ncol = 1)
      }
    }
  }

  Cwdowdowdiwdi = multiplier * (CdgodgiA_fixed * CdgodgiA_moving + mpdoA_moving * mpdiA_moving * CdgodgiA_fixed + mpdoA_fixed * mpdiA_fixed * CdgodgiA_moving)


  # Sum covariances together to come back (B*ni x no x no*ni) array. Iterations for sum are next layer weights that change.
  sum = array(matrix(Cwdowdowdiwdi, nrow = B*ni*no), c(B*ni*no, no2, no*ni))
  Cwdowdowdiwdi = array(apply(sum, 3, rowSums), c(B*ni, no, no*ni))

  return(Cwdowdowdiwdi)

}

#' Covariance between Next Layer Multiplied Products and Current Layer Multiplied Products (Minimum 4 Hidden Layers)
#'
#' This function calculates covariance cov(wdowdo,wdiwdi) where all terms can be different.
#' It is used when there are at least 4 hidden layers.
#'
#' @param mpdi Mean vector of the first order derivative product wd of current layer
#' @param mpdo Mean vector of the first order derivative product wd of next layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param Cdgodgi Covariance between weights times derivatives from consecutive layers
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in second next layer
#' @param B Batch size
#' @return Covariance cov(wdowdo,wdiwdi) where all terms can be different
#' @export
fcCwdowdowdiwdi_4hl <- function(mpdi, mpdo, mdgo2, Cdgodgi, ni, no, no2, B){
  mpdo_original = mpdo
  mpdo = array(aperm(array(t(mpdo), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))

  # Prepare "moving" elements for iterations
  CdgodgiA_moving = array(Cdgodgi, c(B*ni, no*no2, no*ni*no2)) # Replicate Cdgodgi matrix for each dimension of the array
  mpdiA_moving = array(mpdi, c(B*ni, no*no2, no*ni*no2))
  mpdoA_moving = array(t(matrix(mpdo[, rep(1:ncol(mpdo), each = ni),], no*no2,B*ni)), c(B*ni,no*no2, no*ni*no2))

  # Prepare "fixed" elements for iterations
  Cdgodgi_temp = array(aperm(array(t(Cdgodgi), c(no*no2,ni,B)), perm=c(2, 1, 3)), c(ni*no*no2,1,B))
  CdgodgiA_fixed = array(rep(t(matrix(rep(matrix(Cdgodgi_temp, ncol = B), each = no*no2), ncol = B)), each = ni), c(B*ni, no*no2, ni*no*no2))

  mpdi_temp = array(aperm(array(t(mpdi), c(no,ni,B)), perm=c(2, 1, 3)), c(ni*no,1,B))
  mpdiA_fixed = array(rep(t(matrix(rep(matrix(mpdi_temp, ncol = B), each = no*no2), ncol = B)), each = ni), c(B*ni, no*no2, ni*no*no2))

  mpdoA_fixed = array(rep(t(matrix(rep(matrix(mpdo, ncol = B), each = no*no2*ni), ncol = B)), each = ni), c(B*ni, no*no2, ni*no*no2))

  # Prepare mdgo2
  mdgo2_temp1 = matrix(rep(mdgo2, each=ni*no), ncol = no2)
  mdgo2_temp2= array(array(t(mdgo2_temp1), c(ni*no*no2,no2,B)), c(ni*no*no2*no2,1,B))
  mdgo2_temp3 = matrix(mdgo2_temp2[, rep(1:ncol(mdgo2_temp2), each = ni*no),], ni*no*no2*no2,B*no*ni)
  mdgo2_temp4=array(matrix(t(mdgo2_temp3), nrow=B), c(no,B*ni, ni*no*no2*no2 ))
  mdgo2A = array(aperm(mdgo2_temp4, c(2,1,3)), c(B*ni, no*no2, ni*no*no2))

  # Multiplier (multiply by 2 when arrive at same node, no matter the weights)
  multiplier = array(1, c(ni*B, no*no2, no*ni))
  for (i in 1:no){
    for (j in 1:ni){
      for (b in 0:(no2-1)){
        multiplier[,b*no+i,j + i*ni - ni] = matrix(2, ncol = 1)
      }
    }
  }
  multiplier = array(multiplier, c(ni*B, no*no2, no*ni*no2))

  Cwdowdowdiwdi = multiplier * mdgo2A * (CdgodgiA_fixed * CdgodgiA_moving + mpdoA_moving * mpdiA_moving * CdgodgiA_fixed + mpdoA_fixed * mpdiA_fixed * CdgodgiA_moving)

  # Sum covariances together to come back (B*ni x no x no) array. Iterations for sum are next layer weights that change + across array dimensions (each (ni*no)th array)
  sum = array(matrix(Cwdowdowdiwdi, nrow = B*ni*no), c(B*ni*no, no2, ni*no*no2)) # Prepare to sum each (no)th columns in all matrices
  sum2 = matrix(apply(sum, 3, rowSums), nrow = B*ni*no*ni*no) # Once summed, results in (B*ni*no X 1) matrices. Rearrange to sum each current layer unique product (no2 terms to sum)
  Cwdowdowdiwdi = array(rowSums(sum2), c(B*ni, no, no*ni))

  return(Cwdowdowdiwdi)

}

#' Covariance between Activation and Hidden Units
#'
#' This function calculates covariance between activation and hidden units.
#'
#'
#' @param Jo Jacobian of next layer
#' @param J Jacobian of current layer
#' @param Sz Covariance of the units from current layer
#' @param mw Mean vector of the weights for the current layer
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between activation and hidden layers (same layer)
#' @return Covariance between activation (next layer) and hidden (current layer) layers
#' @export
fcCovaz <- function(Jo, J, Sz, mw, ni, no, B){
  Jo = t(matrix(matrix(rep(t(matrix(Jo, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  Caizi = J*Sz
  Caozi = Jo*matrix(Caizi, nrow(mw), ncol(mw))*mw

  outputs <- list(Caizi, Caozi)
  return(outputs)
}

#' Covariance between Derivatives and Hidden Units
#'
#' This function calculates covariance between derivatives and hidden units.
#'
#'
#' @param mao Mean vector of the activation units from next layer
#' @param mai Mean vector of the activation units from current layer
#' @param Caizi Covariance between activation and hidden layers (same layer)
#' @param Caozi Covariance between activation (next layer) and hidden (current layer) layers
#' @param acto Activation function index for next layer defined by \code{\link{activationFunIndex}}
#' @param acti Activation function index for current layer defined by \code{\link{activationFunIndex}}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Covariance between derivative (next layer) and hidden (current layer) layers
#' @return Covariance between derivative and hidden layers (same layer)
#' @export
fcCovdz <- function(mao, mai, Caizi, Caozi, acto, acti, ni, no, B){
  mao = t(matrix(matrix(rep(t(matrix(mao, no, B)), ni), nrow = no*ni, ncol = B, byrow = TRUE), no, ni*B))
  if (acti == 1){ # tanh
    Cdizi = -2*mai*Caizi
  } else if (acti == 2){ # sigmoid
    Cdizi = (1-2*mai)*Caizi
  } else if (acti == 4){ # relu
    Cdizi = matrix(0, nrow(mai), ncol(mai))
  } else {
    Cdizi = matrix(0, nrow(mai), ncol(mai))
  }

  if (acto == 1){ # tanh
    Cdozi = -2*mao*Caozi
  } else if (acto == 2){ # sigmoid
    Cdozi = (1-2*mao)*Caozi
  } else if (acto == 4){ # relu
    Cdozi = matrix(0, nrow(mao), ncol(mao))
  } else {
    Cdozi = matrix(0, nrow(mao), ncol(mao))
  }

  outputs <- list(Cdozi, Cdizi)
  return(outputs)
}

#' Covariance between Derivatives and Hidden States
#'
#' This function calculates covariance between derivatives and hidden states.
#' It is not related to the derivative calculation process.
#' It could be used infer Z (hidden states) with the constraint that the derivative of g with respect to Z equals 0.
#'
#' @param mwo Mean vector of the weights for the next layer
#' @param mw Mean vector of the weights for the current layer
#' @param mdgo2 Mean vector of derivatives in 2nd next layer
#' @param mpdi TBD
#' @param mdgoe Unit vector of length  (# response variables * B)
#' @param Cdozi Covariance between derivative (next layer) and hidden (current layer) layers
#' @param Cdizi between derivative and hidden layers (same layer)
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param no2 Number of units in 2nd next layer
#' @param B Batch size
#' @return Covariance between derivative and hidden states
#' @export
covdx <- function(mwo, mw, mdgo2, mpdi, mdgoe, Cdozi, Cdizi, ni, no, no2, B){
  mdgo2 = matrix(matrix(rep(mdgo2, no), nrow = no, byrow = TRUE), no*no2, B)
  mw = matrix(rep(matrix(t(mw), ni, no), B), nrow = ni*B, ncol = no, byrow = TRUE)
  mdgoe = array(aperm(array(t(mdgoe), c(no2,no,B)), perm=c(2, 1, 3)), c(no*no2,1,B))
  mdgoe = t(matrix(mdgoe[, rep(1:ncol(mdgoe), each = ni),], no*no2,B*ni))
  m = t(matrix(matrix(rep(t(mdgo2*mwo), ni), nrow = ni*nrow(mdgo2*mwo), byrow = TRUE), no*no2, B*ni))
  Cdozi = matrix(rep(Cdozi, no2), nrow = nrow(Cdozi))
  mpdi = matrix(rep(mpdi, no2), nrow(Cdozi), ncol(Cdozi))
  Cdx1 = Cdozi*m*mpdi

  mwdx2 = matrix(rep(mw, no2), nrow = nrow(mw))
  Cdizi = matrix(Cdizi, nrow(mwdx2), ncol(mwdx2))
  Cdx2 = Cdizi*mwdx2*mdgoe

  Cdx = matrix(rowSums(matrix(Cdx1+Cdx2, B*ni*no, no2)), B*ni, no)
  return(Cdx)
}

#' Mean vector of units
#'
#' This function calculate the mean vector of units \eqn{\mu_{Z}} for a given layer.
#'
#' @param mp Mean vector of the parameters for the current layer
#' @param ma Mean vector of the activation units from previous layer
#' @param idxFmwa List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @param idxFmwab Indices for biases of the current layer
#' @return Mean vector of the units for the current layer \eqn{\mu_{Z}}
#' @export
meanMz <- function(mp, ma, idxFmwa, idxFmwab){
  # mp is the mean of parameters for the current layer
  # ma is the mean of activation unit (a) from previous layer
  # idxFmwa{1} is the indices for weight w
  # idxFmwa{2} is the indices for activation unit a
  # idxFmwab is the indices for bias b

  # *NOTE*: All indices have been built in the way that we bypass
  # the transition step such as F*mwa + F*b

  if (nrow(idxFmwa[[1]]) == 1){
    idxSum = 2 # Sum by column
  } else {
    idxSum = 1 # Sum by row
  }

  mpb = matrix(mp[idxFmwab,], nrow = length(idxFmwab))
  mp = matrix(mp[idxFmwa[[1]],], nrow = max(nrow(idxFmwa[[1]]),ncol(idxFmwa[[1]])))
  ma = matrix(ma[idxFmwa[[2]],], nrow = max(nrow(idxFmwa[[2]]),ncol(idxFmwa[[2]])))

  mWa = matrix(apply(mp * ma, idxSum, sum), nrow = nrow(mpb), ncol = 1)
  mz = mWa + mpb
  return(mz)
}

#' Covariance matrix of units
#'
#' This function calculate the covariance matrix of the units \eqn{\Sigma_{Z}} for a given layer.
#'
#' @param mp Mean vector of the parameters for the current layer
#' @param ma Mean vector of the activation units from previous layer
#' @param Sp Covariance matrix of the parameters for the current layer
#' @param Sa Covariance matrix of the activation units from previous layer
#' @param idxFSwaF List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @param idxFSwaFb Indices for biases of the current layer
#' @return Covariance matrix of units for the current layer \eqn{\Sigma_{Z}}
#' @export
covarianceSz <- function(mp, ma, Sp, Sa, idxFSwaF, idxFSwaFb){
  # mp is the mean of parameters for the current layer
  # ma is the mean of activation unit (a) from previous layer
  # Sp is the covariance matrix for parameters p
  # Sa is the covariance matrix for a from the previous layer
  # idxFSwaF{1} is the indices for weight w
  # idxFSwaF{2} is the indices for activation unit a
  # idxFSwaFb is the indices for bias

  # *NOTE*: All indices have been built in the way that we bypass
  # the transition step such as Sa = F*Cwa*F' + F*Cb*F'

  if (nrow(idxFSwaF[[1]]) == ncol(mp)){
    idxSum = 2 # Sum by column
  } else {
    idxSum = 1 # Sum by row
  }

  Spb = matrix(Sp[idxFSwaFb,], nrow = length(idxFSwaFb))
  Sp = matrix(Sp[idxFSwaF[[1]],], nrow = max(nrow(idxFSwaF[[1]]),ncol(idxFSwaF[[1]])))
  ma = matrix(ma[idxFSwaF[[2]],], nrow = max(nrow(idxFSwaF[[2]]),ncol(idxFSwaF[[2]])))

  if (is.null(Sa)){
    Sz = apply(Sp * ma * ma, idxSum, sum)
  } else {
    mp = matrix(mp[idxFSwaF[[1]],], nrow = max(nrow(idxFSwaF[[1]]),ncol(idxFSwaF[[1]])))
    Sa = matrix(Sa[idxFSwaF[[2]],], nrow = max(nrow(idxFSwaF[[2]]),ncol(idxFSwaF[[2]])))
    Sz = apply(Sp * Sa + Sp * ma * ma + Sa * mp * mp, idxSum, sum)
  }
  Sz = Sz + Spb
  return(Sz)
}

#' Mean and Covariance Vectors of Units (Many Observations)
#'
#' This function calculate the mean vector of units \eqn{\mu_{Z}} and the covariance matrix of the units \eqn{\Sigma_{Z}}for a given layer.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mb Mean vector of the biases for the current layer
#' @param Sb Covariance of the biases for the current layer
#' @param ma Mean vector of the activation units from previous layer
#' @param Sa Covariance of the activation units from previous layer
#' @param ni Number of units in previous layer
#' @param no Number of units in current layer
#' @param B Batch size
#' @param rB Number of times batch size is repeated
#' @return Mean vector of the units for the current layer \eqn{\mu_{Z}}
#' @return Covariance matrix of the units for the current layer \eqn{\Sigma_{Z}}
#' @export
fcMeanVar <- function(mz, Sz, mw, Sw, mb, Sb, ma, Sa, ni, no, B, rB){
  if (any(is.nan(mb))){
    mb = rep(0, 1)
    Sb = rep(0, 1)
  } else {
    mb = matrix(mb, nrow = length(mb)*B)
    Sb = matrix(Sb, nrow = length(Sb)*B)
  }
  mw = matrix(matrix(mw, ni, no), nrow = ni, ncol = no*B)
  Sw = matrix(matrix(Sw, ni, no), nrow = ni, ncol = no*B)
  for (t in 1:rB){
    maloop = matrix(matrix(rep(t(matrix(ma[,t], ni, B)), no), ncol = B, byrow = TRUE), ni, no*B)
    Saloop = matrix(matrix(rep(t(matrix(Sa[,t], ni, B)), no), ncol = B, byrow = TRUE), ni, no*B)
    out_vectorizedMeanVar <- vectorizedMeanVar(maloop, mw, Saloop, Sw)
    mzloop = out_vectorizedMeanVar[[1]]
    Szloop = out_vectorizedMeanVar[[2]]
    mzloop = colSums(mzloop)
    Szloop = colSums(Szloop)

    mz[,t] = mzloop + mb
    Sz[,t] = Szloop + Sb
    }
  outputs <- list(mz, Sz)
  return(outputs)
}

#' Mean and Covariance Vectors of Units (One Observation)
#'
#' This function calculate the mean vector of units \eqn{\mu_{Z}} and the covariance matrix of the units \eqn{\Sigma_{Z}}for a given layer.
#'
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mb Mean vector of the biases for the current layer
#' @param Sb Covariance of the biases for the current layer
#' @param ma Mean vector of the activation units from previous layer
#' @param Sa Covariance of the activation units from previous layer
#' @param ni Number of units in previous layer
#' @param no Number of units in current layer
#' @return Mean vector of the units for the current layer \eqn{\mu_{Z}}
#' @return Covariance matrix of the units for the current layer \eqn{\Sigma_{Z}}
#' @export
fcMeanVarB1 <- function(mw, Sw, mb, Sb, ma, Sa, ni, no){

  if (any(is.nan(mb))){
    mb = matrix(0, no, 1)
    Sb = matrix(0, no, 1)
  } else {
    mb = matrix(mb, no, 1)
    Sb = matrix(Sb, no, 1)
  }

  mw = matrix(mw, ni, no)
  Sw = matrix(Sw, ni, no)
  ma = matrix(ma, ni, no)
  Sa = matrix(Sa, ni, no)

  out_vectorizedMeanVar <- vectorizedMeanVar(ma, mw, Sa, Sw)
  mzloop = out_vectorizedMeanVar[[1]]
  Szloop = out_vectorizedMeanVar[[2]]
  mzloop = matrix(colSums(mzloop), no, 1)
  Szloop = matrix(colSums(Szloop), no, 1)

  mz = mzloop + mb
  Sz = Szloop + Sb

  outputs <- list(mz, Sz)
  return(outputs)
}

#' Backpropagation (Parameters' Deltas) for Fully Connected Layers (Many Observations)
#'
#' This function calculates parameters' deltas at a given layer when using more than one observation at the time.
#'
#' @param deltaMw Next layer delta of mean vector of weights given \eqn{y} \eqn{\mu_{\theta}|y}
#' @param deltaSw Next layer delta of covariance matrix of weights given \eqn{y} \eqn{\Sigma_{\theta}|y}
#' @param deltaMb Next layer delta of mean vector of biases given \eqn{y} \eqn{\mu_{\theta}|y}
#' @param deltaSb Next layer delta of covariance matrix of biases given \eqn{y} \eqn{\Sigma_{\theta}|y}
#' @param Sw Covariance of the weights for the current layer
#' @param Sb Covariance of the biases for the current layer
#' @param ma Mean vector of the activation units for the current layer
#' @param deltaMr Delta of mean vector of the next layer units given \eqn{y} \eqn{\mu_{Z}|y}
#' @param deltaSr Delta of covariance matrix of the next layer units given \eqn{y} \eqn{\Sigma_{Z}|y}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @return - Delta of mean vector of weights given \eqn{y} \eqn{\mu_{\theta}|y}}
#' @return - Delta of covariance matrix of weights given \eqn{y} \eqn{\Sigma_{\theta}|y}}
#' @return - Delta of mean vector of biases given \eqn{y} \eqn{\mu_{\theta}|y}}
#' @return - Delta of covariance matrix of biases given \eqn{y} \eqn{\Sigma_{\theta}|y}}
#' @export
fcParameterBackwardPass <- function(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaMr, deltaSr, ni, no, B, rB){
  Cbz = matrix(rep(Sb, B), nrow = length(Sb))
  deltaMw = matrix(deltaMw, ncol = rB)
  deltaSw = matrix(deltaSw, ncol = rB)
  deltaMb = matrix(deltaMb, ncol = rB)
  deltaSb = matrix(deltaSb, ncol = rB)

  for (t in 1:rB){
    maloop = matrix(rep(t(matrix(ma[,t], ni, B)), no), ncol = B, byrow = TRUE)
    deltaMrw = matrix(matrix(rep(deltaMr[,t], ni), nrow = ni, byrow = TRUE), ni*no, B)
    deltaSrw = matrix(matrix(rep(deltaSr[,t], ni), nrow = ni, byrow = TRUE), ni*no, B)

    # Weights
    Cwz = Sw*maloop
    deltaMrw = Cwz*deltaMrw
    deltaSrw = (Cwz^2)*deltaSrw

    deltaMw[,t] = matrix(rowSums(deltaMrw), nrow(deltaMrw), 1)
    deltaSw[,t] = matrix(rowSums(deltaSrw), nrow(deltaSrw), 1)

    # Bias
    if (any(!is.nan(Sb))){
      deltaMrb = matrix(deltaMr[,t], no, B)
      deltaSrb = matrix(deltaSr[,t], no, B)
      deltaMrb = Cbz*deltaMrb
      deltaSrb = (Cbz^2)*deltaSrb
      deltaMb[,t] = matrix(rowSums(deltaMrb), nrow(deltaMrb), 1)
      deltaSb[,t] = matrix(rowSums(deltaSrb), nrow(deltaSrb), 1)
    }
  }

  deltaMw = matrix(rowSums(deltaMw), nrow(deltaMw), 1)
  deltaSw = matrix(rowSums(deltaSw), nrow(deltaSw), 1)
  deltaMb = matrix(rowSums(deltaMb), nrow(deltaMb), 1)
  deltaSb = matrix(rowSums(deltaSb), nrow(deltaSb), 1)
  outputs <- list(deltaMw, deltaSw, deltaMb, deltaSb)
  return(outputs)
}


#' Backpropagation (Parameters' Deltas) for Fully Connected Layers (One Observation)
#'
#' This function calculates parameters' deltas at a given layer when using one observation at the time.
#'
#' @param Sw Covariance of the weights for the current layer
#' @param Sb Covariance of the biaises for the current layer
#' @param ma Mean vector of the activation units for the current layer
#' @param deltaMr Delta of mean vector of the next layer units given \eqn{y} \eqn{\mu_{Z}|y}
#' @param deltaSr Delta of covariance matrix of the next layer units given \eqn{y} \eqn{\Sigma_{Z}|y}
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @return - Delta of mean vector of weights given \eqn{y} \eqn{\mu_{\theta}|y}}
#' @return - Delta of covariance matrix of weights given \eqn{y} \eqn{\Sigma_{\theta}|y}}
#' @return - Delta of mean vector of biases given \eqn{y} \eqn{\mu_{\theta}|y}}
#' @return - Delta of covariance matrix of biaises given \eqn{y} \eqn{\Sigma_{\theta}|y}}
#' @export
fcParameterBackwardPassB1 <- function(Sw, Sb, ma, deltaMr, deltaSr, ni, no){
  Cbz = Sb
  maloop = matrix(rep(t(ma), no), nrow = nrow(ma)*no, byrow = TRUE)
  deltaMrw = matrix(rep(deltaMr, ni), nrow = ncol(deltaMr)*ni, byrow = TRUE)
  deltaMrw = matrix(deltaMrw, ncol = 1)
  deltaSrw = matrix(rep(deltaSr, ni), nrow = ncol(deltaSr)*ni, byrow = TRUE)
  deltaSrw = matrix(deltaSrw, ncol = 1)

  # Weights
  Cwz = Sw*maloop
  deltaMrw = Cwz*deltaMrw
  deltaSrw = (Cwz^2)*deltaSrw

  deltaMw = matrix(rowSums(deltaMrw), nrow(deltaMrw), 1)
  deltaSw = matrix(rowSums(deltaSrw), nrow(deltaSrw), 1)

  # Bias
  if (any(!is.nan(Sb))){
    out_vectorizedDelta <- vectorizedDelta(Cbz, deltaMr, deltaSr)
    deltaMrb = out_vectorizedDelta[[1]]
    deltaSrb = out_vectorizedDelta[[2]]

    deltaMb = matrix(rowSums(deltaMrb), nrow(deltaMrb), 1)
    deltaSb = matrix(rowSums(deltaSrb), nrow(deltaSrb), 1)
  } else {
    deltaMb = Sb
    deltaSb = Sb
  }
  outputs <- list(deltaMw, deltaSw, deltaMb, deltaSb)
  return(outputs)
}

#' Covariance matrices between units and parameters
#'
#' This function calculate the covariance matrices between units and parameters
#' \eqn{\Sigma_{ZW}} and \eqn{\Sigma_{ZB}} for a given layer.
#'
#' @param ma Mean vector of the activation units from previous layer \eqn{\mu_{A}}
#' @param Sp Covariance matrix of the parameters for the current layer \eqn{\Sigma_{\theta}}
#' @param idxFCwwa List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @param idxFCb Indices for biases of the current layer
#' @return A list that contains:
#' @return - Covariance matrix between units and biases for the current layer \eqn{\Sigma_{ZB}}
#' @return - Covariance matrix between units and weights for the current layer \eqn{\Sigma_{ZW}}
#' @export
covarianceCzp <- function(ma, Sp, idxFCwwa, idxFCb) {
  # ma is the mean of activation unit (a) from previous layer
  # Sp is the covariance matrix for parameters p
  # idxFCwwa{1} is the indices for weight w
  # idxFCwwa{2} is the indices for weight action unit a
  # idxFcb is the indices for bias b

  # *NOTE*: All indices have been built in the way that we bypass
  # the transition step such as Cpa = F*Cpwa + F*Cb

  Czb = matrix(Sp[idxFCb,], nrow = length(idxFCb))
  Sp = matrix(Sp[idxFCwwa[[1]]], nrow = nrow(idxFCwwa[[1]]))
  ma = matrix(ma[idxFCwwa[[2]]], nrow = nrow(idxFCwwa[[2]]))

  Czw = Sp * ma

  outputs <- list(Czb, Czw)
  return(outputs)
}

#' Covariance matrix between units of the previous and current layers
#'
#' This function calculate the covariance matrix between units of the previous
#' and current layers \eqn{\Sigma_{ZZ^{+}}} for a given layer.
#'
#' @param mp Mean vector of the parameters for the current layer \eqn{\mu_{\theta}}
#' @param Sz Covariance matrix of the units for the current layer \eqn{\Sigma_{Z}}
#' @param J Jacobian matrix evaluated at \eqn{\mu_{Z}}
#' @param idxCawa List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @return Covariance matrix between units of the previous and current layers \eqn{\Sigma_{ZZ^{+}}}
#' @export
covarianceCzz <- function(mp, Sz, J, idxCawa) {

  Sz = matrix(Sz[idxCawa[[2]]], nrow = nrow(idxCawa[[2]]))
  mp = matrix(mp[idxCawa[[1]]], nrow = nrow(idxCawa[[1]]))
  J = matrix(J[idxCawa[[2]]], nrow = nrow(idxCawa[[2]]))

  Czz = J * Sz * mp
  return(Czz)
}

# Update Step

#' Backward parameters update
#'
#' This function updates parameters from responses to input data. It updates
#' \eqn{\mu_{\theta|y}} and \eqn{\Sigma_{\theta|y}} from the \eqn{\theta|y}
#' distribution for a given layer.
#'
#' @param mp Mean vector of the parameters for the current layer \eqn{\mu_{\theta}}
#' @param Sp Covariance matrix of the parameters for the current layer \eqn{\Sigma_{\theta}}
#' @param mzF Mean vector of the units for the next layer \eqn{\mu_{Z^{+}}}
#' @param SzF Covariance matrix of the units for the next layer \eqn{\Sigma_{Z^{+}}}
#' @param SzB Covariance matrix of the units for the next layer given \eqn{y} \eqn{\Sigma_{Z^{+}|y}}
#' @param Czp Covariance matrix between units and parameters for the current layer \eqn{\Sigma_{\theta Z^{+}}}
#' @param mzB Mean vector of the units for the next layer given \eqn{y} \eqn{\mu_{Z^{+}|y}}
#' @param idx List that contains the indices for the parameter update step of
#' the current layer
#' @details \eqn{f(\boldsymbol{\theta}|\boldsymbol{y}) = \mathcal{N}(\boldsymbol{\theta};\boldsymbol{\mu_{\theta|y}},\boldsymbol{\Sigma_{\theta|y}})} where
#' @details \eqn{\boldsymbol{\mu_{\theta|y}} =\boldsymbol{\mu_{\theta}} + \boldsymbol{J_{\theta}}(\boldsymbol{\mu_{Z^{+}|y}} - \boldsymbol{\mu_{Z^{+}}})}
#' @details \eqn{\boldsymbol{\Sigma_{\theta|y}} = \boldsymbol{\Sigma_{\theta}} + \boldsymbol{J_{\theta}}(\boldsymbol{\Sigma_{Z^{+}|y}} - \boldsymbol{\Sigma_{Z^{+}}})\boldsymbol{J_{\theta}^{T}}}
#' @details \eqn{\boldsymbol{J_{\theta}} = \boldsymbol{\Sigma_{\theta Z^{+}}}\boldsymbol{\Sigma^{-1}_{Z^{+}}}}
#' @return A list that contains:
#' @return - Mean vector of the parameters for the current layer given \eqn{y} \eqn{\mu_{\theta|y}}
#' @return - Covariance matrix of the parameters for the current layer given \eqn{y} \eqn{\Sigma_{\theta|y}}
#' @export
backwardParameterUpdate <- function(mp, Sp, mzF, SzF, SzB, Czp, mzB, idx){
  dz = mzB - mzF
  dz = matrix(dz[idx,], nrow = length(idx))
  dS = SzB - SzF
  dS  = matrix(dS[idx,], nrow = length(idx))
  SzF = 1 / SzF
  SzF = matrix(SzF[idx,], nrow = length(idx))
  J   = Czp * SzF
  # Mean
  mpUd = mp + rowSums(J * dz)
  # Covariance
  SpUd = Sp + rowSums(J * dS * J)

  outputs <- list(mpUd, SpUd)
  return(outputs)
}

#' Backward hidden states update
#'
#' This function updates hidden units from responses to input data. It updates
#' \eqn{\mu_{Z|y}} and \eqn{\Sigma_{Z|y}} from the \eqn{Z|y}
#' distribution for a given layer.
#'
#' @param mz Mean vector of the units for the current layer \eqn{\mu_{Z}}
#' @param Sz Covariance matrix of the units for the current layer \eqn{\Sigma_{Z}}
#' @param mzF Mean vector of the units for the next layer \eqn{\mu_{Z^{+}}}
#' @param SzF Covariance matrix of the units for the next layer \eqn{\Sigma_{Z^{+}}}
#' @param SzB Covariance matrix of the units for the next layer given \eqn{y} \eqn{\Sigma_{Z^{+}|y}}
#' @param Czz Covariance matrix between units of the previous and currents layers \eqn{\Sigma_{ZZ^{+}}}
#' @param mzB Mean vector of the units for the next layer given \eqn{y} \eqn{\mu_{Z^{+}|y}}
#' @param idx List that contains the indices for the hidden state update step of
#' the current layer
#' @details \eqn{f(\boldsymbol{z}|\boldsymbol{y}) = \mathcal{N}(\boldsymbol{z};\boldsymbol{\mu_{Z|y}},\boldsymbol{\Sigma_{Z|y}})} where
#' @details \eqn{\boldsymbol{\mu_{Z|y}} = \boldsymbol{\mu_{Z}} + \boldsymbol{J_{Z}}(\boldsymbol{\mu_{Z^{+}|y}} - \boldsymbol{\mu_{Z^{+}}})}
#' @details \eqn{\boldsymbol{\Sigma_{Z|y}} = \boldsymbol{\Sigma_{Z}} + \boldsymbol{J_{Z}}(\boldsymbol{\Sigma_{Z^{+}|y}} - \boldsymbol{\Sigma_{Z^{+}}})\boldsymbol{J_{Z}^{T}}}
#' @details \eqn{\boldsymbol{J_{Z}} = \boldsymbol{\Sigma_{Z Z^{+}}}\boldsymbol{\Sigma^{-1}_{Z^{+}}}}
#' @return A list that contains:
#' @return - Mean vector of the units for the current layer given \eqn{y} \eqn{\mu_{Z|y}}
#' @return - Covariance matrix of the units for the current layer given \eqn{y} \eqn{\Sigma_{Z|y}}
#' @export
backwardHiddenStateUpdate <- function(mz, Sz, mzF, SzF, SzB, Czz, mzB, idx){
  dz = mzB - mzF
  dz = matrix(dz[idx,], nrow = nrow(idx))
  dS = SzB - SzF
  dS  = matrix(dS[idx,], nrow = nrow(idx))
  SzF = 1 / SzF
  SzF = matrix(SzF[idx,], nrow = nrow(idx))
  J   = Czz * SzF
  # Mean
  mzUd = mz + rowSums(J * dz)
  # Covariance
  SzUd = Sz + rowSums(J * dS * J)

  outputs <- list(mzUd, SzUd)
  return(outputs)
}

#' Last Hidden Layer States' Deltas Update
#'
#' This function updates hidden layer units' deltas using next hidden layer' deltas. It updates
#' \eqn{\mu_{Z|y}} and \eqn{\Sigma_{Z|y}} from the \eqn{Z|y}
#' distribution.
#'
#' @param SzF Covariance matrix of the units for the next layer \eqn{\Sigma_{y}}
#' @param dMz Delta of mean vector of the units for the next hidden layer \eqn{\mu_{Z}}
#' @param dSz Delta of covariance matrix of the units for the next hidden layer \eqn{\Sigma_{Z}}
#' @details \eqn{f(\boldsymbol{z}|\boldsymbol{y}) = \mathcal{N}(\boldsymbol{z};\boldsymbol{\mu_{Z|y}},\boldsymbol{\Sigma_{Z^|y}})} where
#' @details \eqn{\boldsymbol{\mu_{{Z}|y}} = \boldsymbol{\mu_{Z}} + \boldsymbol{J_{Z}}(\boldsymbol{\mu_{Z^{+}|y}} - \boldsymbol{\mu_{Z^{+}}})}
#' @details \eqn{\boldsymbol{\Sigma_{{Z}|y}} = \boldsymbol{\Sigma_{Z}} + \boldsymbol{J_{Z}}(\boldsymbol{\Sigma_{Z^{+}|y}} - \boldsymbol{\Sigma_{Z^{+}}})}\boldsymbol{J_{Z}^{T}}
#' @details \eqn{\boldsymbol{J_{Z}} = \boldsymbol{\Sigma_{ZZ^{+}}} \boldsymbol{\Sigma_{Z^{+}}^{-1}}}
#' @return A list that contains:
#' @return - Delta of mean vector of the current hidden layer units given \eqn{y} \eqn{\mu_{Z}|y}}
#' @return - Delta of covariance matrix of the current hidden layer units given \eqn{y} \eqn{\Sigma_{Z}|y}}
#' @export
innovationVector <- function(SzF, dMz, dSz){
  iSzF = 1 / SzF
  iSzF[is.infinite(iSzF)] = 0
  out_vectorizedDelta <- vectorizedDelta(iSzF, dMz, dSz)
  deltaM = out_vectorizedDelta[[1]]
  deltaS = out_vectorizedDelta[[2]]

  out <- list(deltaM, deltaS)
  return(out)
}

#' Last hidden layer states update
#'
#' This function updates last hidden layer units using responses. It updates
#' \eqn{\mu_{Z^{(0)}|y}} and \eqn{\Sigma_{Z^{(0)}|y}} from the \eqn{Z^{(0)}|y}
#' distribution.
#'
#' @param mz Mean vector of the units for the last hidden layer \eqn{\mu_{X^{(0)}}}
#' @param Sz Covariance matrix of the units for the last hidden layer \eqn{\Sigma_{Z^{(0)}}}
#' @param mzF Mean vector of the units for the output layer \eqn{\mu_{y}}
#' @param SzF Covariance matrix of the units for the output layer \eqn{\Sigma_{y}}
#' @param Cyz Covariance matrix between last hidden layer units and responses \eqn{\Sigma_{YZ^{(0)}}}
#' @param y Response data
#' @details \eqn{f(\boldsymbol{z}^{(0)}|\boldsymbol{y}) = \mathcal{N}(\boldsymbol{z}^{(0)};\boldsymbol{\mu_{Z^{(0)}|y}},\boldsymbol{\Sigma_{Z^{(0)}|y}})} where
#' @details \eqn{\boldsymbol{\mu_{{Z}^{(0)}|y}} = \boldsymbol{\mu_{Z^{(0)}}} + \boldsymbol{\Sigma_{YZ^{(0)}}^{T}}\boldsymbol{\Sigma^{-1}_{Y}}(\boldsymbol{y} - \boldsymbol{\mu_{Y}})}
#' @details \eqn{\boldsymbol{\Sigma_{{Z}^{(0)}|y}} = \boldsymbol{\Sigma_{Z^{(0)}}} - \boldsymbol{\Sigma_{YZ^{(0)}}^{T}}\boldsymbol{\Sigma^{-1}_{Y}}\boldsymbol{\Sigma_{YZ^{(0)}}}}
#' @return A list that contains:
#' @return - Mean vector of the last hidden layer units given \eqn{y} \eqn{\mu_{Z^{(0)}|y}}
#' @return - Covariance matrix of the last hidden layer units given \eqn{y} \eqn{\Sigma_{Z^{(0)}|y}}
#' @export
forwardHiddenStateUpdate <- function(mz, Sz, mzF, SzF, Cyz, y){
  dz = y - mzF
  SzF = 1 / SzF
  SzF[is.infinite(SzF)] = 0
  K = Cyz * SzF
  # Mean
  mzUd = mz + K * dz
  # Covariance
  SzUd = Sz - K * Cyz

  #Outputs
  out <- list(mzUd, SzUd)
  return(out)
}

#' Backpropagation (Parameters Update)
#'
#' This function updates parameters.
#'
#' @param theta List of parameters
#' @param deltaTheta Parameters' deltas (mean and covariance for each)
#' @return List of updated parameters
#' @export
globalParameterUpdate <- function(theta, deltaTheta){

  # Initialization
  out_extractParameters <- extractParameters(theta)
  mw = out_extractParameters[[1]]
  Sw = out_extractParameters[[2]]
  mb = out_extractParameters[[3]]
  Sb = out_extractParameters[[4]]
  mwx = out_extractParameters[[5]]
  Swx = out_extractParameters[[6]]
  mbx = out_extractParameters[[7]]
  Sbx = out_extractParameters[[8]]
  out_extractParameters <- extractParameters(deltaTheta)
  deltaMw = out_extractParameters[[1]]
  deltaSw = out_extractParameters[[2]]
  deltaMb = out_extractParameters[[3]]
  deltaSb = out_extractParameters[[4]]
  deltaMwx = out_extractParameters[[5]]
  deltaSwx = out_extractParameters[[6]]
  deltaMbx = out_extractParameters[[7]]
  deltaSbx = out_extractParameters[[8]]

  out_twoPlus <- twoPlus(mw, Sw, deltaMw, deltaSw)
  mw = out_twoPlus[[1]]
  Sw = out_twoPlus[[2]]
  out_twoPlus <- twoPlus(mb, Sb, deltaMb, deltaSb)
  mb = out_twoPlus[[1]]
  Sb = out_twoPlus[[2]]
  out_twoPlus <- twoPlus(mwx, Swx, deltaMwx, deltaSwx)
  mwx = out_twoPlus[[1]]
  Swx = out_twoPlus[[2]]
  out_twoPlus <- twoPlus(mbx, Sbx, deltaMbx, deltaSbx)
  mbx = out_twoPlus[[1]]
  Sbx = out_twoPlus[[2]]

  theta = compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
  return(theta)
}

#' Reformat covariance matrix between units and parameters
#'
#' This function properly reformats covariance matrix between units and parameters
#' \eqn{\Sigma_{Z\theta}} for the update step.
#'
#' @param Czw Covariance matrix between units and weights for the current layer
#' @param Czb Covariance matrix between units and baises for the current layer
#' @param currentHiddenUnit Number of units in the current layer
#' @param prevHiddenUnit Number of units in the previous layer
#' @param batchSize Number of observations trained at the same time
#' @return Reformatted covariance matrix between units and parameters
#' @export
buildCzp <- function(Czw, Czb, currentHiddenUnit, prevHiddenUnit, batchSize){
  Czp = rbind(Czb, Czw)
  Czp = t(matrix(Czp, nrow = batchSize, ncol = currentHiddenUnit*prevHiddenUnit + currentHiddenUnit))
  return(Czp)
}

#' Reformat covariance matrix between units of the previous and current layers
#'
#' This function properly reformats covariance matrix between units of the
#' previous and current layers \eqn{\Sigma_{ZZ^{+}}} for the update step.
#'
#' @param Czz Covariance matrix between units of the previous and current layers
#' @param currentHiddenUnit Number of units in the current layer
#' @param prevHiddenUnit Number of units in the previous layer
#' @param batchSize Number of observations trained at the same time
#' @return Reformatted covariance matrix between of the previous and current layers
#' @export
buildCzz <- function(Czz, currentHiddenUnit, prevHiddenUnit, batchSize){
  Czz = t(matrix(Czz, nrow = currentHiddenUnit, ncol = prevHiddenUnit*batchSize))
  return(Czz)
}

#' Combination of Products of First Order Derivative (Iterations on Nodes)
#'
#' This function calculates mean of combination of products of first order derivatives (wd)*(wd).
#' Each node is multiplied to another node in the same layer (including itself).
#' Their weights are both pointing to the same node in the next layer.
#'
#' @param mpdi Mean vector of the current product of derivative and weight
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mda Mean vector of the activation units' derivative from current layer
#' @param Sda Covariance of the activation units' derivative from current layer
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Mean array of combination of products of first order derivatives
#' @export
fcCombinaisonDnode <- function(mpdi, mw, Sw, mda, Sda, ni, no, B){
  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  Sw = matrix(rep(t(matrix(Sw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  mda = matrix(mda, nrow(mw), ncol(mw))
  Sda = matrix(Sda, nrow(Sw), ncol(Sw))

  mpdi2 = array(0, c(ni*B, no, ni))
  for (b in 0:(B-1)){
    for (k in 1:ni){
      for (j in 1:no){
        for (i in (b*ni+1):(b*ni+ni)){
          if (b*ni+k == i){
            var = Sw[i,j]*Sda[i,j] + Sw[i,j]*mda[i,j]^2 + Sda[i,j]*mw[i,j]^2
          } else{
            var = 0
          }
          mpdi2[i,j,k] = mpdi[i,j]*mpdi[b*ni+k,j] + var
        }
      }
    }
  }

  return(mpdi2)
}

#' Combination of Products of First Order Derivative (Iterations on Weights)
#'
#' This function calculates mean of combination of products of first order derivatives (wd)*(wd).
#' Each weight is multiplied to another weight (including itself) from the same node.
#' Each node is multiplied to the same node (in the same layer).
#'
#' @param mpdi Mean vector of the current product of derivative and weight
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mda Mean vector of the activation units' derivative from current layer
#' @param Sda Covariance of the activation units' derivative from current layer
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Mean array of combination of products of first order derivatives
#' @export
fcCombinaisonDweight <- function(mpdi, mw, Sw, mda, Sda, ni, no, B){
  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  Sw = matrix(rep(t(matrix(Sw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  mda = matrix(mda, nrow(mw), ncol(mw))
  Sda = matrix(Sda, nrow(Sw), ncol(Sw))

  mpdi2w = array(0, c(ni*B, no, no))
  for (k in 1:no){
    for (j in 1:no){
      if (j == k){
        var = Sw[,j]*Sda[,j] + Sw[,j]*mda[,j]^2 + Sda[,j]*mw[,j]^2
      } else {
        var = mw[,k]*mw[,j]*Sda[,k]
      }
      mpdi2w[,j,k] = mpdi[,k]*mpdi[,j] + var
    }
  }

  return(mpdi2w)
}

#' Combination of Squared Products of First Order Derivative
#'
#' This function calculates mean of squared products of first order derivatives (wd)^2.
#' Every products (weight times node) from current layer are considered which results in a (B*ni x no)-matrix.
#'
#' @param mpdi Mean vector of the current product of derivative and weight
#' @param mw Mean vector of the weights for the current layer
#' @param Sw Covariance of the weights for the current layer
#' @param mda Mean vector of the activation units' derivative from current layer
#' @param Sda Covariance of the activation units' derivative from current layer
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Mean matrix of squared products of first order derivatives
#' @export
fcCombinaisonDweightNode <- function(mpdi, mw, Sw, mda, Sda, ni, no, B){
  mw = matrix(rep(t(matrix(mw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  Sw = matrix(rep(t(matrix(Sw, ni, no)), B), nrow = ni*B, ncol = no, byrow = TRUE)
  mda = matrix(mda, nrow(mw), ncol(mw))
  Sda = matrix(Sda, nrow(Sw), ncol(Sw))

  mpdi2wn = mpdi^2 + Sw*Sda + Sw*mda^2 + Sda*mw^2

  return(mpdi2wn)
}

#' All Possible Combinations of Products of First Order Derivatives
#'
#' This function calculates mean of products of first order derivatives wd*wd.
#' Since both weight and node are iterated over all products, every products
#' (weight times node) from current layer are considered which results in a
#' (Bni x no x noni)-array. I.e. each dimension of the array represents a single
#' product being multiplied to all other possible products from current layer.
#' Order is as followed: w11d1, w12d2, w13d3, ..., w1nidni, w21d1, w22d2, ..., w2nidni, ... wno1d1, ..., wnonidni
#'
#' @param mpdi Mean matrix of the current product of derivative and weight
#' @param mpdin Mean array of combination of products of first order derivatives (iterations on nodes)
#' @param mpdiw Mean array of combination of products of first order derivatives (iterations on weights)
#' @param ni Number of units in current layer
#' @param no Number of units in next layer
#' @param B Batch size
#' @return Mean array of combination of products of first order derivatives
#' @export
fcCombinaisonDweightNodeAll <- function(mpdi, mpdin, mpdiw, ni, no, B){
  mpdi2wnAll = array(0, c(ni*B, no, no*ni))
  seq = c(mpdi)
  for (k in 1:(no*ni)){
    mpdi2wnAll[,,k] = seq[k]*mpdi
  }

  # Adjust expectations when there is a covariance term to consider
  for (k in 1:no*ni){
    i = as.numeric(which(mpdi == seq[k], arr.ind = TRUE)[,"row"])
    j = as.numeric(which(mpdi == seq[k], arr.ind = TRUE)[,"col"])
    mpdi2wnAll[i,,k] = mpdiw[i,,j]
    mpdi2wnAll[,j,k] = mpdin[,j,i]
  }

  return(mpdi2wnAll)
}

#' Weights and biases initialization
#'
#' This function initializes the first weights and biases of the neural network.
#'
#' @param NN List that contains the structure of the neural network
#' @return A list that contains:
#' @return - List that contains initial mean vectors of the parameters for each layer
#' @return - List that contains initial covariance matrices of the parameters
#' for each layer
#' @export
initializeWeightBias <- function(NN){
  # Initialization
  NN$dropWeight = NULL
  NN <- c(NN, dropWeight = 0)
  nodes = NN$nodes
  numLayers = length(nodes)
  idxw = NN$idxw
  idxb = NN$idxb
  factor4Bp = NN$factor4Bp
  factor4Wp = NN$factor4Wp
  mp = matrix(list(), nrow = numLayers - 1, ncol = 1)
  Sp = matrix(list(), nrow = numLayers - 1, ncol = 1)

  for (j in 2:numLayers){
    # Bias variance
    Sbwloop_1 = factor4Bp[j-1] * matrix(1L, nrow = 1, ncol = length(idxb[[j-1, 1]]))

    # Bias mean
    bwloop_1 = stats::rnorm(ncol(Sbwloop_1)) * sqrt(Sbwloop_1)

    # Weight variance
    if (NN$hiddenLayerActivation == "relu" || NN$hiddenLayerActivation == "softplus" || NN$hiddenLayerActivation == "sigm"){
      Sbwloop_2 = factor4Wp[j-1] * (1/nodes[j-1]) * matrix(1L, nrow = 1, ncol = length(idxw[[j-1, 1]]))
    } else {
      Sbwloop_2 = factor4Wp[j-1] * (2/nodes[j-1] + nodes[j]) * matrix(1L, nrow = 1, ncol = length(idxw[[j-1, 1]]))
    }

    # Weight mean
    if (NN$dropWeight == 0){
      bwloop_2 = stats::rnorm(ncol(Sbwloop_2)) * sqrt(Sbwloop_2)
    } else {
      bwloop_2 = matrix(0, nrow = 1, ncol = length(idxw[[j-1, 1]]))
    }

    mp[[j-1, 1]] = t(cbind(bwloop_1, bwloop_2))
    Sp[[j-1, 1]] = t(cbind(Sbwloop_1, Sbwloop_2))
  }

  outputs <- list(mp, Sp)
  return(outputs)
}

#' Weights and biases initialization for calculating derivatives
#'
#' This function initializes the first weights and biases of the neural network.
#'
#' @param NN List that contains the structure of the neural network
#' @return theta: List that contains all parameters required in the neural network to perform derivative calculations
#' @export
initializeWeightBiasD <- function(NN){
  # Initialization
  nodes = NN$nodes
  numLayers = length(nodes)
  idxw = NN$idxw
  idxb = NN$idxb
  biasStd = 0.01
  noParam = NaN
  gainM = NN$gainM
  gainS = NN$gainS
  mw = createInitCellwithArray(numLayers-1)
  Sw = createInitCellwithArray(numLayers-1)
  mb = createInitCellwithArray(numLayers-1)
  Sb = createInitCellwithArray(numLayers-1)
  mwx = createInitCellwithArray(numLayers-1)
  Swx = createInitCellwithArray(numLayers-1)
  mbx = createInitCellwithArray(numLayers-1)
  Sbx = createInitCellwithArray(numLayers-1)

  for (j in 2:numLayers){
    if (!is.null(idxw[[j-1,1]])){
      fanIn = nodes[j-1]
      fanOut = nodes[j]
      if (NN$initParamType == "Xavier"){
        scale = 2 / (fanIn +fanOut)
        Sw[[j-1,1]] = gainS[j-1] * scale * rep(1, nrow(idxw[[j-1,1]]))
      }
      else if (NN$initParamType == "He"){
        scale = 1 / (fanIn +fanOut)
        Sw[[j-1,1]] = gainS[j-1] * scale * rep(1, nrow(idxw[[j-1,1]]))
      }
      mw[[j-1,1]] = gainM[j-1] * stats::rnorm(length(Sw[[j-1,1]])) * sqrt(Sw[[j-1,1]])
      if (!is.null(idxb[[j-1,1]])){
        Sb[[j-1,1]] = gainS[j-1] * scale * rep(1, nrow(idxb[[j-1,1]]))
        mb[[j-1,1]] = stats::rnorm(length(Sb[[j-1,1]])) * sqrt(Sb[[j-1,1]])
      }
    else {
      mw[[j-1,1]] = noParam
      Sw[[j-1,1]] = noParam
      Sb[[j-1,1]] = noParam
      mb[[j-1,1]] = noParam
      }
    }
  }
  out_catParameters = catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
  mw = out_catParameters[[1]]
  Sw = out_catParameters[[2]]
  mb = out_catParameters[[3]]
  Sb = out_catParameters[[4]]
  mwx = out_catParameters[[5]]
  Swx = out_catParameters[[6]]
  mbx = out_catParameters[[7]]
  Sbx = out_catParameters[[8]]

  theta = compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
  return(theta)
}

#' States Initialization
#'
#' Initiliazes neural network states.
#'
#' @param nodes Vector that contains the number of nodes at each layer
#' @param B Batch size
#' @param rB Number of times batch size is repeated
#' @param xsc TBD
#' @return states: States of the neural network
#' @export
initializeStates <- function(nodes, B, rB, xsc){
  # Normal network
  numLayers = length(nodes)
  mz <- createStateCellarray(nodes, numLayers, B, rB)
  Sz = mz
  ma = mz
  Sa = mz
  J = mz
  # Residual network
  idx = ifelse(xsc, no = 0)
  mdxs = matrix(list(), nrow = numLayers, ncol = 1)
  for (i in 1:numLayers){
    if (idx[i] == 1){
      mdxs[[i, 1]] = mz[[i, 1]]
    }
  }
  Sdxs = mdxs
  mxs = mdxs
  Sxs = mdxs
  states = compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs)
  return(states)
}

#' Input Initialization
#'
#' Initializes neural network inputs.
#'
#' @param states States of the neural network
#' @param mz0 TBD
#' @param Sz0 TBD
#' @param ma0 TBD
#' @param Sa0 TBD
#' @param J0 TBD
#' @param mdxs0 TBD
#' @param Sdxs0 TBD
#' @param mxs0 TBD
#' @param Sxs0 TBD
#' @param xsc TBD
#' @return \code{states}: States of the neural network
#' @export
initializeInputs <- function(states, mz0, Sz0, ma0, Sa0, J0, mdxs0, Sdxs0, mxs0, Sxs0, xsc){
  out_extractStates = extractStates(states)
  mz  = out_extractStates[[1]]
  Sz  = out_extractStates[[2]]
  ma  = out_extractStates[[3]]
  Sa  = out_extractStates[[4]]
  J = out_extractStates[[5]]
  mdxs = out_extractStates[[6]]
  Sdxs = out_extractStates[[7]]
  mxs = out_extractStates[[8]]
  Sxs = out_extractStates[[9]]

  # Normal network
  mz[[1,1]] = mz0
  if (any(is.null(Sz0))){
    Sz[[1,1]] = matrix(0, nrow(mz0), ncol(mz0))
  } else {
    Sz[[1,1]] = Sz0
  }
  if (any(is.null(ma0))){
    ma[[1,1]] = mz0
  } else {
    ma[[1,1]] = ma0
  }
  if (any(is.null(Sa0))){
    Sa[[1,1]] = Sz[[1,1]]
  } else {
    Sa[[1,1]] = Sa0
  }
  if (any(is.null(J0))){
    J[[1,1]] = matrix(1, nrow(mz0), ncol(mz0))
  } else {
    J[[1,1]] = J0
  }

  # Residual network
  if (any(is.null(mdxs0)) & !all(xsc == 0)){
    mdxs[[1,1]] = mz0
  } else {
    mdxs[1,1] = list(mdxs0) # In this case, mdxs0 might be NULL (only way to code it)
  }
  if (any(is.null(Sdxs0)) & !all(xsc == 0)){
    Sdxs[[1,1]] = matrix(0, nrow(mz0), ncol(mz0))
  } else {
    Sdxs[1,1] = list(Sdxs0)
  }
  if (any(is.null(mxs0)) & !all(xsc == 0)){
    mxs[[1,1]] = mz0
  } else {
    mxs[1,1] = list(mxs0)
  }
  if (any(is.null(Sxs0)) & !all(xsc == 0)){
    Sxs[[1,1]] = matrix(0, nrow(mz0), ncol(mz0))
  } else {
    Sxs[1,1] = list(Sxs0)
  }
  states = compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs)
  return(states)
}

#' Initialization (Matrix of Lists)
#'
#' Initializes a matrix containing lists.
#'
#' @param numLayers Number of layers in the neural network
#' @return x: Matrix containing empty lists
#' @export
createInitCellwithArray <- function(numLayers){
  x = matrix(list(), nrow = numLayers, ncol = 1)
  for (j in 1:numLayers){
    x[[j, 1]] = NaN
  }
  return(x)
}

#' States Initialization (Zero-Matrices)
#'
#' Initiliazes neural network states at 0.
#'
#' @param nodes Vector that contains the number of nodes at each layer
#' @param numLayers Number of layers in the neural network
#' @param B Batch size
#' @param rB Number of times batch size is repeated
#' @return Zero-matrices for each layer
#' @export
createStateCellarray <- function(nodes, numLayers, B, rB){
  z = matrix(list(), nrow = numLayers, ncol = 1)
  for (j in 2:numLayers){
    z[[j, 1]] = matrix(0, nrow = nodes[j]*B, ncol = rB)
  }
  return(z)
}

#' States Initialization (UnitMatrices)
#'
#' Initiliazes neural network derivative states at 1.
#'
#' @param nodes Vector that contains the number of nodes at each layer
#' @param numLayers Number of layers in the neural network
#' @param B Batch size
#' @param rB Number of times batch size is repeated
#' @return Unit matrices for each layer
#' @export
createDevCellarray <- function(nodes, numLayers, B, rB){
  d = matrix(list(), nrow = numLayers, ncol = 1)
  for (j in 1:numLayers){
    d[[j, 1]] = matrix(1, nrow = nodes[j]*B, ncol = rB)
  }
  return(d)
}

#' Concatenate Parameters
#'
#' Combines in a single column vector each parameter for all layers.
#'
#' @param mw TBD
#' @param Sw TBD
#' @param mb TBD
#' @param Sb TBD
#' @param mwx TBD
#' @param Swx TBD
#' @param mbx TBD
#' @param Sbx TBD
#' @return mw TBD in vector format
#' @return Sw TBD in vector format
#' @return mb TBD in vector format
#' @return Sb TBD in vector format
#' @return mwx TBD in vector format
#' @return Swx TBD in vector format
#' @return mbx TBD in vector format
#' @return Sbx TBD in vector format
#' @export
catParameters <- function(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx){
  var <- list(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
  for (n in 1:length(var)){
    start = var[[n]][[1]]
    for (i in 1:(length(var[[n]])-1)){
      start = c(start, var[[n]][[i+1]])
    }
    var[[n]] = start
  }
  mw  = var[[1]]
  Sw  = var[[2]]
  mb  = var[[3]]
  Sb  = var[[4]]
  mwx = var[[5]]
  Swx = var[[6]]
  mbx = var[[7]]
  Sbx = var[[8]]

  outputs <- list(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
  return(outputs)
}

#' Compress States
#'
#' Put together states into a list of states.
#'
#' @param mz TBD
#' @param Sz TBD
#' @param ma TBD
#' @param Sa TBD
#' @param J TBD
#' @param mdxs TBD
#' @param Sdxs TBD
#' @param mxs TBD
#' @param Sxs TBD
#' @return states: List of states
#' @export
compressStates <- function(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs){
  states = matrix(list(), nrow = 9, ncol = 1)
  states[[1, 1]] = mz
  states[[2, 1]] = Sz
  states[[3, 1]] = ma
  states[[4, 1]] = Sa
  states[[5, 1]] = J
  states[[6, 1]] = mdxs
  states[[7, 1]] = Sdxs
  states[[8, 1]] = mxs
  states[[9, 1]] = Sxs
  return(states)
}

#' Extract States
#'
#' Extract states from list of states.
#'
#' @param states List of states
#' @return mz TBD
#' @return Sz TBD
#' @return ma TBD
#' @return Sa TBD
#' @return J TBD
#' @return mdxs TBD
#' @return Sdxs TBD
#' @return mxs TBD
#' @return Sxs TBD
#' @export
extractStates <- function(states){
  mz = states[[1, 1]]
  Sz = states[[2, 1]]
  ma = states[[3, 1]]
  Sa = states[[4, 1]]
  J = states[[5, 1]]
  mdxs = states[[6, 1]]
  Sdxs = states[[7, 1]]
  mxs = states[[8, 1]]
  Sxs = states[[9, 1]]
  outputs <- list(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs)
  return(outputs)
}

#' Compress Parameters
#'
#' Put together parameters into a list of parameters.
#'
#' @param mw TBD
#' @param Sw TBD
#' @param mb TBD
#' @param Sb TBD
#' @param mwx TBD
#' @param Swx TBD
#' @param mbx TBD
#' @param Sbx TBD
#' @return theta: List of parameters
#' @export
compressParameters <- function(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx){
  theta = matrix(list(), nrow = 8, ncol = 1)
  theta[[1, 1]] = mw
  theta[[2, 1]] = Sw
  theta[[3, 1]] = mb
  theta[[4, 1]] = Sb
  theta[[5, 1]] = mwx
  theta[[6, 1]] = Swx
  theta[[7, 1]] = mbx
  theta[[8, 1]] = Sbx
  return(theta)
}

#' Extract Parameters
#'
#' Extract parameters from list of parameters.
#'
#' @param theta List of parameters
#' @return mw: TBD
#' @return Sw: TBD
#' @return mb: TBD
#' @return Sb: TBD
#' @return mwx: TBD
#' @return Swx: TBD
#' @return mbx: TBD
#' @return Sbx: TBD
#' @export
extractParameters <- function(theta){
  mw = theta[[1, 1]]
  Sw = theta[[2, 1]]
  mb = theta[[3, 1]]
  Sb = theta[[4, 1]]
  mwx = theta[[5, 1]]
  Swx = theta[[6, 1]]
  mbx = theta[[7, 1]]
  Sbx = theta[[8, 1]]
  outputs <- list(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
  return(outputs)
}

#' Compress Normalized Statistics TBD
#'
#' Put together normalized statistics into a list.
#'
#' @param mra TBD
#' @param Sra TBD
#' @return normStat: TBD
#' @export
compressNormStat <- function(mra, Sra){
  normStat = matrix(list(), nrow = 2, ncol = 1)
  normStat[[1, 1]] = mra
  normStat[[2, 1]] = Sra
  return(normStat)
}

#' Initialization (TBD)
#'
#' Initializes a matrix containing lists of 0 for TBD.
#'

#' @param NN List that contains the structure of the neural network
#' @return normStat: TBD
#' @export
createInitNormStat <- function(NN){
  mra = matrix(list(), nrow = length(NN$nodes) - 1, ncol = 1)
  Sra = matrix(list(), nrow = length(NN$nodes) - 1, ncol = 1)
  for (i in 1:(length(NN$nodes) - 1)){
    mra[[i,1]] = 0
    Sra[[i,1]] = 0
  }
  normStat = compressNormStat(mra, Sra)
  return(normStat)
}
