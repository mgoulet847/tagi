#' One iteration of the Tractable Approximate Gaussian Inference (TAGI)
#'
#' This function goes through one learning iteration of the neural network model
#' using TAGI.
#'
#' @param NN List that contains the structure of the neural network
#' @param mp List that contains mean vectors of the parameters for each layer
#' @param Sp List that contains covariance matrices of the
#' parameters for each layer
#' @param x Set of input data
#' @param y Set of corresponding responses
#' @return A list that contains:
#' @return - mp: List that contains the updated mean vectors of the parameters
#' for each layer
#' @return - Sp: List that contains the updated covariance matrices of the
#' parameters for each layer
#' @return - zn: Predicted responses
#' @return - Szn: Variance vector of predicted responses
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

# Forward network

#' Forward uncertainty propagation
#'
#' This function feeds the neural network forward from input data to
#' responses.
#'
#' @param NN List that contains the structure of the neural network
#' @param mp List that contains mean vectors of the parameters for each layer
#' @param Sp List that contains covariance matrices of the
#' parameters for each layer
#' @param x Set of input data
#' @return A list that contains:
#' @return - mz: List that contains the mean vectors of the units
#' for each layer
#' @return - Sz: List that contains the covariance matrices of the
#' units for each layer
#' @return - Czw: List that contains the covariance matrices between
#' units and weights for each layer
#' @return - Czb: List that contains the covariance matrices between
#' units and biases for each layer
#' @return - Czz: List that contains the covariance matrices between
#' previous and current units for each layer
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
      out_act = meanA(mz[[2,1]], mz[[2,1]], hiddenLayerActFunIdx)
      ma[[2,1]] = out_act[[1]]
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

# Backward network

#' Backpropagation
#'
#' This function feeds the neural network backward from responses to input data.
#'
#' @param NN List that contains the structure of the neural network
#' @param mp List that contains mean vectors of the parameters for each layer
#' @param Sp List that contains covariance matrices of the
#' parameters for each layer
#' @param mz List that contains the mean vectors of the units
#' for each layer
#' @param Sz List that contains the covariance matrices of the
#' units for each layer
#' @param Czw List that contains the covariance matrices between
#' units and weights for each layer
#' @param Czb List that contains the covariance matrices between
#' units and biases for each layer
#' @param Czz List that contains the covariance matrices between units of the
#' previous and current layers for each layer
#' @param y A vector or a matrix of responses
#' @return A list that contains:
#' @return - mpUd: List that contains the updated mean vectors of the parameters
#' for each layer
#' @return - SpUd: List that contains the updated covariance matrices of the
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
  out_fowardHiddenStateUpdate = fowardHiddenStateUpdate(mz[[lHL+1]], Sz[[lHL+1]], mz[[lHL+1]], Szv, Sz[[lHL+1]], y)
  mzUd[[lHL+1]] = out_fowardHiddenStateUpdate[[1]]
  SzUd[[lHL+1]] = out_fowardHiddenStateUpdate[[2]]

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

#' Mean vector of units
#'
#' This function calculate the mean vector of the units \eqn{\mu_{Z}} for a given layer.
#'
#' @param mp Mean vector of the parameters for the current layer
#' @param ma Mean vector of the activation units from previous layer
#' @param idxFmwa List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @param idxFmwab Indices for biases of the current layer
#' @return Mean vector of the units for the current layer
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
  mp = matrix(mp[idxFmwa[[1]]], nrow = max(nrow(idxFmwa[[1]]),ncol(idxFmwa[[1]])))
  ma = matrix(ma[idxFmwa[[2]]], nrow = max(nrow(idxFmwa[[2]]),ncol(idxFmwa[[2]])))

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
#' @return Covariance matrix of units for the current layer
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

  if (nrow(idxFSwaF[[1]]) == 1){
    idxSum = 2 # Sum by column
  } else {
    idxSum = 1 # Sum by row
  }

  Spb = matrix(Sp[idxFSwaFb,], nrow = length(idxFSwaFb))
  Sp = matrix(Sp[idxFSwaF[[1]]], nrow = max(nrow(idxFSwaF[[1]]),ncol(idxFSwaF[[1]])))
  ma = matrix(ma[idxFSwaF[[2]]], nrow = max(nrow(idxFSwaF[[2]]),ncol(idxFSwaF[[2]])))

  if (is.null(Sa)){
    Sz = apply(Sp * ma * ma, idxSum, sum)
  } else {
    mp = matrix(mp[idxFSwaF[[1]]], nrow = max(nrow(idxFSwaF[[1]]),ncol(idxFSwaF[[1]])))
    Sa = matrix(Sa[idxFSwaF[[2]]], nrow = max(nrow(idxFSwaF[[2]]),ncol(idxFSwaF[[2]])))
    Sz = apply(Sp * Sa + Sp * ma * ma + Sa * mp * mp, idxSum, sum)
  }
  Sz = Sz + Spb
  return(Sz)
}

#' Covariance matrices between units and parameters
#'
#' This function calculate the covariance matrices between units and parameters
#' \eqn{\Sigma_{ZW}} and \eqn{\Sigma_{ZB}} for a given layer.
#'
#' @param ma Mean vector of the activation units from previous layer
#' @param Sp Covariance matrix of the parameters for the current layer
#' @param idxFCwwa List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @param idxFCb Indices for biases of the current layer
#' @return A list that contains:
#' @return - Covariance matrix between units and biases for the current layer
#' @return - Covariance matrix between units and weights for the current layer
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
#' @param mp Mean vector of the parameters for the current layer
#' @param Sz Covariance matrix of the units for the current layer
#' @param J Jacobian matrix evaluated at \eqn{\mu_{z}}
#' @param idxCawa List that contains the indices for weights and for activation
#' units for the current and previous layers respectively
#' @return Covariance matrix between units of the previous and current layers
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
#' @param mp Mean vector of the parameters for the current layer
#' @param Sp Covariance matrix of the parameters for the current layer
#' @param mzF Mean vector of the units for the next layer
#' @param SzF Covariance matrix of the units for the next layer
#' @param SzB Updated covariance matrix of the units for the next layer
#' @param Czp Covariance matrix between units and parameters for the current layer
#' @param mzB Updated mean vector of the units for the next layer
#' @param idx List that contains the indices for the parameter update step of
#' the current layer
#' @return A list that contains:
#' @return - Updated mean vector of the parameters for the current layer
#' @return - Updated covariance matrix of the parameters for the current layer
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
#' \eqn{\mu_{z|y}} and \eqn{\Sigma_{z|y}} from the \eqn{z|y}
#' distribution for a given layer.
#'
#' @param mz Mean vector of the units for the current layer
#' @param Sz Covariance matrix of the units for the current layer
#' @param mzF Mean vector of the units for the next layer
#' @param SzF Covariance matrix of the units for the next layer
#' @param SzB Updated covariance matrix of the units for the next layer
#' @param Czz Covariance matrix between units of the previous and currents layers
#' @param mzB Updated mean vector of the units for the next layer
#' @param idx List that contains the indices for the hidden state update step of
#' the current layer
#' @return A list that contains:
#' @return - Updated mean vector of the units for the current layer
#' @return - Updated covariance matrix of the units for the current layer
#' @export
backwardHiddenStateUpdate <- function(mz, Sz, mzF, SzF, SzB, Czz, mzB, idx){
  dz = mzB - mzF
  dz = matrix(dz[idx,], nrow = length(idx))
  dS = SzB - SzF
  dS  = matrix(dS[idx,], nrow = length(idx))
  SzF = 1 / SzF
  SzF = matrix(SzF[idx,], nrow = length(idx))
  J   = Czz * SzF
  # Mean
  mzUd = mz + rowSums(J * dz)
  # Covariance
  SzUd = Sz + rowSums(J * dS * J)

  outputs <- list(mzUd, SzUd)
  return(outputs)
}

#' Last hidden layer states update
#'
#' This function updates last hidden layer units using responses. It updates
#' \eqn{\mu_{z^{(0)}|y}} and \eqn{\Sigma_{z^{(0)}|y}} from the \eqn{z^{(0)}|y}
#' distribution.
#'
#' @param mz Mean vector of the units for the output layer
#' @param Sz Covariance matrix of the units for the output layer
#' @param mzF Mean vector of the units for the output layer
#' @param SzF Covariance matrix of the units for the output layer
#' @param Cyz Covariance matrix between last hidden layer units and responses
#' @param y Response data
#' @return A list that contains:
#' @return - Updated mean vector of the last hidden layer units
#' @return - Updated covariance matrix of the last hidden layer units
#' @export
fowardHiddenStateUpdate <- function(mz, Sz, mzF, SzF, Cyz, y){
  dz = y - mzF
  SzF = 1 / SzF
  K = Cyz * SzF
  # Mean
  mzUd = mz + K * dz
  # Covariance
  SzUd = Sz - K * Cyz

  #Outputs
  out <- list(mzUd, SzUd)
  return(out)
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
