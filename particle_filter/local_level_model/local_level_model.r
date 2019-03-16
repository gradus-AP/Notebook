#'@description 
#'implimentation of particle filter 
#'
#'We apply local level model to Nile river flow data.
#'Local level model (x[t], z[t]) is defined the following equations
#'eq1  x[t] = x[t-1] + sysNoise[t] (sysNoise[] ~ i.i.d. Norm(0, sigma ^ 2))
#'eq2  z[t] ~ x[t] + obsNoise[t] (obsNoise[] ~ i.i.d. Normal(0, tau ^ 2)).
#'
#'First, we estimate system noise & observation noise & state vectors x[t]
#'using Kalman filter , which is implementedpackage{dlm}.
#'
#'Second ,we estimate state vectors x[t] using particle filter.
#'
#'Finally, we compare state vectors x[t] estimated using Kalman filter  
#'with that using particle filter. 

library(ggplot2)
library(dlm)

#fitting to local level model using Kalman filter
build_model <- function(parm){
  dlmModPoly(
    order = 1, 
    
    dV = exp(parm[1]),
    dW = exp(parm[2]))
}
T = length(Nile)

fit <- dlmMLE(Nile, parm = c(0, 0), build = build_model)
obsVar <- fit$par[1]
sysVar <- fit$par[2]

model.fitted <- build_model(c(obsVar, sysVar))
model.filtered <- dlmFilter(y = Nile,mod = model.fitted)
kf_filtered <- model.filtered$m[2:T]

df <- data.frame(time = seq(1, T-1, 1), kf_estimate = kf_filtered, org_seq = Nile[1: T-1])
ggplot <- ggplot(data = df, aes(time))
ggplot <- ggplot + geom_line(aes(y =  org_seq), colour='#339900')
ggplot <- ggplot + geom_line(aes(y =  kf_filtered), colour='#000099')
plot(ggplot)

#particle filter
getOneStepaheadState <- function(currentState, sysNoise){
  return(currentState + sysNoise)
}

getSystemNoise <- function(Num) {
  return(rnorm(Num,mean = 0,sd = exp(sysVar / 2)))
}

getLikelihood <- function(observed, currentState){
  return(dnorm(observed, mean = currentState, sd = exp(obsVar / 2)))
}

#vs Particle Filter
ParticleFilter <- function(getOneStepaheadState, getSystemNoise, getLikelihood, T, observed, Num, initParticles) {
  
  predictiveParticlesList = NULL
  filteringParticlesList = matrix(initParticles, nrow = 1)
  particleWeightsList = NULL
  loglikelihood = - T * log(Num)
  
  for (t in 1:T) {
    #prediction
    predictiveParticles <- mapply(getOneStepaheadState, filteringParticlesList[t,], getSystemNoise(Num))
    predictiveParticlesList <- rbind(predictiveParticlesList, predictiveParticles)
    
    #calculate likelihood
    particleWeights <- mapply(getLikelihood, rep(observed[t], Num), predictiveParticlesList[t,])
    loglikelihood = loglikelihood + log(sum(particleWeights))
    
    #resampling
    particleWeights <- particleWeights / sum(particleWeights)
    particleWeightsList <- rbind(particleWeightsList, particleWeights)
    
    filteringParticles <- sample(predictiveParticles, Num, replace = TRUE, prob = particleWeights)
    filteringParticlesList <- rbind(filteringParticlesList, filteringParticles)
  }
  return(list(predictiveParticlesList = predictiveParticlesList, filteringParticlesList = filteringParticlesList[2:T,], loglikelihood = loglikelihood))
}

Num <- 100
initParticles <- rep(1100, Num)
pf <- ParticleFilter(getOneStepaheadState, getSystemNoise, getLikelihood, T, observed =  Nile[1:T], Num, initParticles)

pf$filteringParticlesList
estimate <- apply(pf$filteringParticlesList, MARGIN = 1, mean)

df <- data.frame(time = seq(1, T-1, 1), estimate = estimate, org_seq = Nile[1:T-1])
ggplot <- ggplot(data = df, aes(time))
ggplot <- ggplot + geom_line(aes(y =  org_seq), colour='#339900')
ggplot <- ggplot + geom_line(aes(y =  estimate), colour='#000099')
plot(ggplot)

#compare kalman filter v.s. particle filter
df <- data.frame(time = seq(1, T-1, 1), orig_seq = Nile[1:T-1], kf_filtered = kf_filtered, pf_filtered = estimate[1:T-1])
ggplot <- ggplot(data = df, aes(time))
ggplot <- ggplot + geom_line(aes(y =  orig_seq), colour='#339900')
ggplot <- ggplot + geom_line(aes(y =  kf_filtered), colour='#000099')
ggplot <- ggplot + geom_line(aes(y =  pf_filtered), colour='#ffa500')
plot(ggplot)