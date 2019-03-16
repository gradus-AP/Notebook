#@description
#Fitting a generalized state space model (GSSM Model) to data.
#GSSM model
#x[t] : latant variables
#z[t] : observed variables
# 
# system equation
# x[t] = x[t-1] + systemNoise[t] (systemNoise[] i.i.d. ~ Exponential(lambda), lambda > 0)
# 
# observation equation
# z[t] ~ Poisson(x[t]) 
library(ggplot2)

#model definition
gssm <- function(lambda) {
    getOneStepaheadState <- function(currentState, sysNoise){
        return(currentState + sysNoise)
    }

    getSystemNoise <- function(Num) {
        return(rexp(Num, rate = lambda))
    }

    getLikelihood <- function(observed, currentState){
        return(dpois(observed, lambda = currentState))
    }

    return(list(getOneStepaheadState = getOneStepaheadState, getSystemNoise = getSystemNoise, getLikelihood = getLikelihood))
}

#particle filter
ParticleFilter <- function(model, T, observed, Num, initParticles) {
  
  predictiveParticlesList = NULL
  filteringParticlesList = matrix(initParticles, nrow = 1)
  particleWeightsList = NULL
  loglikelihood = - T * log(Num)
  
  for (t in 1:T) {
    #prediction
    predictiveParticles <- mapply(gssm$getOneStepaheadState, filteringParticlesList[t,], gssm$getSystemNoise(Num))
    predictiveParticlesList <- rbind(predictiveParticlesList, predictiveParticles)
    
    #calculate likelihood
    particleWeights <- mapply(gssm$getLikelihood, rep(observed[t], Num), predictiveParticlesList[t,])
    loglikelihood = loglikelihood + log(sum(particleWeights))
    
    #resampling
    particleWeights <- particleWeights / sum(particleWeights)
    particleWeightsList <- rbind(particleWeightsList, particleWeights)
    
    filteringParticles <- sample(predictiveParticles, Num, replace = TRUE, prob = particleWeights)
    filteringParticlesList <- rbind(filteringParticlesList, filteringParticles)
  }
  return(list(predictiveParticlesList = predictiveParticlesList, filteringParticlesList = filteringParticlesList[2:(T + 1),], loglikelihood = loglikelihood))
}

#generating test data
T <- 30
x_0 <- 10
lambda <- 0.2
v <- rexp(T, rate = lambda)
x <- cumsum(v) + x_0
z <- rpois(T, lambda = x)
  
df <- data.frame(value = v)
ggplot <- ggplot(df, aes(x = value))
ggplot <- ggplot + geom_histogram(bins = 30)
plot(ggplot)

#fit 
Num <- 2000
initParticles <- runif(Num, 0, 30)
model <- gssm(lambda = lambda)
pf <- ParticleFilter(model, T, observed =  z, Num, initParticles)

pf$filteringParticlesList
estimate <- apply(pf$filteringParticlesList, MARGIN = 1, mean)

#visualization 
df_org <- data.frame(time = seq(1, T, 1), val = z[1:T], obj = rep('original sequence', T))
df_est <- data.frame(time = seq(1, T, 1), val = estimate, obj = rep('estimated state', T))
df_true <- data.frame(time = seq(1, T, 1), val = x[1:T], obj = rep('true state', T))
df <- rbind(rbind(df_org, df_est), df_true)

ggplot <- ggplot(df, aes(x = time, y = val, colour = obj, group = obj))
ggplot <- ggplot + geom_line()
plot(ggplot)