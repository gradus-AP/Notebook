#'@description 
#'We consider fitting of Geometric brownian motion.
#'
#'model:Geometric brownian motion
#'mu , sigma > 0: parameters 
#'{X_t} : random variables 
#'h : step (|h| << 1)
#'X_(t+1) = X_t + (mu * h X_t + h^{1/2} sigma W_t)
#'
library(ggplot2)
library(Rmisc)

#simulation
geom_brown_sim <- function(T_max, step, mu, sigma, x0) {
  sim <- x0
  w <- rnorm(T_max)
  for (t in 1:T_max) {
    sim <- append(sim, sim[t] * ((1 + step * mu) + sqrt(step) * sigma * w[t]))
  }
  return(data.frame(t = seq(0, T_max, length = T_max + 1), val = sim))
}

T_max = 100
step = 0.01
mu = 1.3
sigma = 0.50

sim <- geom_brown_sim(T_max = T_max, step = step, mu = mu, sigma = sigma, x0 = 1)
ggplot <- ggplot(data = sim, aes(x = t)) +
  geom_line(aes(y =  val), colour='blue')
plot(ggplot)

#particle filter
particle_filter <- function(T_max, step, observed, Num, initParticles){
  
  getOneStepaheadState <- function(currentState){
    sysNoise <- rnorm(Num)
    currentState$state <- currentState$state * ((1 + step * currentState$mu) + 
                                                  sqrt(step) * currentState$sigma * sysNoise)
    return(currentState)
  }
  
  getLikelihood <- function(observed, currentState){
    return(mapply(dnorm, rep(observed, Num), mean = (1 + step * currentState$mu) * currentState$state, 
                  sd = abs(sqrt(step) * currentState$sigma * currentState$state)))
  }
  
  predictiveParticles= NULL
  filteringParticles = initParticles
  particleWeights= NULL
  loglikelihood = - T_max * log(Num)
  
  estimated = data.frame(predicted = 0, filtered = mean(initParticles$state))
  
  for (t in 1:T_max) {
    #prediction
    predictiveParticles <- getOneStepaheadState(filteringParticles)
    
    #calculate likelihood
    particleWeights <- getLikelihood(observed[t], predictiveParticles) 
    loglikelihood = loglikelihood + log(sum(particleWeights))
    
    #resampling
    particleWeights <- particleWeights / sum(particleWeights)
    
    index <- sample(seq(1, Num, length = Num), Num, replace = TRUE, prob = particleWeights[1:Num])
    filteringParticles <- predictiveParticles[index,]
    
    estimated <- rbind(estimated , c(mean(predictiveParticles$state), mean(filteringParticles$state)))
  }
  return(list(estimated = estimated[2:(T_max + 1),], mu = filteringParticles$mu, sigma = filteringParticles$sigma, loglikelihood = loglikelihood))
}

#estimation
Num <- 20000
initParticles <- data.frame(state = exp(rnorm(Num, sd = 0.3)), mu = runif(Num, -0.5, 3.0), sigma = runif(Num, 0.2, 2.0))
pf <- particle_filter(T_max = T_max, step = step, observed =  sim$val[1:T_max], Num = Num, initParticles = initParticles)

estimated <- pf$estimated$filtered
df <- data.frame(time = seq(1, T_max, 1), estimate = estimated, true = sim$val[1:T_max])
ggplot <- ggplot(data = df, aes(time)) +
  geom_line(aes(y =  estimate), colour='navy') +
  geom_line(aes(y =  true), colour='darkorange')
plot(ggplot)

#parameters
df_mu <- data.frame(mu = pf$mu)
ggplot1 <- ggplot(df_mu, aes(x = mu)) +
  geom_histogram(bins = 20) +
  geom_vline(xintercept=mean(pf$mu),colour='blue')

df_sigma <- data.frame(sigma = pf$sigma)
ggplot2 <- ggplot(df_sigma, aes(x = sigma)) + 
  geom_histogram(bins = 20) + 
  geom_vline(xintercept=mean(pf$sigma),colour='blue')

multiplot(ggplot1, ggplot2,cols=2)