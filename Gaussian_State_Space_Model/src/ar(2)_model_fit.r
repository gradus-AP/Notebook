#'description
#'Fitting AR(2) model to Nile data using Gaussian state space model.
#'Model
#'Let a[1], a[2] be parameters. AR(2) models is defined as the
#'following relation:
#'x[t] = -a[1] * x[t-1] - a[2] * x[t-2] + err[t], err[t] ~ i.i.d. Normal(0, sigma ^2)
#'
#'We estimate parameters a[1], a[2] in the following code.
library(rstan)
library(ggplot2)

x_init <- c(1120, 1120)
Q_init <- rbind(c(1000, 0.0), c(0.0, 1000))

z <- matrix(Nile,nrow = 100)
len = length(z)
plot.ts(z)
summary(z)
acf(z)

W = rbind(c(1000, 0.0), c(0.0, 1000))
V = matrix(c(100), nrow = 1)
H = matrix(c(0.0, 0.0, 0.0, 1.0), nrow = 2)

data <- list(T = len,state_dim =  as.integer(2),obs_dim = as.integer(1), l = as.integer(2)
             ,x_init = x_init,Q_init =  Q_init,z = z,W = W,V = V,H = H)

#sampling @stan
fit <- stan(file='./model.stan',data = data)

summary(fit)$summary

#posterior plot
stan_hist(fit, pars = c('a[1]', 'a[2]'))

x_pred_chain_1 <- summary(fit)$summary[seq(10, 208,2),1]
df <- data.frame(year = seq(1871, 1970, 1), estimated = x_pred_chain_1, origin_data = Nile[1:100])
df
ggplot <- ggplot(data = df, aes(year)) 
ggplot <- ggplot + geom_line(aes(y =  origin_data), colour='#000099')
ggplot <- ggplot + geom_line(aes(y =  estimated), colour='#D55E00')
plot(ggplot)