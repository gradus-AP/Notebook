#'description
#'Fitting state space model to data UKgas.
#'Model
#'We consider a decomposition of into dwo sequences 
#'d_t log(origin_sequence)= trend[t] + season[t] + err[t], err[t] ~ i.i.d. Normal(0, sigma ^2)
#',where d_t denotes difference.
setwd("C:\\Users\\Notebook\\Seasonal_Trend\\src")
library(rstan)
library(ggplot2)

z <- UKgas
len = length(z)
summary(z)

#plot
df <- data.frame(time = seq(0, len -1, 1), val = z[1:len])
ggplot <- ggplot(data = df,mapping = aes(x = time)) + geom_line(aes(y = val), colour='#000099')
plot(ggplot)

#acf plot
z_acf <- acf(z, plot = FALSE)
df <- data.frame(lag = z_acf$lag, acf = z_acf$acf)
ggplot <- ggplot(data = df,mapping = aes(x = lag, y =  acf)) + geom_hline(aes(yintercept = 0), colour='#000099')
ggplot <- ggplot + geom_segment(mapping = aes(xend = lag, yend = 0))
plot(ggplot)
#high auto correlationwith lag = 4


#log
z_log <- matrix(log(z), nrow = len)

#check
plot(z_log)
plot(filter(z_log, c(1, 1, 1, 1))/4)
acf(z_log)

W = rbind(c(0.1, 0.0), c(0.0, 0.1))
V = matrix(c(0.1), nrow = 1)
H = matrix(c(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), nrow = 5)

data <- list(T = len ,state_dim =  as.integer(5),obs_dim = as.integer(1), l = as.integer(2)
             ,z = z_log, W = W,V = V,H = H)

#sampling @stan
fit <- stan(file='./model.stan',data = data)

summary(fit)$summary

#posterior plot
stan_hist(fit, pars = c('x_init', 'tau_trend', 'tau_rand', 'tau_season'))

ind_of_trend_1 = 64
ind_of_rand_1 = 65
ind_of_seasonal_trend_1 = 68

trend <- summary(fit)$summary[seq(ind_of_trend_1, ind_of_trend_1 + 5 * len -1,5),]
random_walk <- summary(fit)$summary[seq(ind_of_rand_1, ind_of_rand_1 + 5 * len -1,5),]
seasonal_trend <- summary(fit)$summary[seq(ind_of_seasonal_trend_1, ind_of_seasonal_trend_1 + 5 * len -1,5),]

df <- data.frame(time = seq(1, len , 1), trend = trend[,'mean'], random_walk = random_walk[,'mean'], seasonal_trend = seasonal_trend[,'mean'])
df
ggplot <- ggplot(data = df, aes(time))
ggplot <- ggplot + geom_line(aes(y =  trend), colour='#000099')
ggplot <- ggplot + geom_line(aes(y =  random_walk), colour='#339900')
ggplot <- ggplot + geom_line(aes(y =  seasonal_trend), colour='#D55E00')
plot(ggplot)

#estimate vs log(origin sequence)
estimate <- trend[,'mean'] + random_walk[,'mean'] + seasonal_trend[,'mean']
df <- data.frame(time = seq(1, len, 1), estimate = estimate, origin_seq = z_log)

ggplot <- ggplot(data = df, aes(time)) 
ggplot <- ggplot + geom_line(aes(y =  estimate), colour='#000099')
ggplot <- ggplot + geom_line(aes(y =  origin_seq), colour='#D55E00')
plot(ggplot)