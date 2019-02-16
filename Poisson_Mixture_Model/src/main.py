import numpy as np
import  matplotlib.pyplot as plt
from numpy.random import poisson as pois_sampler
from Poisson_Mixture_Model import Poisson_Mixture_model
#@memo generate test data
# cluster_num = 2 , data_num =1000
# s[300] ~ Poisson(10)
# t[700] ~ Poisson(5)
s = pois_sampler(10, 300)
t = pois_sampler(5, 700)
data = np.hstack((s,t))
data_num = len(data)

pm_model = Poisson_Mixture_model(2,np.array([0.5, 0.5]), np.array([2.0, 3.0]))
pm_model._fit(1000, data, iteration = 10)

print("mixture_rate:",pm_model.alpha)
print("poisson_parameter", pm_model.theta)    

#かきかき
x = np.linspace(0, 9, 10)
y = pm_model._log_likelihood

plt.plot(x, y, label = "log_likelihood")
plt.title("log likelihood")
plt.show()