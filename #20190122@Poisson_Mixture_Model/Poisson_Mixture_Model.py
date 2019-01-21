import pandas as pd
import numpy as np
from numpy.random import poisson as pois_sampler
from scipy.stats import poisson as pois
import  matplotlib.pyplot as plt

class Poisson_Mixture_model:
    '''
    Poisson_Mixture_model(self, cluster_num, alpha, theta):
    cluster_num : @type integer 
    alpha:mixture rate @type np.array(cluster_num)
    theta:poisson parameter @type np.array(cluster_num)
    '''
    
    _log_likelihood = []
    log_likelihood = 0
    def __init__(self, cluster_num, alpha, theta):
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.theta = theta
    
    def _fit(self, data_num, data, trial_num):
        '''
        run EM-algorithm [trial_num] times
        data:observed values @type list[data_num] 
        '''
        #initial value 
        Z = np.array([[0  for j in range(self.cluster_num)] for i in range(data_num)])
        W = np.array([[0  for j in range(self.cluster_num)] for i in range(data_num)])
       
        normalize =np.array( [0 for i in range(data_num)] )
        eta = np.array([0 for i in range(data_num)])
        #run EM-algorithm
        for step in range(trial_num):
            #E_step_update
            W = np.matmul(np.array([pois.pmf(data, theta_) for theta_ in  self.theta]).T, np.diag(self.alpha))
            normalize = np.exp(- np.log(np.sum(W, axis = 1)))

            Z =  np.matmul( np.diag(normalize), W)
            '''
            log_likelihood
            '''
            log_likelihood = np.sum(np.log(np.sum(W, axis = 1)))
            (self._log_likelihood).append(log_likelihood)
            #M_step_update  
            self.alpha =  np.sum(Z, axis=0) / data_num
           
            eta = np.exp(- np.log(np.sum(Z, axis = 0)))
            
            self.theta = (np.matmul(Z, np.diag(eta)).T).dot(data)
#@memo generate test data
# cluster_num = 2 , data_num =1000
# s[300] ~ Poisson(10)
# t[700] ~ Poisson(5)
s = pois_sampler(10, 300)
t = pois_sampler(5, 700)
data = np.hstack((s,t))
data_num = len(data)

pm_model = Poisson_Mixture_model(2,np.array([0.5, 0.5]), np.array([2.0, 3.0]))
pm_model._fit(1000, data, 500)

print("mixture_rate:",pm_model.alpha)
print("poisson_parameter", pm_model.theta)    

#かきかき
x = np.linspace(0, 499, 500)
y = pm_model._log_likelihood

plt.plot(x, y, label = "log_likelihood")
plt.show()