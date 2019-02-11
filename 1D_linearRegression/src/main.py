#2019/02/10 
from LinearRegression import LinearRegression as LR 
import numpy as np 
from scipy import stats as st
from matplotlib import pyplot as plt

#generating test data with size d = 10
d = 30

#np.random.seed(123)
X = np.random.normal(loc = 0.0, scale = 10.0, size = d)

#Z[i] = 3.0 + 1.5 * X[i] + epsilon_i
#epsilon_i ~ N(0, 2.0) (i.i.d)
epsilon = np.random.normal(loc = 0, scale = 3.0, size = d)
Z = 3.0 * np.ones(d) + 4.0 * X + epsilon

#model fitting
model = LR(0.0, 0.01, 1.0, 0.01, 0.1, 0.1)
model._fit(Z, X, 4.0, 0.5, 6.0, 0.5, iteration = 10)

mu_1 = model.post.mu_1
sd_1 = np.sqrt( 1.0 / model.post.tau_1)

mu_2 = model.post.mu_2
sd_2 = np.sqrt( 1.0 / model.post.tau_2)
print("beta_1 : mean = ",mu_1, "sd =", sd_1)
print("beta_2 : mean = ", mu_2, "sd = ", sd_2)
print("a : ",model.post.a,"b :", model.post.b)

#plot
plt.xlim(-20, 20)
plt.ylim(-25, 25)
plt.legend("scatter plot")

plt.plot(X, Z, '.')

#regression
lin = np.linspace(-20,20, num = 100)
plt.plot(lin, mu_1 + mu_2 * lin)

plt.show()