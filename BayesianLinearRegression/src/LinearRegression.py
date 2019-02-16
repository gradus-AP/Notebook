import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt 
class LinearRegression:
    """
    A class for bayesian linear regression.
    @model
    --------------------------------------------
    z : response variable  
    x : predictor variable  
    z | x , alpha, beta, tau ~ Normal(alpha + beta * x^T, tau^{-1})    
    alpha ~ Normal(mu, theta^{-1})   
    beta ~ Normal(beta , Phi^{-1})    
    tau ~ Gamma(a, b)   
    @params
    --------------------------------------------
    * mu , theta : float  
        parameter of the bias alpha
    * beta : np.array(), Phi : numpy.ndarray()
        parameter of the coefficients   
    * a, b: float > 0
        parameter of the precision tau
    """
    def __init__(self, L, mu, theta, beta, Phi, a, b):
        """
        A method for generating instance
        @params
        ---------------------------------------- 
        L : int 
        dimension of predictor variable
        """
        self.L = L
        self.prior = self.HyperParams(mu, theta, beta, Phi, a, b) 
        self.post = self.HyperParams(0.0 , 1.0, np.array([0.0 for i in range(L)]), np.identity(L), 1.0,1.0)

    class HyperParams:
        """
        A class for describing parameters of post destribution.
        """
        def __init__(self, mu, theta, beta, Phi, a, b):
            self.mu = mu
            self.theta = theta
            self.beta = beta
            self.Phi = Phi
            self.a = a
            self.b = b

        def getExpectedPrecision(self):
            return self.a / self.b

    def plotHyperParams(self):
        ax = np.linspace(-10.0, 10.0, 2000)
        ax_positive = np.linspace(0, 10, 2000)

        plt.plot(ax, stats.norm.pdf(ax,self.post.mu, 1 / self.post.theta), label=r'$\alpha$')   
        for i in range(self.L):
            plt.plot(ax, stats.norm.pdf(ax, self.post.beta[i], 1.0), label=r'$\beta$['+str(i)+']')
                
        plt.ylim([0, 4.0])

        plt.legend()
        plt.title("The posterior distributions")
        plt.show()
    def _fit(self, Z, X, mu, theta, a, b, iteration):
        """
        @params
        ------------------------------------------------
        * Z : numpy.array()
            response data sample
        * X : numpy.array([[]]) 
            predictor data sample
        * mu, theta :float  
            initial values for estimator of parameters of a bias  
        * a, b : float  
            initial values for estimiator of parameters of a precision    
        * iteration : int
        """
        d = len(Z)
        self.post.mu = mu
        self.post.theta = theta
        self.post.a = a
        self.post.b = b

        ones = np.ones(d)
        Z_tot = np.sum(Z)
        X_tot = np.sum(X, axis = 0)
        Z_dot_X = np.dot(Z, X)
        H = np.dot(X.T, X)

        for count in range(iteration):
            #update@parameter of precision
            error = Z - self.post.mu * ones - np.dot(self.post.beta, X.T)
            ess = np.dot(error, error) 
            self.post.a = self.prior.a + d / 2
            self.post.b = self.prior.b + ess + d / self.post.theta \
                + np.trace(np.dot(X, np.dot(np.linalg.inv(self.post.Phi), X.T)))

            #update@parameter of bias 
            theta = d * self.post.getExpectedPrecision() + self.prior.theta
            self.post.theta = theta
            self.post.mu = (self.prior.theta * self.prior.mu \
                + self.post.getExpectedPrecision() *  (Z_tot - np.dot(self.post.beta, X_tot.T))) / theta

            #update@parameter of coefficients
            Q = self.prior.Phi + self.post.getExpectedPrecision() * H
            self.post.Phi = Q
            self.post.beta = np.dot((np.dot(self.prior.beta, self.prior.Phi) + \
                + self.post.getExpectedPrecision() * (Z_dot_X - self.post.mu * X_tot)), np.linalg.inv(Q))