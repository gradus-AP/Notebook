import numpy as np
class LinearRegression:
    """
    The class for Linear Regression 
    Z | beta_1, beta_2, gamma ~ N(beta_1 + beta_2 * X, gamma^{-1})
    beta_1 ~ N(mu1, tau_1^{-1})
    beta_2 ~ N(mu2, tau_2^{-1})
    gamma ~ Gamma(a, b) 
    @params
    ----------------------------------------------
    mu_1, tau_1(>0), mu_2, tau_2(>0), a(>0), b(>0)  
    """
    def __init__(self, mu_1, tau_1, mu_2, tau_2, a, b):
        self.mu_1 = mu_1
        self.tau_1 = tau_1 
        self.mu_2 = mu_2
        self.tau_2 = tau_2
        self.a = a
        self.b = b
        self.post = self.Post()
    
    class Post:
        def __init__(self):
            self.mu_1 = 0.0
            self.tau_1 = 0.0
            self.mu_2 = 0.0
            self.tau_2 = 0.0
            self.a = 0.0
            self.b = 0.0

    def _fit(self, Z, X, mu_1, tau_1, mu_2, tau_2,iteration):
        """
        Estimate parameters of posterior distribution
        using variational inferrence.
        @params
        ------------------------------------------------
        Z : numpy.array([d])               
        X : numpy.array([d])    
        mu_1, tau_1, mu_2, tau_2 : initial value     
        iteration : int > 0
        """
        #initialize 
        self.post.mu_1 = mu_1
        self.post.tau_1 = tau_1
        self.post.mu_2 = mu_2
        self.post.tau_2 = tau_2

        d = len(Z)
        Z_bar = np.sum(Z)
        X_bar = np.sum(X)
        Z_dot_X = np.dot(Z, X)
        X_norm = np.dot(X, X)
        for count in range(iteration):
            residue = Z - self.post.mu_1 * np.ones(d) - self.post.mu_2 * X
            residue_norm = np.dot(residue, residue)
            self.post.a = self.a + 0.5 * d
            self.post.b = self.b + 0.5 * (residue_norm + d / self.post.tau_1 + X_norm / self.post.tau_2)

            self.post.tau_1 = d * self.post.a / self.post.b + self.tau_1
            self.post.mu_1 = ((self.post.a / self.post.b ) * (Z_bar - self.post.mu_2 * X_bar) + self.tau_1 * self.mu_1 ) / self.post.tau_1

            self.post.tau_2 = X_norm * self.post.a / self.post.b + self.tau_2
            self.post.mu_2 = ((self.post.a / self.post.b ) * (Z_dot_X - self.post.mu_1 * X_bar) + self.tau_2 * self.mu_2 ) / self.post.tau_2