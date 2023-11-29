from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
import scipy

class KernelDensityRatio(BaseEstimator):
    def __init__(self, target_class=1, c=1, kernel='gaussian', 
            random_state=None, kernel_parameters={}):
        super(KernelDensityRatio, self).__init__()
        self.target_class = target_class
        self.kernel = kernel
        self.kernel_parameters = kernel_parameters
        self.random_state = random_state
        self.c = c
    
    def fit(self, X, y):
        self.X_num = X[y == self.target_class]
        self.X_den = X[y != self.target_class]
        
        n_1 = self.X_num.shape[0]
        n_2 = self.X_den.shape[0]
        
        # Compute the relevant parts of the Gram matrix
        K_11 = pairwise_kernels(self.X_num, metric=self.kernel, **self.kernel_parameters)
        K_12 = pairwise_kernels(self.X_num, self.X_den, metric=self.kernel,
                                **self.kernel_parameters)
        
        ones = np.ones(shape=(n_2, 1))
        I_n_1 = np.identity(n_1)
        coeff = (-1.0 / (n_1 * n_2 * self.c))
        
        b = np.multiply(coeff, np.matmul(K_12, ones))
        A = np.multiply(1.0 / n_1, K_11) + np.multiply(self.c, I_n_1)
        
        if A.shape[1] >= 1000:
            self._fit_numerically(A, b)
        else:
            try:
                self.coefficients = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self._fit_numerically(A, b)
            
        return self
    
    def _fit_numerically(self, A, b):
        def f_opt(x):
            if len(x.shape) == 1:
                x = x[:, np.newaxis]
            
            difference = np.subtract(np.matmul(A, x), b)
            
            difference = difference.squeeze()

            return difference
        
        guess = np.random.rand(A.shape[1], 1)
        
        self.coefficients = scipy.optimize.fsolve(f_opt, guess)
    
    def score_samples(self, X):
        n_1 = self.X_num.shape[0]
        n_2 = self.X_den.shape[0]
        
        # calculate the influence of the nontarget data
        raw_results = pairwise_kernels(X, self.X_den, metric=self.kernel,
                                       **self.kernel_parameters)
        results_per_sample = raw_results.sum(axis=1)
        influence_den = np.multiply(1.0 / (n_2 * self.c), results_per_sample)
        
        # calculate the influence of the target data
        raw_results = pairwise_kernels(X, self.X_num, metric=self.kernel,
                                       **self.kernel_parameters)
        influence_num = np.matmul(raw_results, self.coefficients).squeeze()

        # get raw ratio scores
        ratio_scores = influence_num + influence_den
        
        # make sure the ratio scores are non-negative
        ratio_scores = np.maximum(ratio_scores, 0)
        
        return ratio_scores