import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MDA:
    
    """
    Supervised Mixture Discriminant Analysis Model. The Model creates a 
    Gaussian Mixture Model for each class in the classification problem.
    Classification probabilities are then computed for each Gaussian
    Mixture is defining how likely each is to contain the data points.
    
    Parameters
    ----------
    K_list: array of ints
        The number of Gaussian components/clusters in each Mixture.
    
    tol: float, defaults to 0.0001.
        The tollerance limit of when the loglikelihood has converged.
        
    max_iter: int, defaults to 200.
        The max number of iterations of the EM-Algorithm used in the
        Gaussian Mixture Model. This limit overrules the tollerance.
        
    init_max_iter: int, defaults to 10.
        The max number of iterations for the KMeans initialization
        of the mu parameters of the Gaussian Mixtures.
    
    Attributes
    ----------
    mus: array-like, shape (n_classes, n_components, n_features)
        The mean of each mixture component.
    
    covariances: array-like, shape (n_classes, n_components, n_features)
        The covariance of each mixture component.
    
    priors: array-like, shape (n_classes, n_components)
        The Mixing Coefficient/Weighting of each component.
    
    rho: array-like, shape (n_classes)
        The class weight of each class in the MDA Model.
        
    ll_list: array-like, shape (n_classes, n_iterations)
        Loglikelihood history used to check for convergance
        and to plot parameter training.
    """

    def __init__(self, K_list, tol = 0.0001, max_iter = 200, init_max_iter = 10):
        self.K_list = K_list
        self.tol = tol
        self.max_iter = max_iter
        self.init_max_iter = init_max_iter
        self.mus = []
        self.covariances = []
        self.priors = []
        self.rho = []
        self.ll_list = []

    def _KMeans(self, X, K, init_max_iter = 10):
        centroids = X[np.random.choice(np.arange(len(X)), K), :]
        for i in range(init_max_iter):
            C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        return np.array(centroids)

    def _multivariate_gaussian(self, x, mu, covariance):
        return 1/np.sqrt(((2*np.pi)**2) * np.linalg.det(covariance)) * np.exp((-0.5) * ((x-mu).T.dot(np.linalg.inv(covariance))).dot((x-mu)))
    
    def _re_estimate_posterior(self, X, mu, covariance, prior, K):
        gamma = np.zeros((len(X), K))
        gamma_final = np.zeros((len(X), K))
        for n in range(len(X)):
            for k in range(K):
                gamma[n,k] = prior[k] * self._multivariate_gaussian(X[n], mu[k], covariance[k])
            gamma_final[n,:] = gamma[n,:] / gamma[n,:].sum()
        return gamma_final

    def fit(self, X):

        for c, K in enumerate(self.K_list):
            print('Running Class:', c + 1, ' with K =', K)

            #Initialization of parameters
            self.mus.append(self._KMeans(X[c], K))

            self.covariances.append([np.cov(X[c].T) for i in range(K)])

            self.priors.append([1/K for i in range(K)])
            
            self.rho.append(len(X[c]))

            ll_old = 0
            self.ll_list.append([])
            for step in range(self.max_iter):

                if step % 10 == 0:
                    print('   Currently at step:', step, "Loglikelihood:", ll_old)

                #Evaluation step
                posterior = self._re_estimate_posterior(X[c], self.mus[c], self.covariances[c], self.priors[c], K)

                #M-Step Re-Estimations
                N_k = posterior.sum(axis=0)

                for k in range(K):
                    for i in range(X[c].shape[1]):
                        self.mus[c][k][i] = (posterior[:,k] * X[c].T[i]).sum() / N_k[k]

                    covariance_sum = 0
                    for n in range(len(X[c])):
                        covariance_sum += (1 / N_k[k]) * posterior[n,k] * np.outer((X[c][n] - self.mus[c][k]), np.transpose((X[c][n] - self.mus[c][k])))
                    self.covariances[c][k] = covariance_sum

                self.priors[c] = N_k / len(X[c])

                #Loglikelihood computation
                ll_new = 0
                for n in range(len(X[c])):
                    s = 0
                    for k in range(K): 
                        s += self.priors[c][k] * self._multivariate_gaussian(X[c][n], self.mus[c][k], self.covariances[c][k])
                    
                    ll_new += np.log(s)
                if np.abs(ll_new - ll_old) < self.tol:
                    break

                ll_old = ll_new
                self.ll_list[c].append(ll_old)
                
        self.rho = np.divide(self.rho, sum(self.rho))
        
    def predict_proba(self, X):
        point_probs = []
        for n in range(len(X)):
            
            n_probs = []
            for c in range(len(self.rho)):
                
                n_c_probs = 0
                for k in range(self.K_list[c]):
                    n_c_probs += self.priors[c][k] * self._multivariate_gaussian(X[n], self.mus[c][k], self.covariances[c][k])
                
                n_probs.append(n_c_probs)
            point_probs.append(n_probs)
            
        target_probs = []
        for c in range(len(self.rho)):
            
            point_target_probs = []
            for p_n in point_probs:
                point_target_probs.append(self.rho[c] * p_n[c])
            
            target_probs.append(point_target_probs)
            
        return (target_probs / np.sum(target_probs, axis = 0))
    
    def predict(self, X):
        class_probs = self.predict_proba(X)
        predictions = []
        for n in range(len(X)):
            predictions.append(np.argmax(class_probs[:,n]) + 1)
        return np.array(predictions)
    
    def plot_gaussian(self, pt, X, c):
        delta = 0.025
        x = np.arange(-3, 3, delta)
        y = np.arange(-3, 3, delta)
        X, Y = np.meshgrid(x, y)
        
        if pt == 1:
            for k in range(self.K_list[c]):
                Z1 = self._multivariate_gaussian((X,Y), self.mus[c][k], self.covariances[c][k])
                plt.contour(x, y, Z1, linewidths=0.5)

        for k in range(self.K_list[c]):
            plt.plot(self.mus[c][k][0], self.mus[c][k][1], '+', markersize=13, mew=3)

        k = 0
        f = np.vectorize(lambda x1, y1: self.priors[c][k] * self._multivariate_gaussian((x1,y1), self.mus[c][k], self.covariances[c][k]))
        Z2 = f(X, Y)
        for k in range(1, self.K_list[c]):
            f = np.vectorize(lambda x1, y1: self.priors[c][k] * self._multivariate_gaussian((x1,y1), self.mus[c][k], self.covariances[c][k]))        
            Z2 = Z2 + f(X,Y)
        plt.contour(x, y, Z2)