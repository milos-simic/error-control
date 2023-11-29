from decimal import getcontext, Decimal, ROUND_DOWN

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import sklearn.base
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from sklearn.utils import check_array, check_random_state
import scipy.stats as st
from scipy.spatial.distance import cdist
import scipy.optimize

#from .evaluate import empirical_rates, empirical_rates_for_p_values

def split_the_data(X, y, target_class, size, random_state=None):
    X_target = X[y == target_class]
    X_nontarget = X[y == 1 - target_class]
    y_target = y[y == target_class]
    y_nontarget = y[y == 1 - target_class]
        
    X_target_1, X_target_2, y_target_1, y_target_2 = train_test_split(
        X_target, y_target, test_size=size, random_state=random_state)

    if isinstance(X, pd.DataFrame):
        X_fit = pd.concat([X_target_1, X_nontarget], ignore_index=True)
    else:
        X_fit = np.concatenate([X_target_1, X_nontarget])
        
    y_fit = np.concatenate([y_target_1, y_nontarget])
    
    return X_fit, X_target_2, y_fit, y_target_2

class ClassificationTest(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, classifier, scorer, alpha=0.05, reserve=101,
                 sample_size=1, target_class=1, augmentor=None, ci=None,
                replace=False, random_state=None):
        super(ClassificationTest, self).__init__()
        self.classifier = classifier
        self.scorer = scorer
        self.alpha = alpha
        self.target_class = target_class
        self.reserve = reserve
        self.sample_size = sample_size
        self.augmentor = augmentor
        self.ci = ci
        self.replace = replace
        self.random_state = random_state

    def describe(self):
        return {'method' : 'SCT'}
        
    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        np.random.seed(self.random_state)

        if self.reserve > 1:
            ratio = self.reserve / (y == self.target_class).sum()
        else:
            ratio = self.reserve
        self.ratio = ratio

        split_data = split_the_data(X, y, self.target_class, ratio, self.random_state_)
        X_fit, X_calc, y_fit, _ = split_data

        
        if isinstance(self.classifier, Pipeline):
            self.classifier.steps[-1][1].set_params(random_state=self.random_state_)
        else:
            self.classifier.set_params(random_state=self.random_state_)

        self.classifier.fit(X_fit, y_fit)
                
        baseline_scores = self.scorer(self.classifier, X_calc)

        if self.augmentor is not None:
            baseline_scores = self.augmentor.augment(baseline_scores)

        self.baseline_scores = np.sort(baseline_scores).squeeze()
        
        #self._determine_sample_size()
        
        return self 
    
    def _determine_sample_size(self):
        #print(self.sample_size, self.reserve)
        if self.sample_size < 1:
            n = len(self.baseline_scores)
            s = self.sample_size  * n
            self.sample_size = min(int(s) + 1, n)
        elif self.sample_size == 1 or self.sample_size == self.reserve:
            self.sample_size = self.reserve
        elif self.sample_size < self.reserve:
            self.sample_size = self.sample_size
            # reserve < sample_size
        else:
            raise ValueError('Invalid sample size')
    
    def transform(self, X):
        return self.p_values(X)
    
    def p_values(self, X):
        X_scores = self.scorer(self.classifier, X)
        n_objects = X.shape[0]
        #if self.ci is not None:
            #z = st.norm.ppf(1 - (1 - self.ci) / 2)
    
        if True: # always do this
            # use all baseline scores to calculate the p values
            total = len(self.baseline_scores)
            
            if self.target_class == 1:
                indices = np.searchsorted(self.baseline_scores, X_scores, side='right')
                more_extreme = indices
            else:
                indices = np.searchsorted(self.baseline_scores, X_scores, side='left')
                more_extreme = total - indices + 1

            if self.ci is None:
                return (more_extreme + 1) / (total + 1) # +1
            else:
                p = more_extreme / total
                #return more_extreme / total + z * p * (1 - p) / (total)
                intervals = proportion_confint(more_extreme, total, 1-self.ci, 'jeffreys')
                return intervals[1]

        else:
            # use a sample of the baseline scores
            sample_size = self.sample_size

            calculation_scores = np.zeros((n_objects, sample_size),
                                              dtype=self.baseline_scores.dtype)
            for i in range(n_objects):
                sample_scores = self.random_state_.choice(self.baseline_scores, 
                    replace=self.replace, size=sample_size)
                sample_scores = np.sort(sample_scores).squeeze()
                calculation_scores[i, :] = sample_scores
                
            total = calculation_scores.shape[0]
        
            if self.target_class == 1:
                indices = [np.searchsorted(calculation_scores[i, :],
                                       X_scores[i], side='right') for i in range(n_objects)]
                more_extreme = np.array(indices)
            else:
                indices = [np.searchsorted(calculation_scores[i, :],
                                       X_scores[i], side='left') for i in range(n_objects)]
                more_extreme = total_count - np.array(indices) + 1

            if self.ci is None:
                return more_extreme / total
            else:
                intervals = proportion_confint(more_extreme, total, 1-self.ci, 'jeffreys')
                return intervals[1]
    
    def predict(self, X):
        p_values = self.p_values(X)
        predictions = np.zeros_like(p_values)
        predictions[p_values >= self.alpha] = self.target_class
        predictions[p_values < self.alpha] = 1 - self.target_class
        
        return predictions
    
    def set_params(self, **params):
        super().set_params(**params)

        if 'sample_size' in params:
            self._determine_sample_size()
            
    def decision_function(self, X):
        return self.p_values(X)
    
    def p_value(self, x):
        if isinstance(x, list):
            n = len(x)
            x = x[np.newaxis, :]
        elif isinstance(x, pd.Series):
            x = x.to_frame().T

        result = self.p_values(x)
        
        return result[0]

def get_threshold_index(n, alpha, delta):
    #print(n, alpha, delta)
    alpha = Decimal(alpha)
    delta = Decimal(delta)

    #print(n, alpha, delta)
    violation_rates = [Decimal(1) for k in range(0, n + 1)]
    violation_rates[n] = (Decimal(1) - alpha) ** n
    binom_coeff = Decimal(1)
    a1 = (Decimal(1) - alpha) ** n
    a2 = Decimal(1)

    k = n - 1
    while k > 0:
        a1 = a1 / (Decimal(1) - alpha)
        a2 = a2 * alpha
        binom_coeff = (Decimal(k + 1) * binom_coeff) / (n - k)
        violation_rates[k] = violation_rates[k + 1] + a1 * a2 * binom_coeff
        
        k = k - 1
    
    candidates = [k for k in range(1, n + 1) if violation_rates[k] <= delta]
    
    if len(candidates) == 0:
        quantized_delta = delta.quantize(Decimal('.001'), rounding=ROUND_DOWN)
        quantized_alpha = alpha.quantize(Decimal('.001'), rounding=ROUND_DOWN)
        raise NoIndexException(
            f'''n={n} is too small for delta={quantized_delta} and alpha={quantized_alpha}''', 
            [n, delta, alpha]
        )
    else:
        return min(candidates) - 1

class NoIndexException(ValueError):
    """n is too small for the current settings of delta and ensemble_size"""
    def __init__(self, message, foo, *args):
        self.message = message
        super(NoIndexException, self).__init__(message, foo, args)

#def get_threshold_index(n, alpha, delta):
#    print(n, alpha, delta)
#    violation_rates = [1 for k in range(0, n + 1)]
#    violation_rates[n] = (1 - alpha)**n
#    binom_coeff = 1
#    a1 = (1 - alpha)**n
#    a2 = 1

#    k = n - 1
#    while k > 0:
#        a1 = a1 / (1 - alpha)
#        a2 = a2 * alpha
#        binom_coeff = ((k + 1) * binom_coeff) / (n-k)
#        violation_rates[k] = violation_rates[k + 1] + a1 * a2 * binom_coeff
        
#        k = k - 1
    
#    candidates = [k for k in range(1, n+1) if violation_rates[k] <= delta]
    
#    if len(candidates) == 0:
#        raise ValueError('No index')
#    else:
#        return min(candidates) - 1
    
class Umbrella(BaseEstimator, TransformerMixin):
    def __init__(self, classifier, scorer, target_class=1, alpha=0.05, delta=0.05,
                 ensemble_size=1, precision=28, thresholds_size=0.5, random_state=None):
        super(Umbrella, self).__init__()
        self.classifier = classifier
        self.scorer = scorer
        self.target_class = target_class
        self.alpha = alpha
        self.delta = delta
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        self.thresholds = []
        self.ensemble = []
        self.thresholds_size = thresholds_size
        self.precision = precision

    def describe(self):
        return {
            'method' : 'UA', 
            'delta' : self.delta,
            'ensemble_size' : self.ensemble_size
        }
    
    def fit(self, X, y):
        getcontext().prec = self.precision
        self.random_state_ = check_random_state(self.random_state)
        if self.target_class == 1:
            y = 1 - np.copy(y)
        
        self.ensemble = []
        self.thresholds = []
        for i in range(self.ensemble_size):
            classifier, thresholds = self.fit_one_classifier(X, y)
            
            n = thresholds.shape[0]

            threshold_index = get_threshold_index(n, self.alpha, self.delta)
            threshold = thresholds[threshold_index]
            
            self.ensemble.append({'classifier': classifier, 'threshold': threshold})
            self.thresholds.append(thresholds)
        
        return self
    
    def set_params(self, **params):
        super().set_params(**params)

        if 'alpha' in params or 'delta' in params:
            for i in range(self.ensemble_size):
                thresholds = self.thresholds[i]
                n = thresholds.shape[0]
                threshold_index = get_threshold_index(n, self.alpha, self.delta)
                threshold = thresholds[threshold_index]
                self.ensemble[i]['threshold'] = threshold
            
    def transform(self, X):
        return self.mean_votes(X)
            
    def mean_votes(self, X):
        n_objects = X.shape[0]
        votes = np.zeros(shape = (self.ensemble_size, n_objects))
        
        for i in range(self.ensemble_size):
            classifier_dict = self.ensemble[i]
            classifier = classifier_dict['classifier']
            threshold = classifier_dict['threshold']
            
            scores = self.scorer(classifier, X)

            leq_than_threshold = scores <= threshold
            
            if self.target_class == 0:
                votes[i, leq_than_threshold] = 0
                votes[i, ~leq_than_threshold] = 1
            else:
                votes[i, leq_than_threshold] = 1
                votes[i, ~leq_than_threshold] = 0
        
        mean_votes = np.mean(votes, axis=0)
        
        return mean_votes
    
    def decision_function(self, X):
        return self.mean_votes(X)
    
    def predict(self, X):
        mean_votes = self.mean_votes(X)
        y_pred = np.rint(mean_votes)
        return y_pred
    
    def fit_one_classifier(self, X, y):
        # private
        split_data = split_the_data(X, y, 0, self.thresholds_size, self.random_state_)
        X_fit, X_0, y_fit, _ = split_data
        
        classifier = sklearn.base.clone(self.classifier)
        if isinstance(classifier, Pipeline):
            classifier.steps[-1][1].set_params(random_state=self.random_state_)
        else:
            classifier.set_params(random_state=self.random_state_)
        classifier.fit(X_fit, y_fit)
        
        thresholds = self.scorer(classifier, X_0)
        thresholds = np.sort(thresholds)
        
        return (classifier, thresholds)
    
class Typicality(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, density_estimator,alpha=0.05, target_class=1, random_state=None):
        super(Typicality, self).__init__()
        self.density_estimator = density_estimator
        self.alpha = alpha
        self.target_class = target_class
        self.random_state = random_state

    def describe(self):
        return {'method' : 'TI'}
        
    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        X_target = X[y == self.target_class]
        self.baseline_scores = None
        
        #if isinstance(self.density_estimator, Pipeline):
        #    self.density_estimator.steps[-1][1].set_params(random_state=self.random_state_)
        #else:
        #    self.density_estimator.set_params(random_state=self.random_state_)

        self.density_estimator = self.density_estimator.fit(X_target)
        
        self.baseline_scores = self.density_estimator.score_samples(X_target)
        self.baseline_scores = np.sort(self.baseline_scores).squeeze()

        return self 
    
    def transform(self, X):
        return self.typicality_indices(X)
    
    def typicality_indices(self, X):
        X_scores = self.density_estimator.score_samples(X)
    
        total_count = self.baseline_scores.shape[0]
            
        lower_density_count = np.searchsorted(self.baseline_scores, X_scores, side='right')

        return lower_density_count / total_count
    
    def predict(self, X):
        typicality_indices = self.typicality_indices(X)
        
        predictions = np.zeros_like(typicality_indices)
        predictions[typicality_indices > self.alpha] = self.target_class
        predictions[typicality_indices <= self.alpha] = 1 - self.target_class
        
        return predictions
    
    def set_params(self, **params):
        super().set_params(**params)
            
    def decision_function(self, X):
        return self.typicality_indices(X)
    
    def typicality_index(self, x):
        if isinstance(x, list):
            n = len(x)
            x = x[np.newaxis, :]
        elif isinstance(x, pd.Series):
            x = x.to_frame().T

        result = self.typicality_indices(x)
        
        return result[0]
    
class DirectNP(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, ratio_estimator, alpha=0.05, target_class=1,
                 threshold_subset_size=101, random_state=None):
        super(DirectNP, self).__init__()
        self.ratio_estimator = ratio_estimator
        self.alpha = alpha
        self.target_class = target_class
        self.threshold_subset_size = threshold_subset_size
        self.random_state = random_state

    def describe(self):
        return {'method' : 'DNP'}
        
    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        self.baseline_scores = None
        X_target = X[y == self.target_class]
        X_nontarget = X[y == 1 - self.target_class]
        y_target = y[y == self.target_class]
        y_nontarget = y[y == 1 - self.target_class]
        
        if self.threshold_subset_size < 1:
            size = self.threshold_subset_size
        else:
            size = self.threshold_subset_size / X_target.shape[0]
            if size > 1:
                raise ValueError('Not enough data. Check the size parameter.')
        
        split = train_test_split(X_target, y_target, test_size=size,
            random_state=self.random_state_)
        X_target_fit, X_threshold, y_target_fit, _ = split

        X_fit = np.concatenate([X_nontarget, X_target_fit])
        y_fit = np.concatenate([y_nontarget, y_target_fit])

        if isinstance(self.ratio_estimator, Pipeline):
            self.ratio_estimator.steps[-1][1].set_params(random_state=self.random_state_)
        else:
            self.ratio_estimator.set_params(random_state=self.random_state_)
        
        self.ratio_estimator = self.ratio_estimator.fit(X_fit, y_fit)
        
        self.ratio_scores = self.ratio_estimator.score_samples(X_threshold)
        self.threshold = np.quantile(self.ratio_scores, self.alpha)

        return self 
    
    def transform(self, X):
        return self.score_samples(X)
    
    def decision_function(self, X):
        return self.ratio_estimator.score_samples(X)
    
    def predict(self, X):
        scores = self.decision_function(X)
        
        predictions = np.zeros_like(scores)
        predictions[scores > self.threshold] = self.target_class
        predictions[scores <= self.threshold] = 1 - self.target_class
        
        return predictions
    
    def set_params(self, **params):
        super().set_params(**params)

        if 'alpha' in params:
            self.threshold = np.quantile(self.ratio_scores, self.alpha)
            
    def sample_ratio(self, x):
        if isinstance(x, list):
            n = len(x)
            x = x[np.newaxis, :]
        elif isinstance(x, pd.Series):
            x = x.to_frame().T

        result = self.decision_function(x)
        
        return result[0]


# TBC

class TBC(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, alpha=0.05, target_class=1,
                 k = 2, distance = 'cityblock',
                 test = 'mwu', weights='reciprocal',
                 random_state=None):
        super(TBC, self).__init__()
        self.alpha = alpha
        self.target_class = target_class
        self.random_state = random_state
        self.k = k
        self.distance = distance
        self.weights = weights
        self.test = test

    def describe(self):
        if self.weights != 'none':
            method = 'WTBC'
        else:
            method = 'TBC'
        return {
            'method' : method,
            'k' : self.k,
            'distance' : self.distance,
            'test' : self.test,
            'weights' : self.weights
        }
        
    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)

        self.X_target = X[y == self.target_class]
        self.X_other = X[y == 1 - self.target_class]

        if self.test == 'mwu':
            self.two_sample_test = st.mannwhitneyu
        elif self.test == 'ttest_rel':
            self.two_sample_test = st.ttest_rel
        elif self.test == 'ttest_ind':
            self.two_sample_test = st.ttest_ind

        ranks = np.arange(1, self.k + 1, 1)

        if self.weights == 'reciprocal':
            self.weights_ = [1 / i for i in ranks]
        elif self.weights == 'proportional':
            self.weights_ = ranks.tolist()
        elif self.weights != 'none':
            raise ValueError('Not recognized weights parameter')
        #elif self.weights == 'optimize':
        #    if self.weight_opt_nominal_rates is None:
        #        self.weight_opt_nominal_rates = np.linspace(0.01, 0.99, 99)
        #    self.optimize_weights(X, y)

        return self 

    def optimize_weights(self, X, y):
        pass
        #weights = [1 / i for i in range(1, self.k + 1)]
        #nearest_target, nearest_other = self.nearest(X)
        #def f(weights):
        #    results = self.two_sample_test(
        #        nearest_target * weights,
        #        nearest_other * weights,
        #        alternative='greater', axis=1
        #    )
        #
        #    p_values = results.pvalue
        #
        #    rates = empirical_rates_for_p_values(p_values,
        #        y,
        #        nominal_rates=self.weight_opt_nominal_rates,
        #        target_class=self.target_class
        #    ) 
        #
        #    residuals = np.subtract(rates,
        #        self.weight_opt_nominal_rates
        #    )
        #    value = np.mean(np.square(residuals))
        #
        #    return value
        #
        #def f_opt(weights):
        #    print('* annealing', weights)
        #
        #    value = f(weights)
        #    
        #
        #    #step = 0.05
        #    
        #    #gradient = np.zeros_like(weights)
        #    #for i in range(len(gradient)):
        #    #    new_weights = np.copy(weights)
        #    #    new_weights[i] += step
        #    #    new_value = f(new_weights)
        #    #    gradient[i] = (new_value - value) / step
        #
        #    #print('\tGradient:', gradient, np.sign(gradient))
        #
        #    return value#, gradient
        #
        #opt_results = scipy.optimize.dual_annealing(f_opt,
        #    x0=np.array([1 / self.k for i in range(self.k)]),
        #    bounds=[(0, 1) for i in range(self.k)],
        #    maxiter=100
        #    #jac=True,#'3-step',
        #    #method='Nelder-Mead',
        #    #options = {'maxiter' : 10}
        #)
        #
        #print(opt_results)
        #
        #self.weights_ = opt_results.x
    
    def transform(self, X):
        return self.p_values(X)
    
    def decision_function(self, X):
        return self.p_values(X)
    
    def predict(self, X):
        scores = self.decision_function(X)
        
        predictions = np.zeros_like(scores)
        predictions[scores > self.alpha] = self.target_class
        predictions[scores <= self.alpha] = 1 - self.target_class
        
        return predictions
    
    def set_params(self, **params):
        super().set_params(**params)

    def nearest(self, X):
        distances_target = cdist(X, self.X_target, self.distance)
        distances_other = cdist(X, self.X_other, self.distance)

        nearest_target = np.partition(distances_target, self.k, axis=1)[:, :self.k]
        nearest_other = np.partition(distances_other, self.k, axis=1)[:, :self.k]
        # mannwhitneyu

        nearest_target = np.sort(nearest_target, axis=1)
        nearest_other = np.sort(nearest_other, axis=1)

        return nearest_target, nearest_other
            
    def p_values(self, X):
        distances_target = cdist(X, self.X_target, self.distance)
        distances_other = cdist(X, self.X_other, self.distance)

        nearest_target = np.partition(distances_target, self.k, axis=1)[:, :self.k]
        nearest_other = np.partition(distances_other, self.k, axis=1)[:, :self.k]
        # mannwhitneyu

        nearest_target = np.sort(nearest_target, axis=1)
        nearest_other = np.sort(nearest_other, axis=1)

        #np.save('target_unweighted.p', nearest_target)
        #np.save('other_unweighted.p', nearest_other)

        if self.weights != 'none':
            nearest_target = nearest_target * self.weights_
            nearest_other = nearest_other * self.weights_

        #np.save('target_weighted.p', nearest_target)
        #np.save('other_weighted.p', nearest_other)

        #other_minima = np.min(nearest_other, axis=1, keepdims=True)
        #target_minima = np.min(nearest_target, axis=1, keepdims=True)
        
        #nearest_target_ = np.divide(nearest_target, other_minima)
        #nearest_other_ = np.divide(nearest_other, target_minima)

        results = self.two_sample_test(
            nearest_target,
            nearest_other,
            alternative='greater', axis=1)

        p_values = results.pvalue

        return p_values

class EnsembleTBC(BaseEstimator, TransformerMixin, ClassifierMixin):
    """docstring for EnsembleTBC"""
    def __init__(self, ensemble, alpha, target_class=1, random_state=None):
        super(EnsembleTBC, self).__init__()
        self.ensemble = ensemble
        self.alpha = alpha
        self.target_class = 1
        self.random_state = random_state

    def fit(self, X, y):
        for tbc in self.ensemble:
            tbc.fit(X, y)
    
        return self 
    
    def transform(self, X):
        return self.p_values(X)
    
    def decision_function(self, X):
        return self.p_values(X)
    
    def predict(self, X):
        scores = self.decision_function(X)
        
        predictions = np.zeros_like(scores)
        predictions[scores > self.alpha] = self.target_class
        predictions[scores <= self.alpha] = 1 - self.target_class
        
        return predictions
    
    def set_params(self, **params):
        super().set_params(**params)

        #if 'alpha' in params:
        #    for tbc in self.ensemble:
        #        tbc.set_params(**{'alpha' : params['alpha']})
            
    def p_values(self, X):
        p_values = np.mean(
            [tbc.p_values(X) for tbc in self.ensemble],
            axis=0
        )

        return p_values

        

# fsoILCMC


class ForcedInductiveConformal(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, classifier, scorer, 
        target_class=1, alpha=0.05, reserve=101,
        nonconformity='avgdev', random_state=None):
        super(ForcedInductiveConformal, self).__init__()
        self.classifier = classifier
        self.nonconformity = nonconformity
        self.alpha = alpha
        self.target_class = target_class
        self.random_state = random_state
        self.reserve = reserve
        self.scorer = scorer

    def describe(self):
        return {
            'method' : 'CPF',
            'nonconformity' : self.nonconformity
        }

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        np.random.seed(self.random_state)

        if self.reserve > 1:
            ratio = self.reserve / len(X)
        else:
            ratio = self.reserve


        if isinstance(self.classifier, Pipeline):
            self.classifier.steps[-1][1].set_params(random_state=self.random_state_)
        else:
            self.classifier.set_params(random_state=self.random_state_)

        if self.nonconformity == 'score':
            split_data = train_test_split(X, y,
                test_size=ratio,
                stratify=y.tolist(),
                random_state=self.random_state)
        
            X_fit, X_cal, y_fit, y_cal = split_data
            self.classifier.fit(X_fit, y_fit)
        else:
            X_cal = X
            y_cal = y

        self.compute_frozen_nonconformity(X_cal, y_cal)
        

    def compute_frozen_nonconformity(self, X, y):
        self.X_cal = X
        self.y_cal = y
        mask_target = (y == self.target_class)
        nonconformity_scores = np.zeros_like(y)
        if self.nonconformity == 'score':
            scores = self.scorer(self.classifier, X)
            self.clf_cal_scores = scores
            nonconformity_scores = np.copy(scores)
            nonconformity_scores[y == 1] = np.subtract(1, nonconformity_scores[y == 1])
        elif self.nonconformity == 'avgdev':
            #mean_target = np.mean(scores[mask_target])
            #mean_nontarget = np.mean(scores[~mask_target])
            #nonconformity_scores[mask_target] = np.abs(
            #    np.subtract(nonconformity_scores[mask_target], mean_target)
            #)
            #nonconformity_scores[~mask_target] = np.abs(
            #    np.subtract(nonconformity_scores[~mask_target], mean_nontarget)
            #)
            mean_target = np.mean(X[mask_target], axis=0).reshape(1, -1)
            mean_nontarget = np.mean(X[~mask_target], axis=0).reshape(1, -1)
            #print(X[mask_target])
            target_distances = cdist(X[mask_target], mean_target, metric='cityblock').squeeze()
            nontarget_distances = cdist(X[~mask_target], mean_nontarget, metric='cityblock').squeeze()
            
            nonconformity_scores[mask_target] = target_distances
            nonconformity_scores[~mask_target] = nontarget_distances
        elif self.nonconformity == 'nearest_neighbor':
            D = cdist(X, X, metric='cityblock')
            D[np.arange(D.shape[0]), np.arange(D.shape[0])] = np.inf
            
            nearest_target = D[:, mask_target].min(axis=1)
            #print(X.shape, D.shape, nearest_target.shape, nonconformity_scores.shape)
            nonconformity_scores[mask_target] = nearest_target[mask_target]
            
            nearest_nontarget = D[:, ~mask_target].min(axis=1)
            nonconformity_scores[~mask_target] = nearest_nontarget[~mask_target]
        else:
            raise ValueError(f"Nonconformity is {self.nonconformity} but must be 'score', 'avgdev', or 'nearest_neighbor'.")

        self.ncf_cal_scores = {0 : None, 1 : None}
        
        target_ncf_cal_scores = np.copy(nonconformity_scores[mask_target])
        self.ncf_cal_scores[self.target_class] = np.sort(target_ncf_cal_scores)
        
        nontarget_ncf_cal_scores = np.copy(nonconformity_scores[~mask_target])
        self.ncf_cal_scores[1-self.target_class] = np.sort(nontarget_ncf_cal_scores)

    def predict(self, X):
        p_values = self.p_values(X, self.target_class)
        ####
        predictions = np.zeros(X.shape[0])
        mask = p_values[:, self.target_class] >= self.alpha
        #mask = mask | (p_values[:, self.target_class] > p_values[:, 1 - self.target_class])
        #mask = p_values[:, self.target_class] > p_values[:, 1 - self.target_class]
        predictions[mask] = self.target_class
        predictions[~mask] = 1 - self.target_class
        
        return predictions
    
    def set_params(self, **params):
        super().set_params(**params)
            
    def decision_function(self, X):
        #return self.p_values(X)
        pass

    def p_values(self, X, c):
        if self.nonconformity == 'score':
            if self.target_class == 0:
                target_scores = self.scorer(self.classifier, X)
                nontarget_scores = 1 - target_scores
            else:
                nontarget_scores = self.scorer(self.classifier, X)
                target_scores = 1 - nontarget_scores
        elif self.nonconformity == 'avgdev':
            mask_target = (self.y_cal == self.target_class)
            mean_target = np.mean(self.X_cal[mask_target], axis=0).reshape(1, -1)
            mean_nontarget = np.mean(self.X_cal[~mask_target], axis=0).reshape(1, -1)
            
            target_scores = cdist(X, mean_target, metric='cityblock').squeeze()
            nontarget_scores = cdist(X, mean_nontarget, metric='cityblock').squeeze()
        elif self.nonconformity == 'nearest_neighbor':
            mask_target = (self.y_cal == self.target_class)
            
            D = cdist(X, self.X_cal, metric='cityblock')
            #D[np.arange(D.shape[0]), np.arange(D.shape[0])] = np.inf
            
            target_scores = D[:, mask_target].min(axis=1)
            nontarget_scores = D[:, ~mask_target].min(axis=1)
        else:
            raise ValueError(f"Nonconformity is {self.nonconformity} but must be 'score', 'avgdev', or 'nearest_neighbor'.")

        scores = {}
        scores[self.target_class] = target_scores
        scores[1-self.target_class] = nontarget_scores

        n = X.shape[0]
        pvals = np.zeros((n, 2))

        tmp_ncf = {0: [], 1 : []}

        for i in range(n):
            for c in [0, 1]:
                k = np.searchsorted(self.ncf_cal_scores[c],
                                scores[c][i],
                                side='left')
                k += len([s for s in tmp_ncf[c] if s >= scores[c][i]])

                pval = (k + 1) / (len(self.ncf_cal_scores[c]) + len(tmp_ncf[c]) + 1)
                pvals[i, c] = pval

                if pval >= self.alpha: #
                    tmp_ncf[c].append(scores[c][i])

        return pvals