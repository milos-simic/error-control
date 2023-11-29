from time import time

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .control import NoIndexException

def empirical_rates(estimator, X, y, target_class,
                     nominal_rates=None, ci_coverage=None):
    if nominal_rates is None:
        nominal_rates = np.linspace(0.01, 0.99, 99) 
        
    n_objects = X.shape[0]
    
    estimates = []
    
    for nominal_rate in nominal_rates:
        try:
            if not isinstance(estimator, Pipeline):
                estimator.set_params(alpha=nominal_rate)
            else:
                estimator.steps[-1][1].set_params(alpha=nominal_rate)
        except NoIndexException as e:
            print(e.message)
            if ci_coverage is not None:
                estimate = (np.nan, np.nan, np.nan, 
                        np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan)
            else:
                estimate = (np.nan, np.nan, np.nan)
            estimates.append(estimate)
            continue

        
        y_pred = estimator.predict(X)
        # Check the accuracy, FPR, and FNR
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        [[TN, FP], [FN, TP]] = cm
        
        FPR = FP / (TN + FP)
        FNR = FN / (TP + FN)
        hits = TN + TP
        accuracy = hits / n_objects
        
        if ci_coverage is not None:
            
            accuracy_lower, accuracy_upper = proportion_confint(hits, n_objects, 1 - ci_coverage, 'beta')
            
            if target_class == 1:
                false_target_rate = FNR
                false_nontarget_rate = FPR
            
                target_lower, target_upper = proportion_confint(FN, FN + TP, 1 - ci_coverage, 'beta')
                nontarget_lower, nontarget_upper = proportion_confint(FP, FP + TN, 1 - ci_coverage, 'beta')
            else:
                false_target_rate = FPR
                false_nontarget_rate = FNR
            
                target_lower, target_upper = proportion_confint(FP, FP + TN, 1 - ci_coverage, 'beta')
                nontarget_lower, nontarget_upper = proportion_confint(FN, FN + TP, 1 - ci_coverage, 'beta')
            
            estimate = (target_lower, false_target_rate, target_upper, 
                        nontarget_lower, false_nontarget_rate, nontarget_upper,
                        accuracy_lower, accuracy, accuracy_upper)
        else:
            if target_class == 1:
                false_target_rate = FNR
                false_nontarget_rate = FPR
            else:
                false_target_rate = FPR
                false_nontarget_rate = FNR
            
            estimate = (false_target_rate, false_nontarget_rate, accuracy)
            
        estimates.append(estimate)
    
    if ci_coverage is not None:
        columns = ['target_lower', 'target_estimate', 'target_upper',
                  'nontarget_lower', 'nontarget_estimate', 'nontarget_upper',
                  'accuracy_lower', 'accuracy_estimate', 'accuracy_upper']
    else:
        columns = ['target_estimate', 'nontarget_estimate', 'accuracy']
        
    df = pd.DataFrame(estimates, columns=columns)
    df.insert(0, 'nominal', nominal_rates)
    
    return df

def empirical_rates_for_p_values(p_values,
    y,
    nominal_rates=None,
    target_class=1):
    if nominal_rates is None:
        nominal_rates = np.linspace(0.01, 0.99, 99)
    y_pred = np.zeros_like(p_values)

    rates = []

    for alpha in nominal_rates:
        y_pred = np.zeros_like(p_values)
        y_pred[p_values > alpha] = target_class
        y_pred[p_values <= alpha]= 1 - target_class
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        [[TN, FP], [FN, TP]] = cm
        
        FPR = FP / (TN + FP)
        FNR = FN / (TP + FN)
        if target_class == 1:
            rates.append(FNR)
        else:
            rates.append(FPR)

    return np.array(rates)

def evaluate_once(clfs, X, y, target_class, eval_size, seed, nominal_rates, confidence_level=0.99):
    all_results = None
    clf_times = []
    fit_times = []
    
    X_cv, X_eval, y_cv, y_eval = train_test_split(X, y, 
                                                  test_size=eval_size, 
                                                  stratify = y.tolist(),
                                                 random_state=seed)
    print('\tFit the Controllers...')
    descriptions = {}
    for name in clfs:
        clf = clfs[name]
        print('\t\t', name)
        if any([method in name for method in ['TBC', 'WTBC', 'DNP', 'CPF']]):
            clf.steps[-1][1].set_params(random_state=seed)
            descriptions[name] = clf.steps[-1][1].describe()
        else:
            clf.set_params(random_state=seed)
            descriptions[name] = clf.describe()
        
        start = time()
        clfs[name] = clf.fit(X_cv, y_cv)
        end = time()
        fit_times.append(descriptions[name] | {'time' : end - start})
        print(f'\t\t{np.round(end - start, 2)} seconds!\n')
        
    print('\tEvaluate the Controllers...')
    for name in clfs:
        clf = clfs[name]
        print(f'\t{name}', end='')
        start = time()
        #print(controller.predict(X_eval))
        #v=v
        results = empirical_rates(clf, X_eval, y_eval, target_class,
                                  nominal_rates=nominal_rates,
                                 ci_coverage=confidence_level)
        end = time()
        print(f'\t: {np.round(end - start, 2)} seconds')
    
        #results.insert(0, 'method', name)
        #results.insert(0, 'split', i + 1)
        for param in descriptions[name]:
            results[param] = descriptions[name][param]
        
        if all_results is None:
            all_results = results
        else:
            all_results = pd.concat([all_results, results])
            
        clf_time = (end - start) / (len(X_eval) * len(nominal_rates))
        clf_times.append(descriptions[name] | {'time' : clf_time})
    
    return all_results, pd.DataFrame(clf_times), pd.DataFrame(fit_times)