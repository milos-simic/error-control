import numpy as np

class Interpolator(object):
    """docstring for Augmentor"""
    def __init__(self, add_between=1):
        super(Interpolator, self).__init__()
        self.add_between = add_between
    
    def augment(self, scores, fitting_data=None):
        n = len(scores)

        scores = np.sort(scores)

        additional = []
        
        for i in range(n - 1):
            new_scores = np.linspace(scores[i], scores[i+1], self.add_between + 2)
            new_scores = new_scores[1: len(new_scores) - 1]
            additional.extend(new_scores)

        scores = np.append(scores, additional)

        scores = np.sort(scores)

        return scores