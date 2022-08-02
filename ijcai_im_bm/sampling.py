# train processing head

import numpy as np
import smote_variants as sv
# import smote_variants.OverSampling as OverSampling
from smote_variants import OverSampling
from sklearn.neighbors import NearestNeighbors
import random

class Sampling(OverSampling):
    def __init__(self,  k1 = 5, k2 = 5, k3 = 5, n_jobs = 1):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            k1 (int): parameter of the NearestNeighbors component
            k2 (int): parameter of the NearestNeighbors component
            k3 (int): parameter of the NearestNeighbors component
            M (int): number of clusters
            cf_th (float): cutoff threshold
            cmax (float): maximum closeness value
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()

        
        self.k1= k1
        self.k2= k2
        self.k3= k3

        self.n_jobs= n_jobs

    def getTrainingSample(self, X, y):
        self.class_label_statistics(X, y)

        minority = np.where(y == self.minority_label)[0]
        majority = np.where(y == self.majority_label)[0]

        X_min= X[minority]
        X_maj= X[majority]
        
        # X_min= X[y == self.minority_label]
        # X_maj= X[y == self.majority_label]

        # minority= np.where(y == self.minority_label)[0]

        # Step 1
        nn= NearestNeighbors(n_neighbors= min([len(X), self.k1 + 1]), n_jobs= self.n_jobs)
        nn.fit(X)
        dist1, ind1= nn.kneighbors(X)

        # Step 2
        filtered_minority = np.array([i for i in minority if np.sum(y[ind1[i][1:]] == self.minority_label) > 0])
        
        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        nn_maj= NearestNeighbors(n_neighbors= self.k2, n_jobs= self.n_jobs)
        nn_maj.fit(X_maj)
        dist2, ind2= nn_maj.kneighbors(X[filtered_minority])
        
        # Step 4
        border_majority_index = np.unique(ind2.flatten())

        length_border_majority = len(border_majority_index)

        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        nn_min= NearestNeighbors(n_neighbors= min([self.k3, len(X_min)]), n_jobs= self.n_jobs)
        nn_min.fit(X_min)
        dist3, ind3= nn_min.kneighbors(X_maj[border_majority_index])

         # Step 6 - informative minority indexes X_min
        informative_minority_index = np.unique(ind3.flatten())

        length_informative_minority = len(informative_minority_index)

        if length_border_majority > length_informative_minority:
            border_majority_index = list(border_majority_index)
            border_majority_index = random.sample(border_majority_index, length_informative_minority)
            border_majority_index = sorted(border_majority_index)
        if length_border_majority < length_informative_minority:
            border_majority_index = list(border_majority_index)
            length_majority = X_maj.shape[0]
            majority_index = [i for i in range(length_majority)]
            left_border_majority_index = [i for i in majority_index if i not in border_majority_index]
            exter_border_majority_index = random.sample(left_border_majority_index, (length_informative_minority - length_border_majority))
            border_majority_index = border_majority_index + exter_border_majority_index
            border_majority_index = sorted(border_majority_index)
        return border_majority_index, informative_minority_index
        # border_majority = X_maj[border_majority_index]
        # informative_minority = X_min[informative_minority_index]
        # return border_majority, informative_minority

    def getTestingSample(self, X, y):
        self.class_label_statistics(X, y)

        minority = np.where(y == self.minority_label)[0]
        majority = np.where(y == self.majority_label)[0]

        X_min= X[minority]
        X_maj= X[majority]

        # X_min= X[y == self.minority_label]
        # X_maj= X[y == self.majority_label]

        # minority= np.where(y == self.minority_label)[0]


        # Step 1
        nn= NearestNeighbors(n_neighbors= min([len(X), self.k1 + 1]), n_jobs= self.n_jobs)
        nn.fit(X)
        dist1, ind1= nn.kneighbors(X)

        # Step 2
        filtered_minority = np.array([i for i in minority if np.sum(y[ind1[i][1:]] == self.minority_label) > 0])
        
        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        nn_maj= NearestNeighbors(n_neighbors= self.k2, n_jobs= self.n_jobs)
        nn_maj.fit(X_maj)
        dist2, ind2= nn_maj.kneighbors(X[filtered_minority])
        
        # Step 4
        border_majority_index = np.unique(ind2.flatten())

        # length_border_majority = len(border_majority_index)

        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        nn_min= NearestNeighbors(n_neighbors= min([self.k3, len(X_min)]), n_jobs= self.n_jobs)
        nn_min.fit(X_min)
        dist3, ind3= nn_min.kneighbors(X_maj[border_majority_index])

         # Step 6 - informative minority indexes X_min
        informative_minority_index = np.unique(ind3.flatten())
        return border_majority_index, informative_minority_index
        # length_informative_minority = len(informative_minority_index)

        # border_majority = X_maj[border_majority_index]
        # informative_minority = X_min[informative_minority_index]
        # return border_majority, informative_minority