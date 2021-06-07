import numpy as np

from deslib.des.base import BaseDES

class my_method(BaseDES):
    """description of class"""
    def __init__(self, pool_classifiers=None, k=15, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30, random_state=None, knn_classifier='knn', knne=False, DSEL_perc=0.5, n_jobs=-1):
        super(my_method, self).__init__(pool_classifiers, k,
                                     DFP=DFP,
                                     with_IH=with_IH,
                                     safe_k=safe_k,
                                     IH_rate=IH_rate,
                                     mode='selection',
                                     random_state=random_state,
                                     knn_classifier=knn_classifier,
                                     DSEL_perc=DSEL_perc)

    def estimate_competence(self, query, neighbors, distances=None, predictions=None):
        #competences = np.sum(self.DSEL_processed_[neighbors, :], axis=1, dtype=np.float)
        number_of_true_predictions = np.sum(self.DSEL_processed_[neighbors, :], axis=1, dtype=np.float)
        competences = number_of_true_predictions
        selected = (competences > self.k*0.51)
        selectedn = (competences > 0)
        #print("---------------------NUMBER OF TRUE PREDICTIONS-----------------------")
        #print(competences)
        #print("--------------------- > K *0.51 -----------------------")
        #print(selected)
        #print("--------------------- > 0 -----------------------")
        #print(selectedn)
        return competences.astype(np.float)

    def estimate_competence2(self, query, neighbors, distances=None, predictions=None):
        results_neighbors = self.DSEL_processed_[neighbors, :]

        # Get the shape of the vector in order to know the number of samples,
        # base classifiers and neighbors considered.
        shape = results_neighbors.shape

        # add an row with zero for the case where the base classifier correctly
        # classifies the whole neighborhood. That way the search will always
        # find a zero after comparing to self.K + 1 and will return self.K
        # as the Competence level estimate (correctly classified the whole
        # neighborhood)
        addition = np.zeros((shape[0], shape[2]))
        results_neighbors = np.insert(results_neighbors, shape[1], addition, axis=1)

        # Look for the first occurrence of a zero in the processed predictions
        # (first misclassified sample). The np.argmax can be used here, since
        # in case of multiple occurrences of the maximum values, the indices_
        # corresponding to the first occurrence are returned.
        competences = np.argmax(results_neighbors == 0, axis=1)
        selected = (competences > self.k*9.0)
        selectedn = (competences > 0)
        #print("---------------------Results neighbours-----------------------")
        #print(results_neighbors)
        #print("---------------------NUMBER OF TRUE PREDICTIONS-----------------------")
        #print(competences)
        #print("--------------------- > K *0.51 -----------------------")
        #print(selected)
        #print("--------------------- > 0 -----------------------")
        #print(selectedn)
        return competences.astype(np.float)

    def select(self, competences):
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Select classifier if it correctly classified at least one sample
        selected_classifiers = (competences >= 0.8)
        #print(selected_classifiers)
        #print(selected_classifiers)
        # For the rows that are all False (i.e., no base classifier was
        # selected, select all classifiers (set all True)
        #selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers

    def select2(self, competences):
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Checks which was the max value for each sample
        # (i.e., the maximum number of consecutive predictions)
        max_value = np.max(competences, axis=1)

        # Select all base classifiers with the maximum number of
        #  consecutive correct predictions for each sample.
        selected_classifiers = (competences > self.k*self.passer)

        return selected_classifiers