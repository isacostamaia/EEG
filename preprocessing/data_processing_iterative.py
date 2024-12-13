#%%

import copy
from re import T
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg


import mne
from moabb.datasets import BI2013a


from preprocessing.power import field_root_mean_square, plot_fmrs
from preprocessing.data_processing import get_clean_epochs, \
                            Filter


class AltFilters:
    """
    Compute iterative alternating spatial and temporal filters
    """
    def __init__(self, epochs, p):
        """
        Params:
            epochs: epochs object of all classes
            p: number of spatial and temporal components
        """
        self.epochs = epochs
        self.p = p
    def fit_and_apply(self, class_ = "Target", plot_it = True):
        """
        Fit Spatial and Temporal filters for class_ and applies them to all classes
        Params:
            class_: class of epochs to which filters should be fitted
            plot_it: True if iterations should be plotted
        returns: Spatially and Temporally filtered epochs from all classes
        """

        self.S_list = []
        self.T_list = []
        diverg_hist = []

        spat_filt_epochs = self.epochs
        temp_filt_epochs = self.epochs

        tol = 1e-6
        max_it = 16

        for it in range(max_it):
            diverg = 0
            # compute spatial filter using temporally filtered epochs only
            spatial_filter= Filter(epochs=temp_filt_epochs, p=self.p, spatial=True)
            spatial_filter.fit(class_= class_)
            spat_filt_epochs = spatial_filter.apply(temp_filt_epochs)
            self.S_list.append(spatial_filter.A_p@spatial_filter.B_p.T)

            # compute temporal filter using spatially filtered epochs only
            temporal_filter= Filter(epochs=spat_filt_epochs, p=self.p, spatial=False)
            temporal_filter.fit(class_= class_)
            temp_filt_epochs = temporal_filter.apply(spat_filt_epochs)
            self.T_list.append(temporal_filter.B_p@temporal_filter.A_p.T) #D@E.T


            #plot spatially and filtered epochs to verify
            if plot_it:
                frms = field_root_mean_square(temp_filt_epochs[class_])
                plot_fmrs(frms)


            #compute convergence
            if it>0:
                diverg += np.linalg.norm(self.S_list[it] - self.S_list[it-1])/ np.linalg.norm(self.S_list[it-1])
                diverg += np.linalg.norm(self.T_list[it] - self.T_list[it-1])/ np.linalg.norm(self.T_list[it-1])
                diverg_hist.append(diverg)
                if plot_it: print("divergence ", diverg)
                if diverg <= tol:
                    print("Spatial and Temporal Filters converged.")
                    return temp_filt_epochs, diverg_hist

                if it>1 and abs(diverg_hist[it-1] - diverg_hist[it-2]) < 1e-6 : #Check if divergence does not diminish (diverg_hist idx is -1 compared to global loop)
                    print("Spatial and Temporal Filters no longer changing after {} iterations".format(it))
                    return temp_filt_epochs, diverg_hist    
                
        print("Maximum number of iterations was reached and Spatial and Temporal Filters did not converge.") 
        return temp_filt_epochs, diverg_hist 
    
    def apply(self, epochs):
        """
        Apply Spatial and Temporal filters obtained in fit_and_apply() to epochs object
        """
        epochs = copy.deepcopy(epochs)
        S_list = reversed(self.S_list)
        id = np.eye(N=self.S_list[0].shape[0])
        for m in S_list:
            res = np.matmul(id, m)
        left_mult = lambda epoch: res@epoch
        epochs.apply_function(left_mult, picks="all", channel_wise=False)
        id = np.eye(N=self.T_list[0].shape[0])
        for m in self.T_list:
            res = np.matmul(id, m)
        right_mult = lambda epoch: epoch@res
        epochs.apply_function(right_mult, picks="all", channel_wise=False)
        return epochs


# %%
def main():
    dataset=BI2013a()
    epochs = get_clean_epochs(dataset, subjects_list=[1])
    alt_filter = AltFilters(epochs, p=4)
    filtered_epochs, _ = alt_filter.fit_and_apply(class_="Target")

    print("Plots target and non target")
    frms_tg = field_root_mean_square(filtered_epochs["Target"])
    plot_fmrs(frms_tg)

    filtered_epochs_non_target = alt_filter.apply(epochs["NonTarget"])
    frms_ntg = field_root_mean_square(filtered_epochs_non_target)
    plot_fmrs(frms_ntg, heat_min=np.min(frms_tg.values), heat_max=np.max(frms_tg.values))   

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
