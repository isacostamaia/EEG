#%%
import sys
import os

import mne
sys.path.append(os.path.abspath(".."))

import panel as pn
import numpy as np
import mne
from moabb.datasets import BI2013a

from preprocessing.power import FRMS
from preprocessing.data_processing import get_clean_epochs, Lagger
from preprocessing.data_processing_iterative import AltFilters


def dash_peaks_tg_ntg(epochs_tg, epochs_ntg, spec, avg_ERP = True):
    """
        Creates dashboard for peak extraction from FRMS of average Target and FRMS of average NonTarget
    Params:
        epochs_tg: epochs of Target class
        epochs_ntg: epochs of Non Target class
    """
    figs =[]

    #FRMS of evoked potentials or mean evoked potential
    if avg_ERP:
        e_tg = epochs_tg.average()
        e_ntg = epochs_tg.average()
        addtitle = "Average "
    else:
        e_tg = epochs_tg
        e_ntg = epochs_ntg
        addtitle = ""

    frms_tg = FRMS(e_tg) #epochs_tg.average()
    frms_ntg = FRMS(e_ntg) #epochs_ntg.average()
    
    # Determine shared y-limits for trimmed mean plots
    trim_mean_min = min(np.min(frms_tg.trim_mean_frms), np.min(frms_ntg.trim_mean_frms))
    trim_mean_max = max(np.max(frms_tg.trim_mean_frms), np.max(frms_ntg.trim_mean_frms))

    # Determine shared colorbar limits for heatmap plots
    heat_min = min(np.min(frms_tg.frms_df.values), np.min(frms_ntg.frms_df.values))
    heat_max = max(np.max(frms_tg.frms_df.values), np.max(frms_tg.frms_df.values))

    # 1 - FRMS plot target and non target
    tg_plot = frms_tg.plot(heat_min=heat_min, heat_max=heat_max, trim_mean_min=trim_mean_min, trim_mean_max=trim_mean_max)
    ntg_plot = frms_ntg.plot(heat_min=heat_min, heat_max=heat_max, trim_mean_min=trim_mean_min, trim_mean_max=trim_mean_max)
    figs.append(tg_plot)
    figs.append(ntg_plot)

    #compare_plot = frms_tg.plot_compare(frms_ntg)

    # 2 - peak finder plot
    peaks_idx_tg, peaks_tg_plot = frms_tg.peaks_idx()
    peaks_idx_ntg, peaks_ntg_plot = frms_ntg.peaks_idx()
    figs.append(peaks_tg_plot)
    figs.append(peaks_ntg_plot)

    #Topomaps Target and Non Target at peaks of FRMS Target and FRMS Non Target
    peaks_times_tg = epochs_tg.times[peaks_idx_tg]
    peaks_times_ntg = epochs_ntg.times[peaks_idx_ntg]

    # 3 - Topomaps plots

    topo_tg_plot = epochs_tg.average().plot_topomap(times= peaks_times_tg, ch_type="eeg", average=0.05, show = False)
    topo_ntg_plot = epochs_ntg.average().plot_topomap(times= peaks_times_ntg, ch_type="eeg", average=0.05, show = False)
    figs.append(topo_tg_plot)
    figs.append(topo_ntg_plot)

    # 4 - Average Target ERP Response
    avg_tg = epochs_tg["Target"].average().plot(show = False)
    figs.append(avg_tg)

    mpls = [pn.pane.Matplotlib(f, dpi=144, tight=True) for f in figs]

    title = pn.pane.Markdown("# FRMS of {}Target and {}Non Target {}".format(addtitle, addtitle, spec), align="center")

    # Arrange plots in a grid
    dashboard = pn.GridSpec(sizing_mode='stretch_both')

    dashboard[0, 0] = mpls[0]
    dashboard[0, 1] = mpls[1]
    dashboard[1, 0] = mpls[2]
    dashboard[1, 1] = mpls[3]
    dashboard[2, 0] = mpls[4]
    dashboard[2, 1] = mpls[5]
    dashboard[3, 0] = mpls[6]
    
    # Combine the title and dashboard
    layout = pn.Column(
        title,  # Title at the top
        dashboard  # Grid of plots
    )

    #save
    dash_imgs_path = os.path.join(os.getcwd(), 'dashboards_imgs')
    #verify if folder exists, create otherwise
    os.makedirs(dash_imgs_path, exist_ok=True)
    # Filepath for the image
    image_path = os.path.join(dash_imgs_path, '{}.html'.format(spec))

    layout.save(image_path)
    #layout.show()




def main():
    """
    Do dashboard for all sessions at once for a subject.
    """
    dataset=BI2013a()
    for subj in dataset.subject_list[6:]:
        epochs = get_clean_epochs(dataset, subjects_list=[subj])

        alt_filter = AltFilters(epochs, p=4)
        del epochs
        filtered_epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)
        del alt_filter
        #Average filtered and lag-corrected target FRMS 
        lagger = Lagger(filtered_epochs["Target"])
        lag_corrected_epochs_tg = lagger.compute_and_correct_lags()

        #Average filtered and lag-corrected non-target FRMS 
        lagger = Lagger(filtered_epochs["NonTarget"])
        lag_corrected_epochs_ntg = lagger.compute_and_correct_lags()
        del lagger, filtered_epochs


        dash_peaks_tg_ntg(lag_corrected_epochs_tg, lag_corrected_epochs_ntg, subj=subj)

def main_per_session():
    #this one lags an then filters

    """
    Do dashboard for a single session for a subject.
    """
    dataset=BI2013a()
    subj = 1
    session = "0"
    p = 10
    lag_correction = True

    epochs = get_clean_epochs(dataset, subjects_list=[subj])
    epochs = epochs[epochs.metadata.session.values == session]

    if lag_correction:
        #lag-corrected target FRMS 
        lagger = Lagger(epochs["Target"])
        epochs_tg = lagger.compute_and_correct_lags()
    
        #lag-corrected non-target FRMS 
        lagger = Lagger(epochs["NonTarget"])
        epochs_ntg = lagger.compute_and_correct_lags()
        lag_correction = "BeforeFilt"
        
        epochs = mne.concatenate_epochs([epochs_tg, epochs_ntg])

    spec = "subj{}_sess{}_p{}_lag{}".format(subj, session, p, lag_correction)

    if p: #Filter
        alt_filter = AltFilters(epochs, p = p)
        epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)

    dash_peaks_tg_ntg(epochs["Target"], epochs["NonTarget"], spec, avg_ERP=False)

# def main_per_session():
# #this one filters and then lags
#     """
#     Do dashboard for a single session for a subject.
#     """
#     dataset=BI2013a()
#     subj = 2
#     session = "0"
#     p = 10
#     lag_correction = True

#     epochs = get_clean_epochs(dataset, subjects_list=[subj])
#     epochs = epochs[epochs.metadata.session.values == session]

#     if p: #Filter
#         alt_filter = AltFilters(epochs, p = p)
#         epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)

#     epochs_tg = epochs["Target"]
#     epochs_ntg = epochs["NonTarget"]

#     if lag_correction:
#         #lag-corrected target FRMS 
#         lagger = Lagger(epochs_tg)
#         epochs_tg = lagger.compute_and_correct_lags()
    
#         #lag-corrected non-target FRMS 
#         lagger = Lagger(epochs_ntg)
#         epochs_ntg = lagger.compute_and_correct_lags()    
#         lag_correction = "AfterFilt"

#     spec = "subj{}_sess{}_p{}_lag{}".format(subj, session, p, lag_correction)

#     dash_peaks_tg_ntg(epochs_tg, epochs_ntg, spec)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   #main()
   main_per_session()

# %%
