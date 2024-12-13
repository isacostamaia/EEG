#%%
import sys
import os
sys.path.append(os.path.abspath(".."))

from io import BytesIO
import panel as pn
import numpy as np
from moabb.datasets import BI2013a

from preprocessing.power import FRMS
from preprocessing.data_processing import get_clean_epochs, Lagger
from preprocessing.data_processing_iterative import AltFilters


def dash_peaks_tg_ntg(epochs_tg, epochs_ntg, subj=None):
    """
        Creates dashboard for peak extraction from FRMS of average Target and FRMS of average NonTarget
    Params:
        epochs_tg: epochs of Target class
        epochs_ntg: epochs of Non Target class
    """
    figs =[]

    #FRMS of mean evoked potential
    frms_tg = FRMS(epochs_tg.average())
    frms_ntg = FRMS(epochs_ntg.average())
    
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

    mpls = [pn.pane.Matplotlib(f, dpi=144, tight=True) for f in figs]

    title = pn.pane.Markdown("# FRMS of Average Target and Average Non Target subject #{}".format(subj), align="center")

    # Arrange plots in a grid
    dashboard = pn.GridSpec(sizing_mode='stretch_both')

    dashboard[0, 0] = mpls[0]
    dashboard[0, 1] = mpls[1]
    dashboard[1, 0] = mpls[2]
    dashboard[1, 1] = mpls[3]
    dashboard[2, 0] = mpls[4]
    dashboard[2, 1] = mpls[5]
    
    # Combine the title and dashboard
    layout = pn.Column(
        title,  # Title at the top
        dashboard  # Grid of plots
    )

    #save
    dash_imgs_path = os.path.join(os.path.dirname(__file__), '../dashboards_imgs')
    #verify if folder exists, create otherwise
    os.makedirs(dash_imgs_path, exist_ok=True)
    # Filepath for the image
    image_path = os.path.join(dash_imgs_path, 'subj{}.html'.format(subj))

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
    """
    Do dashboard for a single session for a subject.
    """
    dataset=BI2013a()
    subj = 2
    session = "0"
    epochs = get_clean_epochs(dataset, subjects_list=[subj])
    epochs = epochs[epochs.metadata.session.values == session]

    alt_filter = AltFilters(epochs, p=4)
    filtered_epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)
    #Average filtered and lag-corrected target FRMS 
    lagger = Lagger(filtered_epochs["Target"])
    lag_corrected_epochs_tg = lagger.compute_and_correct_lags()

    #Average filtered and lag-corrected non-target FRMS 
    lagger = Lagger(filtered_epochs["NonTarget"])
    lag_corrected_epochs_ntg = lagger.compute_and_correct_lags()
    dash_peaks_tg_ntg(lag_corrected_epochs_tg, lag_corrected_epochs_ntg, subj="Subj {}, Session {}".format(subj,session))

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   #main()
   main_per_session()

# %%
