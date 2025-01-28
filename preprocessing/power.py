#%%
from io import BytesIO
import panel as pn
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema



import mne

def field_root_mean_square(class_epochs):
    """
    Calculates Field Root Mean Square
        Parameters:
            class_epochs: object corresponding to a class, could be EpochsArraay or ndarray
        Returns:
            df: instance of dataframe with row sorted field root mean square
    """
    
    if isinstance(class_epochs,mne.EpochsArray) or isinstance(class_epochs, mne.Epochs):
        class_epochs_data = class_epochs.get_data() #array of shape (n_epochs, n_channels, n_times)
    elif isinstance(class_epochs, np.ndarray):
        class_epochs_data = class_epochs  #array of shape (n_epochs, n_channels, n_times)
    else:
        raise TypeError(f"class_epochs must be EpochsArray or ndarray type, not {type(class_epochs)}")
    
    centered_Xks = class_epochs_data
    
    #frms_i is the std of the columns of centered_Xk
    frms = np.stack([Xk.std(axis=0) for Xk in centered_Xks])  #each pfmrs_i has dim = n_times, frms will be n_epochs x n_times

    #create df and sort by sum of the rows 
    df = pd.DataFrame(frms, index = np.arange(1, frms.shape[0]+1), columns = ['%.f' % elem for elem in class_epochs.times*1e3])
    df["row_sum"]=df.apply(lambda x: sum(x), axis=1)
    df = df.sort_values("row_sum", ascending=False).drop(columns=['row_sum'])

    return df

def trim_mean_frms(frms, plot=False):
    """
    Columns mean of FRMS dataframe (n_epochs x n_times)
    returns: cols_mean, times
    """
    cols_mean = stats.trim_mean(frms, proportiontocut=0.1, axis=0)
    times = np.array([int(e) for e in frms.columns]) #TODO replace the columns of frms by exact value instead of str and reuse it here
    if plot:
        fig, ax1 = plt.subplots(figsize=(12,4))
        ax1.plot(times, cols_mean)
        ax1.tick_params(axis = 'x', which='major', labelrotation=90)
        ax1.set_title("Average FRMS")
        ax1.set_xlabel("ms")
        ax1.set_ylabel("V")
    return cols_mean, times

def plot_fmrs(frms, heat_min=None, heat_max=None):

    if not heat_min:
        heat_min = np.min(frms.values)
    if not heat_max:
        heat_max = np.max(frms.values)


    fig, ((ax1, cbar_ax, ax3), (ax2, dummy_ax1, dummy_ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex='col',
                                                      gridspec_kw={'height_ratios': [5, 1], 'width_ratios': [30, 1, 8]})

    #FRMS
    g1 = sns.heatmap(frms, ax=ax1, cbar_ax=cbar_ax, vmin=heat_min, vmax=heat_max)
    ax1.set_title("FRMS")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Epoch")
    
    #Row sum
    ax3.plot(np.arange(0,len(frms)),frms.iloc[::-1].apply(lambda x: sum(x), axis=1))
    ax3.set_title("Row sum")
    ax3.yaxis.tick_right()
    
    #Trimmed mean of columns taking out 10% oh highest and lowest values
    col_mean, _ = trim_mean_frms(frms)
    ax2.plot(frms.columns, col_mean)
    ax2.tick_params(which='major', labelrotation=90)
    
    dummy_ax1.set_axis_off()
    dummy_ax2.set_axis_off()

    plt.show()

class FRMS:
    def __init__(self, class_epochs):
        """
        Calculates Field Root Mean Square
            Parameters:
                class_epochs: object corresponding to a class, could be EpochsArraay or ndarray
            Attributes:
                frms_df: instance of dataframe with row sorted field root mean square (each row is an epoch)
                trim_mean_frms: trimmed mean of FRMS across epochs
        """
        
        if isinstance(class_epochs,mne.EpochsArray) or isinstance(class_epochs, mne.Epochs):
            class_epochs_data = class_epochs.get_data() #array of shape (n_epochs, n_channels, n_times)
            columns = ['%.f' % elem for elem in class_epochs.times*1e3]
        elif isinstance(class_epochs, mne.EvokedArray):
            class_epochs_data = np.expand_dims(class_epochs.get_data(), axis=0) #array of shape (1, n_channels, n_times)
            columns = ['%.f' % elem for elem in class_epochs.times*1e3]
        elif isinstance(class_epochs, np.ndarray):
            class_epochs_data = class_epochs  #array of shape (n_epochs, n_channels, n_times)
            columns = np.linspace(0,1000,class_epochs.shape[-1])
        else:
            raise TypeError(f"class_epochs must be EpochsArray, Evoked Array or ndarray type, not {type(class_epochs)}")
        
        centered_Xks = class_epochs_data
        
        #frms_i is the std of the columns of centered_Xk
        self.frms_df = np.stack([Xk.std(axis=0) for Xk in centered_Xks])  #each pfmrs_i has dim = n_times, frms will be n_epochs x n_times

        #create df and sort by sum of the rows 
        #['%.f' % elem for elem in class_epochs.times*1e3] [round(e) for e in class_epochs.times*1e3]
        self.frms_df = pd.DataFrame(self.frms_df, index = np.arange(1, self.frms_df.shape[0]+1), columns = columns )
        self.frms_df["row_sum"]=self.frms_df.apply(lambda x: sum(x), axis=1)
        self.frms_df = self.frms_df.sort_values("row_sum", ascending=False).drop(columns=['row_sum'])

        #Trimmed mean of columns taking out 10% oh highest and lowest values
        self.trim_mean_frms = stats.trim_mean(self.frms_df, proportiontocut=0.1, axis=0)

        #for smoothing in peaks()
        self.sfreq =  class_epochs.info['sfreq']         

        

    def plot(self, heat_min=None, heat_max=None, trim_mean_min=None, trim_mean_max=None, show = False):
        """
            Plot FRMS heatmap, trimmed mean across epochs and row sum across time
        """
        if not heat_min:
            heat_min = np.min(self.frms_df.values)
        if not heat_max:
            heat_max = np.max(self.frms_df.values)
        if not trim_mean_min:
            trim_mean_min = np.min(self.trim_mean_frms)
        if not trim_mean_max:
            trim_mean_max = np.max(self.trim_mean_frms)


        fig, ((ax1, cbar_ax, ax3), (ax2, dummy_ax1, dummy_ax2)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharex='col',
                                                        gridspec_kw={'height_ratios': [5, 1], 'width_ratios': [30, 1, 8]})

        #FRMS
        g1 = sns.heatmap(self.frms_df, ax=ax1, cbar_ax=cbar_ax, vmin=heat_min, vmax=heat_max)
        ax1.set_title("FRMS")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Epoch")
        
        #Row sum
        ax3.plot(np.arange(0,len(self.frms_df)),self.frms_df.iloc[::-1].apply(lambda x: sum(x), axis=1))
        ax3.set_title("Row sum")
        ax3.yaxis.tick_right()
        
        #Trimmed mean 
        ax2.plot(self.frms_df.columns, self.trim_mean_frms)
        ax2.tick_params(which='major', labelrotation=90)
        ax2.set_ylim(trim_mean_min, trim_mean_max)

        
        dummy_ax1.set_axis_off()
        dummy_ax2.set_axis_off()
        if show:
            plt.show()
        plt.close(fig)
        return fig
    
    def plot_compare(self, second_frms, show = False):
        """
        Plot for comparing FRMS 1 (self) to a FRMS 2 (second_frms) side by side with a shared color bar and shared trimmed mean ylim.
        """
        # Set heatmap bounds
        heat_min = np.min(list(self.frms_df.values)+list(second_frms.frms_df.values))

        heat_max = np.max(list(self.frms_df.values)+list(second_frms.frms_df.values))

        # Create the layout for two plots with a single color bar on the right
        fig, axes = plt.subplots(
            nrows=2, ncols=3,
            figsize=(24, 6),
            gridspec_kw={'height_ratios': [5, 1], 'width_ratios': [30, 30, 1]},
            sharex='col'
        )

        # Unpack axes
        ((ax1, ax2, cbar_ax), (ax_trim1, ax_trim2, dummy_ax)) = axes

        # Use Seaborn's default colormap for heatmaps
        cmap = sns.color_palette("rocket", as_cmap=True)

        # Set color normalization for consistent colors across heatmaps
        norm = plt.Normalize(vmin=heat_min, vmax=heat_max)

        # Plot the first heatmap
        sns.heatmap(self.frms_df, ax=ax1, cbar=False, vmin=heat_min, vmax=heat_max, norm=norm, cmap=cmap)
        ax1.set_title("FRMS 1")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Epoch")

        # Determine shared y-limits for trimmed mean plots
        trimmed_min = min(np.min(self.trim_mean_frms), np.min(second_frms.trim_mean_frms))
        trimmed_max = max(np.max(self.trim_mean_frms), np.max(second_frms.trim_mean_frms))

        # Plot the second heatmap
        sns.heatmap(second_frms.frms_df, ax=ax2, cbar=False, vmin=heat_min, vmax=heat_max, norm=norm, cmap=cmap)
        ax2.set_title("FRMS 2")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Epoch")

        ax_trim2.plot(second_frms.frms_df.columns, second_frms.trim_mean_frms)
        ax_trim2.tick_params(which='major', labelrotation=90)
        ax_trim2.set_ylim(trimmed_min, trimmed_max)

        # Add the shared color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.set_title("Value")

        # Plot the first trimmed mean
        ax_trim1.plot(self.frms_df.columns, self.trim_mean_frms)
        ax_trim1.tick_params(which='major', labelrotation=90)
        ax_trim1.set_ylim(trimmed_min, trimmed_max)

        # Hide unused dummy axis
        dummy_ax.set_axis_off()

        # Show the plot
        plt.tight_layout()
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def plot_trimmed_mean(self, subtract_frms=None, show=False):
        """
            Plot trimmed mean. 
            If subtract_frms is provided, plot the difference (self.trim_mean_frms - sibtract_frms.trim_mean_frms)
        """
        title = "FRMS Trimmed Mean"
        times = [int(e) for e in self.frms_df.columns]
        if subtract_frms:
            y = self.trim_mean_frms -  subtract_frms.trim_mean_frms
            title += " Difference"
        else:
            y = self.trim_mean_frms
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(times, y)
        ax.tick_params(which='major', labelrotation=90)
        ax.set_title(title)
        if show:
            plt.show()
        plt.close(fig)
        return fig


    def peaks_idx(self, subtract_frms=None, t_interval = 70, thr_per = 0.45, show = False, smooth = False):

        """
        Get peaks idx of mean FRMS 
        Params:
            subtract_frms: if provided, will find the peaks of self.trim_mean_frms - subtract_frms.trim_mean_frms instead
            t_interval: time interval in which samples should be Gaussian-smoothed in ms
            thr_per: threshold percentage of valuable peaks. Only peaks above thr_per*max of (smoothed) mean FRMS are acceptable
            plot: True if visualization desired
        returns:
            valid_idx: peaks idx 
        """
        
        #get number of samples corresponding to ~t_interval ms
        ns = int(t_interval*1e-3*self.sfreq) #around t_interval as number of samples

        title = "FRMS Trimmed Mean"
        mean_frms = self.trim_mean_frms
        if subtract_frms:
            mean_frms = mean_frms - subtract_frms.trim_mean_frms
            title += " Difference"
        if smooth:
            mean_frms = gaussian_filter1d(mean_frms, sigma=ns/6)

        max_idx = argrelextrema(mean_frms, np.greater)
        #  local max >= percentage threshold of maxmean_frms
        valid_idx = np.where(mean_frms>=thr_per*np.max(mean_frms))[0]
        #get valid local extrema idx
        valid_idx = valid_idx[np.isin(valid_idx, max_idx)]
        
        times = np.array([int(e) for e in self.frms_df.columns])
        fig = None
        
        fig = plt.figure(figsize=(12,3))
        plt.plot(times, mean_frms, label="filtered mean FRMS")
        if smooth:
            plt.plot(times, mean_frms, linestyle = "dashed", color = "b", label="smoothed filtered mean FRMS")
        plt.scatter(times[max_idx], mean_frms[max_idx], color = "b", label="local max of (smoothed) filtered mean FRMS ")
        plt.scatter(times[valid_idx], mean_frms[valid_idx], color = "g", label = "local max >= {} max of (smoothed) filtered mean FRMS".format(thr_per))
        plt.xlabel("ms")
        plt.ylabel("V")
        plt.legend()
        plt.title(title)
        if show:
            plt.show()
        plt.close(fig)
        return valid_idx, fig


def main():

    from preprocessing.data_processing import get_clean_epochs
    from moabb.datasets import BI2013a

    dataset=BI2013a()
    epochs = get_clean_epochs(dataset, subjects_list=[2])
    frms = field_root_mean_square(epochs["Target"])
    plot_fmrs(frms)
    trim_mean_frms(frms, plot=True)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
# %%
