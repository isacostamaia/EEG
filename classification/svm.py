#%%

import sys
import os
sys.path.append(os.path.abspath(".."))

import pickle 
import numpy as np
import pandas as pd
from moabb.datasets import BI2013a

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from preprocessing.power import FRMS
from preprocessing.data_processing import get_clean_epochs, Lagger
from preprocessing.data_processing_iterative import AltFilters



def preprocess_epochs(epochs_subj, session = "0"):
    """
    Given an epochs object from a subject and a session of interest, treat the data within that session.
    Returns:
        Filtered and lag-corrected Target and Non-Target epochs
    """


    #check available sessions and filter the session of interest
    unique_sessions = np.unique(epochs_subj.metadata['session'])
    if session not in unique_sessions:
        raise ValueError("Session {} does not exist. Existing sessions: {}".format(session, unique_sessions))
    session_epochs = epochs_subj[epochs_subj.metadata.session.values == session]
    
    #preprocess
    alt_filter = AltFilters(session_epochs, p=4)
    del session_epochs
    filtered_epochs, _ = alt_filter.fit_and_apply(class_="Target", plot_it=False)
    del alt_filter
    #Average filtered and lag-corrected target FRMS 
    lagger = Lagger(filtered_epochs["Target"])
    lag_corrected_epochs_tg = lagger.compute_and_correct_lags()

    #Average filtered and lag-corrected non-target FRMS 
    lagger = Lagger(filtered_epochs["NonTarget"])
    lag_corrected_epochs_ntg = lagger.compute_and_correct_lags()
    del lagger, filtered_epochs

    return lag_corrected_epochs_tg, lag_corrected_epochs_ntg


def extract_window_mean(epochs_tg, epochs_ntg, t_interval = 30, step = 3, save = False, db_name = "db"):
    """
    Extracts sample data (n_channels x 1) for Target and Non-Target responses
    and saves them as .pkl files (one for each subject). Samples will be the average
    ERP value within a window of t_interval ms. Samples are created every "step" time steps.
    Parameters:
        epochs_tg: Target preprocessed (filtered and lag-corrected) epochs object from a single subject and session
        epochs_ntg: Non-Target preprocessed (filtered and lag-corrected) epochs object from a single subject and session
        t_interval: time window duration in ms in which average ERP will be computed
        step:       Step size for advancing the window

    """


    XY = {"ERP":[], "Target":[], "Timeidx":[], "Session":[], "Subject":[], "Epoch":[]}
    #
    # Get data and preprocess
    #
    #Get subject id and check if epochs object contains a single subj

    assert epochs_tg.info['sfreq'] == epochs_ntg.info['sfreq'], "Provided epochs have different sampling rates"
    assert len(epochs_tg.metadata.subject.unique()) == 1 and len(epochs_ntg.metadata.subject.unique()), "One of the epochs object contain more than one subject"
    assert epochs_tg.metadata.subject.unique()[0] == epochs_ntg.metadata.subject.unique()[0], "Epochs object do not contain same subject"
    assert len(epochs_tg.metadata.session.unique()) == 1 and len(epochs_ntg.metadata.session.unique()), "One of the epochs object contain more than one session"
    assert epochs_tg.metadata.session.unique()[0] == epochs_ntg.metadata.session.unique()[0], "Epochs object do not contain same session"
    
    session = epochs_tg.metadata.session.unique()[0]
    subj = epochs_tg.metadata.subject.unique()[0]
    sfreq = epochs_tg.info['sfreq'] #sampling frequency
    #
    # Extract window mean
    #
    # Parameters
    #get number of samples corresponding to ~t_interval ms

    len_window = int(t_interval*1e-3*sfreq) # Length of the window ~ around t_interval as number of samples

    tg_samples = epochs_tg.get_data()
    ntg_samples = epochs_ntg.get_data()

    _, n_channels, n_times = tg_samples.shape
    # Slide the window over the time axis
    for start in range(0, n_times - len_window + 1, step):
        end = start + len_window
        # Compute the average within the window and reshape to (n_epochs, n_channels, 1)
        window_avg_tg = np.mean(tg_samples[:, :, start:end], axis=2) #(n_epochs, n_channels,)
        window_avg_tg = np.expand_dims(window_avg_tg, 2) #(n_epochs,n_channels,1)

        window_avg_ntg = np.mean(ntg_samples[:, :, start:end], axis=2) #(n_epochs, n_channels,)
        window_avg_ntg = np.expand_dims(window_avg_ntg, 2) #(n_epochs,n_channels,1)

        XY["ERP"] = XY["ERP"] + list(window_avg_tg) #len = n_epochs
        XY["Target"] = XY["Target"] + [1]*window_avg_tg.shape[0] #n_epochs
        XY["Session"] = XY["Session"] + [session]*window_avg_tg.shape[0] #n_epochs
        XY["Subject"] = XY["Subject"] + [subj]*window_avg_tg.shape[0] #n_epochs
        XY["Timeidx"] = XY["Timeidx"] + [(start, end)]*window_avg_tg.shape[0] 
        XY["Epoch"] = XY["Epoch"] + list(epochs_tg.metadata.index) #n_epochs


        XY["ERP"] = XY["ERP"] + list(window_avg_ntg)
        XY["Target"] = XY["Target"] + [0]*window_avg_ntg.shape[0] #n_epochs
        XY["Session"] = XY["Session"] + [session]*window_avg_ntg.shape[0] #n_epochs
        XY["Subject"] = XY["Subject"] + [subj]*window_avg_ntg.shape[0] #n_epochs
        XY["Timeidx"] = XY["Timeidx"] + [(start, end)]*window_avg_ntg.shape[0] 
        XY["Epoch"] = XY["Epoch"] + list(epochs_ntg.metadata.index) #n_epochs

    del epochs_tg, epochs_ntg

    df = pd.DataFrame(XY)
    #flatten the 'ERP' n_channelsx1 arrays into n_channels-element vectors
    df['ERP'] = df['ERP'].apply(lambda x: np.array(x).flatten())

    if save:
        #save
        path = os.path.join(os.path.dirname(__file__), '../data/window_mean_data/{}/'.format(db_name))
        #verify if folder exists, create otherwise
        os.makedirs(path, exist_ok=True)
        # Filepath for the image
        dict_path = os.path.join(path, 'df_{}_subj{}_session{}.pkl'.format(db_name,subj,session))
        with open(dict_path, 'wb') as f:
            pickle.dump(df, f)

    return df

def pkl_to_df(db_name, subject, session, folder = os.path.join("..", "data", "window_mean_data")):
    """
    Reads .pkl file in folder and returns df 
    for db_name directory 
        Params: 
            db_name: (e.g. db_name = 'BI2013a' or '*' for all db directories in folder)
            subject: string number of subject file inside folder/db_name. 
            session: string number of session file inside folder/db_name
        Returns: df 
    """
    direc = os.path.join(os.path.dirname(__file__), folder)

    file = "df_" + db_name + "_subj" + subject + "_session" + session + ".pkl"
    #list desired files inside desired db folder
    file_path = os.path.join(direc, db_name, file)
    print("Provided path",file_path)
    df = pd.read_pickle(file_path)
    if df.empty:
        raise ValueError("No files found with the specified arguments.")
    return df


def scaled_df(df):
    """
        Given a df from a subject and session, returns X and y data
        Params:
  
        Returns: feature-scaled df
            
    """
    #stack the flattened ERP into a feature matrix
    X = np.vstack(df['ERP'].values)  # Shape: (n_samples, 16)
    y = df['Target'].values  # Class labels
    
    #normalize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df["ERP_scaled"] = pd.Series(list(X_scaled), index = df.index)
    return df


def plot_roc_and_acc_time(df, db_name = None, subject = None, session = None):

    """
    Plot ROC curve for SVM model and its accuracy on time for a given subject in given a session. Prediction probabilities are
    the mean of prediction probabilities obtained in cross validation. These mean predictions are
    used for tracing the ROC curve.
    """

    if db_name and subject and session:
        title = "SVM ROC Curve " + db_name + " subject " + subject + " session " + session
    else:
        title = "SVM ROC Curve"
    
    X, y = np.vstack(df["ERP_scaled"].values), df["Target"].values

    #set up the SVM classifier and try to compensate class imbalance
    svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')

    #cross-validation and AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(svm, X, y, cv=cv, method='predict_proba', n_jobs = -1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    optimal_idx = np.argmax(tpr - fpr)  # Maximizing Youden's index  (sensitivity + specificity - 1)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)

    #plot AUC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    #apply the optimal threshold
    y_pred_custom = (y_proba >= optimal_threshold).astype(int)
    #print general accuracy
    acc = (y == y_pred_custom).mean()
    print("global accuracy: ", acc)

    timeidx_groups = df.groupby("Timeidx").indices
    acc_time = {}
    for k,v in timeidx_groups.items():
        pred_timeidx = y_pred_custom[list(v)]
        label_timeidx = y[list(v)]
        acc_time[k] = (pred_timeidx == label_timeidx).mean()

    times = np.array([int(np.mean(list(l))) for l in list(acc_time.keys())])
    #target and non-target
    plt.figure(figsize=(12,3))
    plt.plot(epochs_subj.times[times]*1e3, list(acc_time.values()), color='blue', lw=2)
    plt.xlabel("ms")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on time for all samples")
    plt.grid()
    plt.show()


#%%
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here

    dataset = BI2013a()
    subject = "2"
    epochs_subj = get_clean_epochs(dataset, subjects_list = [int(subject)])
    #preprocess and extract data
    session = "0"
    epochs_tg, epochs_ntg = preprocess_epochs(epochs_subj, 
                                            subject = subject, 
                                            session = session)
    df = extract_window_mean(epochs_tg, epochs_ntg, t_interval = 30, step = 3, save = False, db_name = "db")
    #get features 
    df = scaled_df(df)

    db_name = 'BI2013a'

    #plot ROC ang get optimal threshold
    #svm_opt_threshold  = plot_roc_and_acc_time(df, db_name = db_name, subject = subject, session = session)
    


    """
    Plot ROC curve for SVM model and its accuracy on time for a given subject in given a session. Prediction probabilities are
    the mean of prediction probabilities obtained in cross validation. These mean predictions are
    used for tracing the ROC curve.
    """

    if db_name and subject and session:
        title = "SVM ROC Curve " + db_name + " subject " + subject + " session " + session
    else:
        title = "SVM ROC Curve"
    
    X, y = np.vstack(df["ERP_scaled"].values), df["Target"].values

    #set up the SVM classifier and try to compensate class imbalance
    svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')

    #cross-validation and AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(svm, X, y, cv=cv, method='predict_proba', n_jobs = -1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    optimal_idx = np.argmax(tpr - fpr)  # Maximizing Youden's index  (sensitivity + specificity - 1)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)

    #plot AUC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    #apply the optimal threshold
    y_pred_custom = (y_proba >= optimal_threshold).astype(int)
    #print general accuracy
    acc = (y == y_pred_custom).mean()
    print("global accuracy: ", acc)

    timeidx_groups = df.groupby("Timeidx").indices
    acc_time = {}
    for k,v in timeidx_groups.items():
        pred_timeidx = y_pred_custom[list(v)]
        label_timeidx = y[list(v)]
        acc_time[k] = (pred_timeidx == label_timeidx).mean()

    times = np.array([int(np.mean(list(l))) for l in list(acc_time.keys())])
    #target and non-target
    plt.figure(figsize=(12,3))
    plt.plot(epochs_subj.times[times]*1e3, list(acc_time.values()), color='blue', lw=2)
    plt.xlabel("ms")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on time for all samples")
    plt.grid()
    plt.show()




    #plot FRMS
    frms_tg = FRMS(epochs_tg.average())
    frms_ntg = FRMS(epochs_ntg.average())

    #peaks idx
    peaks_idx_tg, _ = frms_tg.peaks_idx(show=True)
    peaks_idx_ntg, _ = frms_ntg.peaks_idx(show=True)

    #del frms_tg, frms_ntg
    tg_samples = epochs_tg.get_data()[:,:,peaks_idx_tg]
    ntg_samples = epochs_ntg.get_data()[:,:,peaks_idx_ntg]
# %%
