import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from neurodsp import spectral
from fooof import FOOOFGroup
from fooof import analysis, utils

from matplotlib import rcParams
from matplotlib import pyplot as plt
rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] =15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

import os, requests

def analyze():

    fname = ['steinmetz_st.npz']
    fname.append('steinmetz_wav.npz')
    fname.append('steinmetz_lfp.npz')

    url = ["https://osf.io/4bjns/download"]
    url.append("https://osf.io/ugm9v/download")
    url.append("https://osf.io/kx3v9/download")

    for j in range(len(url)):
      if not os.path.isfile(fname[j]):
        try:
          r = requests.get(url[j])
        except requests.ConnectionError:
          print("!!! Failed to download data !!!")
        else:
          if r.status_code != requests.codes.ok:
            print("!!! Failed to download data !!!")
          else:
            with open(fname[j], "wb") as fid:
              fid.write(r.content)

    import os, requests

    fname = []
    for j in range(3):
      fname.append('steinmetz_part%d.npz'%j)
    url = ["https://osf.io/agvxh/download"]
    url.append("https://osf.io/uv3mw/download")
    url.append("https://osf.io/ehmw2/download")

    for j in range(len(url)):
      if not os.path.isfile(fname[j]):
        try:
          r = requests.get(url[j])
        except requests.ConnectionError:
          print("!!! Failed to download data !!!")
        else:
          if r.status_code != requests.codes.ok:
            print("!!! Failed to download data !!!")
          else:
            with open(fname[j], "wb") as fid:
              fid.write(r.content)

    dat_LFP = np.load('steinmetz_lfp.npz', allow_pickle=True)['dat']
    # dat_WAV = np.load('steinmetz_wav.npz', allow_pickle=True)['dat']
    # dat_ST = np.load('steinmetz_st.npz', allow_pickle=True)['dat']

    alldat = np.array([])
    for j in range(len(fname)):
      alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

    # select just one of the recordings here. 11 is nice because it has some neurons in vis ctx.
    dat = dat_LFP[11]
    print(dat.keys())
    dat_k = alldat[11]
    print(dat_k.keys())

    regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
    brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                    ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                    ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                    ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                    ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                    ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia
                    ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                    ]


    def get_rec_dat(alldat, dat_LFP, i):
      "gets information for single recording session"

      mouse_name = alldat[i]['mouse_name']
      n_shanks = dat_LFP[i]['lfp'].shape[0]
      n_trials = dat_LFP[i]['lfp'].shape[1]
      n_samples = dat_LFP[i]['lfp'].shape[2]
      sfreq = n_samples/2.5 # 100 Hz sampling rate (nyquist @ 50 Hz)

      return mouse_name, n_shanks, n_trials, n_samples, sfreq

    def get_trial_dat(alldat, dat_LFP, i, shank, trial):
      "gets data from single trial"

      sig = dat_LFP[i]['lfp'][shank][trial]
      contrast_left = alldat[i]['contrast_left'][trial]
      contrast_right = alldat[i]['contrast_right'][trial]
      response = alldat[i]['response'][trial]
      response_time = alldat[i]['response_time'][trial]

      return sig, contrast_left, contrast_right, response, response_time

    def spec_param_features(freqs, spectrum, fm):
      "returns features of parameterized power spectrum"

      fm.fit(freqs, spectrum, freq_range=[1,50])
      ap_exp = fm.aperiodic_params_[1]
      offset = fm.aperiodic_params_[0]
      gamma = analysis.get_band_peak_fm(fm, [30,50])
      gamma_cf = gamma[0]
      gamma_pow = gamma[1]

      return ap_exp, offset, gamma_cf, gamma_pow

    g = FOOOFGroup(peak_width_limits=(2,6))

    recordings = np.array([])
    mouse_names = np.array([])
    n_recording = np.array([])
    shanks = np.array([])
    probe_locs = np.array([])
    trials = np.array([])
    contrast_lefts = np.array([])
    contrast_rights = np.array([])
    contrast_diffs = np.array([])
    responses = np.array([])
    response_times = np.array([])
    ap_exps = np.array([])
    offsets = np.array([])
    theta_cfs = np.array([])
    theta_pows = np.array([])
    theta_bands = np.array([])
    beta_cfs = np.array([])
    beta_pows = np.array([])
    beta_bands = np.array([])
    gamma_cfs = np.array([])
    gamma_pows = np.array([])
    gamma_bands = np.array([])

    for i in range(0, 1): #alldat.shape[0]

      mouse_name, n_shanks, n_trials, n_samples, sfreq = get_rec_dat(alldat, dat_LFP, i)
      print('loading recording '+ str(i) +', mouse: '+ mouse_name)

      for shank in range(0,n_shanks):

        probe_loc = dat_LFP[i]['brain_area_lfp'][shank]

        sig = dat_LFP[i]['lfp'][shank]
        freqs, spectra = spectral.compute_spectrum_welch(sig[:,25:125], sfreq, avg_type='median', nperseg=sfreq, noverlap=sfreq/1.5)
        fg.fit(freqs, spectra, freq_range=[1,50])
        ap_exp = fg.get_params('aperiodic_params', 'exponent')
        offset = fg.get_params('aperiodic_params', 'offset')
        theta = analysis.get_band_peak_fg(fg, [3,8]) #tuples of cf, pwr
        beta = analysis.get_band_peak_fg(fg, [13,30]) #tuples of cf, pwr
        gamma = analysis.get_band_peak_fg(fg, [30,50]) #tuples of cf, pwr
        theta_ext = utils.trim_spectrum(freqs, spectra, [8,13])
        theta_band = np.trapz(theta_ext[1], theta_ext[0])
        beta_ext = utils.trim_spectrum(freqs, spectra, [13,30])
        beta_band = np.trapz(beta_ext[1], beta_ext[0])
        gamma_ext = utils.trim_spectrum(freqs, spectra, [30,50])
        gamma_band = np.trapz(gamma_ext[1], gamma_ext[0])

        shanks = np.concatenate([shanks, np.tile(shank, n_trials)], axis=None)
        probe_locs = np.concatenate([probe_locs, np.tile(probe_loc, n_trials)], axis=None)
        trials = np.concatenate([trials, np.arange(0,n_trials)], axis=None)

        contrast_left = alldat[i]['contrast_left']
        contrast_right = alldat[i]['contrast_right']
        contrast_lefts = np.concatenate([contrast_lefts, contrast_left], axis=None)
        contrast_rights = np.concatenate([contrast_rights, contrast_right], axis=None)
        contrast_diffs = np.concatenate([contrast_diffs, contrast_left-contrast_right], axis=None)
        responses = np.concatenate([responses, alldat[i]['response']], axis=None)
        response_times = np.concatenate([response_times, alldat[i]['response_time']], axis=None)

        ap_exps = np.concatenate([ap_exps, ap_exp], axis=None)
        offsets = np.concatenate([offsets, offset], axis=None)
        theta_cfs = np.concatenate([theta_cfs, theta[:,0]], axis=None)
        theta_pows = np.concatenate([theta_pows, theta[:,1]], axis=None)
        theta_bands = np.concatenate([theta_bands, theta_band], axis=None)
        beta_cfs = np.concatenate([beta_cfs, beta[:,0]], axis=None)
        beta_pows = np.concatenate([beta_pows, beta[:,1]], axis=None)
        beta_bands = np.concatenate([beta_bands, beta_band], axis=None)
        gamma_cfs = np.concatenate([gamma_cfs, gamma[:,0]], axis=None)
        gamma_pows = np.concatenate([gamma_pows, gamma[:,1]], axis=None)
        gamma_bands = np.concatenate([gamma_bands, gamma_band], axis=None)

      # recording
      recordings = np.concatenate([recordings, np.tile(i,n_shanks*n_trials)], axis=None)
      # mouse name
      mouse_names = np.concatenate([mouse_names, np.tile(mouse_name,n_shanks*n_trials)], axis=None)


    keys = ['recording', 'mouse_name', 'shank', 'brain_area', 'trial', 'contrast_left',
            'contrast_right', 'contrast_diff', 'response', 'response_time', 'exponent',
            'offset', 'theta_cf', 'theta_pow', 'theta_band', 'beta_cf', 'beta_pow', 'beta_band',
            'gamma_cf', 'gamma_pow', 'gamma_band']
    values = [recordings, mouse_names, shanks, probe_locs, trials, contrast_lefts,
              contrast_rights, contrast_diffs, responses, response_times, ap_exps,
              offsets, theta_cfs, theta_pows, theta_bands, beta_cfs, beta_pows, beta_bands,
              gamma_cfs, gamma_pows, gamma_bands]
    exp_dict = dict(zip(keys, values))



    #     for trial in range(0,n_trials):

    #       sig, contrast_left, contrast_right, response, response_time = get_trial_dat(alldat, dat_LFP, i, shank, trial)
    #       freqs, spectrum = spectral.compute_spectrum_welch(sig[25:125], sfreq, avg_type='median', nperseg=sfreq, noverlap=sfreq/1.5) #wavelet instead for such a short time window?
    #       ap_exp, offset, gamma_cf, gamma_pow = spec_param_features(freqs, spectrum, fm)

    #       mouse_names.append(mouse_name)
    #       shanks.append(shank)
    #       probe_locs.append(probe_loc)
    #       trials.append(trial)
    #       contrast_lefts.append(contrast_left)
    #       contrast_rights.append(contrast_right)
    #       responses.append(response)
    #       response_times.append(response_times)
    #       ap_exps.append(ap_exp)
    #       offsets.append(offset)
    #       # beta
    #       gamma_cfs.append(gamma_cf)
    #       gamma_pows.append(gamma_pow)

    # keys = ['mouse_name', 'shank', 'brain_area', 'trial', 'contrast_left', 'contrast_right', 'response', 'response_time', 'exponent', 'offset', 'gamma_cf', 'gamma_pow']
    # lists = [mouse_names, shanks, probe_locs, trials, contrast_lefts, contrast_rights, responses, response_times, ap_exps, offsets, gamma_cfs, gamma_pows]

    # exp_dict = dict(zip(keys, lists))

    df = pd.DataFrame(exp_dict)
