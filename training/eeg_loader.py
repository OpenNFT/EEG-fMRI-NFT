import os
import json
import numpy as np
import mne
from scipy.interpolate import CubicSpline
from stockwell import st
import multiprocessing as mp
import warnings
import glob
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_channel_params(config_path):
    """
    Load parameters for all channels from subject_params.json
    
    Returns:
        bands: dict, channel_name -> band boundaries (B+1,)
        norm_means: dict, channel_name -> norm_mean (B,)
        norm_sds: dict, channel_name -> norm_sd (B,)
    """
    with open(config_path, 'r') as f:
        all_params = json.load(f)
    
    bands = {}
    norm_means = {}
    norm_sds = {}
    
    for channel, params in all_params.items():
        bands[channel] = np.array(params['bands'])
        norm_means[channel] = np.array(params['norm_mean'])
        norm_sds[channel] = np.array(params['norm_sd'])
    
    return bands, norm_means, norm_sds

def init_pool_worker():
    pass

def cubic_spline_interpolator(signal, source_f, target_f, axis=-1):
    n_points = signal.shape[axis]
    interpolator = CubicSpline(np.arange(n_points), signal, axis=axis)
    target_length = int(n_points * target_f / source_f)
    target_grid = np.linspace(0, n_points - 1, target_length)
    return interpolator(target_grid)

def average_into_bands(stock_power, stock_freqs, band_boundaries):
    band_means = []
    
    if stock_power.ndim == 2:
        stock_power = stock_power[np.newaxis, ...]
    
    for i in range(len(band_boundaries) - 1):
        band_idx = (stock_freqs >= band_boundaries[i]) & (stock_freqs < band_boundaries[i + 1])
        selected = stock_power[:, band_idx, :]
        
        # Handle zero values
        nonzero_vals = selected[selected != 0]
        if nonzero_vals.size > 0:
            min_nonzero = nonzero_vals.min()
            selected[selected == 0] = min_nonzero
        
        selected = np.log10(selected)
        band_avg = selected.mean(axis=1, keepdims=True)
        band_means.append(band_avg)
    
    return np.concatenate(band_means, axis=1)

def stockwell_one_channel(data, fmin_samples, fmax_samples, df, band_boundaries,
                          gamma=15, window_type='kazemi'):
    data = data.ravel()
    n = len(data)
    
    # Mirror edges
    extended_data = np.concatenate([data[::-1], data, data[::-1]])
    
    stock = st.st(extended_data, fmin_samples, fmax_samples,
                  gamma, win_type=window_type)
    stock = stock[:, n:2 * n]
    stock = np.abs(stock)
    freqs = np.arange(stock.shape[0]) * df
    
    bands_result = average_into_bands(stock, freqs, band_boundaries)
    return bands_result

def preprocess_data(sample, sample_rate, pool, bands_dict, norm_means_dict, norm_sds_dict,
                    channel_names, fmin=0, fmax=60, target_freq=4):
    df = 0.05
    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)
    
    # Process each channel with its specific parameters
    banded_list = []
    for i, ch_data in enumerate(sample):
        ch_name = channel_names[i]
        bands = bands_dict[ch_name]
        result = stockwell_one_channel(ch_data, fmin_samples, fmax_samples, df, bands)
        banded_list.append(result)
    
    stacked = np.stack(banded_list, axis=0).squeeze()
    C, B, raw_pts = stacked.shape
    
    flat_for_interp = stacked.transpose(1, 0, 2).reshape(C * B, raw_pts)
    interpolated = cubic_spline_interpolator(flat_for_interp, sample_rate, target_freq).squeeze()
    
    CxB_T = interpolated.reshape(B, C, -1).transpose(1, 0, 2)  # (C, B, T')
    
    # Normalize using per-channel parameters
    normalized = np.zeros_like(CxB_T)
    for i, ch_name in enumerate(channel_names):
        norm_mean = norm_means_dict[ch_name].reshape(-1, 1)  # (B, 1)
        norm_sd = norm_sds_dict[ch_name].reshape(-1, 1)      # (B, 1)
        normalized[i] = (CxB_T[i] - norm_mean) / norm_sd
    
    return normalized

class DataCruncher:
    def __init__(self, bands_dict, norm_means_dict, norm_sds_dict, channel_names, n_procs=mp.cpu_count()):
        self.bands_dict = bands_dict
        self.norm_means_dict = norm_means_dict
        self.norm_sds_dict = norm_sds_dict
        self.channel_names = channel_names
        self.pool = mp.Pool(processes=n_procs, initializer=init_pool_worker)
    
    def __call__(self, sample, sampling_rate):
        return preprocess_data(
            sample,
            sampling_rate,
            self.pool,
            self.bands_dict,
            self.norm_means_dict,
            self.norm_sds_dict,
            self.channel_names
        )
    
    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

def load_and_concatenate_eeg(eeg_paths, trigger_label='TR'):
    """Load and concatenate multiple EEG files while preserving trigger positions"""
    all_raws = []
    trigger_offsets = []
    total_samples = 0
    
    print(f"Loading and concatenating {len(eeg_paths)} EEG files...")
    for path in eeg_paths:
        print(f"  Loading {os.path.basename(path)}")
        raw = mne.io.read_raw_eeglab(path, preload=True)
        all_raws.append(raw)
        trigger_offsets.append(total_samples)
        total_samples += raw.n_times
    
    # Concatenate all files
    concatenated = mne.concatenate_raws(all_raws)
    sfreq = concatenated.info['sfreq']
    channel_names = concatenated.ch_names
    
    # Collect triggers from all files with correct offsets
    all_events = []
    for i, raw in enumerate(all_raws):
        events, event_id = mne.events_from_annotations(raw)
        if trigger_label in event_id:
            run_trigger_events = events[events[:, 2] == event_id[trigger_label]]
            
            if run_trigger_events.size > 0:
                # Apply offset to sample numbers
                offset_events = run_trigger_events.copy()
                offset_events[:, 0] += trigger_offsets[i]
                all_events.append(offset_events)
    
    if not all_events:
        raise ValueError(f"No trigger events found for label '{trigger_label}' in any files")
    
    all_events = np.concatenate(all_events)
    trigger_times = all_events[:, 0] / sfreq
    
    return concatenated, trigger_times, sfreq, channel_names

def process_eeg_session(eeg_paths, output_dir, cruncher,
                        warmup_time=0, window_duration_sec=12,
                        trigger_label='TR', save_outputs=True,
                        apply_preprocessing=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and concatenate all EEG files for this session
    concatenated, trigger_times, sfreq, eeg_channel_names = load_and_concatenate_eeg(
        eeg_paths, trigger_label
    )
    
    # Verify all EEG channels have parameters
    param_channels = set(cruncher.channel_names)
    eeg_channels = set(eeg_channel_names)
    
    if param_channels != eeg_channels:
        missing_params = eeg_channels - param_channels
        missing_eeg = param_channels - eeg_channels
        
        if missing_params:
            print(f"Warning: EEG channels without parameters: {sorted(missing_params)}")
        if missing_eeg:
            print(f"Warning: Parameters for non-existent EEG channels: {sorted(missing_eeg)}")
    
    # Reorder EEG channels to match parameter order
    channel_order = [eeg_channel_names.index(ch) for ch in cruncher.channel_names if ch in eeg_channel_names]
    eeg_data = concatenated.get_data()
    
    window_samples = int(window_duration_sec * sfreq)
    
    print("Extracting EEG windows from concatenated data...")
    eeg_windows = []
    valid_trigger_times = []
    for t in trigger_times:
        if t < warmup_time:
            continue
        start = int(t * sfreq)
        end = start + window_samples
        if end > eeg_data.shape[1]:
            continue
        eeg_windows.append(eeg_data[:, start:end])
        valid_trigger_times.append(t)
    
    eeg_windows = np.array(eeg_windows)
    print(f"Extracted {len(eeg_windows)} EEG windows (shape {eeg_windows.shape})")
    
    if apply_preprocessing:
        print("Preprocessing EEG data...")
        preprocessed = []
        for i, win in enumerate(eeg_windows):
            # Select only channels that have parameters
            win = win[channel_order, :] if channel_order else win
            print(f"  Processing window {i+1}/{len(eeg_windows)}")
            preprocessed.append(cruncher(win, sfreq))
        preprocessed = np.stack(preprocessed)
    else:
        preprocessed = eeg_windows
    
    if save_outputs:
        # Save metadata
        metadata = {
            "eeg_files": [os.path.basename(p) for p in eeg_paths],
            "sfreq": sfreq,
            "channel_names": list(cruncher.channel_names),
            "window_duration_sec": window_duration_sec,
            "trigger_label": trigger_label,
            "total_triggers": len(trigger_times),
            "valid_triggers": len(valid_trigger_times),
            "window_count": len(eeg_windows),
            "warmup_time": warmup_time
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data
        np.save(os.path.join(output_dir, 'eeg_windows.npy'), eeg_windows)
        if apply_preprocessing:
            np.save(os.path.join(output_dir, 'eeg_preprocessed.npy'), preprocessed)
    
    return eeg_windows, preprocessed

def process_all_sessions(config_path, eeg_base_dir, output_base_dir):
    """Process all sessions found in the EEG directory"""
    # Load channel parameters
    bands_dict, norm_means_dict, norm_sds_dict = load_channel_params(config_path)
    param_channels = sorted(bands_dict.keys())
    
    # Find all EEG files
    eeg_files = glob.glob(os.path.join(eeg_base_dir, "*.set"))
    print(f"Found {len(eeg_files)} EEG files in directory {eeg_base_dir}")
    
    if not eeg_files:
        print("No EEG files found. Exiting.")
        return
    
    # Extract unique session identifiers
    session_files = {}
    for file_path in eeg_files:
        file_name = os.path.basename(file_path)
        
        # Extract session key - this handles multiple naming conventions
        parts = file_name.split('_')
        session_key = None
        
        # Try to find sub-xxx_ses-yyy pattern
        for i in range(len(parts)-1):
            if parts[i].startswith("sub-") and parts[i+1].startswith("ses-"):
                session_key = f"{parts[i]}_{parts[i+1]}"
                break
        
        # If not found, use first 2 parts as session key
        if not session_key and len(parts) >= 2:
            session_key = f"{parts[0]}_{parts[1]}"
        
        if session_key:
            if session_key not in session_files:
                session_files[session_key] = []
            session_files[session_key].append(file_path)
    
    print(f"Identified {len(session_files)} sessions")
    
    for session_key, eeg_paths in session_files.items():
        session_output_dir = os.path.join(output_base_dir, session_key)
        
        # Skip if already processed
        if os.path.exists(os.path.join(session_output_dir, 'metadata.json')):
            print(f"\nSession {session_key} already processed. Skipping.")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing session: {session_key}")
        print(f"  Found {len(eeg_paths)} EEG files")
        
        try:
            # Create cruncher with parameter channels
            cruncher = DataCruncher(
                bands_dict, 
                norm_means_dict, 
                norm_sds_dict, 
                param_channels,
                n_procs=4
            )
            
            # Process session
            process_eeg_session(
                eeg_paths,
                session_output_dir,
                cruncher,
                apply_preprocessing=True
            )
            
            print(f"  Successfully processed session {session_key}")
            cruncher.close()
            
        except Exception as e:
            print(f"  ! Error processing session {session_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'cruncher' in locals():
                cruncher.close()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Examine a collection of .set files and prepare samples for training regressors.")
    parser.add_argument("directory", type=str, help="Path to the directory containing .set files. Non-recursive.")
    parser.add_argument("json_path", type=str, help="JSON file containing information about frequency bands, generated by psd.py.")
    parser.add_argument("output_base_dir", type=str, help="Output directory.")

    args = parser.parse_args()
    config_path = args.json_path
    eeg_base_dir = args.directory
    output_base_dir = args.output_base_dir

    os.makedirs(output_base_dir, exist_ok=True)
    process_all_sessions(config_path, eeg_base_dir, output_base_dir)
    print("\nProcessing complete!")
