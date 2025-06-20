import os
import glob
import json
import logging
import numpy as np
import mne                
from scipy import signal
from typing import List, Tuple, Dict
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_eeg_from_set(set_path: str) -> Tuple[np.ndarray, float, List[str]]:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    return raw.get_data(), raw.info['sfreq'], raw.info['ch_names']

def concatenate_eeg_sets(set_files: List[str]) -> Tuple[np.ndarray, float, List[str]]:
    all_data = None
    sfreq_ref = None
    ch_names_ref = None

    for idx, filepath in enumerate(sorted(set_files)):
        data, sfreq, ch_names = load_eeg_from_set(filepath)

        if idx == 0:
            all_data = data.copy()
            sfreq_ref = sfreq
            ch_names_ref = ch_names
            logger.info(f"Initialized with {os.path.basename(filepath)}: {data.shape}")
        else:
            # Validate consistency
            if not np.isclose(sfreq, sfreq_ref):
                raise ValueError(
                    f"Sampling rate mismatch: '{os.path.basename(filepath)}' "
                    f"has {sfreq} Hz, expected {sfreq_ref} Hz."
                )
            if ch_names != ch_names_ref:
                raise ValueError(
                    f"Channel name mismatch in '{os.path.basename(filepath)}'."
                )
            
            all_data = np.hstack((all_data, data))
            
            if (idx + 1) % 10 == 0:  # Log progress every 10 files
                logger.info(f"Processed {idx + 1}/{len(set_files)} files")

    logger.info(f"Concatenation complete: {all_data.shape}")
    return all_data, sfreq_ref, ch_names_ref


def compute_psd_all_channels(
    eeg_data: np.ndarray, 
    sfreq: float, 
    ch_names: List[str], 
    nperseg: int = None, 
    noverlap: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD for all channels using Welch's method.
    
    Args:
        eeg_data: EEG data array (n_channels, n_samples)
        sfreq: Sampling frequency (Hz)
        ch_names: List of channel names
        nperseg: Length of each segment for Welch's method
        noverlap: Number of points to overlap between segments
        
    Returns:
        freqs: Frequency vector
        psd_array: PSD array (n_channels, n_freqs)
    """
    n_channels, n_samples = eeg_data.shape

    if nperseg is None:
        nperseg = int(4 * sfreq)  
    if noverlap is None:
        noverlap = nperseg // 2    # 50% overlap

    logger.info(f"Computing PSD for {n_channels} channels...")
    
    # Compute PSD for first channel to get frequency vector
    freqs, _ = signal.welch(
        eeg_data[0, :], 
        fs=sfreq, 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling='density'
    )
    
    psd_array = np.zeros((n_channels, len(freqs)))
    
    # Compute PSD for all channels
    for ch_idx in range(n_channels):
        _, psd_array[ch_idx, :] = signal.welch(
            eeg_data[ch_idx, :],
            fs=sfreq, 
            nperseg=nperseg, 
            noverlap=noverlap,
            scaling='density'
        )
    
    logger.info(f"PSD computation complete: {psd_array.shape}")
    return freqs, psd_array


def get_band_boundaries_from_arrays(
    psd_array: np.ndarray, 
    freqs: np.ndarray, 
    num_bands: int, 
    fmin: float, 
    fmax: float
) -> Dict[int, List[float]]:
    """
    Divide frequency range into equal-energy bands for each channel.
    
    Args:
        psd_array: PSD array (n_channels, n_freqs)
        freqs: Frequency vector
        num_bands: Number of frequency bands
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        
    Returns:
        band_boundaries: Dictionary mapping channel index to band boundaries
    """
    valid_mask = (freqs >= fmin) & (freqs <= fmax)
    idx_min = np.where(valid_mask)[0].min()
    idx_max = np.where(valid_mask)[0].max()
    
    freqs_roi = freqs[idx_min:idx_max + 1]
    psd_roi = psd_array[:, idx_min:idx_max + 1]
    
    n_channels = psd_roi.shape[0]
    band_boundaries = {}

    for ch in range(n_channels):
        # Normalize power to sum to 1
        power = psd_roi[ch, :]
        power = power / np.sum(power)
        
        cum_energy = np.cumsum(power)
        energy_per_band = 1.0 / num_bands

        boundaries = [fmin]
        for b in range(1, num_bands):
            target = b * energy_per_band
            idx_boundary = np.argmin(np.abs(cum_energy - target))
            freq_boundary = freqs_roi[idx_boundary]
            boundaries.append(freq_boundary)
        boundaries.append(fmax)

        band_boundaries[ch] = boundaries

    return band_boundaries


def compute_band_stats(
    psd_array: np.ndarray, 
    freqs: np.ndarray, 
    band_boundaries: Dict[int, List[float]], 
    ch_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute band statistics for all channels.
    
    Args:
        psd_array: PSD array (n_channels, n_freqs)
        freqs: Frequency vector
        band_boundaries: Dictionary mapping channel index to band boundaries
        ch_names: List of channel names
        
    Returns:
        band_means: Mean values for each band (n_channels, n_bands)
        band_stds: Standard deviations for each band (n_channels, n_bands)
    """
    n_channels = psd_array.shape[0]
    n_bands = len(band_boundaries[0]) - 1

    band_means = np.zeros((n_channels, n_bands))
    band_stds = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        # Z-score normalization
        psd_ch = psd_array[ch, :]
        psd_ch = (psd_ch - np.mean(psd_ch)) / np.std(psd_ch)
        
        boundaries = band_boundaries[ch]
        for b in range(n_bands):
            f_low = boundaries[b]
            f_high = boundaries[b + 1]
            mask = (freqs >= f_low) & (freqs <= f_high)
            
            if not np.any(mask):
                # Handle edge case where no exact frequency matches
                idx_low = np.argmin(np.abs(freqs - f_low))
                idx_high = np.argmin(np.abs(freqs - f_high))
                idxs = np.arange(min(idx_low, idx_high), max(idx_low, idx_high) + 1)
            else:
                idxs = np.where(mask)[0]

            values = psd_ch[idxs]
            band_means[ch, b] = np.mean(values)
            band_stds[ch, b] = np.std(values, ddof=0)

    return band_means, band_stds


def process_eeg_folder(
    eeg_folder: str,
    output_json_path: str,
    num_bands: int = 10,
    fmin: float = 0.0,
    fmax: float = 60.0
) -> None:
    """
    Process all EEG files in a folder and save PSD parameters to JSON.
    
    Args:
        eeg_folder: Path to folder containing .set files
        output_json_path: Path for output JSON file
        num_bands: Number of frequency bands
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
    """
    pattern = os.path.join(eeg_folder, "*.set")
    set_files = sorted(glob.glob(pattern))
    
    if not set_files:
        raise FileNotFoundError(f"No '.set' files found in '{eeg_folder}'.")
    
    logger.info(f"Found {len(set_files)} .set files")

    # Concatenate all EEG files
    concatenated_data, sfreq, ch_names = concatenate_eeg_sets(set_files)
    duration_minutes = concatenated_data.shape[1] / sfreq / 60
    logger.info(f"Total duration: {duration_minutes:.1f} minutes")

    freqs, psd = compute_psd_all_channels(concatenated_data, sfreq, ch_names)
    band_bounds = get_band_boundaries_from_arrays(psd, freqs, num_bands, fmin, fmax)
    band_means, band_stds = compute_band_stats(psd, freqs, band_bounds, ch_names)

    # Prepare output data
    output_data = {}
    for ch, ch_name in enumerate(ch_names):
        output_data[ch_name] = {
            "bands": band_bounds[ch],
            "norm_mean": band_means[ch].tolist(),
            "norm_sd": band_stds[ch].tolist()
        }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to: {output_json_path}")


def main(eeg_folder, output_json_path, N, fmin, fmax) -> None:

    try:
        process_eeg_folder(
            eeg_folder=eeg_folder,
            output_json_path=output_json_path,
            num_bands=N,
            fmin=fmin,
            fmax=fmax
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examine a collection of .set files and extract N bands of equal area.")
    parser.add_argument("directory", type=str, help="Path to the directory containing .set files. Non-recursive.")
    parser.add_argument("json_path", type=str, help="Where to write the JSON file with the bands as an output.")
    parser.add_argument("N", type=int, help="Number of bands.", default=10)
    parser.add_argument("fmin", type=float, help="Minimum frequency (e.g., Hz).", default=0)
    parser.add_argument("fmax", type=float, help="Maximum frequency (e.g., Hz).", default=60.0)

    args = parser.parse_args()
    main(args.directory, args.json_path, args.N, args.fmin, args.fmax)
