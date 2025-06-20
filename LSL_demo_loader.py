## LSL ##

import nest_asyncio
import time
from pylsl import StreamInfo, StreamOutlet, local_clock
import mne
import os
import shutil
import multiprocessing

nest_asyncio.apply()
eeg_data = '/path/to/file.vhdr'
mri_data_dir_source = '/home/example/Documents/files/loose_nii'
mri_data_dir_target = '/media/example'
TR = 1
warmup_time = 20  # seconds, simulates starting stream and API before starting the MRI sequence

for old_file in os.listdir(mri_data_dir_target):
    os.remove(os.path.join(mri_data_dir_target, old_file))


def copy_files_periodically(src_dir, dest_dir, interval=1.0):
    """Copies files from src_dir to dest_dir in order, at regular intervals."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = sorted(f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)))

    for file in files:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy2(src_path, dest_path)
        print(f"Copied {file} to {dest_dir}")
        time.sleep(interval)


def start_copy_process(src_dir, dest_dir, interval=1.0):
    """Starts the file copying process in a separate process."""
    process = multiprocessing.Process(target=copy_files_periodically, args=(src_dir, dest_dir, interval))
    process.start()
    return process


# Load EEG data
raw = mne.io.read_raw_brainvision(
    eeg_data, preload=False
)

# List of channels
channels_list = raw.ch_names

# Create StreamInfo object
info = StreamInfo('EEG_Stream', 'EEG', len(channels_list), raw.info['sfreq'], 'float32', 'myuid2424')

# Add metadata
chns = info.desc().append_child("channels")
for label in channels_list:
    ch = chns.append_child("channel")
    ch.append_child_value("label", label)
    ch.append_child_value("unit", "microvolts")
    ch.append_child_value("type", "EEG")

# Create StreamOutlet
outlet = StreamOutlet(info, 32, 360)

# Extract data and start streaming
data, times = raw[:]

print("Starting pre-paradigm EEG stream")
for k in range(0, int(warmup_time * raw.info['sfreq'])):
    sample = data[:, k].tolist()
    timestamp = local_clock()
    outlet.push_sample(sample, timestamp)
    time.sleep(1 / raw.info['sfreq'])

print('Starting MRI copy...')
mri_copy_process = start_copy_process(mri_data_dir_source, mri_data_dir_target, interval=TR)

print("Now streaming EEG data...")
for i in range(data.shape[1]):
    sample = data[:, i].tolist()
    timestamp = local_clock()
    outlet.push_sample(sample, timestamp)
    time.sleep(1 / raw.info['sfreq'])

mri_copy_process.join()
print('Finished')
