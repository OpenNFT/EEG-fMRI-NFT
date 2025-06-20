import pylsl
from collections import deque
import threading
import time
import numpy as np
import warnings


def get_channel_indexes(requested_channels, stream_channels):
    indexes = []
    lowered_stream = [k.lower() for k in stream_channels]
    lowered_requested = [k.lower() for k in requested_channels]
    for channel in lowered_requested:
        if channel in lowered_stream:
            try:
                index = int(np.argwhere([k == channel for k in lowered_stream]).squeeze())
                indexes.append(index)
            except Exception as e:
                raise IndexError(f'Could not find channel {channel} in {lowered_stream} (case insensitive) '
                                 f'or multiple channels sharing the same name')
        else:
            warnings.warn(f"Channel {channel} not found in {lowered_stream}")
    return indexes


class LSLStreamListener:
    def __init__(self, buffer_duration_seconds=12, stream_name=None, selected_channels=None, data_cruncher=None):
        """
        Args:
            buffer_duration: Time window to retain (seconds)
            max_samples: Max samples to buffer (safety limit)
            stream_name: Target LSL stream name
        """
        self.buffer = deque(maxlen=1)
        self.buffer_duration = buffer_duration_seconds
        self.stream_name = stream_name
        self.running = False
        self.thread = None
        self.sample_rate = None
        self.last_timestamp = 0
        self.resolved = False
        self.selected_channels = selected_channels
        self.selected_channel_indexes = None
        self.data_cruncher = data_cruncher
        self.updating = False

    def set_stream_name(self, stream_name):
        print(f"Stream name set to {stream_name}")
        self.stream_name = stream_name
        if self.resolved:
            self.stop()
            self.start()

    def _connect_to_stream(self):
        """Connect to LSL stream with retry logic"""
        print("Searching for LSL stream...")
        while self.running:
            if self.stream_name is not None:
                streams = pylsl.resolve_streams(1)
                print(f"Looking for stream: {self.stream_name}")
                print("Detected streams:", [k.name() for k in streams])
                streams = [k for k in streams if k.name() == self.stream_name]
            else:
                streams = False
            if streams:
                inlet = pylsl.StreamInlet(streams[0])
                self.sample_rate = inlet.info().nominal_srate()
                max_samples = int(self.sample_rate * self.buffer_duration)
                self.buffer = deque(maxlen=max_samples)
                print(f"Connected to {self.stream_name} @ {self.sample_rate}Hz")
                if self.selected_channels is not None:
                    self.selected_channel_indexes = get_channel_indexes(
                        self.selected_channels, inlet.info().get_channel_labels())
                self.resolved = True
                return inlet
            time.sleep(1)

    def _listen(self):
        """Main listening loop"""
        inlet = self._connect_to_stream()
        while self.running and inlet:

            samples, timestamps = inlet.pull_chunk()
            if samples:
                self.updating = True
                self.buffer.extend(samples)
                self.last_timestamp = timestamps[-1]

                # Trim buffer to time window
                if self.sample_rate and self.buffer_duration:
                    max_samples = int(self.sample_rate * self.buffer_duration)
                    while len(self.buffer) > max_samples:
                        self.buffer.popleft()
                self.updating = False
            time.sleep(0.01)

    def start(self):
        """Start background listening thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._listen)
            self.thread.start()

    def stop(self):
        """Stop listening thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def get_buffer(self):
        """Get current buffer as numpy array"""
        # guard clauses, don't pull data if listener is off
        if not self.resolved:
            return None
        # And wait for the listener to update if trying to get while updating
        while self.updating:
            time.sleep(0.001)

        array = np.array(self.buffer.copy())
        if self.selected_channel_indexes is not None:

            array = array[:, self.selected_channel_indexes]
        if self.data_cruncher is not None:
            array = self.data_cruncher(array, self.sample_rate)

        return array

    def __del__(self):
        self.stop()


if __name__ == '__main__':
    listener = LSLStreamListener(stream_name='LSLExampleAmp', selected_channels=['Fp1', 'cz', 'c3'])
    listener.start()
    time.sleep(5)
    while True:
        time.sleep(0.1)
        result = listener.get_buffer()
        if result is not None:
            print(result.shape)
            print(result.mean())
