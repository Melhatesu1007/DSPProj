import streamlit as st
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, filtfilt, find_peaks

# --- Function Definitions ---
def load_pcg_signal(filename):
    """Loads a PCG signal from an audio file."""
    try:
        fs, audio = wavfile.read(filename)
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)  # Ensure it's a single channel (mono)
    except Exception as e:
        audio = np.array([])
        fs = None
        st.error(f'Error loading audio file: {e}')
    return audio, fs

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal

def calculate_energy_envelope(signal):
    """Calculates the energy envelope of the signal."""
    energy_envelope = np.abs(signal)**2
    return energy_envelope

def calculate_heart_rate(energy_envelope, fs):
    """Estimates heart rate from the energy envelope."""

    peaks, _ = find_peaks(energy_envelope, prominence=np.max(energy_envelope)/3) # location-based

    if len(peaks) < 2:
        heart_rate = np.nan
        return heart_rate
    peak_times = peaks / fs
    ibi = np.diff(peak_times)  # inter-beat intervals
    if len(ibi) > 0:
        heart_rate = 60 / np.mean(ibi)
    else:
        heart_rate = np.nan
    return heart_rate

# --- Main Script ---
def main():
    st.title('PCG Signal Analysis')

    uploaded_file = st.file_uploader("Upload PCG Audio File (WAV)", type=['wav'])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio, fs = load_pcg_signal("temp_audio.wav")


        if len(audio) > 0 and fs is not None:
            # 2. Filtering Parameters
            lowcut = 25  # Hz
            highcut = 600 # Hz
            filter_order = 5

            # 3. Apply Bandpass Filter
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, filter_order)

            # 4. Calculate Energy Envelope
            energy_envelope = calculate_energy_envelope(filtered_signal)

            # 5. Calculate Heart Rate
            heart_rate = calculate_heart_rate(energy_envelope, fs)

            # --- Display Results ---
            st.header('PCG Signal Analysis Output')

            # Subplot 1: Filtered PCG Signal
            st.subheader('Filtered PCG Signal')
            t_audio = np.arange(len(filtered_signal)) / fs
            st.line_chart(filtered_signal)
            st.text("Time (s)")
            st.text("Amplitude")

            # Subplot 2: Energy Envelope
            st.subheader('Energy Envelope')
            t_envelope = np.arange(len(energy_envelope)) / fs
            st.line_chart(energy_envelope)
            st.text("Time (s)")
            st.text("Energy")

            # Subplot 3: Heart Rate Display
            st.subheader('Heart Rate Display')
            if not np.isnan(heart_rate):
                st.markdown(f'**Estimated Heart Rate:  {heart_rate:.2f} BPM**')
            else:
                st.markdown('**Could not estimate heart rate.**')
        else:
            st.write('No audio file loaded.')

if __name__ == "__main__":
    main()
