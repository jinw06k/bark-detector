import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "dog_barking_copy.wav"

signal, sr = librosa.load(file, sr=22050)

# librosa.display.waveshow(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()


# Fast Fourier Transform
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

half_frequency = frequency[:int(len(frequency)/2)]
half_magnitude = magnitude[:int(len(magnitude)/2)]

# plt.plot(half_frequency, half_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()


# Short Time Fourier Transform
stft = librosa.core.stft(signal, hop_length=512, n_fft=2048)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=512)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()


# Mel Frequency Cepstral Coefficients (MFCCs)
MFCCs = librosa.feature.mfcc(y=signal, n_fft=2048, hop_length=512, n_mfcc=13)

librosa.display.specshow(MFCCs, sr=sr, hop_length=512)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()