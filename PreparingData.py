import sounddevice as sd
from scipy.io.wavfile import write

def record_audio_and_save(save_path, n_times=50):

    input("To start recording press Enter: ")
    i = 0
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press [Enter] to record next or [ctrl + C] to stop ({i + 1}/{n_times}): ")


print("Recording the Wake Word:\n")
record_audio_and_save("audio_data/", n_times=100) 
