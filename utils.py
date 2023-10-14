import numpy as np
import matplotlib.pyplot as plt

def visualize_signal(signal, length=1000):

    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(fft), 1.0/200)

    fig, axs = plt.subplots(2, 1, figsize=(20,10))

    # plot waves
    axs[0].set_title('Wave signals')
    axs[0].plot(signal)
    axs[0].set_xlabel('Time (ms)')
    
    # Plot the FFT
    axs[1].set_title('Frequency spectrum waves')
    axs[1].plot(np.abs(freq), np.abs(fft))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Power')
    plt.show()