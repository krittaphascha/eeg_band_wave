import numpy as np
import matplotlib.pyplot as plt

def sine_wave_generator(sfreq,duration, wave_band='delta', 
                        white_noise = 0.2,
                        verbose=False, plot=None):

    # randomize amplitude, phase, and frequency
    times = np.arange(0,duration,1/sfreq)
    amplitude = 2 * np.random.rand()
    phase = np.random.uniform(0, 2 * np.pi)

    # generate wave frequency
    if wave_band == 'delta':
        sin_freq = np.random.uniform(0.5, 4) # 0.5 - 4 Hz
    if wave_band == 'theta':
        sin_freq = np.random.uniform(4, 8) # 4 - 8 Hz
    if wave_band == 'alpha':
        sin_freq = np.random.uniform(8, 13) # 8 - 13 Hz
    if wave_band == 'beta':
        sin_freq = np.random.uniform(13, 30) # 13 - 30 Hz
    if wave_band == 'gamma':
        sin_freq = np.random.uniform(30, 100) # 30 - 100 Hz
    
    # generate wave 
    signals = amplitude * np.sin(2 * np.pi * sin_freq * times + phase) + white_noise * np.random.randn(len(times))

    if verbose:
        print("Wave amplitude:", amplitude)
        print("Wave frequency:", sin_freq)
        print(f"Wave phase: {phase/np.pi} * pi")

    if plot is not None: # require integer for plot points
        plt.plot(times[:plot], signals[:plot])
        plt.show()
    
    return signals


def nonst_narrow_wave_generator(sfreq, duration, wave_band='delta', fwhm_ratio=0.5, white_noise=0.2):
    
    freq = np.linspace(0, sfreq, sfreq*duration)  

    if wave_band == 'delta':
        peakfreq = 2
        fwhm = fwhm_ratio * peakfreq
    if wave_band == 'theta':
        peakfreq = 6
        fwhm = fwhm_ratio * peakfreq
    if wave_band == 'alpha':
        peakfreq = 10
        fwhm = fwhm_ratio * peakfreq
    if wave_band == 'beta':
        peakfreq = 20
        fwhm = fwhm_ratio * peakfreq
    if wave_band == 'gamma':
        peakfreq = 60
        fwhm = fwhm_ratio * peakfreq

    s = fwhm * (2*np.pi-1)/(4*np.pi)
    x = freq - peakfreq
    fg = np.exp(-.5*(x/s)**2)

    fc = np.random.rand(1,sfreq*duration) * np.exp(1j*2*np.pi*np.random.rand(1,sfreq*duration))

    fc = fc * fg

    signals = np.real(np.fft.ifft(fc))

    return signals


def generating_signal_batchs(wave_generator, n_samples, n_channels, sfreq, duration, wave_band,
                             random_channels=False,
                             white_noise=0.2):

    # generate signals
    whole_signals = np.zeros((n_samples, n_channels, sfreq*duration))

    for i in range(n_samples):

        signals = wave_generator(sfreq, duration, wave_band=wave_band, white_noise=white_noise)
        signals = np.expand_dims(signals, axis=0)
        for j in range(n_channels):
            if random_channels:
                signals = wave_generator(sfreq, duration, wave_band=wave_band)
                signals = np.expand_dims(signals, axis=0)
            whole_signals[i, j, :] = signals
    
    return whole_signals