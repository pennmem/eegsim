# EEG Simulator

This repository contains the EEG Generator class EEGGen, which provides physiologically authentic 1/f^exponent pink noise, and the ability to add transient oscillatory components and delta spikes.

Example usage is as follows:

    from eegsim import EEGGen
    import matplotlib.pyplot as plt

    gen = EEGGen()
    # Set the pink noise amplitude.
    gen.EnablePinkNoise(30)
    # 7Hz, amplitude 100, at 0.85s with 3 reps.
    # Note:  Generator time starts from 0.
    gen.AddWave(7, 100, 0.85, 3)
    # Generate a 1.5s long numpy array.
    eeg = gen.Generate(1.5)
    # Recenter 0 time in the middle.
    # The added oscillation now starts at 0.1s.
    time = gen.time_coords - 0.75
    # Plot the result.
    plt.plot(time, eeg)
    plt.show()

