import numpy as np


class EEGComponent():
  '''A class for storing the parameters of a sinusoidal component of
     an EEG signal.  It can be passed to EEGGen with AddComp.'''
  def __init__(self, freq, amp, start=0, reps=np.inf):
    '''Specify the frequency, amplitude, start time, and the number of
       repetitions for a sinusoidal component.'''
    self.freq = freq
    self.amp = amp
    self.start = start
    if reps != np.inf:
      reps = np.round(reps)
    self.reps = reps

  def AtTime(self, time):
    '''Returns the numerical value of this sinusoidal component at the
       given time point.'''
    cycle = (time-start)*freq
    if (time < start) or (cycle > self.reps):
      return 0

    return self.amp * np.sin(2*np.pi*cycle)

  def AddToEEG(self, eeg, sr):
    '''Adds the component to the given eeg numpy array using the provided
       sampling rate.'''
    startN = int(round(self.start * sr))
    stopN = startN + np.round((self.reps / self.freq) * sr)
    if stopN > len(eeg):
      stopN = len(eeg)
    stopN = int(stopN)
    width = stopN - startN
    tvals = np.linspace(0, width/sr, width)
    eeg[startN:stopN] += self.amp * np.sin(2*np.pi*tvals*self.freq)


class EEGGen():
  '''An EEG Generator class for simulating EEG signals with pink noise and
     transient sinusoidal components.  Units should be Hz and seconds.'''
  def __init__(self, sampling_rate=250):
    '''Initializes the generator with the given sampling rate.'''
    self.sampling_rate = sampling_rate
    self.components = []
    self.spikes = []
    self.pinknoise_amp = 0
    self.pinknoise_exponent = 0.51
    self.time_coords = None

  def AddComp(self, component):
    '''Add an EEGComponent object to this generator.'''
    self.components.append(component)

  def AddWave(self, freq, amp, start=0, reps=np.inf):
    '''Add a sinusoidal oscillation to this generator with the given
       frequency, amplitude, start time, and repetition counts.'''
    self.AddComp(EEGComponent(freq, amp, start, reps))

  def AddSpike(self, time, amp):
    '''Add a delta spike at the given time point.'''
    self.spikes.append((time, amp))

  def EnablePinkNoise(self, amp, exponent=0.51):
    '''Generates pink noise with the specified 1/f^exponent.  The amplitude
       value sets the average standard deviation of the resulting pink noise.
       The default 0.51 exponent is from:
       Linkenkaer-Hansen et al. 2001, J. Neurosci.'''
    self.pinknoise_amp = amp
    self.pinknoise_exponent = exponent


  def _GetPinkNoise(self, N):
    out_N = N
    if N&1 == 1:
      N += 1
    scales = np.linspace(0, 0.5, N//2+1)[1:]
    scales = scales**(-self.pinknoise_exponent/2)
    pinkf = (np.random.normal(scale=scales) *
        np.exp(2j*np.pi*np.random.random(N//2)))
    fdata = np.concatenate([[0], pinkf])
    sigma = np.sqrt(2*np.sum(scales**2))/N
    data = self.pinknoise_amp * np.real(np.fft.irfft(fdata))/sigma
    return data[:out_N]


  def Generate(self, duration, random_state=None):
    '''Generate an EEG signal of the specified duration including the
       configured pink noise, components, and spikes.'''
    if random_state is not None:
        np.random.seed(random_state)
    N = int(round(duration * self.sampling_rate))
    eeg = np.zeros(N)
    self.time_coords = np.linspace(0, N/self.sampling_rate, N)

    for comp in self.components:
      comp.AddToEEG(eeg, self.sampling_rate)

    for spike in self.spikes:
      si = int(round(spike[0]*self.freq))
      if (si >= 0) and (si < N):
        eeg[si] += spike[1]

    if self.pinknoise_amp != 0:
      eeg += self._GetPinkNoise(N)

    return eeg

