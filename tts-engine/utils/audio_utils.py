from scipy.io.wavfile import write


def save_audio(numpy_array, audio_path='test.wav', sample_rate=16000):
    data = numpy_array * 32768.0
    data = data.astype('int16')
    write(audio_path, sample_rate, data)
