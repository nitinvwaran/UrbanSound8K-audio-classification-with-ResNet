import sounddevice as sd

def record():
    sd.default.samplerate = 16000
    sd.default.channels = 1

    myrec = sd.rec(2 * fs)


