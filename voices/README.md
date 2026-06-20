Optional place for local reference voice samples.

The runner first looks for a reference sample in the project root:

    reference.wav

It can also use files from this folder when passed explicitly:

    .\run.ps1 -Speaker myvoice.wav

Use a clean WAV sample of the target voice. A short clip is enough to start, but 10-30 seconds of clear speech usually gives better voice cloning than a noisy or heavily processed sample.
