# SpeechEngine

**Created: November 2025** | NodeNestor

> **Status: Archived** -- This project is published as-is for reference. No active development.

A trainless audio tokenizer for LLMs. Converts speech to discrete tokens at ~7 tokens/sec with 2000-3000x compression, using only classical signal processing -- no neural networks required. Fully reversible (lossy reconstruction via Griffin-Lim).

## How It Works

```
Audio (16kHz, 128ms windows)
    -> 41-dim MFCCs (13 MFCC + 13 delta + 13 delta2 + energy + delta-energy)
    -> k-means VQ (2048 clusters) -> integer per frame
    -> BPE merging (up to 16384 vocab) -> ~7 tokens/sec
    -> Reversible: centroid lookup -> inverse MFCC -> Griffin-Lim -> audio
```

The pipeline discovers phoneme-level structure through clustering and compresses repeated patterns via BPE, producing token sequences directly compatible with LLM context windows.

## Key Numbers

| Metric | Value |
|--------|-------|
| Token rate | ~7 tokens/sec |
| Compression | 2000-3000x |
| VQ codebook | 2048 clusters |
| BPE vocab | up to 16384 |
| Reconstruction | Intelligible, vocoder-like |
| Training | Zero neural nets |

## Files

- `audio_tokenizer.py` -- Core tokenizer: feature extraction, VQ encoding, BPE, reconstruction
- `train_librispeech.py` -- Multithreaded training script (downloads LibriSpeech, trains VQ + BPE)
- `tokenization_visualization.png` -- Visualization of the tokenization pipeline

## Usage

```bash
pip install numpy librosa scikit-learn soundfile tqdm
```

```python
from audio_tokenizer import AudioTokenizer
import librosa

# Train on speech corpus
tokenizer = AudioTokenizer(n_clusters=2048, bpe_vocab_size=16384)
tokenizer.train_vq(audio_files)
# ... encode to VQ, train BPE, then:

audio, _ = librosa.load("speech.wav", sr=16000)
vq_tokens = tokenizer.encode_to_vq(audio)
bpe_tokens = tokenizer.encode_with_bpe(vq_tokens)
reconstructed = tokenizer.decode_from_vq(vq_tokens)
```

For full training on LibriSpeech:

```bash
python train_librispeech.py
```

## How It Compares

This is the classical equivalent of Meta's Encodec, Google's SoundStream, and AudioLM -- but using k-means instead of neural VQ and BPE instead of learned merging. No GPU training required.

## License

MIT
