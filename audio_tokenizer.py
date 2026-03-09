"""
Trainless Audio LLM Tokenizer
Converts speech audio to discrete tokens matching LLM token rates (~7 tokens/sec)
Fully reversible (lossy) - can reconstruct intelligible speech from tokens
"""

import numpy as np
import librosa
from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter
import soundfile as sf


class AudioTokenizer:
    def __init__(self, 
                 sample_rate: int = 16000,
                 window_ms: int = 128,
                 n_mfcc: int = 13,
                 n_clusters: int = 2048,
                 bpe_vocab_size: int = 16384):
        """
        Initialize the audio tokenizer
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_ms: Window size in milliseconds
            n_mfcc: Number of MFCC coefficients
            n_clusters: VQ codebook size
            bpe_vocab_size: Final BPE vocabulary size
        """
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.n_mfcc = n_mfcc
        self.n_clusters = n_clusters
        self.bpe_vocab_size = bpe_vocab_size
        
        # Calculate window parameters
        self.window_length = int(sample_rate * window_ms / 1000)
        self.hop_length = self.window_length  # Non-overlapping
        
        # Feature dimension: MFCCs + deltas + delta-deltas + energy + delta-energy
        self.feature_dim = n_mfcc * 3 + 2
        
        # Models (to be trained)
        self.kmeans = None
        self.feature_mean = None
        self.feature_std = None
        self.bpe_merges = []
        self.bpe_vocab = {}
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 41-dimensional features from audio
        
        Returns: (n_frames, 41) array
        """
        # Pre-emphasis filter
        emphasized = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Extract MFCCs (skip 0th coefficient)
        mfccs = librosa.feature.mfcc(
            y=emphasized,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc + 1,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            window='hann'
        )[1:, :]  # Skip 0th coefficient
        
        # Compute deltas
        delta_mfccs = librosa.feature.delta(mfccs, order=1)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Log energy
        rms = librosa.feature.rms(
            y=emphasized,
            frame_length=self.window_length,
            hop_length=self.hop_length
        )
        log_energy = np.log(rms + 1e-8)
        delta_energy = librosa.feature.delta(log_energy, order=1)
        
        # Concatenate all features
        features = np.vstack([
            mfccs,           # 13
            delta_mfccs,     # 13
            delta2_mfccs,    # 13
            log_energy,      # 1
            delta_energy     # 1
        ])  # Shape: (41, n_frames)
        
        return features.T  # Return (n_frames, 41)
    
    def train_vq(self, audio_files: List[str], max_samples: int = 1000000, n_jobs: int = 1):
        """
        Train VQ codebook using k-means on audio corpus
        
        Args:
            audio_files: List of paths to audio files
            max_samples: Maximum number of feature vectors to use for training
            n_jobs: Number of parallel jobs for k-means (-1 = use all cores)
        """
        print(f"Extracting features from {len(audio_files)} files...")
        
        all_features = []
        
        for i, audio_file in enumerate(audio_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(audio_files)}")
            
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Extract features
            features = self.extract_features(audio)
            all_features.append(features)
            
            # Stop if we have enough samples
            total_samples = sum(f.shape[0] for f in all_features)
            if total_samples >= max_samples:
                break
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        print(f"Total feature vectors: {all_features.shape[0]}")
        
        # Compute normalization statistics
        print("Computing normalization statistics...")
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0) + 1e-8
        
        # Normalize
        normalized_features = (all_features - self.feature_mean) / self.feature_std
        
        # Train k-means
        print(f"Training k-means with {self.n_clusters} clusters...")
        if n_jobs != 1:
            print(f"Using {n_jobs} parallel threads via OpenMP...")
            import os
            os.environ['OMP_NUM_THREADS'] = str(n_jobs if n_jobs > 0 else os.cpu_count())
        
        # Use regular KMeans (automatically parallelized via OpenMP in newer scikit-learn)
        if n_jobs > 1 or n_jobs == -1:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=100,
                verbose=1,
                random_state=42,
                n_init=3,
                algorithm='elkan'
            )
        else:
            # Use MiniBatchKMeans for single-threaded
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=1024,
                max_iter=100,
                verbose=1,
                random_state=42
            )
        
        self.kmeans.fit(normalized_features)
        
        print("VQ training complete!")
        
    def encode_to_vq(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode audio to VQ token sequence
        
        Returns: 1D array of cluster indices
        """
        if self.kmeans is None:
            raise ValueError("VQ model not trained! Call train_vq() first")
        
        # Extract features
        features = self.extract_features(audio)
        
        # Normalize
        normalized = (features - self.feature_mean) / self.feature_std
        
        # VQ encoding
        tokens = self.kmeans.predict(normalized)
        
        return tokens
    
    def train_bpe(self, vq_sequences: List[np.ndarray], num_merges: Optional[int] = None):
        """
        Train BPE on VQ token sequences
        
        Args:
            vq_sequences: List of VQ token sequences
            num_merges: Number of BPE merges (if None, computed from vocab size)
        """
        if num_merges is None:
            num_merges = self.bpe_vocab_size - self.n_clusters
        
        print(f"Training BPE with {num_merges} merges...")
        
        # Initialize vocabulary with base tokens
        self.bpe_vocab = {i: [i] for i in range(self.n_clusters)}
        next_token_id = self.n_clusters
        
        # Convert sequences to lists for easier manipulation
        sequences = [list(seq) for seq in vq_sequences]
        
        for merge_idx in range(num_merges):
            if merge_idx % 100 == 0:
                print(f"Merge {merge_idx}/{num_merges}")
            
            # Count all adjacent pairs
            pair_counts = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # Create new token
            new_token = next_token_id
            self.bpe_merges.append(best_pair)
            self.bpe_vocab[new_token] = self.bpe_vocab[best_pair[0]] + self.bpe_vocab[best_pair[1]]
            next_token_id += 1
            
            # Replace pair in all sequences
            for seq in sequences:
                i = 0
                while i < len(seq) - 1:
                    if (seq[i], seq[i + 1]) == best_pair:
                        seq[i:i+2] = [new_token]
                    else:
                        i += 1
        
        print(f"BPE training complete! Vocabulary size: {len(self.bpe_vocab)}")
    
    def encode_with_bpe(self, vq_tokens: np.ndarray) -> List[int]:
        """
        Apply BPE encoding to VQ token sequence
        """
        tokens = list(vq_tokens)
        
        for pair in self.bpe_merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair:
                    # Find the merged token ID
                    for token_id, decomposition in self.bpe_vocab.items():
                        if decomposition == self.bpe_vocab[pair[0]] + self.bpe_vocab[pair[1]]:
                            tokens[i:i+2] = [token_id]
                            break
                else:
                    i += 1
        
        return tokens
    
    def decode_from_vq(self, vq_tokens: np.ndarray) -> np.ndarray:
        """
        Reconstruct audio from VQ tokens (lossy)
        
        Returns: Audio waveform
        """
        if self.kmeans is None:
            raise ValueError("VQ model not trained!")
        
        # Get cluster centroids
        centroids = self.kmeans.cluster_centers_
        
        # Map tokens to centroids
        features = centroids[vq_tokens]
        
        # Denormalize
        features = features * self.feature_std + self.feature_mean
        
        # Reconstruct audio using Griffin-Lim
        # We'll use just the MFCCs (first 13 dimensions) for reconstruction
        mfccs = features[:, :self.n_mfcc].T  # (13, n_frames)
        
        # Inverse MFCC to mel spectrogram
        mel_spec = librosa.feature.inverse.mfcc_to_mel(
            mfccs,
            n_mels=128,
            dct_type=2
        )
        
        # Mel to linear spectrogram
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.window_length,
            n_mels=128
        )
        
        # Approximate inverse (pseudo-inverse)
        spec = np.dot(np.linalg.pinv(mel_basis), mel_spec)
        spec = np.maximum(0, spec)  # Ensure non-negative
        
        # Griffin-Lim for phase reconstruction
        audio = librosa.griffinlim(
            spec,
            n_iter=32,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window='hann'
        )
        
        return audio
    
    def save(self, path: str):
        """Save trained model"""
        data = {
            'kmeans': self.kmeans,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'bpe_merges': self.bpe_merges,
            'bpe_vocab': self.bpe_vocab,
            'config': {
                'sample_rate': self.sample_rate,
                'window_ms': self.window_ms,
                'n_mfcc': self.n_mfcc,
                'n_clusters': self.n_clusters,
                'bpe_vocab_size': self.bpe_vocab_size
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.kmeans = data['kmeans']
        self.feature_mean = data['feature_mean']
        self.feature_std = data['feature_std']
        self.bpe_merges = data['bpe_merges']
        self.bpe_vocab = data['bpe_vocab']
        
        config = data['config']
        self.sample_rate = config['sample_rate']
        self.window_ms = config['window_ms']
        self.n_mfcc = config['n_mfcc']
        self.n_clusters = config['n_clusters']
        self.bpe_vocab_size = config['bpe_vocab_size']
        
        self.window_length = int(self.sample_rate * self.window_ms / 1000)
        self.hop_length = self.window_length
        
        print(f"Model loaded from {path}")


def calculate_token_rate(audio_duration: float, num_tokens: int) -> float:
    """Calculate tokens per second"""
    return num_tokens / audio_duration


if __name__ == "__main__":
    print("Audio Tokenizer - Example Usage")
    print("=" * 50)
    
    # This is example code - you'll need actual audio files
    print("\nTo use this tokenizer:")
    print("1. Collect ~100 hours of diverse speech audio (e.g., LibriSpeech)")
    print("2. Train VQ: tokenizer.train_vq(audio_files)")
    print("3. Encode all audio to VQ sequences")
    print("4. Train BPE: tokenizer.train_bpe(vq_sequences)")
    print("5. Use tokenizer.encode_to_vq() to tokenize new audio")
    print("6. Use tokenizer.decode_from_vq() to reconstruct audio")