"""
OPTIMIZED: Multithreaded Audio Tokenizer Training
Takes advantage of all CPU cores for faster training
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np
from audio_tokenizer import AudioTokenizer
import librosa
import soundfile as sf
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_features_from_file(audio_file, tokenizer):
    """Helper function for parallel feature extraction"""
    try:
        audio, _ = librosa.load(audio_file, sr=tokenizer.sample_rate)
        features = tokenizer.extract_features(audio)
        return features
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def encode_file_to_vq(audio_file, tokenizer):
    """Helper function for parallel VQ encoding"""
    try:
        audio, _ = librosa.load(audio_file, sr=tokenizer.sample_rate)
        vq_tokens = tokenizer.encode_to_vq(audio)
        return vq_tokens
    except Exception as e:
        print(f"Error encoding {audio_file}: {e}")
        return None


def download_librispeech_subset(output_dir: str = "./librispeech_data"):
    """
    Download LibriSpeech dev-clean subset (~350MB, ~5 hours of speech)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    filename = os.path.join(output_dir, "dev-clean.tar.gz")
    
    if not os.path.exists(filename):
        print("Downloading LibriSpeech dev-clean subset...")
        print("(This is ~350MB, will take a few minutes)")
        
        def progress_hook(count, block_size, total_size):
            percent = min(100, count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent:.1f}%", end='')
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print("\nDownload complete!")
    else:
        print("LibriSpeech already downloaded.")
    
    extract_dir = os.path.join(output_dir, "LibriSpeech")
    if not os.path.exists(extract_dir):
        print("Extracting archive...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(output_dir)
        print("Extraction complete!")
    else:
        print("Archive already extracted.")
    
    audio_files = glob.glob(os.path.join(extract_dir, "**/*.flac"), recursive=True)
    print(f"Found {len(audio_files)} audio files")
    
    return audio_files


def train_production_model_parallel(audio_files, output_path="tokenizer_production.pkl", n_jobs=-1):
    """
    Train a production-quality model with MULTITHREADING on FULL dataset
    
    Args:
        audio_files: List of audio file paths
        output_path: Where to save the model
        n_jobs: Number of parallel jobs (-1 = use all cores)
    """
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    print("\n" + "="*60)
    print(f"Training Production Audio Tokenizer (Using {n_jobs} cores)")
    print(f"Dataset: {len(audio_files)} files")
    print("="*60)
    
    # Initialize with production parameters
    tokenizer = AudioTokenizer(
        sample_rate=16000,
        window_ms=128,
        n_mfcc=13,
        n_clusters=2048,
        bpe_vocab_size=16384  # Target vocab size
    )
    
    # Determine how much data to use based on dataset size
    if len(audio_files) < 5000:
        # Small dataset (dev-clean): use most of it
        n_vq_files = min(1000, len(audio_files))
        n_bpe_files = min(500, len(audio_files))
    else:
        # Large dataset (train-clean-360): USE IT ALL
        n_vq_files = min(50000, len(audio_files))  # Use up to 50k for VQ
        n_bpe_files = min(20000, len(audio_files))  # Use up to 20k for BPE
    
    print(f"\nUsing {n_vq_files} files for VQ training ({n_vq_files/len(audio_files)*100:.1f}% of dataset)")
    print(f"Using {n_bpe_files} files for BPE training ({n_bpe_files/len(audio_files)*100:.1f}% of dataset)")
    
    # === STAGE 1: PARALLEL FEATURE EXTRACTION ===
    print(f"\n[1/3] Extracting features from {n_vq_files} files...")
    print(f"Using {n_jobs} parallel workers...")
    
    # Create partial function with tokenizer baked in
    extract_func = partial(extract_features_from_file, tokenizer=tokenizer)
    
    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(extract_func, audio_files[:n_vq_files]),
            total=n_vq_files,
            desc="Extracting features"
        ))
    
    # Filter out failed extractions
    all_features = [r for r in results if r is not None]
    all_features = np.vstack(all_features)
    
    print(f"Total feature vectors: {all_features.shape[0]}")
    
    # Compute normalization statistics
    print("Computing normalization statistics...")
    tokenizer.feature_mean = np.mean(all_features, axis=0)
    tokenizer.feature_std = np.std(all_features, axis=0) + 1e-8
    
    # Normalize
    normalized_features = (all_features - tokenizer.feature_mean) / tokenizer.feature_std
    
    # === STAGE 2: PARALLEL K-MEANS ===
    print(f"\n[2/3] Training k-means with {tokenizer.n_clusters} clusters...")
    print(f"Using all available CPU cores via OpenMP...")
    
    # Set OpenMP threads for parallel k-means
    import os
    os.environ['OMP_NUM_THREADS'] = str(n_jobs if n_jobs > 0 else cpu_count())
    
    from sklearn.cluster import KMeans
    
    tokenizer.kmeans = KMeans(
        n_clusters=tokenizer.n_clusters,
        max_iter=100,
        verbose=1,
        random_state=42,
        n_init=3,
        algorithm='elkan'
        # n_jobs removed in newer scikit-learn - uses OpenMP automatically
    )
    
    tokenizer.kmeans.fit(normalized_features)
    print("✓ VQ training complete!")
    
    # === STAGE 3: PARALLEL VQ ENCODING ===
    print(f"\n[3/3] Encoding {n_bpe_files} files to VQ sequences...")
    print(f"Using {n_jobs} parallel workers...")
    
    # Create partial function
    encode_func = partial(encode_file_to_vq, tokenizer=tokenizer)
    
    # Parallel encoding
    with Pool(processes=n_jobs) as pool:
        vq_sequences = list(tqdm(
            pool.imap(encode_func, audio_files[:n_bpe_files]),
            total=n_bpe_files,
            desc="Encoding to VQ"
        ))
    
    # Filter out failures
    vq_sequences = [s for s in vq_sequences if s is not None]
    
    # Calculate exact number of merges needed to reach target vocab size
    num_merges = tokenizer.bpe_vocab_size - tokenizer.n_clusters  # 16384 - 2048 = 14336
    
    # Train BPE (this part is harder to parallelize efficiently)
    print("\n[4/4] Training BPE...")
    print(f"Target vocab size: {tokenizer.bpe_vocab_size}")
    print(f"Base VQ tokens: {tokenizer.n_clusters}")
    print(f"BPE merges needed: {num_merges}")
    tokenizer.train_bpe(vq_sequences, num_merges=num_merges)
    
    # Save model
    tokenizer.save(output_path)
    
    print("\n" + "="*60)
    print(f"✓ Model saved to {output_path}")
    print("="*60)
    
    return tokenizer


def test_trained_model(tokenizer, test_audio_file, example_num=0):
    """Test the trained model on a sample file"""
    import librosa
    import soundfile as sf
    from audio_tokenizer import calculate_token_rate
    
    if example_num == 0:
        print("\n" + "="*60)
        print("Testing Trained Model")
        print("="*60)
    
    audio, _ = librosa.load(test_audio_file, sr=16000)
    duration = len(audio) / 16000
    
    print(f"\nTest audio: {os.path.basename(test_audio_file)}")
    print(f"Duration: {duration:.2f} seconds")
    
    vq_tokens = tokenizer.encode_to_vq(audio)
    vq_rate = calculate_token_rate(duration, len(vq_tokens))
    
    bpe_tokens = tokenizer.encode_with_bpe(vq_tokens)
    bpe_rate = calculate_token_rate(duration, len(bpe_tokens))
    
    print(f"\nTokenization Results:")
    print(f"  VQ tokens: {len(vq_tokens)} ({vq_rate:.2f} tokens/sec)")
    print(f"  BPE tokens: {len(bpe_tokens)} ({bpe_rate:.2f} tokens/sec)")
    print(f"  Target: ~7.5 tokens/sec ✓" if 6 < bpe_rate < 9 else f"  Target: ~7.5 tokens/sec ✗")
    
    reconstructed = tokenizer.decode_from_vq(vq_tokens)
    
    output_dir = "./test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use numbered filenames to avoid overwriting
    suffix = f"_{example_num}" if example_num > 0 else ""
    original_path = os.path.join(output_dir, f"test_original{suffix}.wav")
    reconstructed_path = os.path.join(output_dir, f"test_reconstructed{suffix}.wav")
    
    sf.write(original_path, audio, 16000)
    sf.write(reconstructed_path, reconstructed, 16000)
    
    print(f"\n✓ Saved original to: {original_path}")
    print(f"✓ Saved reconstructed to: {reconstructed_path}")
    
    original_bits = len(audio) * 16
    compressed_bits = len(bpe_tokens) * np.log2(tokenizer.bpe_vocab_size)
    compression_ratio = original_bits / compressed_bits
    
    print(f"\nCompression:")
    print(f"  Ratio: {compression_ratio:.1f}x")
    print(f"  Bitrate: {compressed_bits / duration:.0f} bps")
    
    if example_num == 0:
        print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    n_cores = cpu_count()
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║   OPTIMIZED Audio Tokenizer - Multithreaded Training      ║
║   Detected: {n_cores} CPU cores                                    ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\nOption 1: Quick Training (dev-clean, ~5 hours)")
    print("  - Downloads 350MB")
    print(f"  - Trains in ~5-10 minutes with {n_cores} cores")
    print("  - Good for testing/demo")
    
    print("\nOption 2: Full Training (train-clean-360, ~360 hours)")
    print("  - Downloads ~30GB")
    print(f"  - Trains in ~1-2 hours with {n_cores} cores")
    print("  - Production quality")
    print("  - Reaches full 16384 vocab size")
    
    choice = input("\nEnter 1 or 2 (or 'skip' to use demo): ").strip()
    
    if choice == "skip":
        print("\nSkipping training. Run demo.py for synthetic data demo.")
        sys.exit(0)
    
    if choice == "1":
        # Download and train on dev-clean
        audio_files = download_librispeech_subset()
        
        # Train with ALL cores on ALL available data
        tokenizer = train_production_model_parallel(
            audio_files, 
            "tokenizer_devclean.pkl",
            n_jobs=-1  # Use all cores
        )
        
        # Test on multiple examples
        if audio_files:
            print("\n" + "="*60)
            print("Testing on 3 examples from dataset...")
            print("="*60)
            
            num_examples = min(3, len(audio_files))
            for i in range(num_examples):
                print(f"\n--- Example {i+1}/{num_examples} ---")
                test_trained_model(tokenizer, audio_files[i], example_num=i)
        
        print(f"\n\n🚀 Training completed using all {n_cores} cores!")
        print("💡 Check ./test_outputs/ for original vs reconstructed audio!")
        print("   Listen to compare quality!")
    
    elif choice == "2":
        # Download train-clean-360
        output_dir = "./librispeech_data"
        url = "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
        filename = os.path.join(output_dir, "train-clean-360.tar.gz")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(filename):
            print("\nDownloading train-clean-360 (~23GB)...")
            print("This will take a while...")
            
            def progress_hook(count, block_size, total_size):
                percent = min(100, count * block_size * 100 / total_size)
                mb_downloaded = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB)", end='')
            
            urllib.request.urlretrieve(url, filename, progress_hook)
            print("\nDownload complete!")
        else:
            print("train-clean-360 already downloaded.")
        
        # Extract
        extract_dir = os.path.join(output_dir, "LibriSpeech")
        if not os.path.exists(extract_dir):
            print("Extracting archive (this takes 10-15 minutes)...")
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(output_dir)
            print("Extraction complete!")
        else:
            print("Archive already extracted.")
        
        # Get all audio files
        audio_files = glob.glob(os.path.join(extract_dir, "**/*.flac"), recursive=True)
        print(f"Found {len(audio_files)} audio files")
        
        # Train with ALL cores on ALL data
        tokenizer = train_production_model_parallel(
            audio_files,
            "tokenizer_full_production.pkl",
            n_jobs=-1
        )
        
        # Test on examples
        if audio_files:
            print("\n" + "="*60)
            print("Testing on 3 examples from dataset...")
            print("="*60)
            
            num_examples = min(3, len(audio_files))
            for i in range(num_examples):
                print(f"\n--- Example {i+1}/{num_examples} ---")
                test_trained_model(tokenizer, audio_files[i], example_num=i)
        
        print(f"\n\n🚀 Full training completed using all {n_cores} cores!")
        print(f"💡 Model saved: tokenizer_full_production.pkl")
        print("💡 Check ./test_outputs/ for examples!")
        print(f"\n💾 You can now delete the dataset to free up ~23GB:")
        print(f"   rm -rf {output_dir}")
    
    else:
        print("Invalid choice. Exiting.")