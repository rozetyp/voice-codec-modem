#!/usr/bin/env python3
"""
AUTOENCODER VALIDATION: Compare Approach B vs Phase 2 Champion

This script:
1. Loads trained autoencoder
2. Tests at various bitrates (1.2 kbps → 10 kbps)
3. Measures BER vs Phase 2 champion (2.7 kbps, 1.15% BER)
4. Validates if ML approach beats hand-tuned classical DSP

Expected outcome:
  If successful: 5-10x bitrate at same BER
  If not: Hybrid approach (classical + ML refinement)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_autoencoder(
    checkpoint_path: Path = Path("checkpoints/codec_agnostic_autoencoder.pth"),
) -> Dict:
    """Load trained autoencoder."""
    
    if not checkpoint_path.exists():
        print(f"❌ Model not found: {checkpoint_path}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    from voice_detector.codec_agnostic_autoencoder import NeuralEncoder, NeuralDecoder
    
    bits_per_sequence = checkpoint.get('bits_per_sequence', 16)
    
    encoder = NeuralEncoder(bits_per_sequence=bits_per_sequence).to(device)
    decoder = NeuralDecoder(bits_per_sequence=bits_per_sequence).to(device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    return {
        'encoder': encoder,
        'decoder': decoder,
        'bits_per_sequence': bits_per_sequence,
        'device': device,
    }


def test_autoencoder_bitrate(
    autoencoder: Dict,
    proxy_codec: nn.Module = None,
    num_test_sequences: int = 100,
    add_noise: bool = True,
) -> Dict:
    """
    Test autoencoder at configured bitrate.
    Optionally runs through proxy codec to simulate real transmission.
    
    Returns:
      {
        'bits_per_sequence': 16,
        'bitrate_bps': 800,
        'ber_percent': 2.5,
        'num_errors': 3,
      }
    """
    
    encoder = autoencoder['encoder']
    decoder = autoencoder['decoder']
    device = autoencoder['device']
    bits_per_sequence = autoencoder['bits_per_sequence']
    
    total_bits = 0
    total_errors = 0
    
    with torch.no_grad():
        for i in range(num_test_sequences):
            # Random bits
            bits = torch.randint(0, 2, (1, bits_per_sequence)).float().to(device)
            
            # Encode
            audio = encoder(bits)
            
            # Pass through proxy codec if available
            if proxy_codec is not None:
                audio = proxy_codec(audio)
            
            # Add noise if requested
            if add_noise:
                snr_db = 20
                noise_power = torch.mean(audio ** 2) / (10 ** (snr_db / 10))
                noise = torch.sqrt(noise_power) * torch.randn_like(audio)
                audio = audio + noise
            
            # Decode
            bit_logits = decoder(audio)
            bit_probs = torch.sigmoid(bit_logits)
            bit_predictions = (bit_probs > 0.5).float()
            
            # Count errors
            errors = torch.sum((bit_predictions != bits).float()).item()
            total_errors += errors
            total_bits += bits_per_sequence
    
    ber = (total_errors / total_bits) * 100
    
    # Bitrate estimate: bits per 20ms segment
    bitrate_bps = (bits_per_sequence / 20) * 1000  # (bits / ms) * 1000
    
    return {
        'bits_per_sequence': bits_per_sequence,
        'bitrate_bps': bitrate_bps,
        'bitrate_kbps': bitrate_bps / 1000,
        'ber_percent': ber,
        'total_errors': int(total_errors),
        'total_bits': int(total_bits),
    }


def compare_with_phase2_champion():
    """
    Phase 2 Champion (Classical DSP):
      - Bitrate: 2.7 kbps (2666 bps)
      - BER: 1.15%
      - Codec: Opus (32 kbps)
      - Symbol duration: 1ms
      - Modulation: Music mode (chirps + harmonics)
    
    Autoencoder Target:
      To beat Phase 2, we need:
        - Higher bitrate at same BER (e.g., 5+ kbps at 1% BER)
        OR
        - Same bitrate with lower BER (e.g., 2.7 kbps at <0.5% BER)
    """
    
    print("\n" + "=" * 100)
    print("PHASE 2 CHAMPION (Classical DSP) vs APPROACH B (ML)")
    print("=" * 100)
    
    phase2_champion = {
        'name': 'Music Mode Modem (Phase 2)',
        'bitrate_kbps': 2.7,
        'bitrate_bps': 2666,
        'ber_percent': 1.15,
        'codec': 'Opus (32 kbps)',
        'modulation': 'Chirps + 60Hz + 200Hz harmonics',
        'symbol_duration_ms': 1.0,
    }
    
    print(f"\n{'Property':<30} {'Phase 2 Champion':<30}")
    print("-" * 60)
    for key, value in phase2_champion.items():
        print(f"{key:<30} {str(value):<30}")
    
    return phase2_champion


def validate_autoencoder():
    """Full validation pipeline."""
    
    print("\n" + "=" * 100)
    print("AUTOENCODER VALIDATION PIPELINE")
    print("=" * 100)
    
    # Load model
    print("\n[Step 1/4] Loading autoencoder...")
    autoencoder = load_autoencoder()
    
    if autoencoder is None:
        print("\n❌ Cannot proceed without trained model")
        print("   Run: python3 -m src.voice_detector.codec_agnostic_autoencoder")
        return
    
    print(f"✓ Loaded autoencoder")
    print(f"  - Bits per sequence: {autoencoder['bits_per_sequence']}")
    print(f"  - Device: {autoencoder['device']}")
    
    # Test current configuration
    print("\n[Step 2/4] Testing autoencoder at current bitrate...")
    results = test_autoencoder_bitrate(autoencoder, num_test_sequences=100, add_noise=True)
    print(f"✓ Test complete")
    print(f"  - Bits per sequence: {results['bits_per_sequence']}")
    print(f"  - Estimated bitrate: {results['bitrate_kbps']:.1f} kbps")
    print(f"  - BER: {results['ber_percent']:.2f}%")
    print(f"  - Total errors: {results['total_errors']} / {results['total_bits']}")
    
    # Compare with Phase 2
    print("\n[Step 3/4] Comparing with Phase 2 champion...")
    phase2 = compare_with_phase2_champion()
    
    print("\n[Step 4/4] Analysis...")
    bitrate_ratio = results['bitrate_kbps'] / (phase2['bitrate_kbps'])
    ber_comparison = phase2['ber_percent'] / results['ber_percent']
    
    print(f"\n{'Metric':<40} {'Result':<20} {'Verdict':<30}")
    print("-" * 90)
    
    # Bitrate
    if bitrate_ratio > 1.5:
        verdict = f"✓ {bitrate_ratio:.1f}x improvement!"
    elif bitrate_ratio > 1.0:
        verdict = f"~ {bitrate_ratio:.1f}x (marginal)"
    else:
        verdict = f"✗ {bitrate_ratio:.1f}x (inferior)"
    print(f"{'Bitrate: ML vs Phase 2':<40} {bitrate_ratio:.2f}x{'':<14} {verdict:<30}")
    
    # BER
    if ber_comparison > 1.5:
        verdict = "✓ Better (lower BER)"
    elif ber_comparison > 0.8:
        verdict = "~ Similar"
    else:
        verdict = "✗ Worse (higher BER)"
    print(f"{'BER quality: Phase 2 vs ML':<40} {ber_comparison:.2f}x{'':<14} {verdict:<30}")
    
    # Overall
    print("\n" + "=" * 100)
    if bitrate_ratio > 2.0 and results['ber_percent'] < 2.0:
        print("✓✓✓ SUCCESS: ML approach beats Phase 2 champion!")
        print("\nRECOMMENDATION:")
        print("  1. Increase bits_per_sequence and retrain")
        print("  2. Test on real Opus/AAC codec (not just noise)")
        print("  3. Scale to DPI evasion (add perceptual loss)")
        print("  4. Deploy as production modem")
    elif bitrate_ratio > 1.0 and results['ber_percent'] < phase2['ber_percent']:
        print("~ PROGRESS: ML approach shows promise")
        print("\nRECOMMENDATION:")
        print("  1. Tune hyperparameters (learning rate, epochs)")
        print("  2. Add more training data")
        print("  3. Retrain with higher bits_per_sequence")
    else:
        print("✗ Not yet successful")
        print("\nRECOMMENDATION:")
        print("  1. Debug: Is proxy codec learning correctly?")
        print("  2. Check: Is decoder receiving good features?")
        print("  3. Consider: Hybrid approach (classical preprocessing + ML refinement)")
        print("  4. Fallback: Use Phase 2 champion as baseline")
    
    print("=" * 100)


if __name__ == "__main__":
    validate_autoencoder()
