#!/usr/bin/env python3
"""
AUTOENCODER VALIDATION: Compare Approach B vs Phase 2 Champion (WITH CODEC)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_with_codec():
    """Test autoencoder end-to-end through proxy codec."""
    
    device = torch.device("cpu")
    
    # Load autoencoder
    from voice_detector.codec_agnostic_autoencoder import NeuralEncoder, NeuralDecoder
    checkpoint = torch.load("checkpoints/codec_agnostic_autoencoder.pth", map_location=device)
    bits_per_seq = checkpoint['bits_per_sequence']
    
    encoder = NeuralEncoder(bits_per_sequence=bits_per_seq).to(device)
    decoder = NeuralDecoder(bits_per_sequence=bits_per_seq).to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()
    
    # Load proxy codec
    from voice_detector.proxy_codec import ProxyCodec
    proxy_codec = ProxyCodec().to(device)
    proxy_codec.load_state_dict(torch.load("checkpoints/proxy_codec_opus.pth", map_location=device))
    proxy_codec.eval()
    
    # Test
    print("\n" + "="*100)
    print("APPROACH B VALIDATION: Autoencoder with Proxy Codec")
    print("="*100)
    
    total_errors = 0
    total_bits = 0
    
    with torch.no_grad():
        for i in range(100):
            bits = torch.randint(0, 2, (1, bits_per_seq)).float().to(device)
            audio = encoder(bits)
            audio = proxy_codec(audio)
            
            # Add noise (20dB SNR)
            noise_power = torch.mean(audio ** 2) / 100
            noise = torch.sqrt(noise_power) * torch.randn_like(audio)
            audio = audio + noise
            
            logits = decoder(audio)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            errors = torch.sum((preds != bits).float()).item()
            total_errors += errors
            total_bits += bits_per_seq
    
    ber = (total_errors / total_bits) * 100
    bitrate = (bits_per_seq / 20) * 1000 / 1000  # kbps
    
    print(f"\nRESULTS:")
    print(f"  Bitrate: {bitrate:.2f} kbps")
    print(f"  BER: {ber:.2f}%")
    print(f"  Errors: {int(total_errors)}/{int(total_bits)}")
    
    print(f"\nPHASE 2 CHAMPION (Baseline):")
    print(f"  Bitrate: 2.70 kbps")
    print(f"  BER: 1.15%")
    
    print(f"\nCOMPARISON:")
    ratio = bitrate / 2.70
    ber_comp = 1.15 / ber
    print(f"  Bitrate ratio: {ratio:.2f}x")
    print(f"  BER quality: {ber_comp:.2f}x")
    
    print(f"\nVERDICT:")
    if ratio > 1.5 and ber < 2.0:
        print(f"  ✓ SUCCESS: Beats Phase 2!")
    elif ratio > 1.0:
        print(f"  ~ Progress: Competitive")
    else:
        print(f"""  ⚠ First iteration needs improvement
  
  Autoencoder learned well (49% → 26% BER training)
  But with codec simulation, needs refinement
  
  Options:
  A) Retrain longer (NUM_EPOCHS = 50)
  B) Improve proxy codec (more training data)
  C) Hybrid: Use Phase 2 + ML refinement
  D) Accept Phase 2 (2.7 kbps proven)
  """)
    
    print("="*100)


if __name__ == "__main__":
    try:
        test_with_codec()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
