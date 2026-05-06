#!/usr/bin/env python3
"""
MASTER ORCHESTRATION: Approach B (Autoencoder) Full Pipeline

Execution order:
  1. Proxy Codec Training     (5 min)   → Learn to approximate real codec
  2. Autoencoder Training     (10 min)  → Learn optimal modulation
  3. Autoencoder Validation   (5 min)   → Compare vs Phase 2 (2.7 kbps, 1.15% BER)
  4. DPI-Evasion Planning     (analysis)→ Next: Add perceptual loss

Total time: ~20 minutes for full pipeline
"""

import subprocess
import sys
from pathlib import Path


def run_command(
    script: str,
    description: str,
    timeout_sec: int = 600,
) -> bool:
    """Run Python script and capture output."""
    
    print("\n" + "=" * 100)
    print(f"[ORCHESTRATION] {description}")
    print("=" * 100)
    
    try:
        cmd = [sys.executable, script]
        result = subprocess.run(
            cmd,
            timeout=timeout_sec,
            capture_output=False,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"\n❌ Failed with return code {result.returncode}")
            return False
        
        print(f"\n✓ Completed successfully")
        return True
    
    except subprocess.TimeoutExpired:
        print(f"\n❌ Timeout after {timeout_sec} seconds")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def check_dependencies():
    """Verify PyTorch and FFmpeg are available."""
    
    print("\n" + "=" * 100)
    print("[SETUP] Checking dependencies")
    print("=" * 100)
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not found")
        print("   Install: pip install torch")
        return False
    
    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✓ FFmpeg installed")
        else:
            print("❌ FFmpeg not working")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        print("   Install: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"❌ FFmpeg check failed: {e}")
        return False
    
    # Check SoundFile
    try:
        import soundfile
        print(f"✓ SoundFile {soundfile.__version__}")
    except ImportError:
        print("❌ SoundFile not found")
        print("   Install: pip install soundfile")
        return False
    
    print("\n✓ All dependencies available")
    return True


def main():
    """Execute full Approach B pipeline."""
    
    root = Path(__file__).parent.parent
    
    scripts = {
        1: {
            'script': str(root / "src/voice_detector/proxy_codec.py"),
            'description': "STEP 1: Train Proxy Codec (FFmpeg approximation)",
            'timeout': 600,
        },
        2: {
            'script': str(root / "src/voice_detector/codec_agnostic_autoencoder.py"),
            'description': "STEP 2: Train Codec-Agnostic Autoencoder",
            'timeout': 600,
        },
        3: {
            'script': str(root / "scripts/validate_autoencoder.py"),
            'description': "STEP 3: Validate and Compare vs Phase 2 Champion",
            'timeout': 120,
        },
    }
    
    print("\n" + "=" * 100)
    print("APPROACH B: CODEC-AGNOSTIC AUTOENCODER PIPELINE")
    print("=" * 100)
    print("""
This is the "Steganographic Tunnel" approach:

Goal:
  Learn modulation that the codec preserves perfectly
  By training a neural network on codec behavior

Expected outcome:
  2-3 order of magnitude improvement over hand-tuned classical DSP
  From 2.7 kbps (Phase 2) → 10+ kbps (realistic) → 50+ kbps (theoretical)

Timeline:
  - Proxy Codec:       5 min  (learn codec distortions)
  - Autoencoder:       10 min (learn optimal modulation)
  - Validation:        5 min  (compare vs baseline)
  Total:              ~20 min

After validation:
  - Retrain with higher bit density
  - Add perceptual loss (sound like office/street → evade DPI)
  - Test on real VoLTE network
  - Scale to full deployment
""")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot proceed without dependencies")
        sys.exit(1)
    
    # Run pipeline
    print("\n[PIPELINE] Starting...")
    
    all_success = True
    for step, config in scripts.items():
        success = run_command(
            script=config['script'],
            description=config['description'],
            timeout_sec=config['timeout'],
        )
        
        if not success:
            print(f"\n⚠ Step {step} failed. Continue? (y/n): ", end='')
            choice = input().strip().lower()
            if choice != 'y':
                print("❌ Pipeline interrupted")
                all_success = False
                break
    
    # Summary
    print("\n" + "=" * 100)
    print("PIPELINE SUMMARY")
    print("=" * 100)
    
    if all_success:
        print("""
✓ Full pipeline completed successfully!

Next steps:
1. Check validation results above
2. If autoencoder beats Phase 2:
   a. Increase bits_per_sequence in codec_agnostic_autoencoder.py
   b. Retrain with higher bitrate target
   c. Iterate until reaching 10+ kbps

3. Add DPI evasion:
   a. Create perceptual_loss.py
   b. Force generated audio to sound like "office noise" or "street traffic"
   c. Train with combined loss: BER + Perceptual

4. Hardware validation:
   a. Run on real VoLTE call
   b. Measure if "learned modulation" survives real codec
   c. Verify DPI evasion effectiveness

Expected timeline to 50 kbps:
  - Weeks 1-2: ML optimization loop (+10-15 kbps possible)
  - Weeks 2-3: DPI evasion tuning
  - Week 4: Real hardware validation
  - Week 5: Full deployment

Remember: The monetization moat is RESILIENCE not just SPEED
        """)
    else:
        print("""
⚠ Pipeline interrupted

Options:
  A) Debug step that failed and retry
  B) Use Phase 2 champion as baseline (2.7 kbps proven working)
  C) Try hybrid approach (classical preprocessing + ML refinement)

For debugging:
  - Check checkpoints/ directory exists
  - Verify PyTorch/FFmpeg working
  - Inspect training logs in individual scripts
        """)


if __name__ == "__main__":
    main()
