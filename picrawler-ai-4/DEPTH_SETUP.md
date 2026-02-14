# Depth Estimation Setup Guide

This guide explains how to set up monocular depth estimation on your Raspberry Pi.

## Overview

v4 now includes **learned depth estimation** using MiDaS, enabling 3D spatial understanding from a single camera without hardware depth sensors.

## What You Get

- **Full field of view depth**: Not just one ultrasonic point, but depth across entire image
- **Directional awareness**: Separate depth for front, left, and right regions
- **Sensor fusion**: Combines ultrasonic + learned depth for higher confidence
- **Visualization**: Save colorized depth maps to see what the robot "sees"

## Installation on Raspberry Pi

### Option 1: Quick Install (Recommended)

```bash
cd ~/picrawler-ai/picrawler-ai-4
source .venv/bin/activate  # or use v3's venv

# Install PyTorch for Pi (CPU-only, lightweight)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip3 install timm

# Test it works
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

### Option 2: Full Requirements Install

```bash
pip3 install -r requirements.txt
```

**Note**: This may take 10-20 minutes on Pi as it compiles some packages.

## First Run - Test Mode

Test depth estimation before running full exploration:

```bash
python main.py --mode test
```

This will:
1. Capture one image
2. Estimate depth (~2-5 seconds on Pi)
3. Save both RGB image and colorized depth map
4. Show directional depth values
5. Display fused world model

**Expected output:**
```
Depth inference: 2847.3ms
Directional depths: {'front': 0.45, 'left': 0.72, 'right': 0.38}
Image saved: logs/images/frame_20260123-170430.jpg
Depth map saved: logs/images/frame_20260123-170430_depth.jpg
World model: WorldModel(front=89.5cm, free_space=0.75, best_dir=turn_left)
```

## Performance Notes

### Inference Speed
- **MiDaS_small at 256x256**: ~2-5 seconds per frame on Pi 4
- **Cached results**: Reuses depth for 1 second (configurable)
- **No GPU needed**: Runs on CPU

### Model Size
- **MiDaS_small**: ~100MB download (first run only)
- **Stored in**: `~/.cache/torch/hub/`

### Memory Usage
- **Additional RAM**: ~300-500MB during inference
- **Total system**: Should work fine on Pi 4 with 2GB+ RAM

## Depth Estimation in Action

### How It Works

1. **Capture**: Pi camera takes RGB image
2. **Estimate**: MiDaS predicts depth for every pixel
3. **Extract**: Get average depth in front/left/right regions
4. **Fuse**: Combine with ultrasonic sensor data
5. **Decide**: WorldModel uses fused data for navigation

### Depth Map Regions

The depth estimator divides the view into:
- **Front** (center 30% width, lower 30% height): Primary navigation
- **Left** (left 30% width): Left turn assessment
- **Right** (right 30% width): Right turn assessment

### Depth Values

Normalized 0-1 where:
- **0.0** = Very close (~10cm)
- **0.5** = Medium distance (~100cm)
- **1.0** = Far away (~200cm)

**Note**: Learned depth is *relative*, not absolute. It's accurate for comparing "close vs far" but not precise distances.

## Troubleshooting

### "PyTorch not available - depth estimation disabled"

**Solution**: PyTorch not installed. Run:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Depth estimation very slow (>10 seconds)

**Solutions**:
1. Lower input size in `main.py`:
   ```python
   depth_estimator = DepthEstimator(input_size=192)  # Was 256
   ```
2. Increase cache duration:
   ```python
   depth_estimator = DepthEstimator(cache_duration_s=2.0)  # Was 1.0
   ```

### Out of memory error

**Solutions**:
1. Close other applications on Pi
2. Reduce input size to 192 or 128
3. Use swap space:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=1024
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### Model download fails

**Solution**: Download manually:
```bash
cd ~/.cache/torch/hub/
git clone https://github.com/isl-org/MiDaS.git intel-isl_MiDaS_master
```

## Comparison: With vs Without Depth

### Without Depth (v3 behavior)
- Ultrasonic: "Something 15cm ahead"
- Vision AI: "I see a wall"
- Decision: "Turn" (but which way?)

### With Depth (v4 behavior)
- Ultrasonic: "Something 15cm ahead"
- Depth: "Front blocked, left 120cm clear, right 40cm blocked"
- Vision AI: "I see a wall"
- **Fused Decision**: "Turn left - most clearance"

## Advanced Configuration

Edit `main.py` to customize:

```python
depth_estimator = DepthEstimator(
    model_type="MiDaS_small",     # or "DPT_Hybrid" for higher quality (slower)
    input_size=256,               # Lower = faster, higher = more accurate
    cache_duration_s=1.0          # How long to reuse depth maps
)
```

Edit `world_model.py` to adjust depth-to-distance mapping:
```python
def depth_to_distance(depth_value: float) -> float:
    # Current: 10-200cm range
    # Adjust if your environment is different
    return 10 + (depth_value * 190)
```

## Next Steps

With depth estimation working:

1. **Run full exploration**: `python main.py --duration 5`
2. **Enable verbose mode**: `python main.py --verbose` to save all depth maps
3. **Compare to v3**: Notice how it makes smarter turn decisions
4. **Tune parameters**: Adjust cache duration and input size for your Pi

## Future Enhancements

This is just the beginning. With depth working, you can add:
- **Visual SLAM**: Build 3D maps of explored areas
- **Object distance**: "How far is that chair?"
- **Terrain analysis**: "Is this a step down?"
- **Manipulation planning**: "Can I fit through that gap?"

The infrastructure is now in place for cutting-edge robotics research!
