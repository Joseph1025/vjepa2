#!/usr/bin/env python3
"""
Simple inference script for V-JEPA 2
Extracts visual features from a video using the pretrained encoder
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from decord import VideoReader, cpu
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_video_transform(img_size=256):
    """Build preprocessing transform for video"""
    short_side_size = int(256.0 / 224 * img_size)
    transform = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def load_video(video_path, num_frames=16, fps=None):
    """
    Load video frames from file
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        fps: Target FPS (None = use original fps)
    
    Returns:
        video: numpy array of shape [T, H, W, C]
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample frames uniformly
    if total_frames <= num_frames:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Get frames
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames


def load_model(checkpoint_path, img_size=256, num_frames=64, device='cuda'):
    """
    Load V-JEPA 2 encoder from checkpoint
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        img_size: Image size (256 or 384)
        num_frames: Max number of frames
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded encoder model
    """
    # Initialize model
    model = vit_giant_xformers_rope(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Clean state dict keys
    state_dict = checkpoint.get('encoder', checkpoint.get('target_encoder', checkpoint))
    cleaned_state_dict = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned_state_dict[key] = val
    
    # Load weights
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model


@torch.no_grad()
def extract_features(model, video, transform, device='cuda'):
    """
    Extract features from video using V-JEPA 2 encoder
    
    Args:
        model: V-JEPA 2 encoder
        video: numpy array [T, H, W, C]
        transform: preprocessing transform
        device: 'cuda' or 'cpu'
    
    Returns:
        features: torch tensor [1, num_patches, embed_dim]
    """
    # Preprocess: T x H x W x C -> T x C x H x W
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    
    # Apply transform
    video_tensor = transform(video_tensor)  # [T, C, H, W]
    
    # Add batch dimension: [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    print(f"Input shape: {video_tensor.shape}")
    
    # Extract features
    features = model(video_tensor)
    
    print(f"Output features shape: {features.shape}")
    print(f"Feature dimension: {features.shape[-1]}")
    
    return features


def main():
    # Configuration
    video_path = "sample_video.mp4"  # Change to your video path
    checkpoint_path = "vitg.pt"  # Path to downloaded checkpoint
    img_size = 256
    num_frames = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("V-JEPA 2 Inference Script")
    print("="*60)
    
    # Step 1: Load video
    print(f"\n1. Loading video from: {video_path}")
    video = load_video(video_path, num_frames=num_frames)
    print(f"   Video shape: {video.shape} (T, H, W, C)")
    
    # Step 2: Build transform
    print(f"\n2. Building video transform (img_size={img_size})")
    transform = build_video_transform(img_size=img_size)
    
    # Step 3: Load model
    print(f"\n3. Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, img_size=img_size, num_frames=64, device=device)
    
    # Step 4: Extract features
    print(f"\n4. Extracting features (device={device})")
    features = extract_features(model, video, transform, device=device)
    
    # Step 5: Results
    print(f"\n5. Feature extraction complete!")
    print(f"   Features shape: {features.shape}")
    print(f"   Features mean: {features.mean().item():.4f}")
    print(f"   Features std: {features.std().item():.4f}")
    print(f"   Features min: {features.min().item():.4f}")
    print(f"   Features max: {features.max().item():.4f}")
    
    # Optional: Save features
    output_path = "video_features.pt"
    torch.save(features.cpu(), output_path)
    print(f"\n6. Features saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Done! You can now use these features for downstream tasks.")
    print("="*60)


if __name__ == "__main__":
    main()

