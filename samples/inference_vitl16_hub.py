#!/usr/bin/env python3
"""
Simple inference script for V-JEPA 2 ViT-Large using torch.hub
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


def load_video(video_path, num_frames=16):
    """
    Load video frames from file
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
    
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


@torch.no_grad()
def extract_features(encoder, video, transform, device='cuda'):
    """
    Extract features from video using V-JEPA 2 encoder
    
    Args:
        encoder: V-JEPA 2 encoder
        video: numpy array [T, H, W, C]
        transform: preprocessing transform
        device: 'cuda' or 'cpu'
    
    Returns:
        features: torch tensor [1, num_patches, embed_dim]
        inference_time: float (seconds)
    """
    import time
    
    # Preprocess: T x H x W x C -> T x C x H x W
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    
    # Apply transform
    video_tensor = transform(video_tensor)  # [T, C, H, W]
    
    # Add batch dimension: [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    print(f"Input shape: {video_tensor.shape}")
    
    # Extract features with timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    features = encoder(video_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    inference_time = time.time() - start_time
    
    batch_size, num_patches, embed_dim = features.shape
    print(f"Output features shape: {features.shape}")
    print(f"  ├─ Batch size: {batch_size}")
    print(f"  ├─ Number of patches: {num_patches}")
    print(f"  └─ Embedding dimension: {embed_dim}")
    print(f"Inference time: {inference_time*1000:.2f} ms ({inference_time:.4f} s)")
    
    return features, inference_time


@torch.no_grad()
def predict_future(encoder, predictor, video, transform, context_ratio=0.9, device='cuda'):
    """
    Predict future frames from context frames
    
    Args:
        encoder: V-JEPA 2 encoder
        predictor: V-JEPA 2 predictor
        video: numpy array [T, H, W, C]
        transform: preprocessing transform
        context_ratio: Ratio of frames to use as context
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with context_features, predicted_features, ground_truth_features, metrics, timings
    """
    import time
    import torch.nn.functional as F
    
    timings = {}
    
    # Preprocess
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    video_tensor = transform(video_tensor)
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    B, C, T, H, W = video_tensor.shape
    
    # Split into context and future
    split_frame = int(T * context_ratio)
    split_frame = max(1, min(split_frame, T-1))
    
    context_video = video_tensor[:, :, :split_frame, :, :]
    future_video = video_tensor[:, :, split_frame:, :, :]
    
    print(f"\nContext frames: {split_frame} ({split_frame/T*100:.1f}%)")
    print(f"Future frames: {T-split_frame} ({(T-split_frame)/T*100:.1f}%)")
    
    # Encode context
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    context_features = encoder(context_video)
    if device == 'cuda':
        torch.cuda.synchronize()
    timings['context_encoding'] = time.time() - start
    
    print(f"Context features: {context_features.shape} | Time: {timings['context_encoding']*1000:.2f} ms")
    
    # Encode ground truth future
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    ground_truth_features = encoder(future_video)
    if device == 'cuda':
        torch.cuda.synchronize()
    timings['gt_encoding'] = time.time() - start
    
    print(f"Ground truth features: {ground_truth_features.shape} | Time: {timings['gt_encoding']*1000:.2f} ms")
    
    # Predict future
    num_context_patches = context_features.shape[1]
    num_future_patches = ground_truth_features.shape[1]
    
    mask_context = torch.arange(num_context_patches, device=device).unsqueeze(0)
    mask_future = torch.arange(num_future_patches, device=device).unsqueeze(0)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    predicted_features = predictor(context_features, [mask_context], [mask_future])
    if device == 'cuda':
        torch.cuda.synchronize()
    timings['prediction'] = time.time() - start
    
    if isinstance(predicted_features, list):
        predicted_features = predicted_features[0]
        if isinstance(predicted_features, list):
            predicted_features = predicted_features[0]
    
    print(f"Predicted features: {predicted_features.shape} | Time: {timings['prediction']*1000:.2f} ms")
    print(f"  ├─ Batch size: {predicted_features.shape[0]}")
    print(f"  ├─ Number of patches: {predicted_features.shape[1]}")
    print(f"  └─ Embedding dimension: {predicted_features.shape[2]}")
    
    # Compute metrics
    metrics = {}
    if predicted_features.shape == ground_truth_features.shape:
        mae = torch.abs(predicted_features - ground_truth_features).mean().item()
        cos_sim = F.cosine_similarity(
            predicted_features, ground_truth_features, dim=-1
        ).mean().item()
        metrics = {'mae': mae, 'cosine_similarity': cos_sim}
        print(f"\nPrediction Quality:")
        print(f"  ├─ MAE: {mae:.4f}")
        print(f"  └─ Cosine Similarity: {cos_sim:.4f}")
    
    timings['total'] = sum(timings.values())
    
    return {
        'context_features': context_features,
        'predicted_features': predicted_features,
        'ground_truth_features': ground_truth_features,
        'metrics': metrics,
        'timings': timings
    }


def main():
    # Configuration
    video_path = "sample_video.mp4"  # Change to your video path
    img_size = 256
    num_frames = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("V-JEPA 2 ViT-Large Inference using torch.hub")
    print("="*80)
    
    # Step 1: Load models using torch.hub
    print(f"\n1. Loading ViT-Large models via torch.hub...")
    print("   (This will download pretrained weights if not cached)")
    
    encoder, predictor = torch.hub.load(
        'facebookresearch/vjepa2',
        'vjepa2_vit_large',
        pretrained=True,
        num_frames=num_frames
    )
    
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    
    print(f"   ✓ Encoder loaded: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M parameters")
    print(f"   ✓ Predictor loaded: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M parameters")
    print(f"   ✓ ViT-Large config: embed_dim={encoder.embed_dim}, depth=24, num_heads=16")
    
    # Step 2: Load video
    print(f"\n2. Loading video from: {video_path}")
    video = load_video(video_path, num_frames=num_frames)
    print(f"   Video shape: {video.shape} (T, H, W, C)")
    
    # Step 3: Build transform
    print(f"\n3. Building video transform (img_size={img_size})")
    transform = build_video_transform(img_size=img_size)
    
    # Step 4: Extract features
    print(f"\n4. Extracting features (device={device})")
    features, inference_time = extract_features(encoder, video, transform, device=device)
    
    print(f"\n5. Feature extraction complete!")
    print(f"   Features shape: {features.shape}")
    print(f"   Features statistics:")
    print(f"     ├─ Mean: {features.mean().item():.4f}")
    print(f"     ├─ Std: {features.std().item():.4f}")
    print(f"     ├─ Min: {features.min().item():.4f}")
    print(f"     └─ Max: {features.max().item():.4f}")
    print(f"   Throughput: {num_frames/inference_time:.2f} fps")
    
    # Step 5: Predict future frames (optional)
    print(f"\n{'='*80}")
    print("Bonus: Future Frame Prediction")
    print("="*80)
    
    results = predict_future(
        encoder, predictor, video, transform,
        context_ratio=0.9, device=device
    )
    
    # Save features (optional)
    output_path = "vitl_features.pt"
    torch.save({
        'features': features.cpu(),
        'context_features': results['context_features'].cpu(),
        'predicted_features': results['predicted_features'].cpu(),
        'ground_truth_features': results['ground_truth_features'].cpu(),
        'metrics': results['metrics'],
        'inference_time': inference_time,
        'timings': results['timings'],
    }, output_path)
    print(f"\n6. Features saved to: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: ViT-Large (embed_dim={encoder.embed_dim})")
    print(f"Input: {num_frames} frames @ {img_size}x{img_size}")
    print(f"Output: {features.shape[1]} patches x {features.shape[2]} dimensions")
    print(f"\nInference Times:")
    print(f"  ├─ Feature extraction: {inference_time*1000:.2f} ms ({num_frames/inference_time:.2f} fps)")
    print(f"  ├─ Context encoding: {results['timings']['context_encoding']*1000:.2f} ms")
    print(f"  ├─ GT encoding: {results['timings']['gt_encoding']*1000:.2f} ms")
    print(f"  ├─ Prediction: {results['timings']['prediction']*1000:.2f} ms")
    print(f"  └─ Total prediction time: {results['timings']['total']*1000:.2f} ms")
    print(f"\nPrediction Quality:")
    print(f"  ├─ MAE: {results['metrics']['mae']:.4f}")
    print(f"  └─ Cosine Similarity: {results['metrics']['cosine_similarity']:.4f}")
    print("="*80)
    
    return features, results


if __name__ == "__main__":
    main()

