#!/usr/bin/env python3
"""
V-JEPA 2 Feature Extraction and Future Frame Prediction

Unified inference on images: extract info-rich features and predict future frames.

Main Function:
    inference_frame_features() - Extract features from images and predict future

Input Options:
    • Series of image files (load_images_from_files)
    • Images from directory (load_images_from_directory)
    • Single image (repeat to create sequence)
    • Your own numpy array: (T, H, W, C)

Example:
    # Load models
    encoder, predictor = load_models("vitg.pt")
    transform = build_video_transform()
    
    # Load your images
    frames = load_images_from_files(["img1.jpg", "img2.jpg", ...])
    # OR: frames = load_images_from_directory("path/to/images")
    # OR: frames = your_camera_frames  # (T, H, W, C) numpy array
    
    # Run inference
    results = inference_frame_features(
        encoder, predictor, frames, transform,
        num_future_frames=4
    )
    
    # Use the features
    context_features = results['context_features']      # Current state
    predicted_features = results['predicted_features']  # Future prediction
"""

import sys
import os
from pathlib import Path

# Add project root to path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import time
from decord import VideoReader, cpu
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope
from src.models.predictor import vit_predictor
from src.masks.utils import apply_masks
import torch.nn.functional as F
from typing import Union, Dict, Optional

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
    """Load video frames from a video file"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    if total_frames <= num_frames:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames


def load_images_from_files(image_paths, target_size=None):
    """
    Load a series of images from file paths
    
    Args:
        image_paths: List of image file paths
        target_size: Optional (H, W) to resize images. If None, keeps original size
    
    Returns:
        frames: numpy array of shape (T, H, W, C)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required to load images. Install with: pip install Pillow")
    
    frames = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        if target_size is not None:
            img = img.resize((target_size[1], target_size[0]))  # PIL uses (W, H)
        frames.append(np.array(img))
    
    frames = np.stack(frames)  # (T, H, W, C)
    return frames


def load_images_from_directory(directory_path, num_frames=16, pattern="*.jpg"):
    """
    Load images from a directory
    
    Args:
        directory_path: Path to directory containing images
        num_frames: Number of frames to load
        pattern: File pattern (e.g., "*.jpg", "*.png", "frame_*.jpg")
    
    Returns:
        frames: numpy array of shape (T, H, W, C)
    """
    import glob
    from pathlib import Path
    
    # Find all matching images
    image_paths = sorted(glob.glob(str(Path(directory_path) / pattern)))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {directory_path} matching pattern {pattern}")
    
    # Sample num_frames images
    if len(image_paths) > num_frames:
        indices = np.linspace(0, len(image_paths) - 1, num_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]
    
    print(f"Loading {len(image_paths)} images from {directory_path}")
    return load_images_from_files(image_paths)


def load_models(checkpoint_path, img_size=256, num_frames=16, device='cuda'):
    """
    Load both encoder and predictor from checkpoint
    
    Returns:
        encoder: Vision Transformer encoder
        predictor: Predictor for masked frames
    """
    # Initialize encoder
    encoder = vit_giant_xformers_rope(
        img_size=(img_size, img_size),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
    )
    
    # Initialize predictor
    predictor = vit_predictor(
        img_size=(img_size, img_size),
        patch_size=16,
        num_frames=num_frames,
        tubelet_size=2,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        use_mask_tokens=True,
        num_mask_tokens=10,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load encoder weights
    encoder_state = checkpoint.get('encoder', checkpoint.get('target_encoder', checkpoint))
    cleaned_encoder = {k.replace("module.", "").replace("backbone.", ""): v 
                      for k, v in encoder_state.items()}
    encoder.load_state_dict(cleaned_encoder, strict=False)
    
    # Load predictor weights
    predictor_state = checkpoint.get('predictor', {})
    cleaned_predictor = {k.replace("module.", "").replace("backbone.", ""): v 
                        for k, v in predictor_state.items()}
    predictor.load_state_dict(cleaned_predictor, strict=False)
    
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    
    print(f"Models loaded from {checkpoint_path}")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")
    print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M")
    
    return encoder, predictor


def create_temporal_mask(num_frames, patch_h, patch_w, mask_ratio=0.5):
    """
    Create a simple temporal mask - mask future frames
    
    Args:
        num_frames: Number of temporal patches (frames / tubelet_size)
        patch_h: Patches in height
        patch_w: Patches in width
        mask_ratio: Ratio of frames to mask
    
    Returns:
        mask_enc: Boolean mask for encoder (True = visible)
        mask_pred: Boolean mask for predictor (True = predict)
    """
    total_patches = num_frames * patch_h * patch_w
    num_context_frames = int(num_frames * (1 - mask_ratio))
    
    # Create mask: [T, H, W]
    mask_enc = torch.zeros(num_frames, patch_h, patch_w, dtype=torch.bool)
    mask_pred = torch.zeros(num_frames, patch_h, patch_w, dtype=torch.bool)
    
    # Encoder sees first few frames
    mask_enc[:num_context_frames] = True
    
    # Predictor predicts remaining frames
    mask_pred[num_context_frames:] = True
    
    # Flatten to [T*H*W]
    mask_enc = mask_enc.flatten()
    mask_pred = mask_pred.flatten()
    
    return mask_enc, mask_pred


@torch.no_grad()
def inference_frame_features(
    encoder, 
    predictor, 
    frames: Union[np.ndarray, torch.Tensor], 
    transform,
    context_ratio: float = 0.9,
    device: str = 'cuda',
    normalize_features: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Simplified inference: Split video, encode context, predict future, compare with ground truth.
    
    Logic:
        1. Split video into context (90%) and future (10%) parts
        2. Encode context frames → context_features
        3. Encode ground truth future frames → ground_truth_features
        4. Use predictor to predict future → predicted_features
        5. Compare predicted vs ground truth (MAE and Cosine Similarity)
    
    Args:
        encoder: Pretrained V-JEPA encoder model
        predictor: Pretrained V-JEPA predictor model
        frames: Input frames - numpy array (T, H, W, C) or torch tensor
        transform: Video preprocessing transform (from build_video_transform)
        context_ratio: Ratio of frames to use as context (default: 0.9 = 90% context, 10% predict)
        device: Device to run inference on ('cuda' or 'cpu')
        normalize_features: Whether to apply layer normalization to features
        verbose: Whether to print progress information
    
    Returns:
        Dictionary containing:
            'context_features': [1, N_context, D] - Encoded context features
            'predicted_features': [1, N_future, D] - Predicted future features
            'ground_truth_features': [1, N_future, D] - Actual encoded future features
            'metrics': {'mae': float, 'cosine_similarity': float}
            'timings': dict - Inference timing breakdown
    
    Example:
        >>> encoder, predictor = load_models("vitg.pt", num_frames=16)
        >>> transform = build_video_transform()
        >>> frames = load_video("video.mp4", num_frames=16)
        >>> 
        >>> # 90% context (14.4 frames), 10% predict (1.6 frames)
        >>> results = inference_frame_features(
        ...     encoder, predictor, frames, transform,
        ...     context_ratio=0.9
        ... )
        >>> 
        >>> print(f"MAE: {results['metrics']['mae']:.4f}")
        >>> print(f"Cosine Sim: {results['metrics']['cosine_similarity']:.4f}")
    """
    # Start total timing
    start_total = time.time()
    timings = {}
    
    if verbose:
        print(f"\n{'='*80}")
        print("Simplified Inference Pipeline")
        print(f"{'='*80}")
    
    # ========== Step 1: Prepare and preprocess frames ==========
    start_prep = time.time()
    if isinstance(frames, np.ndarray):
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        video_tensor = transform(video_tensor)  # -> (C,T,H,W)
    else:
        if frames.ndim == 4:
            video_tensor = transform(frames) if frames.shape[1] <= 4 else frames
        elif frames.ndim == 5:
            video_tensor = frames.squeeze(0) if frames.shape[0] == 1 else frames[0]
        else:
            raise ValueError(f"Unsupported frame tensor shape: {frames.shape}")
    
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dim: [1, C, T, H, W]
    video_tensor = video_tensor.to(device)
    
    B, C, T, H, W = video_tensor.shape
    timings['preprocessing'] = time.time() - start_prep
    
    # ========== Step 2: Split video into context and future parts ==========
    # Calculate split point (90% context, 10% future)
    split_frame = int(T * context_ratio)
    split_frame = max(1, min(split_frame, T-1))  # Ensure at least 1 frame on each side
    
    context_video = video_tensor[:, :, :split_frame, :, :]  # [1, C, T_context, H, W]
    future_video = video_tensor[:, :, split_frame:, :, :]   # [1, C, T_future, H, W]
    
    if verbose:
        print(f"\n1. Video Split:")
        print(f"   Total frames: {T}")
        print(f"   Context frames: {split_frame} ({split_frame/T*100:.1f}%)")
        print(f"   Future frames: {T-split_frame} ({(T-split_frame)/T*100:.1f}%)")
        print(f"   Context tensor: {context_video.shape}")
        print(f"   Future tensor: {future_video.shape}")
    
    # ========== Step 3: Encode context frames ==========
    if verbose:
        print(f"\n2. Encoding Context Frames...")
    
    start_ctx = time.time()
    context_features = encoder(context_video)  # [1, N_context, D]
    if normalize_features:
        context_features = F.layer_norm(context_features, (context_features.size(-1),))
    timings['context_encoding'] = time.time() - start_ctx
    
    if verbose:
        print(f"   Context features: {context_features.shape}")
    
    # ========== Step 4: Encode ground truth future frames ==========
    if verbose:
        print(f"\n3. Encoding Ground Truth Future Frames...")
    
    start_gt = time.time()
    ground_truth_features = encoder(future_video)  # [1, N_future, D]
    if normalize_features:
        ground_truth_features = F.layer_norm(ground_truth_features, (ground_truth_features.size(-1),))
    timings['ground_truth_encoding'] = time.time() - start_gt
    
    if verbose:
        print(f"   Ground truth features: {ground_truth_features.shape}")
    
    # ========== Step 5: Predict future from context using predictor ==========
    if verbose:
        print(f"\n4. Predicting Future Features...")
    
    start_pred = time.time()
    
    # Create masks for predictor
    # Context mask: all context patches are visible
    num_context_patches = context_features.shape[1]
    mask_context = torch.arange(num_context_patches, device=device).unsqueeze(0)  # [1, N_context]
    
    # Future mask: indices for patches to predict
    num_future_patches = ground_truth_features.shape[1]
    mask_future = torch.arange(num_future_patches, device=device).unsqueeze(0)  # [1, N_future]
    
    # Predictor takes context features and mask indices
    predicted_features = predictor(context_features, [mask_context], [mask_future])
    
    # Handle predictor output format
    if isinstance(predicted_features, list):
        predicted_features = predicted_features[0]
        if isinstance(predicted_features, list):
            predicted_features = predicted_features[0]
    
    timings['prediction'] = time.time() - start_pred
    
    if verbose:
        print(f"   Predicted features: {predicted_features.shape}")
    
    # ========== Step 6: Compare predicted vs ground truth ==========
    if verbose:
        print(f"\n5. Computing Metrics...")
    
    metrics = {}
    if predicted_features.shape == ground_truth_features.shape:
        mae = torch.abs(predicted_features - ground_truth_features).mean().item()
        cos_sim = F.cosine_similarity(
            predicted_features, ground_truth_features, dim=-1
        ).mean().item()
        metrics = {'mae': mae, 'cosine_similarity': cos_sim}
        
        if verbose:
            print(f"   MAE: {mae:.4f}")
            print(f"   Cosine Similarity: {cos_sim:.4f}")
    else:
        if verbose:
            print(f"   ⚠️  Shape mismatch!")
            print(f"   Predicted: {predicted_features.shape}")
            print(f"   Ground truth: {ground_truth_features.shape}")
    
    timings['total'] = time.time() - start_total
    
    # ========== Step 7: Summary ==========
    if verbose:
        print(f"\n{'='*80}")
        print("Timing Summary:")
        print(f"  Preprocessing: {timings['preprocessing']*1000:.2f} ms")
        print(f"  Context encoding: {timings['context_encoding']*1000:.2f} ms")
        print(f"  Ground truth encoding: {timings['ground_truth_encoding']*1000:.2f} ms")
        print(f"  Prediction: {timings['prediction']*1000:.2f} ms")
        print(f"  Total: {timings['total']*1000:.2f} ms ({timings['total']:.3f} s)")
        print(f"{'='*80}\n")
    
    return {
        'context_features': context_features,
        'predicted_features': predicted_features,
        'ground_truth_features': ground_truth_features,
        'metrics': metrics,
        'timings': timings,
        'feature_dim': context_features.shape[-1],
        'num_context_patches': context_features.shape[1],
        'num_predicted_patches': predicted_features.shape[1],
    }


def save_results(results, output_path='inference_results.pt', save_features=True):
    """
    Save inference results to disk
    
    Args:
        results: Dictionary returned from inference_frame_features()
        output_path: Path to save results (default: 'inference_results.pt')
        save_features: If True, saves full feature tensors. If False, only saves metadata
    
    Returns:
        Path to saved file
    """
    import os
    
    save_dict = {
        'feature_dim': results['feature_dim'],
        'num_context_patches': results['num_context_patches'],
        'num_predicted_patches': results['num_predicted_patches'],
        'spatial_info': results['spatial_info'],
        'metrics': results['metrics'],
        'timings': results['timings'],
    }
    
    if save_features:
        save_dict.update({
            'context_features': results['context_features'].cpu(),
            'predicted_features': results['predicted_features'].cpu(),
        })
        if results['full_features'] is not None:
            save_dict['full_features'] = results['full_features'].cpu()
        if results['ground_truth_features'] is not None:
            save_dict['ground_truth_features'] = results['ground_truth_features'].cpu()
    
    torch.save(save_dict, output_path)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")
    
    return output_path


def main():
    """
    Demonstrate unified inference on a series of images (or single image)
    """
    # Configuration
    checkpoint_path = "vitg.pt"
    img_size = 256
    num_frames = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("V-JEPA 2: Inference on Images")
    print("="*80)
    print("\nThis script demonstrates how to run inference on:")
    print("  • A series of images from files")
    print("  • Images from a directory")
    print("  • A single image (repeated)")
    print("  • Your own numpy arrays (e.g., from camera)")
    
    # ========== Choose your input method ==========
    
    # Option 1: Load from video file (for demo purposes)
    video_path = "sample_episode.mp4"
    print(f"\n[DEMO] Loading frames from video: {video_path}")
    frames = load_video(video_path, num_frames=num_frames)
    print(f"✓ Loaded frames shape: {frames.shape} (T, H, W, C)")
    
    # Option 2: Load from list of image files
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", ...]
    # frames = load_images_from_files(image_paths)
    # print(f"✓ Loaded {len(frames)} images: {frames.shape}")
    
    # Option 3: Load from directory
    # frames = load_images_from_directory("path/to/images", num_frames=16, pattern="*.jpg")
    # print(f"✓ Loaded frames: {frames.shape}")
    
    # Option 4: Use your own numpy array (e.g., from camera feed)
    # frames = your_camera_frames  # Must be shape (T, H, W, C)
    # Where T = number of frames, H = height, W = width, C = 3 (RGB)
    
    # Option 5: Single image (repeat it to create temporal sequence)
    # from PIL import Image
    # single_image = np.array(Image.open("single_frame.jpg"))  # (H, W, C)
    # frames = np.repeat(single_image[np.newaxis, ...], num_frames, axis=0)  # (T, H, W, C)
    # print(f"✓ Created sequence from single image: {frames.shape}")
    
    # Build transform
    print(f"\nBuilding transform (img_size={img_size})")
    transform = build_video_transform(img_size=img_size)
    
    # Load models
    print(f"\nLoading encoder and predictor from: {checkpoint_path}")
    encoder, predictor = load_models(checkpoint_path, img_size=img_size, 
                                     num_frames=num_frames, device=device)
    
    # Run simplified inference
    results = inference_frame_features(
        encoder=encoder,
        predictor=predictor,
        frames=frames,  # Your frames: (T, H, W, C) numpy array
        transform=transform,
        context_ratio=0.9,  # Use 90% for context, 10% for prediction
        device=device,
        normalize_features=True,
        verbose=True
    )
    
    # Display final summary
    print("="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n✓ Context Features: {results['context_features'].shape}")
    print(f"✓ Predicted Features: {results['predicted_features'].shape}")
    print(f"✓ Ground Truth Features: {results['ground_truth_features'].shape}")
    print(f"\n✓ MAE: {results['metrics']['mae']:.4f}")
    print(f"✓ Cosine Similarity: {results['metrics']['cosine_similarity']:.4f}")
    print(f"\n✓ Total Time: {results['timings']['total']:.3f} s")
    
    # Optionally save results
    # save_results(results, 'inference_results.pt', save_features=True)
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Example: Save results for later use
    # save_results(results, 'my_features.pt', save_features=True)
    
    # Example: Load saved results
    # loaded = torch.load('my_features.pt')
    # context_features = loaded['context_features']
    # predicted_features = loaded['predicted_features']

