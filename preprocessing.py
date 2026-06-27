import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def sample_frames(video_path: str, num_frames: int = 16) -> list[np.ndarray]:
    """
    Uniformly samples a fixed number of frames from a video.
    Handles short videos, corrupt frames, and seeks failures gracefully.
    
    Args:
        video_path: Path to the input video file.
        num_frames: Number of frames to sample.
        
    Returns:
        List of frames as NumPy arrays (BGR format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video frame count reported: {total_frames}")
    
    # If frame count is zero/negative, count manually
    if total_frames <= 0:
        total_frames = 0
        while cap.grab():
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Counted frame count manually: {total_frames}")
        
    if total_frames <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")
        
    # Calculate uniform indices
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    
    # If the video is short (less than 500 frames), sequential reading is often safer
    # and faster than seeking, which can be unstable with some codecs.
    use_sequential = total_frames < 500
    
    if use_sequential:
        frame_dict = {}
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame_dict[idx] = frame
                
        # Build the final frames list (handles duplicates correctly if np.linspace yielded any)
        for idx in indices:
            if idx in frame_dict:
                frames.append(frame_dict[idx])
            elif frame_dict:
                # Fallback to closest available frame
                closest_idx = min(frame_dict.keys(), key=lambda k: abs(k - idx))
                frames.append(frame_dict[closest_idx])
    else:
        # Seek-based reading for long videos
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # Seek failed, try to read next frame sequentially
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                elif frames:
                    # Duplicate the last frame
                    frames.append(frames[-1].copy())
                else:
                    # Try frame 0 as last resort
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_zero, frame_zero = cap.read()
                    if ret_zero:
                        frames.append(frame_zero)
                    else:
                        raise ValueError(f"Could not read frame {idx} and no fallback was found.")
                        
    cap.release()
    
    # Ensure we have exactly num_frames
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            raise ValueError(f"Could not read any frames from video: {video_path}")
            
    return frames[:num_frames]

def preprocess_video(
    frames: list[np.ndarray],
    resize_size: int = 256,
    crop_size: int = 224,
    mean: list[float] = None,
    std: list[float] = None
) -> torch.Tensor:
    """
    Transforms a list of BGR frame images into a PyTorch tensor.
    Resizes smaller edge, center-crops, converts to RGB, normalizes,
    and formats shape to (1, C, T, H, W).
    
    Args:
        frames: List of BGR frames (numpy arrays).
        resize_size: Target size for the smaller edge.
        crop_size: Spatial crop dimensions (height and width).
        mean: Normalization means for RGB channels.
        std: Normalization std deviations for RGB channels.
        
    Returns:
        Preprocessed PyTorch tensor of shape (1, C, T, H, W).
    """
    if mean is None:
        mean = [0.43216, 0.394666, 0.37645]
    if std is None:
        std = [0.22803, 0.22145, 0.216989]
        
    processed_frames = []
    
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Determine new dimensions maintaining aspect ratio (smaller side is resize_size)
        h, w, c = frame_rgb.shape
        if w < h:
            new_w = resize_size
            new_h = int(h * (resize_size / w))
        else:
            new_h = resize_size
            new_w = int(w * (resize_size / h))
            
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center crop
        crop_y = (new_h - crop_size) // 2
        crop_x = (new_w - crop_size) // 2
        frame_cropped = frame_resized[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        
        # Convert to float and scale to [0, 1]
        frame_float = frame_cropped.astype(np.float32) / 255.0
        
        # Normalize
        frame_norm = (frame_float - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        
        # Transpose from (H, W, C) to (C, H, W)
        frame_tensor = frame_norm.transpose(2, 0, 1)
        processed_frames.append(frame_tensor)
        
    # Stack along temporal dimension to get (T, C, H, W)
    video_tensor = np.stack(processed_frames, axis=0)
    
    # Transpose to (C, T, H, W)
    video_tensor = video_tensor.transpose(1, 0, 2, 3)
    
    # Convert to PyTorch tensor and add batch dimension (1, C, T, H, W)
    tensor = torch.from_numpy(video_tensor).float().unsqueeze(0)
    return tensor
