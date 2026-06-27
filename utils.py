import os
import json
import requests
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def download_kinetics_labels(url: str, output_path: str) -> dict[int, str]:
    """
    Downloads Kinetics-400 labels mapping if it doesn't exist locally,
    and returns a mapping from class index (int) to action name (str).
    """
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Downloading Kinetics-400 labels from {url}...")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(output_path, "w") as f:
                f.write(response.text)
            logger.info(f"Saved Kinetics-400 labels to {output_path}")
        except Exception as e:
            logger.error(f"Failed to download Kinetics-400 labels: {e}")
            # Fallback mock map if download fails
            logger.warning("Using partial fallback action labels dictionary.")
            return {i: f"action_class_{i}" for i in range(400)}
            
    try:
        with open(output_path, "r") as f:
            classnames = json.load(f)
        # Reverse mapping: classnames is {classname: id} -> we want {id: classname}
        id_to_name = {int(v): k for k, v in classnames.items()}
        return id_to_name
    except Exception as e:
        logger.error(f"Error reading local labels file {output_path}: {e}")
        return {i: f"action_class_{i}" for i in range(400)}

def download_sample_video(url: str, output_path: str):
    """
    Downloads a sample test video chunk by chunk from a public URL.
    """
    if os.path.exists(output_path):
        logger.info(f"Sample video already exists at {output_path}")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Downloading sample video from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Download chunk by chunk
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Sample video downloaded successfully and saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download sample video: {e}")
        raise

def annotate_video(
    video_path: str,
    predictions: list[tuple[str, float]],
    output_path: str
):
    """
    Overlays action predictions and confidence scores on every frame of the video
    and saves the annotated video.
    
    Args:
        video_path: Path to the original video file.
        predictions: List of tuples (action_name, confidence).
        output_path: Path to save the annotated output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video file for annotation: {video_path}")
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        
    # Use MP4V codec for saving
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Annotating video. Resolution: {width}x{height}, FPS: {fps}")
    
    # Prepare overlay text lines
    text_lines = []
    text_lines.append("Predictions (Kinetics-400):")
    for i, (action, prob) in enumerate(predictions):
        text_lines.append(f"{i+1}. {action.replace('_', ' ').title()}: {prob*100:.1f}%")
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw a semi-transparent black rectangle background for the overlay text
        # to ensure the text is highly readable
        overlay = frame.copy()
        
        # Calculate overlay dimensions based on text length and resolution
        rect_width = int(width * 0.45) if width > 600 else int(width * 0.75)
        rect_height = 30 + len(text_lines) * 25
        
        # Ensure rectangular region is within bounds
        rect_width = min(rect_width, width - 20)
        rect_height = min(rect_height, height - 20)
        
        cv2.rectangle(overlay, (10, 10), (10 + rect_width, 10 + rect_height), (0, 0, 0), -1)
        
        # Apply semi-transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text lines
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55 if width > 640 else 0.45
        font_color = (255, 255, 255) # White
        thickness = 1
        
        y_offset = 30
        for line in text_lines:
            # Highlight the top predicted class in green-yellow
            if line.startswith("1."):
                color = (0, 255, 127) # Spring Green
            elif line.startswith("Predictions"):
                color = (255, 191, 0) # Amber/Gold
            else:
                color = font_color
                
            cv2.putText(frame, line, (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += 25
            
        out.write(frame)
        
    cap.release()
    out.release()
    logger.info(f"Saved annotated video to {output_path}")

def visualize_prediction(
    video_path: str,
    predicted_action: str,
    confidence: float,
    model_name: str,
    inference_time: float
):
    """
    Plays the input video in an OpenCV window, overlaying the model predictions,
    confidence scores, model name, inference time, playback FPS, and frame progress.
    
    Args:
        video_path: Path to the input video file.
        predicted_action: Name of the top predicted action.
        confidence: Confidence score as a float in [0.0, 1.0].
        model_name: Name of the model architecture.
        inference_time: Inference execution time in milliseconds.
    """
    import time
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video file for visualization: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0
        
    # Standard frame delay in ms to match video FPS
    frame_delay = max(1, int(1000 / video_fps))
    
    window_name = "AI Human Action Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Calculate target size: fit comfortably on screen (e.g. width 960)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        orig_w, orig_h = 640, 480
        
    target_w = 960
    aspect_ratio = orig_h / orig_w
    target_h = int(target_w * aspect_ratio)
    
    cv2.resizeWindow(window_name, target_w, target_h)
    
    frame_idx = 0
    t0 = time.time()
    
    # Prepare text overlays
    title_text = "AI Human Action Recognition"
    action_text = f"Action: {predicted_action.replace('_', ' ').title()}"
    confidence_text = f"Confidence: {confidence * 100:.2f}%"
    model_text = f"Model: {model_name}"
    inf_time_text = f"Inference Time: {inference_time:.1f} ms"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Calculate playback FPS dynamically
        t1 = time.time()
        elapsed = t1 - t0
        t0 = t1
        fps_val = 1.0 / elapsed if elapsed > 0 else video_fps
        fps_text = f"Playback FPS: {fps_val:.1f}"
        
        # Resize frame for display
        display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Draw semi-transparent panel on the top-left of the display_frame
        overlay = display_frame.copy()
        panel_w = 420
        panel_h = 190
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 20), -1)
        
        # Semi-transparent opacity
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        # Draw Text
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        
        # Title (Amber/Orange)
        cv2.putText(
            display_frame, title_text, (25, 45),
            font, 0.7, (0, 165, 255), thickness + 1, cv2.LINE_AA
        )
        
        # Divider line
        cv2.line(display_frame, (25, 55), (15 + panel_w - 10, 55), (80, 80, 80), 1, cv2.LINE_AA)
        
        # Action (Green) - OpenCV BGR format: (0, 255, 0)
        cv2.putText(
            display_frame, action_text, (25, 85),
            font, 0.6, (0, 255, 0), thickness, cv2.LINE_AA
        )
        
        # Confidence (Yellow) - OpenCV BGR format: (0, 255, 255)
        cv2.putText(
            display_frame, confidence_text, (25, 113),
            font, 0.6, (0, 255, 255), thickness, cv2.LINE_AA
        )
        
        # Model Name (Cyan) - OpenCV BGR format: (255, 255, 0)
        cv2.putText(
            display_frame, model_text, (25, 141),
            font, 0.6, (255, 255, 0), thickness, cv2.LINE_AA
        )
        
        # Stats: FPS, Frame index, Inference time (White / Gray)
        stats_text = f"Frame: {frame_idx}/{total_frames} | {inf_time_text}"
        cv2.putText(
            display_frame, stats_text, (25, 169),
            font, 0.5, (220, 220, 220), thickness, cv2.LINE_AA
        )
        
        # Playback FPS
        cv2.putText(
            display_frame, fps_text, (25, 190),
            font, 0.45, (160, 160, 160), thickness, cv2.LINE_AA
        )
        
        cv2.imshow(window_name, display_frame)
        
        # Wait key and allow exit on 'q' or ESC
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
