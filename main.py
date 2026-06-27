import argparse
import sys
import os
import logging

# Ensure project root is in python path to resolve local imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    KINETICS_LABELS_URL,
    KINETICS_LABELS_PATH,
    DEFAULT_VIDEO_URL,
    DEFAULT_VIDEO_PATH,
    NUM_FRAMES,
    RESIZE_SIZE,
    CROP_SIZE,
    NORM_MEAN,
    NORM_STD
)
from preprocessing import sample_frames, preprocess_video
from models import ActionRecognitionModel
from utils import download_sample_video, annotate_video, visualize_prediction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("action_recognition")

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Human Action Recognition System"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to the input video file. If not provided, a default sample video will be downloaded."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(SUPPORTED_MODELS.keys()),
        help=f"TorchVision 3D video classification model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save an annotated version of the video with prediction overlays."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_annotated.mp4",
        help="Path to save the annotated video (default: output_annotated.mp4)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force inference to run on CPU even if CUDA/GPU is available."
    )
    
    args = parser.parse_args()
    
    logger.info("=== Starting Human Action Recognition System ===")
    
    # 1. Resolve inputs
    video_path = args.video
    if not video_path:
        logger.info("No input video specified. Preparing default test video...")
        try:
            download_sample_video(DEFAULT_VIDEO_URL, DEFAULT_VIDEO_PATH)
            video_path = DEFAULT_VIDEO_PATH
        except Exception as e:
            logger.critical(f"Failed to obtain default test video: {e}")
            sys.exit(1)
            
    if not os.path.exists(video_path):
        logger.error(f"Input video file not found: {video_path}")
        sys.exit(1)
        
    # 2. Model class mapping is retrieved natively from torchvision model weights
    
    # 3. Preprocess Video
    logger.info(f"Sampling {NUM_FRAMES} frames from {video_path}...")
    try:
        raw_frames = sample_frames(video_path, num_frames=NUM_FRAMES)
        logger.info(f"Successfully sampled {len(raw_frames)} frames.")
    except Exception as e:
        logger.error(f"Error during frame sampling: {e}")
        sys.exit(1)
        
    logger.info("Preprocessing frame sequence...")
    try:
        video_tensor = preprocess_video(
            raw_frames,
            resize_size=RESIZE_SIZE,
            crop_size=CROP_SIZE,
            mean=NORM_MEAN,
            std=NORM_STD
        )
        logger.info(f"Input tensor shape: {video_tensor.shape}") # Expect (1, C, T, H, W)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)
        
    # 4. Load Model
    use_gpu = not args.cpu_only
    try:
        model = ActionRecognitionModel(model_name=args.model, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"Error initializing model {args.model}: {e}")
        sys.exit(1)
        
    # 5. Inference
    logger.info("Running model inference...")
    import time
    start_time = time.time()
    try:
        raw_predictions = model.predict(video_tensor, top_k=args.top_k)
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        sys.exit(1)
    inference_time_ms = (time.time() - start_time) * 1000
        
    # Format predictions with class names
    predictions = []
    for idx, score in raw_predictions:
        if idx < len(model.categories):
            class_name = model.categories[idx]
        else:
            class_name = f"unknown_class_{idx}"
        predictions.append((class_name, score))
        
    # Display results
    print("\n" + "="*50)
    print(f" TOP {args.top_k} PREDICTED ACTIONS (Model: {args.model}) ")
    print("="*50)
    for rank, (action, score) in enumerate(predictions, 1):
        clean_action = action.replace("_", " ").title()
        print(f"Rank {rank}: {clean_action:<30} | Confidence: {score*100:6.2f}%")
    print("="*50 + "\n")
    
    # 6. Optional Video Annotation
    if args.annotate:
        logger.info("Annotating and saving video...")
        try:
            annotate_video(video_path, predictions, args.output)
            print(f"[SUCCESS] Annotated video saved to: {os.path.abspath(args.output)}\n")
        except Exception as e:
            logger.error(f"Failed to annotate video: {e}")
            sys.exit(1)
            
    # 7. Real-time Visualization Window
    top_action, top_conf = predictions[0]
    logger.info("Starting video visualization window...")
    try:
        visualize_prediction(
            video_path=video_path,
            predicted_action=top_action,
            confidence=top_conf,
            model_name=args.model,
            inference_time=inference_time_ms
        )
    except Exception as e:
        logger.error(f"Error during video visualization: {e}")
            
    logger.info("Action recognition pipeline completed successfully.")

if __name__ == "__main__":
    main()
