import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Path to cache downloaded files
KINETICS_LABELS_PATH = os.path.join(DATA_DIR, "kinetics_classnames.json")
DEFAULT_VIDEO_PATH = os.path.join(DATA_DIR, "driving.mp4")

# Kinetics-400 dataset download URL for class names
KINETICS_LABELS_URL = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"

# Default test video from Intel IoT sample videos (driver action recognition)
DEFAULT_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/driver-action-recognition.mp4"

# Model architectures and configuration
SUPPORTED_MODELS = {
    "r3d_18": "r3d_18",
    "mc3_18": "mc3_18",
    "r2plus1d_18": "r2plus1d_18",
    "s3d": "s3d"
}

DEFAULT_MODEL = "r3d_18"

# Preprocessing parameters
NUM_FRAMES = 16          # Standard temporal frame sequence length for Kinetics models
RESIZE_SIZE = 256        # Smaller edge is resized to this
CROP_SIZE = 224          # Central crop dimensions (224x224)

# Kinetics-400 normalization parameters (RGB mean/std)
NORM_MEAN = [0.43216, 0.394666, 0.37645]
NORM_STD = [0.22803, 0.22145, 0.216989]
