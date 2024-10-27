# config.py
import logging
from pathlib import Path

# Directory Configuration
BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / "test_images"
BACKUP_DIR = TEST_DIR / 'backup'

# Supported Image Formats
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Model Configuration
MODEL_CONFIG = {
    'resnet': {
        'name': 'ResNet50V2',
        'size': (224, 224)
    },
    'efficient': {
        'name': 'EfficientNetB4',
        'size': (380, 380)
    },
    'dense': {
        'name': 'DenseNet201',
        'size': (224, 224)
    },
    'landscape': {
        'name': 'ResNet50V2',
        'size': (224, 224)
    }
}

# Category Mappings
ANIMAL_CATEGORIES = {
    'ostrich': ['ostrich', 'common_ostrich', 'African_ostrich'],
    'bird': ['bird', 'crane', 'peacock', 'hornbill', 'vulture', 'eagle', 'bustard', 'stork'],
    'elephant': ['elephant', 'African_elephant', 'Indian_elephant', 'tusker'],
    'monkey': ['monkey', 'macaque', 'baboon', 'chimpanzee', 'gorilla', 'orangutan', 'langur', 'colobus'],
    'giraffe': ['giraffe', 'reticulated_giraffe'],
    'zebra': ['zebra', 'plains_zebra', 'Grevy_zebra'],
    'lion': ['lion', 'male_lion', 'lioness', 'African_lion'],
    'lion_cub': ['lion_cub', 'cub', 'young_lion', 'cat_baby'],
    'leopard': ['leopard', 'spotted_leopard'],
    'cheetah': ['cheetah', 'spotted_cat', 'running_cat'],
    'buffalo': ['buffalo', 'African_buffalo', 'water_buffalo', 'cape_buffalo'],
    'antelope': ['antelope', 'gazelle', 'impala', 'kudu', 'springbok', 'gemsbok', 'oryx'],
    'wildebeest': ['wildebeest', 'gnu', 'blue_wildebeest'],
    'hippopotamus': ['hippopotamus', 'hippo', 'river_horse'],
    'crocodile': ['crocodile', 'Nile_crocodile', 'alligator'],
    'hyena': ['hyena', 'spotted_hyena', 'striped_hyena']
}

LANDSCAPE_CATEGORIES = {
    'river': ['river', 'stream', 'waterfall', 'waterway', 'creek', 'rapids'],
    'savannah': ['savannah', 'grassland', 'plain', 'prairie', 'veldt', 'steppe'],
    'forest': ['forest', 'woodland', 'jungle', 'trees', 'rainforest', 'grove'],
    'bush': ['bush', 'shrubland', 'thicket', 'scrub', 'brush', 'undergrowth'],
    'mountain': ['mountain', 'hill', 'cliff', 'ridge', 'peak', 'highland'],
    'wetland': ['wetland', 'marsh', 'swamp', 'bog', 'fen', 'mangrove'],
    'desert': ['desert', 'dune', 'sand', 'arid', 'wasteland'],
    'lake': ['lake', 'pond', 'reservoir', 'lagoon', 'water_body'],
    'valley': ['valley', 'gorge', 'ravine', 'canyon', 'depression']
}

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'lion_cub': 0.3,
    'bird': 0.4,
    'default': 0.35,
    'landscape': 0.15
}

# Logging Configuration
LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'handlers': [logging.StreamHandler()]
}