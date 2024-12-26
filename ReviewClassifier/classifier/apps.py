from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    def ready(self):
        import os
        import pickle
        import nltk
        from .custom_layers import AttentionWeightedSum, ReduceSumCustom
        import tensorflow as tf

        # Initialize NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Paths to model and resources
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.h5')
        TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pickle')
        ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.pickle')
        MAX_LENGTH = 297  # Must match the training max_length

        # Load Tokenizer
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Load Label Encoder
        with open(ENCODER_PATH, 'rb') as handle:
            label_encoder = pickle.load(handle)

        # Load Model
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'AttentionWeightedSum': AttentionWeightedSum,
                    'ReduceSumCustom': ReduceSumCustom
                }
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None

        # Attach to AppConfig for access in views
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = MAX_LENGTH