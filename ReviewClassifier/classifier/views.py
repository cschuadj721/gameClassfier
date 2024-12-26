# classifier/views.py

from django.shortcuts import render
from django.apps import apps
from .custom_layers import AttentionWeightedSum, ReduceSumCustom

def index(request):
    prediction = None
    if request.method == 'POST':
        review = request.POST.get('review', '')
        if review.strip():
            # Access the loaded model and resources from AppConfig
            classifier_config = apps.get_app_config('classifier')

            model = classifier_config.model
            tokenizer = classifier_config.tokenizer
            label_encoder = classifier_config.label_encoder
            max_length = classifier_config.max_length

            if model:
                # Preprocess the review
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                import numpy as np
                from tensorflow.keras.preprocessing.sequence import pad_sequences

                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))

                def preprocess_text(sentence):
                    tokens = word_tokenize(sentence)
                    tokens = [word.lower() for word in tokens if word.isalpha()]
                    tokens = [word for word in tokens if word not in stop_words]
                    tokens = [lemmatizer.lemmatize(word) for word in tokens]
                    return ' '.join(tokens)

                preprocessed_review = preprocess_text(review)

                # Convert to sequence
                sequence = tokenizer.texts_to_sequences([preprocessed_review])

                # Pad sequence
                padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

                # Predict
                prediction_prob = model.predict(padded_sequence)
                predicted_index = np.argmax(prediction_prob, axis=1)
                predicted_genre = label_encoder.inverse_transform(predicted_index)[0]

                prediction = predicted_genre
            else:
                prediction = "Model not loaded."
        else:
            prediction = "Please enter a valid review."

    return render(request, 'classifier/index.html', {'prediction': prediction})
