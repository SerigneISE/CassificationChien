import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Charger le modèle de reconnaissance de races de chien pré-entraîné
model = tf.keras.models.load_model('xception_trained_model.h5')

# Liste des races de chiens prises en charge par le modèle
races_chiens = ['Chihuahua', 'Labrador', 'Berger Allemand', 'Bulldog', 'Husky', 'Golden Retriever', 'Bouledogue Français', 'Beagle', 'Poodle', 'Autre']

# Fonction pour prédire la race de chien
def predict_race(image):
    # Charger et prétraiter l'image
    img = Image.open(image).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
   
    # Faire la prédiction avec le modèle
    predictions = model.predict(img_array)
    race_index = np.argmax(predictions)
    predicted_race = races_chiens[race_index]
    confidence = predictions[0][race_index]
   
    return predicted_race, confidence

# Interface Streamlit
st.title('Reconnaissance de races de chiens')

uploaded_file = st.file_uploader('Choisissez une image de chien...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Image sélectionnée', use_column_width=True)
    predicted_race, confidence = predict_race(uploaded_file)
    st.write('Race prédite :', predicted_race)
    st.write('Confiance :', confidence)

