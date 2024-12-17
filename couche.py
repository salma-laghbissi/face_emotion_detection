from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('facialemotionmodel.h5')

# Afficher le résumé du modèle
model.summary()
