from keras.models import load_model
from create_model import old_prediction_function

def predict(body_mass_grams, brain_mass_grams):
    pred_brain_mass_grams = old_prediction_function(body_mass_grams)
    eq = brain_mass_grams / pred_brain_mass_grams
    
    print("body mass (g): ", body_mass_grams)
    print("pred brain mass (g): ", pred_brain_mass_grams)
    print("brain mass (g): ", brain_mass_grams)
    print("encephilization quotient: ", eq)
    print("_____________________________")
