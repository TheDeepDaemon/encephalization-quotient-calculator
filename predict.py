from keras.models import load_model

model_resource = '2x100-hidden-layers-model'

weight = 0.065
weight = 0.12

model = load_model(model_resource)

def predict(body_mass_grams, brain_mass_grams):
    # pred_brain_mass_grams = weight * (body_mass_grams**(2/3))
    pred_brain_mass_grams = model.predict([body_mass_grams])
    eq = brain_mass_grams / pred_brain_mass_grams
    
    print("body mass (g): ", body_mass_grams)
    print("pred brain mass (g): ", pred_brain_mass_grams)
    print("brain mass (g): ", brain_mass_grams)
    print("encephilization quotient: ", eq)
    print("_____________________________")

# in reality, the best predictor may be based on a point slope formula
# y = base * x + m * (x^(2/3))
# or some slightly more complicated formula
