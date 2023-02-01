from pandas import read_csv
import numpy as np

def load_data():
    data = read_csv('BrainAllometry_Supplement_Data.csv')
    data = data[['Mean_body_mass_g', 'Mean_brain_mass_g']]
    body_mass = data['Mean_body_mass_g']
    brain_mass = data['Mean_brain_mass_g']
    return np.array([body_mass, brain_mass], dtype=np.float32)
