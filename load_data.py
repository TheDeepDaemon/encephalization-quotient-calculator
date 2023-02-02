from pandas import read_csv
import numpy as np

def load_data():
    data = read_csv('BrainAllometry_Supplement_Data.csv')
    data = data[['Mean_body_mass_g', 'Mean_brain_mass_g']]
    body_mass = data['Mean_body_mass_g']
    brain_mass = data['Mean_brain_mass_g']
    body_mass = np.array(body_mass, dtype=np.float32)
    brain_mass = np.array(brain_mass, dtype=np.float32)
    body_mass = np.reshape(body_mass, (body_mass.shape[0], 1))
    brain_mass = np.reshape(brain_mass, (brain_mass.shape[0], 1))
    return body_mass, brain_mass
