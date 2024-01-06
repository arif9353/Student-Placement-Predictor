import pickle
import numpy as np
import os

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_directory = os.path.dirname(script_path)

# Set the working directory to the script directory
os.chdir(script_directory)

# Use relative paths from the script directory
model_path = os.path.join(script_directory, 'placement_model1.pkl')
model = pickle.load(open(model_path, 'rb'))

model2_path = os.path.join(script_directory, 'placement_model2.pkl')
model2 = pickle.load(open(model2_path, 'rb'))

def predictStatus(inputValues):
    inputValues = np.array(inputValues).reshape(1,-1)
    prediction = model.predict(inputValues)
    return prediction

def predictSalary(inp):
    inp = np.array(inp).reshape(1,-1)
    prediction2 = model2.predict(inp)
    return prediction2
    
# predictStatus([0,92,78.6,73.93,73.42,70.9,71.79,0,1,1,0])

# model 2 Parameters ["Branch","10th_p","Sem 3 %","Sem 4 %","Sem 5 %","Sem 6 %","Agg_UG_p","12th_P","Diploma_P"]