import pickle
import numpy as np

model = pickle.load(open('placement_model1.pkl','rb'))

model2 = pickle.load(open('placement_model2.pkl','rb'))



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