import json
import pickle
import numpy as np
# import scikit learn as sklearn
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,total_sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return  round(__model.predict([x])[0],2)



def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
        # as the locations start from the 3 index so in the above line we have performed slicing

    with open("./artifacts/banglore_home_prices_prediction_model",'rb') as f:
        __model = pickle.load(f)
        print(('loading the saved artifacts is done '))
def get_location_names():
    # load_saved_artifacts()
   return __locations
def get_data_columns():
   return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()

    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar',1000,3,3))
    print(get_estimated_price('2nd stage nagarbhavi',1000,2,2))
    # other location test
    print(get_estimated_price('kalhall',1000,2,3))
    print(get_estimated_price('igatpuri',1000,4,5))