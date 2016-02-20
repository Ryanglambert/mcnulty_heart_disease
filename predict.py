import pickle
import numpy as np
from MKRL_test_model import ModelWithDecisionFuncWrapper
from sklearn.naive_bayes import GaussianNB
import sys

def load_prod_model(pickle_model_path):
    with open(pickle_model_path, 'rb') as pickle_model:
        return pickle.load(pickle_model)

def predict(web_form_inputs):
    model = load_prod_model('./prod_model/prod_model.pkl')
    # print web_form_inputs
    return model.predict(web_form_inputs)

def main():
    pretend_patient = [67, 1, 108, 1, 1, 1, 64] #### should throw a [1]
    prediction = predict(pretend_patient)
    print prediction
    return prediction

if __name__ == '__main__':
    main()
