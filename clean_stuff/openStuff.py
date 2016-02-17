import pickle
from pprint import pprint
stuff = {}
morestuff = {}
with open('train.pkl','r') as dmp:
    stuff = pickle.load(dmp)
    dmp.close()
with open('holdout.pkl','r') as dmp:
    morestuff = pickle.load(dmp)
    dmp.close()
print(len(stuff['name']))
print(len(morestuff['name']))
