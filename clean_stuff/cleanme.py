from pprint import pprint
import pandas as pd
import csv
import pickle

def getVarNames():
    varNames = []
    with open('variables.txt','r+') as read:
        reader = read.readlines()
        for line in reader:
            spl = line.split()
            if spl[0].isdigit():
                varNames.append(spl[1].replace(':', ''))
        read.close()
    return varNames
def get_key_from_index(varNames,num):
    for variable in varNames:
        if variable[0] == num:
            return variable[1]


def get_list_dicts(varNames):
    i = 0
    varNamesNew = []
    for variable in varNames:
        varNamesNew.append((i,variable))
        i+=1

    holdout_dict = {}
    train_dict = {}
    for variable in varNames:
        holdout_dict[variable] = []
        train_dict[variable] = []
    i = 0
    with open('clevelandData.txt','r+') as read:
        reader = read.readlines()
        name_count = 0
        for line in reader:
            spl = line.split()
            if name_count > 211:
                for elem in spl:
                    if elem == 'name':
                        k = get_key_from_index(varNamesNew,i)
                        train_dict[k].append(elem)
                        i=0
                        name_count+=1
                    else:
                        k = get_key_from_index(varNamesNew,i)
                        train_dict[k].append(elem)
                        i+=1
            else:
                for elem in spl:
                    if elem == 'name':
                        k = get_key_from_index(varNamesNew,i)
                        holdout_dict[k].append(elem)
                        i=0
                        name_count+=1
                    else:
                        k = get_key_from_index(varNamesNew,i)
                        holdout_dict[k].append(elem)
                        i+=1
        read.close()
    return train_dict, holdout_dict

trainDict,holdout = get_list_dicts(getVarNames())
'''with open('train.pkl','w') as dmp:
    pickle.dump(trainDict,dmp)
    dmp.close()
with open('holdout.pkl','w') as dmp:
    pickle.dump(holdout,dmp)
    dmp.close()'''
