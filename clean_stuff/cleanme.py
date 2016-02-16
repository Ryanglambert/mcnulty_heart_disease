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


    ret_dict = {}
    for variable in varNames:
        ret_dict[variable] = []
    i = 0
    with open('clevelandData.txt','r+') as read:
        reader = read.readlines()
        for line in reader:
            spl = line.split()
            for elem in spl:
                if elem == 'name':
                    k = get_key_from_index(varNamesNew,i)
                    ret_dict[k].append(elem)
                    i=0
                else:
                    k = get_key_from_index(varNamesNew,i)
                    ret_dict[k].append(elem)
                    i+=1
        read.close()
    return ret_dict

theDict = get_list_dicts(getVarNames())
with open('cleaned.pkl','w') as dmp:
    pickle.dump(theDict,dmp)
    dmp.close()
