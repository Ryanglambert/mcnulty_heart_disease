#!/usr/bin/python
​
import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB

import predict
from MKRL_test_model import ModelWithDecisionFuncWrapper

​
print "First line of script.<P>"
argString = str(sys.argv)
​
argArray = argString.split()
​
ageStripped = argArray[1].replace("'","").replace(",","")
age = int(ageStripped)
​
sexStripped = argArray[2].replace("'","").replace(",","")
if sexStripped == "Male":
	sex = 1
elif sexStripped == "Female":
	sex = 0
​
thalachStripped = argArray[3].replace("'","").replace(",","")
thalach = int(thalachStripped)
​
exangStripped = argArray[4].replace("'","").replace(",","")
if exangStripped == "Yes":
	exang = 1
elif exangStripped == "No":
	exang = 0
​
yearsStripped = argArray[5].replace("'","").replace(",","")
if yearsStripped == "Yes":
	years = 1
elif yearsStripped == "No":
	years = 0
	
famhistStripped = argArray[6].replace("'","").replace(",","")
if famhistStripped == "Yes":
	famhist = 1
elif famhistStripped == "No":
	famhist = 0
​
thalrestStripped = argArray[7].replace("'","").replace(",","").replace("]","")
thalrest = int(thalrestStripped)
​
​
#print "Completed section 2.<P>"
​

##### new from ryan
predicted = predict.predict(age, sex, thalach, exang, years, famhist, thalrest)

print "age: " + str(age) + "<br>"
print "sex: " + str(sex) + "<br>"
print "thalach: " + str(thalach) + "<br>"
print "exang: " + str(exang) + "<br>"
print "years: " + str(years) + "<br>"
print "famhist: " + str(famhist)  + "<br>"
print "thalrest: " + str(thalrest) + "<br>"
print "<P>"
print "Done with script."
