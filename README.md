
# McNulty / Heart Disease
## Project 3 For The [Metis](http://www.thisismetis.com/) Data Science Program


A website that takes diagnostic information from someone experiencing chest pain and shows a gauge of their predicted risk of having heart disease.

The demo version (index.html) that you can run on localhost is a logisitic regression classifier. We also built a voting classifier that uses GaussianNB, SVM, and Logistic regression for the base model. Addional feature models--ones with blood pressure & cholesterol--use GuassianNB and SVM. Predictions are made using a `sklearn`  classifier trained on the Cleveland, OH subset of the [UCI Heart Disease Data Set](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). The model has been tuned to maximize recall (minimize false negatives).

Completed as project 'Mcnulty' for the 2016 [Metis](http://www.thisismetis.com/) San Francisco cohort.

### Project Authors

[Mark Jackson](https://github.com/markgjackson)
[Matt Kerrigan](https://github.com/mkerrig)
[Ryan Lambert](https://github.com/Ryanglambert)
[Ben Straate](https://github.com/bstraa)
[Nathan Thompson](https://github.com/Nathan-Thompson)


# DISCLAIMER
* This is a prototype for demonstration purposes only. Nobody involved in its creation has any medical training whatsoever. If you are experiencing chest pain, consult an actual doctor. *