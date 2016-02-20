The model is currently chosen.  


The website will use a pickle of the model (which is wrapped to include the decision threshhold function)

To use the model you will load the pickle and just treat it like a regular sklearn model by using: 
    `model.predict(df_x)` 
    
Where `df_x` is your input variables


To use this you go to the root of this repo then type: 
`python MKRL_test_model.py`

This will put a pickle of the model into the `prod_model/` folder. 
