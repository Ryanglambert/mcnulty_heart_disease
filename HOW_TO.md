The model is currently chosen.  


The website will use a pickle of the model (which is wrapped to include the decision threshhold function)

# How do I set this up on Mark's Wordpress Server? 
1. git clone the repo to your server 
1. make sure you have these packages
    1. numpy 
    1. pickle
    1. sklearn
1. when you call `predict` from within Markscript.py make sure to specify the absolute path of the mcnulty repository on your server
1. Markscript.py will need to be run from withiin the `mcnulty_heart_disease` folder
    1. If you'd rather not, then you can follow the instructions here to point to wherever you put the `mcnulty_heart_disease` repo by following these instructions from stackoverflow: http://stackoverflow.com/questions/4383571/importing-files-from-different-folder-in-python


