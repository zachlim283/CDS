# **Multimodal Sentiment Classifier**
## Requirements
See included requirements.txt. Running in an environment that supports the use of an NVidia GPU in Tensorflow is strongly recommended.

## Training the Model
The code is ready to run with no modifications, simply run 'train_combined.py' from the command line. By default, the script will create and train and then evaluate the best model as defined in our report.
The model and results will be saved to "Saved_Models/combined_model_3" by default.

Simple model parameters can be modified in the "Parameters" section at the top of the file, including the location of the saved model.

## Evaluating the Model
At the end of the train script, the model will automatically be evaluated and the results generated. If for some reason you want to re-run the evaluation, you can do so by running 'evaluate_combined.py' from the command line.

By default, the script will look into "Saved_Models/combined_model_3" to load the model. This can be changed easily in the "Parameters" section at the top of the file as well.