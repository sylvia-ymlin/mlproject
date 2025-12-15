import pickle 
from flask import Flask, request, render_template
# Flask: A lightweight WSGI web application framework in Python.
# request: To handle incoming request data.
# render_template: To render HTML templates with dynamic data.

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler # for feature scaling
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__) # the __name__ is a special variable in Python that represents the name of the current module. When the module is run directly, __name__ is set to "__main__".

app = application  # for AWS deployment compatibility

# route for home page
@app.route('/') 
def index(): # the decorayor above binds the index function to the '/' URL
    return render_template('index.html') # render the index.html template

@app.route('/predictdata', methods=['GET', 'POST']) # route for handling form submission
def predict_datapoint():
    # in this function, we will get the data from the form and make a prediction
    if request.method == 'GET':
        # get the data from the form
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )
        pred_df = data.get_data_as_data_frame() # convert the input data to a dataframe
        print(pred_df)

        predict_pipeline = PredictPipeline() # create an instance of the PredictPipeline class
        results = predict_pipeline.predict(pred_df) # make a prediction

        return render_template('home.html', results=results[0]) # render the home.html template with the prediction result

if __name__ == "__main__":
    # run the app in the local server with debug mode on
    app.run(host='0.0.0.0', port=8080, debug=True) # run the app on the specified host and port with debug mode enabled