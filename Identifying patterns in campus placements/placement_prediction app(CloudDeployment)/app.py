import flask
from flask import Flask, request, render_template
import pickle
import sklearn
import numpy as np

# load the model from the pickle file
model = pickle.load(open(r"C:\Users\HAI\PycharmProjects\placement_prediction\placement_model_v1.pkl", "rb"))

# create the flask app
app = Flask(__name__)


# define the home page
@app.route("/")
def home():
    return render_template("home.html")


# define the prediction page
@app.route("/predict", methods=["POST"])
def predict():
    # get the features from the form
    features = [float(x) for x in request.form.values()]
    # convert the features to a numpy array
    features = np.array(features).reshape(1, -1)
    features_list = features.tolist()
    print(features_list)
    import requests

    # NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
    API_KEY = "Hx36zI3vYKT6MRsIqbScvO5QTycUqqTUPPXpwgAVcJ2c"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token',
                                   data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": ["CGPA",
                                                  "Internships",
                                                  "Projects",
                                                  "Workshops/Certifications",
                                                  "AptitudeTestScore",
                                                  "SoftSkillsRating",
                                                  "ExtracurricularActivities",
                                                  "PlacementTraining",
                                                  "SSC_Marks",
                                                  "HSC_Marks"],
                                       "values": features_list}]}

    response_scoring = requests.post(
        'https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/61c44f84-b5b7-4857-b65d-a9bf78f418a8/predictions?version=2021-05-01',
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())

    # make the prediction using the model
    prediction = response_scoring.json()
    prediction_value = prediction['predictions'][0]['values'][0][0]
    # convert the prediction to a string
    if prediction_value == 0:
        output = "You may Not get Placed. IMPROVE YOUR SKILLS"
    else:
        output = "Wow You have a chance to get Placed !"
    # return the output to the user
    return render_template(
        "home.html", prediction_text="{}".format(output)
    )


# run the app
if __name__ == "__main__":
    app.run(debug=True)