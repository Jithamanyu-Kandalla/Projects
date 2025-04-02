import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd

# Declare the flask app
app = Flask(__name__)

# Enable cross origin request for our application
cors = CORS(app)

model = pickle.load(open("./Output/model.pkl", "rb"))
normalizer = pickle.load(open("./Output/min-max-scaler.pkl", "rb"))

# Enable an api route for status check
@app.route('/check', methods= ['GET'])
@cross_origin()
def return_status():
    return "Yay! Flask App is running"

# Enable api route to get time series predictions
@app.route('/', methods = ['POST'])
@cross_origin()
def return_model_prediction():
        # Get prediction results and respond
        # POST Request: Where we need csv file
    try:
        data_org = pd.read_csv(request.files.get("data"))
        data = data_org.Healthcare
        #Empty lists to be populated using formatted training data
        target_data = []

        # Number of days we want to look into the future based on the past days.
        n_past = 5  # Number of past days we want to use to predict the future.

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        #In my example, my df_for_training_scaled has a shape (?)
        #refers to the number of data points and 5 refers to the columns (multi-variables).
        for i in range(len(data)):
            temp = []
            for j in range(n_past + 1):
                try:
                    temp.append(data[i+j])
                except Exception as e:
                    continue
            if len(temp) > 5:
                target_data.append(temp)
        print(target_data[0:5])
        normalized_data = normalizer.transform(target_data) 
        data_df = pd.DataFrame(normalized_data, columns=["t-4","t-3","t-2","t-1","t","Y"])
        test_data = data_df.drop("Y", axis=1).values
        print("Normalized the data")
        predictions = model.predict(test_data)
        print("Predictions :", predictions[:5], "\n\n\n")
        final_predictions = [float(x[0]) for  x in predictions]
        print(final_predictions[:5])
        return {"status_code":200,"message":"Sucess", "body": {"preds": final_predictions}}

    except Exception as e:
        print(f"Error occured :     {e}")
        return {"status_code":404, "message": f"Error :-    {e}"}

if __name__ == '__main__':
    app.run("0.0.0.0", port= 5000)
