from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("heart.csv")

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    recommendation = ""
    risk_percent = None
    if request.method == "POST":
        input_data = {
            "age": int(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": int(request.form["trestbps"]),
            "chol": int(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": int(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"]),
        }

        input_df = pd.DataFrame([input_data])
        probability = model.predict_proba(input_df)[0][1]
        risk_percent = round(probability * 100, 2)
        if probability >= 0.5:
            prediction = 1
            recommendation = "High Risk: Please consult a cardiologist, follow a healthy diet, and exercise regularly." 
        else:
            prediction = 0
            recommendation = "Low Risk: Maintain a balanced diet, regular exercise, and yearly health checkups."

    return render_template("index.html", prediction=prediction, recommendation=recommendation, risk=risk_percent)
if __name__ == "__main__":
    app.run(debug=True)