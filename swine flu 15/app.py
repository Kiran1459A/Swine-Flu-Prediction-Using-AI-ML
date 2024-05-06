from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__,template_folder='template')

# Load the trained model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

# Load train and test datasets
train_df = pd.read_csv('train_swine_flu_dataset.csv')
test_df = pd.read_csv('test_swine_flu_dataset.csv')

# Separate features and labels
X_train = train_df.drop('SwineFlu', axis=1)
y_train = train_df['SwineFlu']

X_test = test_df.drop('SwineFlu', axis=1)
y_test = test_df['SwineFlu']

# Fit the model
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    

    return render_template('login&signup.html')

@app.route('/pred')
def pred():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from the form
    symptoms = [1 if request.form[symptom] == 'yes' else 0 for symptom in ['Fever', 'Cough', 'SoreThroat', 'BodyAches', 'Fatigue', 'Headache', 'Chills', 'RunnyNose', 'Nausea', 'Diarrhea']]

    # Make prediction
    prediction = model.predict([symptoms])[0]
    if prediction == 1:
        prediction = "Yes"
    else:
        prediction = "No"

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
