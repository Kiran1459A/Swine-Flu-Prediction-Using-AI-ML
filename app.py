import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.models import load_model
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

import cv2
import numpy as np
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

def main():
    background_image_url = "https://images.pexels.com/photos/531880/pexels-photo-531880.jpeg"

    # Add CSS for background image
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url("{background_image_url}");
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Swine Flu Predictor')
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,500;1,500&display=swap');

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: "Montserrat", sans-serif;
            line-height: 1.6;
            overflow-x: hidden;
            background-color: white; /* Set background color to white */
            color: rgb(0, 0, 0); /* Set text color to black */
        }

        
    </style>
    """, unsafe_allow_html=True)

    page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction Form","X-Ray Prediction"])

    if page == "Home":
        

      

        
            # Define the background image URL
            background_image_url = "https://images.pexels.com/photos/531880/pexels-photo-531880.jpeg?cs=srgb&dl=background-blur-clean-531880.jpg&fm=jpg"

            # Set the title and header
            st.title('Swine Flu Information')

            # Add CSS for background image
            st.markdown(
                f"""
                <style>
                    .reportview-container {{
                        background: url("{background_image_url}") no-repeat center center fixed;
                        background-size: cover;
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )

            # Add content
            st.header('Description')
            st.markdown("""
            Swine flu, also known as H1N1 influenza, is a respiratory illness caused by the H1N1 influenza virus. It is called "swine flu" because the virus originally jumped to humans from pigs. Swine flu can cause mild to severe illness and has the potential to spread rapidly from person to person.

            Symptoms of swine flu are similar to those of seasonal flu and include fever, cough, sore throat, body aches, headache, chills, and fatigue. In severe cases, swine flu can lead to pneumonia and respiratory failure, particularly in vulnerable populations such as young children, elderly individuals, pregnant women, and people with underlying health conditions.

            Although swine flu is no longer considered a pandemic, it continues to circulate and cause illness worldwide. Vaccination is the most effective way to prevent swine flu infections, and antiviral medications can help treat and manage symptoms.
            """)

            st.header('History')
            st.markdown("""
            Swine flu first gained widespread attention in 2009 when it emerged as a new strain of influenza virus. The outbreak, known as the H1N1 pandemic, spread rapidly around the world, causing significant illness and mortality. The virus was a combination of genetic material from human, avian, and swine influenza viruses.

            The H1N1 pandemic prompted a global response from public health authorities, including the development and distribution of vaccines, implementation of preventive measures such as hand hygiene and respiratory etiquette, and surveillance and monitoring of the virus.

            Since the pandemic, swine flu has continued to circulate as a seasonal influenza virus, causing sporadic outbreaks and cases of illness. Vaccination campaigns and public health efforts remain crucial in controlling the spread of the virus.
            """)

            st.header('Causes')
            st.markdown("""
            Swine flu is caused by infection with the H1N1 influenza virus. The virus is primarily transmitted through respiratory droplets when an infected person coughs or sneezes. It can also spread by touching contaminated surfaces and then touching the mouth, nose, or eyes.

            Swine flu viruses are a combination of genetic material from human, avian, and swine influenza viruses. The ability of these viruses to undergo genetic reassortment contributes to the emergence of new strains that can infect humans and cause outbreaks.

            Close contact with infected pigs, particularly in agricultural settings such as farms and livestock markets, can also lead to transmission of swine flu to humans.
            """)

            st.header('Effects')
            st.markdown("""
            Swine flu can cause a range of symptoms, from mild to severe, depending on individual factors such as age, overall health, and immune status. Common symptoms include fever, cough, sore throat, body aches, headache, chills, and fatigue. In severe cases, swine flu can lead to complications such as pneumonia, respiratory failure, and death.

            Vulnerable populations such as young children, elderly individuals, pregnant women, and people with underlying health conditions are at higher risk of developing severe illness from swine flu. Prompt diagnosis and treatment, including antiviral medications and supportive care, are essential in managing symptoms and preventing complications.

            In addition to its health effects, swine flu can have significant social and economic impacts, including disruptions to healthcare systems, schools, workplaces, and travel.
            """)

        





    elif page == "Prediction Form":
        st.header('Prediction Form')
        st.write("Please answer the following questions:")

        # Arrange symptoms questions in two columns
        col1, col2 = st.columns(2)

        with col1:
            symptoms1 = {
                'Fever': st.radio("Do you have fever?", ('Yes', 'No')),
                'Cough': st.radio("Do you have cough?", ('Yes', 'No')),
                'SoreThroat': st.radio("Do you have sore throat?", ('Yes', 'No')),
                'BodyAches': st.radio("Do you have body aches?", ('Yes', 'No')),
                'Fatigue': st.radio("Do you feel fatigue?", ('Yes', 'No')),
            }

        with col2:
            symptoms2 = {
                'Headache': st.radio("Do you have headache?", ('Yes', 'No')),
                'Chills': st.radio("Do you have chills?", ('Yes', 'No')),
                'RunnyNose': st.radio("Do you have runny nose?", ('Yes', 'No')),
                'Nausea': st.radio("Do you have nausea?", ('Yes', 'No')),
                'Diarrhea': st.radio("Do you have diarrhea?", ('Yes', 'No'))
            }

        # Combine symptoms from both columns
        symptoms = {**symptoms1, **symptoms2}
        
        st.write('<div style="text-align: center;">', unsafe_allow_html=True)
        if st.button("Predict", key="predict_button", help="Click to predict"):
            input_data = [1 if value == 'Yes' else 0 for value in symptoms.values()]
            prediction = model.predict([input_data])[0]
            result = "Positive" if prediction == 1 else "Negative"
        
            # Center-align the prediction result and change color
            prediction_text = f'<div style="text-align: center; margin-left:-200px;">Prediction: <span style="color:red;">{result}</span></div>' if result == "Positive" else f'<div style="text-align: center;">Prediction: <span style="color:green;">{result}</span></div>'
            st.write(prediction_text, unsafe_allow_html=True)
        
    elif page == "X-Ray Prediction":
        

        # Load the saved model
        models = load_model('swine_flu_model.h5')

        # Define function to preprocess image
        def preprocess_image(image):
            image = cv2.resize(image, (256, 256))
            image = image / 255.0
            return np.expand_dims(image, axis=0)

        # Define function to make prediction
        def predict_image(image):
            processed_image = preprocess_image(image)
            prediction = models.predict(processed_image)
            return prediction[0][0]  # Extract the prediction probability

        # Streamlit app
        st.title('Swine Flu Image Detection')

        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Read the image
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            prediction = predict_image(image)
            formatted_prediction = f"{prediction:.2f}"

            if prediction > 0.5:
                st.write(f'Prediction: <span style="color:red;">{formatted_prediction} Positive</span>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="color:green;">Prediction: {formatted_prediction} Negative</p>', unsafe_allow_html=True)

        

if __name__ == "__main__":
    main()
