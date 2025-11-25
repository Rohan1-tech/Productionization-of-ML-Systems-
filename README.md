# Productionization-of-ML-Systems-


This project is an end-to-end Machine Learning application that predicts the total estimated travel cost using important trip details such as flight price, hotel cost, duration, distance, and travel intensity.

The goal is to demonstrate complete ML productionization:
✔ Model training
✔ Feature engineering
✔ Model scaling
✔ Saving artifacts (joblib)
✔ Building a Streamlit UI
✔ Deploying the app publicly on Streamlit Cloud

 **Model Used**

The prediction engine is powered by a Support Vector Regression (SVR) model, selected for its robustness in non-linear continuous prediction tasks.

Key steps included:

Scaling features using StandardScaler

Deriving additional useful metrics

Hyperparameter tuning

Saving the model and scalers for deployment

**Live Demo**

 The app is deployed and publicly accessible here:

  https://cbjced5xguzcn2hvmq4vbz.streamlit.app/

Interact with the UI to enter trip details and instantly see the predicted travel cost.

**Tech Stack**
Component	Technology
Language	Python
ML Model	Support Vector Regression (SVR)
Libraries	Scikit-Learn, NumPy, Joblib
Frontend	Streamlit
Deployment	Streamlit Cloud

**Project Structure**
travel-cost-prediction/
│── app.py                  # Streamlit UI + prediction code
│── requirements.txt        # Dependencies for deployment
│── best_svr_model.joblib   # Trained SVR model
│── x_scaler.joblib         # Feature scaler
│── y_scaler.joblib         # Target scaler
│── Productionization_of_ML_Systems_.ipynb   # Training notebook
│── README.md               # Project documentation

**Features**

✔ Real-time prediction
✔ Clean, user-friendly UI
✔ Automatic feature engineering (hotel/day, distance/day, intensity)
✔ Fast inference
✔ Fully deployed on the cloud
✔ Ideal for resume + portfolio + interviews

**How to Run Locally**
pip install -r requirements.txt
streamlit run app.py

 **Contact**

Feel free to connect:

 Email: rohanpagare6616@gmail.com

 GitHub: https://github.com/Rohan1-tech
