import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

Dense = layers.Dense
BatchNormalization = layers.BatchNormalization
Dropout = layers.Dropout
import matplotlib.pyplot as plt

def home():
    st.title("Expected Goals Prediction Model Description")

    st.header("Purpose")
    st.write("""
    This project aims to build a model that predicts the expected goals (xG) of football players based on various performance metrics.
    The prediction helps in evaluating a player's offensive contribution and assists teams in making informed strategic decisions.
    """)

    st.header("Problem Type")
    st.write("""
    This is a **regression problem**, where the model predicts a continuous numerical value (expected goals) based on player statistics.
    """)

    st.header("Source Data")
    st.write("The dataset used in this project comes from Kaggle:")
    st.write("[Fantasy Premier League]( https://www.kaggle.com/datasets/meraxes10/fantasy-premier-league-dataset-2024-2025) and contains several important attributes related to football players and their performance, including:")
    st.write("""
    - **name**: The name of the player.             
    - **now_cost**: The current cost of the player.    
    - **position**: The player's on-field position (e.g., Forward, Midfielder, Defender, Goalkeeper).
    - **team**: The club or team the player is currently playing for.
    - **threat_rank**: The ranking of the player's offensive threat.
    - **expected_assists**: The estimated number of assists based on previous performances.
    - **total_points**: Total points accumulated by the player.
    - **influence**: The player's impact on the game. 
    - **creativity**: The player's creative ability to generate chances.
    - **minutes**: Total minutes played.
    - **starts**: Number of times the player started in a match.
    - **goals_scored**: The actual number of goals scored by the player.
    - **expected_goals (target)**: The predicted metric estimating goal probability.
    """)

    st.header("Model")
    st.write("""
    Two machine learning models are used to predict Expected Goals (xG):

    - **Random Forest Regressor**: A robust ensemble learning model that improves prediction accuracy by combining multiple decision trees.

    - **Linear Regression**: A simple and interpretable model that establishes relationships between features and target values.

    The models are trained using historical player performance data, and feature scaling is applied to ensure proper normalization.
    """)

    st.header("Data Preparation")
    st.write("""

    **Filtering Data**:
    1. Only players who have played actual minutes are included.
    2. Managers (MNG) are excluded from the dataset.

    **Feature Selection**:
    Key attributes that contribute to goal-scoring potential are chosen.

    **Data Scaling**:
    Features are normalized using StandardScaler to improve model performance.

    **Train-Test Split**:
    The dataset is split into 80% training and 20% testing to evaluate model accuracy.
    """)

    st.header("Model Performance")
    st.write("""
    The model's performance is evaluated using two key metrics:

    - **Mean Absolute Error (MAE)**: Measures the average prediction error.

    - **R² Score**: Indicates how well the model explains variance in actual goals scored.

    Comparison of model performance is displayed in the UI, along with Predicted vs Actual Goals and Residual Plots to analyze errors.
    """)

    st.header("Deployment")
    st.write("""
    After the model is trained and evaluated, it will be deployed using **Streamlit**, a Python library that allows the creation of web applications. 
    """)

def preexpectedgoal():    
    df = pd.read_csv("players.csv")
    
    df = df[(df["minutes"] > 0) & (df["position"] != "MNG")]
    
    features = ["now_cost", "threat_rank", "expected_assists", "total_points", "influence", "creativity", "minutes", "starts", "goals_scored"]
    target = "expected_goals"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    rf_preds = rf_model.predict(X_test_scaled)
    lr_preds = lr_model.predict(X_test_scaled)
    
    rf_mae = mean_absolute_error(y_test, rf_preds)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    
    st.title("⚽ Expected Goals Prediction")
    
    selected_team = st.selectbox("🎯 Select a team", df["team"].unique())
    players = df[df["team"] == selected_team]["name"].values
    selected_player = st.selectbox("🎯 Select a player", players)
    
    player_data = df[df["name"] == selected_player]
    player_features = player_data[features]
    player_scaled = scaler.transform(player_features)
    
    rf_prediction = rf_model.predict(player_scaled)[0]
    lr_prediction = lr_model.predict(player_scaled)[0]
    
    player_position = player_data["position"].values[0]
    st.write(f"**Position:** {player_position}")
    
    st.subheader("⚽ Expected Goals")
    st.write(f"Random Forest: {rf_prediction:.2f}")
    st.write(f"Linear Regression: {lr_prediction:.2f}")    
    
    st.subheader("📊 Model Performance")
    st.write(f"Random Forest - MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
    st.write(f"Linear Regression - MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")
    

    st.subheader("📊 Predicted vs Actual Goals")    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=rf_preds, label="Random Forest", alpha=0.6)
    sns.scatterplot(x=y_test, y=lr_preds, label="Linear Regression", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual Goals")
    plt.ylabel("Predicted Goals")
    plt.title("Predicted vs Actual Goals")
    plt.legend()
    st.pyplot(plt)
    

    st.subheader("📉 Residual Plot")
    rf_residuals = y_test - rf_preds
    lr_residuals = y_test - lr_preds
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=rf_residuals, label="Random Forest", alpha=0.6)
    sns.scatterplot(x=y_test, y=lr_residuals, label="Linear Regression", alpha=0.6)
    plt.axhline(y=0, color="k", linestyle="--", lw=2)
    plt.xlabel("Actual Goals")
    plt.ylabel("Residuals (Error)")
    plt.title("Residual Plot")
    plt.legend()
    st.pyplot(plt)

def home2():
    st.title("Lung Disease Prediction Model Description")
    st.header("Purpose")
    st.write("""
    The purpose of this project is to build a model that can predict the likelihood of recovery from lung disease based on user data. 
    This model will help healthcare professionals assess the recovery chances of a patient based on a variety of features like age, 
    gender, lung capacity, smoking status, disease type, treatment type, and hospital visits.
    """)

    st.header("Problem Type")
    st.write("""
    This is a **binary classification** problem, where the model predicts the likelihood of recovery (Recovered or Not Recovered) 
    from lung disease based on the input data.
    """)

    st.header("Source Data")
    st.write("""
    The dataset used in this project comes from **Kaggle**:  
    [Lung Diseases Dataset](https://www.kaggle.com/datasets/samikshadalvi/lungs-diseases-dataset). 
    It contains several important attributes related to lung diseases and recovery outcomes, including:
    """)
    st.write("""
    - **Age**: The age of the person.
    - **Gender**: The male or female sex of the person.
    - **Lung Capacity**: The lung capacity of the person.
    - **Smoking Status**: Whether the person is a smoker or non-smoker.
    - **Disease Type**: The type of lung disease.
    - **Treatment Type**: The type of treatment received by the person.
    - **Hospital Visits**: The number of hospital visits made by the person.
    - **Recovered**: Whether the person recovered or not.
    """)

    st.header("Model")
    st.write("""
    A **Neural Network** will be built to predict the likelihood of recovery from lung disease based on the cleaned and processed user data. 
    This model is designed to handle complex relationships in the data and perform well with classification tasks.

    - **Neural Network Architecture**: 
      The model consists of multiple layers:
      - **Dense Layers**: Fully connected layers of neurons.
      - **Activation Functions**: **ReLU** is used in hidden layers, and **Sigmoid** is used in the output layer for probabilities between 0 and 1.
      - **Batch Normalization**: To ensure stable training and faster convergence.
      - **Dropout**: To prevent overfitting by randomly setting a fraction of the input units to zero during training.

    - **Optimizer**: The **Adam optimizer** is used, as it adapts the learning rate during training.

    - **Loss Function**: **Binary Cross-Entropy** loss function is used, as this is a binary classification task.
    """)


    st.header("Data Preparation")
    st.write("""
    Before using the data to train the model, we perform several steps to prepare and clean the data:

    1. **Missing Values**:
       - **Numerical Features** (e.g., Age, Lung Capacity, Hospital Visits): Missing values are replaced with the median of the respective columns.
       - **Categorical Features** (e.g., Gender, Smoking Status, Disease Type, Treatment Type, Recovered): Missing values are replaced with the mode of the respective columns.
       
    2. **Duplicates**: Any duplicate rows in the dataset are removed to prevent bias in the model's predictions.

    3. **Categorical Encoding**: Categorical variables are encoded using **Label Encoding**, which converts each category into a numerical value.

    4. **Data Normalization**: Numerical features like Age, Lung Capacity, and Hospital Visits are normalized using **StandardScaler** to ensure all features are on a similar scale, improving model performance.

    5. **Train-Test Split**: The dataset is divided into a **training set** (80%) and a **testing set** (20%) to evaluate the model's performance on unseen data.
    """)

    st.header("Model Development Process")
    st.write("""
    The development of the model will follow these steps:

    1. **Data Loading**: The dataset is loaded using **Pandas**.
    2. **Data Preprocessing**: This includes filling missing values, encoding categorical variables, and normalizing numerical features.
    3. **Model Design**: A neural network with **dense layers**, **batch normalization**, and **dropout** is defined using the **Keras** library.
    4. **Model Training**: The model is trained using the **train set** and evaluated on the **test set**.
    5. **Model Evaluation**: After training, we evaluate the model's **accuracy** and **loss** to assess its performance.
    """)

    st.header("Model Evaluation")
    st.write("""
    After training, the model is evaluated on the test set to assess its performance in terms of accuracy.
    The loss and accuracy plots are displayed to track the model's learning process.
    """)
   
    st.header("Deployment")
    st.write("""
    After the model is trained and evaluated, it will be deployed using **Streamlit**, a Python library that allows the creation of web applications. 
    """)

def predrecover():
    st.title("Lung Disease Prediction Model")

    @st.cache_resource    
    def train_model():
        df = pd.read_csv("lung_disease_data.csv")

        num_cols = ["Age", "Lung Capacity", "Hospital Visits"]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = ["Gender", "Smoking Status", "Disease Type", "Treatment Type", "Recovered"]
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        X = df.drop(columns=["Recovered"])
        y = df["Recovered"]

        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = keras.Sequential([
            Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

        return model, scaler, label_encoders, history

    model, scaler, label_encoders, history = train_model()

    st.subheader("Model Performance")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history.history["accuracy"], label="Train Accuracy", color="blue")
    ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Model Accuracy")
    ax[0].legend()

    ax[1].plot(history.history["loss"], label="Train Loss", color="blue")
    ax[1].plot(history.history["val_loss"], label="Validation Loss", color="orange")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Model Loss")
    ax[1].legend()

    st.pyplot(fig)   

    st.subheader("Input Data for Prediction")
    age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
    lung_capacity = st.number_input("Enter Lung Capacity", min_value=0.5, max_value=6.0, value=3.5)
    hospital_visits = st.number_input("Enter Number of Hospital Visits", min_value=0, max_value=100, value=3)

    gender = st.selectbox("Select Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Select Smoking Status", ["Smoker", "Non-Smoker"])
    disease_type = st.selectbox("Select Disease Type", ["Asthma", "COPD", "Pneumonia"])
    treatment_type = st.selectbox("Select Treatment Type", ["Medication", "Therapy", "Surgery"])

    if st.button("Predict Outcome"):
        user_data = {
            "Age": age,
            "Lung Capacity": lung_capacity,
            "Hospital Visits": hospital_visits,
            "Gender": gender,
            "Smoking Status": smoking_status,
            "Disease Type": disease_type,
            "Treatment Type": treatment_type
        }

        user_data_encoded = pd.DataFrame([user_data])
        for col in ["Gender", "Smoking Status", "Disease Type", "Treatment Type"]:
            le = label_encoders[col]
            if not all(user_data_encoded[col].isin(le.classes_)):
                user_data_encoded[col] = le.transform(user_data_encoded[col].replace(set(user_data_encoded[col]) - set(le.classes_), le.classes_[0]))
            else:
                user_data_encoded[col] = le.transform(user_data_encoded[col])

        user_data_encoded[["Age", "Lung Capacity", "Hospital Visits"]] = scaler.transform(user_data_encoded[["Age", "Lung Capacity", "Hospital Visits"]])

        prediction = model.predict(user_data_encoded)
        if prediction[0][0] > 0.5:
            st.write("Prediction: The person is likely to recover.")
        else:
            st.write("Prediction: The person is unlikely to recover.")

def main():    
    page = st.sidebar.selectbox("Choose a page", ["Machine Model Description", "Expected Goals Prediction", "Neural Network Description", "Lung Disease Prediction"])
    if page == "Machine Model Description":
        home()
    elif page == "Expected Goals Prediction":
        preexpectedgoal()
    elif page == "Neural Network Description":
        home2()
    elif page == "Lung Disease Prediction":
        predrecover()

if __name__ == "__main__":
    main()