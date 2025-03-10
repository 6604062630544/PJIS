import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

Dense = layers.Dense
BatchNormalization = layers.BatchNormalization
Dropout = layers.Dropout
import matplotlib.pyplot as plt

def home():
    st.title("Recommend Games Model Description")

    st.subheader("Purpose")
    st.write("The purpose of this project is to build a model that can recommend games to users based on their preferences.")

    st.subheader("Problem Type")
    st.write("This is a recommendation problem.")

    st.subheader("Source Data")
    st.write("this dataset come from https://www.kaggle.com/datasets/anandshaw2001/video-game-sales")

    st.subheader("Data")
    st.write("The dataset contains the following columns:")
    st.write("1. Name: The name of the game.")
    st.write("2. Platform: The platform of the game.")
    st.write("3. Year: The year the game was released.")
    st.write("4. Genre: The genre of the game.")
    st.write("5. Publisher: The publisher of the game.")
    st.write("6. NA_Sales: The sales of the game in North America.")
    st.write("7. EU_Sales: The sales of the game in Europe.")
    st.write("8. JP_Sales: The sales of the game in Japan.")
    st.write("9. Other_Sales: The sales of the game in other regions.")
    st.write("10. Global_Sales: The global sales of the game.")
    
    st.subheader("Data Cleaning")
    st.write("We will be cleaning the data by removing missing values and duplicates.")
    st.write("We will also be removing the columns that are not needed for the model.")

    st.subheader("Model")
    st.write("We will be building a Content-Based Filtering model and a Collaborative Filtering model.")
    st.write("The Content-Based Filtering model will recommend games based on the similarity of their features.")
    st.write("The Collaborative Filtering model will recommend games based on the ratings given by users.")
    st.write("We will also be building a Hybrid model that combines the scores of the Content-Based Filtering model and the Collaborative Filtering model.")

    st.subheader("Evaluation")
    st.write("We will be evaluating the models using Mean Cosine Similarity for the Content-Based Filtering model and Explained Variance for the Collaborative Filtering model.")

    st.subheader("Deployment")
    st.write("We will be deploying the model using Streamlit.")
    st.write("The user will be able to select a game and the model will recommend games based on the selected game.")    

def recommend():
    st.title("🎮 Game Recommendation System")

    # โหลดข้อมูล
    df = pd.read_csv("vgsales.csv")

    # ทำความสะอาดข้อมูล
    df.dropna(subset=['Publisher'], inplace=True)
    df['Year'] = df['Year'].interpolate(method='linear')
    df["Year"] = df["Year"].astype(int)

    # สร้าง widget สำหรับเลือกเกม
    game = st.selectbox("🎯 Select a game", df["Name"])

    # บันทึกค่าดั้งเดิมก่อนทำ encoding
    genre_mapping = dict(enumerate(df['Genre'].astype('category').cat.categories))
    publisher_mapping = dict(enumerate(df['Publisher'].astype('category').cat.categories))
    platform_mapping = dict(enumerate(df['Platform'].astype('category').cat.categories))

    # แปลงหมวดหมู่ข้อมูลเป็นตัวเลข
    df['Genre'] = df['Genre'].astype('category').cat.codes
    df['Publisher'] = df['Publisher'].astype('category').cat.codes
    df['Platform'] = df['Platform'].astype('category').cat.codes

    # Feature Scaling
    scaler = StandardScaler()
    features = ['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    X_scaled = scaler.fit_transform(df[features])

    # Elbow Method
    wcss = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
 
    st.subheader("📊 Elbow Method for Optimal k")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 15), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    st.pyplot(plt)

    # use เลือกจำนวน cluster
    k = st.slider("🔢 Select number of clusters for KMeans", 2, 10, 3)

    # เทรนโมเดล KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # แปลงกลับเป็นชื่อเดิม   
    df['Genre'] = df['Genre'].map(genre_mapping)
    df['Publisher'] = df['Publisher'].map(publisher_mapping)
    df['Platform'] = df['Platform'].map(platform_mapping)

    # แนะนำเกม (KMeans)
    st.subheader("🎮 Recommended Games (KMeans)")
    game_cluster = df[df['Name'] == game]['Cluster'].values[0]
    recommended_games_kmeans = df[(df['Cluster'] == game_cluster) & (df['Name'] != game)]

    if recommended_games_kmeans.empty:
        st.write("❌ No similar games found.")
    else:
        st.write(recommended_games_kmeans[['Name', 'Platform', 'Year', 'Genre', 'Publisher']])

    # ประเมินโมเดล KMeans
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    st.subheader("📊 Model Evaluation (KMeans)")
    st.write(f"✔️ Silhouette Score: {silhouette:.4f} (higher is better)")
    st.write('---')

    # เทรนโมเดล DBSCAN
    dbscan = DBSCAN(eps=1.0, min_samples=5)  # Adjust `eps` for better clustering
    dbscan.fit(X_scaled)
    df['Cluster_DBSCAN'] = dbscan.labels_

    # แนะนำเกม DBSCAN
    st.subheader("🎮 Recommended Games (DBSCAN)")
    game_cluster_dbscan = df[df['Name'] == game]['Cluster_DBSCAN'].values[0]

    if game_cluster_dbscan == -1:
        st.write("❌ This game is classified as noise/outlier in DBSCAN.")
    else:
        recommended_games_dbscan = df[(df['Cluster_DBSCAN'] == game_cluster_dbscan) & (df['Name'] != game)]
        if recommended_games_dbscan.empty:
            st.write("❌ No similar games found.")
        else:
            st.write(recommended_games_dbscan[['Name', 'Platform', 'Year', 'Genre', 'Publisher']])

    st.write('---------------------------------')

    # ประเมินโมเดล DBSCAN
    silhouette_dbscan = silhouette_score(X_scaled, dbscan.labels_)
    st.subheader("📊 Model Evaluation (DBSCAN)")
    st.write(f"✔️ Silhouette Score: {silhouette_dbscan:.4f} (higher is better)")

    # แสดงจำนวน cluster
    # st.subheader("📊 Number of Clusters")
    # st.write(f"✅ KMeans: {len(set(kmeans.labels_))} clusters")
    # st.write(f"✅ DBSCAN: {len(set(dbscan.labels_))} clusters (including noise)")

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

   
    st.header("Deployment")
    st.write("""
    After the model is trained and evaluated, it will be deployed using **Streamlit**, a Python library that allows the creation of web applications. 
    """)

def predrecover():
    st.title("Lung Disease Prediction Model")

    age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
    lung_capacity = st.number_input("Enter Lung Capacity", min_value=0.5, max_value=6.0, value=3.5)
    hospital_visits = st.number_input("Enter Number of Hospital Visits", min_value=0, max_value=100, value=3)

    gender = st.selectbox("Select Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Select Smoking Status", ["Smoker", "Non-Smoker"])
    disease_type = st.selectbox("Select Disease Type", ["Asthma", "COPD", "Pneumonia"])
    treatment_type = st.selectbox("Select Treatment Type", ["Medication", "Therapy", "Surgery"])

    # Button to trigger model training
    if st.button("Start Model Training"):
        # Load the dataset
        df = pd.read_csv("lung_disease_data.csv")

        # Handle missing values
        num_cols = ["Age", "Lung Capacity", "Hospital Visits"]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = ["Gender", "Smoking Status", "Disease Type", "Treatment Type", "Recovered"]
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        # Encode categorical variables
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Separate features and target
        X = df.drop(columns=["Recovered"])
        y = df["Recovered"]

        # Normalize numerical features
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model
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

        # Compile model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        with st.spinner('Training model... please wait'):
            history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

            # Evaluate the model
            train_loss, train_acc = model.evaluate(X_train, y_train)
            test_loss, test_acc = model.evaluate(X_test, y_test)

            # Display results in Streamlit
            st.write(f"**Training Accuracy:** {train_acc:.4f}")
            st.write(f"**Validation Accuracy:** {test_acc:.4f}")
            st.write(f"**Training Loss:** {train_loss:.4f}")
            st.write(f"**Validation Loss:** {test_loss:.4f}")

            # Plot accuracy and loss graphs
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Accuracy Plot
            ax[0].plot(history.history["accuracy"], label="Train Accuracy", color="blue")
            ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Accuracy")
            ax[0].set_title("Model Accuracy")
            ax[0].legend()

            # Loss Plot
            ax[1].plot(history.history["loss"], label="Train Loss", color="blue")
            ax[1].plot(history.history["val_loss"], label="Validation Loss", color="orange")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss")
            ax[1].set_title("Model Loss")
            ax[1].legend()

            st.pyplot(fig)

            # User input data
            user_data = {
                "Age": age,
                "Lung Capacity": lung_capacity,
                "Hospital Visits": hospital_visits,
                "Gender": gender,
                "Smoking Status": smoking_status,
                "Disease Type": disease_type,
                "Treatment Type": treatment_type
            }

            # Encode user input as the model expects
            user_data_encoded = pd.DataFrame([user_data])
            for col in ["Gender", "Smoking Status", "Disease Type", "Treatment Type"]:
                le = label_encoders[col]
                # Check for unseen labels
                if not all(user_data_encoded[col].isin(le.classes_)):
                    unseen_labels = set(user_data_encoded[col]) - set(le.classes_)
                     
                    # Handle unseen labels by replacing with the first class
                    user_data_encoded[col] = le.transform(user_data_encoded[col].replace(unseen_labels, le.classes_[0]))
                else:
                    user_data_encoded[col] = le.transform(user_data_encoded[col])

            # Normalize numerical features
            user_data_encoded[num_cols] = scaler.transform(user_data_encoded[num_cols])

            # Predict the outcome (recovered or not)
            prediction = model.predict(user_data_encoded)
            if prediction[0][0] > 0.5:
                st.write("Prediction: The person is likely to recover.")
            else:
                st.write("Prediction: The person is unlikely to recover.")

def main():    
    page = st.sidebar.selectbox("Choose a page", ["Machine Model Description", "Recommend Games", "Neuron Network Description", "Lung Disease Prediction"])
    if page == "Machine Model Description":
        home()
    elif page == "Recommend Games":
        recommend()
    elif page == "Neuron Network Description":
        home2()
    elif page == "Lung Disease Prediction":
        predrecover()

if __name__ == "__main__":
    main()