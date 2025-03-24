import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

# Streamlit App
st.title("Exoplanet Detection App 🌌")
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose an action", ["Train Model", "Test Model", "HUNT"])

# Load Data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Preprocess Data
def preprocess_data(data):
    data = data.replace({'LABEL': {1: 0, 2: 1}})
    X = data.drop(['LABEL'], axis=1)
    y = data['LABEL']
    return X, y

# Train Model
def train_model(X, y, model_type):
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    
    # Model Selection
    if model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=1)
    elif model_type == "SVM":
        model = SVC(probability=True)  # Enable probability for confidence scores
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train_sc, y_train)  # Train the model
    return model, X_test_sc, y_test, sc

# Test Model
def test_model(model, X_test, y_test):
    # Check for NaN values in y_test
    if y_test.isnull().any():
        st.error("The test labels contain NaN values. Please clean your data.")
        return
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues", 
                xticklabels=['Non-Exoplanet', 'Exoplanet'], 
                yticklabels=['Non-Exoplanet', 'Exoplanet'])
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    # ROC Curve
    st.write("ROC Curve:")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Count of Exoplanet and Non-Exoplanet classifications
    exoplanet_count = sum(y_pred == 1)
    non_exoplanet_count = sum(y_pred == 0)
    st.write(f"Number of stars classified as Exoplanets: {exoplanet_count}")
    st.write(f"Number of stars classified as Non-Exoplanets: {non_exoplanet_count}")

# HUNT: Predict Exoplanets from New Data
def hunt_exoplanets(model, scaler, hunt_data):
    # Ensure the data has the correct format
    if 'Star Name' not in hunt_data.columns:
        st.error("The dataset must contain a 'Star Name' column.")
        return
    
    X_hunt = hunt_data.drop(['Star Name'], axis=1)
    X_hunt_sc = scaler.transform(X_hunt)
    
    # Predict exoplanets
    y_pred = model.predict(X_hunt_sc)
    y_pred_proba = model.predict_proba(X_hunt_sc)[:, 1]  # Confidence scores
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Star Name': hunt_data['Star Name'],
        'Exoplanet Prediction': y_pred,
        'Confidence Score': y_pred_proba
    })
    
    # Filter stars predicted as exoplanets
    exoplanet_stars = results[results['Exoplanet Prediction'] == 1]
    
    st.write("### Exoplanet Predictions")
    st.write(exoplanet_stars)

# Main App Logic
if options == "Train Model":
    st.header("Train Model")
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    if train_file is not None:
        # Check file size (1GB = 1024 * 1024 * 1024 bytes)
        if train_file.size > 1024 * 1024 * 1024:
            st.error("File size exceeds 1GB. Please upload a smaller file.")
        else:
            train_data = load_data(train_file)
            X, y = preprocess_data(train_data)
            
            # Model Selection
            model_type = st.selectbox("Select Model", ["KNN", "SVM"])
            
            # Run Button for Training
            if st.button("Train Model"):
                model, X_test, y_test, scaler = train_model(X, y, model_type)
                st.success(f"{model_type} Model Trained Successfully!")
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler

elif options == "Test Model":
    st.header("Test Model")
    test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
    if test_file is not None:
        # Check file size (1GB = 1024 * 1024 * 1024 bytes)
        if test_file.size > 1024 * 1024 * 1024:
            st.error("File size exceeds 1GB. Please upload a smaller file.")
        else:
            test_data = load_data(test_file)
            X_test, y_test = preprocess_data(test_data)
            
            # Check for NaN values in the test data
            if X_test.isnull().any().any() or y_test.isnull().any():
                st.error("The test data contains NaN values. Please clean your data.")
            else:
                if 'model' in st.session_state and 'scaler' in st.session_state:
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    X_test_sc = scaler.transform(X_test)
                    
                    # Run Button for Testing
                    if st.button("Test Model"):
                        test_model(model, X_test_sc, y_test)
                else:
                    st.error("Please train the model first!")

elif options == "HUNT":
    st.header("HUNT for Exoplanets")
    hunt_file = st.file_uploader("Upload HUNT Data (CSV)", type=["csv"])
    if hunt_file is not None:
        # Check file size (1GB = 1024 * 1024 * 1024 bytes)
        if hunt_file.size > 1024 * 1024 * 1024:
            st.error("File size exceeds 1GB. Please upload a smaller file.")
        else:
            hunt_data = load_data(hunt_file)
            
            if 'model' in st.session_state and 'scaler' in st.session_state:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                
                # Run Button for HUNT
                if st.button("HUNT for Exoplanets"):
                    hunt_exoplanets(model, scaler, hunt_data)
            else:
                st.error("Please train the model first!")