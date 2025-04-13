import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from PIL import Image, ImageDraw
import joblib
import os
import time
import io
import base64

# Set page config
st.set_page_config(page_title="ExoHunter 🔭", page_icon="🪐", layout="wide")
def make_circular(image):
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    result = image.copy()
    result.putalpha(mask)
    return result

# Custom CSS to reduce the gap
st.markdown("""
<style>
    div[data-testid="column"] {
        padding: 0px !important;
    }
    .title-container {
        margin-top: -5000px !important;
    }
</style>
""", unsafe_allow_html=True)

# Create tighter columns (notice the column ratio)
col1, col2 = st.columns([1, 19])  # Reduced from [1,4] to bring items closer

# Circular logo
with col1:
    logo = Image.open("logo.png")
    logo = logo.resize((120, 120))  # Slightly reduced size
    circular_logo = make_circular(logo)
    st.image(circular_logo, width=110)  # Smaller display width

# Title with adjusted positioning
with col2:
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("ExoHunter - Advanced Exoplanet Detection")
    st.markdown('</div>', unsafe_allow_html=True)


# Initialize session state for tracking progress
if 'workflow_progress' not in st.session_state:
    # Start with Introduction and About unlocked
    st.session_state.workflow_progress = {
        "Introduction": True,
        "Data Exploration": True,
        "Train Model": False,
        "Test Model": False,
        "HUNT": False,
        "About": True  # About is always accessible
    }
    st.session_state.options = "Introduction"

# Function to update progress when completing a step
def complete_step(step_name):
    # Mark current step as completed
    st.session_state.workflow_progress[step_name] = True
    
    # Find the next step in the sequence
    step_list = list(st.session_state.workflow_progress.keys())
    if step_name != "About" and step_list.index(step_name) < len(step_list) - 2:  # -2 to account for About
        next_step = step_list[step_list.index(step_name) + 1]
        # Unlock the next step
        st.session_state.workflow_progress[next_step] = True

# Create the sidebar with conditional formatting
st.sidebar.title("Navigation")
st.sidebar.markdown("### Workflow Steps:")

# Container for navigation buttons
nav_container = st.sidebar.container()

# Create buttons for each step with appropriate styling
with nav_container:
    for step in st.session_state.workflow_progress.keys():
        # Determine if step is available
        is_available = st.session_state.workflow_progress[step]
        
        # Use different styles for available vs unavailable steps
        if is_available:
            # Create a clickable button for available steps
            if st.sidebar.button(f"🔓 {step}", key=f"nav_{step}", use_container_width=True):
                st.session_state.options = step
                # If this is a new step being visited, mark it as complete and unlock next
                complete_step(step)
        else:
            # Create a disabled button effect for unavailable steps
            st.sidebar.markdown(
                f"""<div style='background-color:#E0E0E0; opacity:0.6; padding:0.5rem; 
                border-radius:0.3rem; margin:0.2rem 0; color:#666666; text-align:center'>
                🔒 {step} (Complete previous steps first)
                </div>""", 
                unsafe_allow_html=True
            )

# Progress bar to visualize workflow completion
steps_list = list(st.session_state.workflow_progress.keys())
steps_list.remove("About")  # Don't count About in progress
completed = sum(1 for step in steps_list if st.session_state.workflow_progress[step])
progress = completed / len(steps_list)

st.sidebar.progress(progress)
st.sidebar.caption(f"Progress: {int(progress*100)}% ({completed}/{len(steps_list)} steps)")

# Connect the options to the main app logic by setting options
options = st.session_state.options

# The main app logic will now use the 'options' variable as before,
# but it will be controlled by the workflow navigation system

# Cache decorator for expensive operations
@st.cache_data
def load_data(file):
    """Load data with caching to improve performance"""
    data = pd.read_csv(file)
    return data

def preprocess_data(data):
    """Preprocess the dataset by converting labels and separating features/target"""
    # Convert labels: 1 (Non-Exoplanet) to 0, 2 (Exoplanet) to 1
    data = data.replace({'LABEL': {1: 0, 2: 1}})
    
    # Check for columns that are not FLUX or LABEL
    flux_cols = [col for col in data.columns if col.startswith('FLUX')]
    
    # Separate features and target
    X = data[flux_cols]
    y = data['LABEL']
    
    return X, y

def extract_features(X, method='raw', n_components=None):
    """Extract features using various dimensionality reduction techniques"""
    if method == 'raw':
        return X
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        # Create DataFrame with new column names
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        X_pca = pd.DataFrame(X_reduced, columns=pca_cols)
        return X_pca, pca
    elif method == 'statistical':
        # Extract statistical features from light curves
        stats_features = pd.DataFrame()
        for col in X.columns:
            series = X[col]
            stats_features[f'{col}_mean'] = [series.mean()]
            stats_features[f'{col}_std'] = [series.std()]
            stats_features[f'{col}_min'] = [series.min()]
            stats_features[f'{col}_max'] = [series.max()]
        return stats_features
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

def train_model(X, y, model_type, params=None, feature_extraction='raw', cv=5, 
                sampling_strategy='auto', n_components=100, use_scaler='standard'):
    """Train a model with specified parameters and preprocessing pipeline"""
    # Define sampling method (for imbalance)
    if sampling_strategy == 'none':
        sampler = None
    elif sampling_strategy == 'smote':
        sampler = SMOTE(random_state=42)
    else:  # Default: random oversampling
        sampler = RandomOverSampler(random_state=42)
    
    # Define scaler
    if use_scaler == 'robust':
        scaler = RobustScaler()
    else:  # Default: standard scaler
        scaler = StandardScaler()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply feature extraction if needed
    if feature_extraction == 'pca':
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        feature_names = [f'PC{i+1}' for i in range(n_components)]
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)
        
    # Select model
    if model_type == "KNN":
        default_params = {'n_neighbors': 5, 'weights': 'uniform'}
        if params:
            default_params.update(params)
        model = KNeighborsClassifier(**default_params)
    elif model_type == "SVM":
        default_params = {'C': 1.0, 'kernel': 'rbf', 'probability': True}
        if params:
            default_params.update(params)
        model = SVC(**default_params)
    elif model_type == "RF":
        default_params = {
            'n_estimators': 100, 
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        model = RandomForestClassifier(**default_params)
    elif model_type == "Ensemble":
        # Create an ensemble of models
        knn = KNeighborsClassifier(n_neighbors=3)
        svm = SVC(kernel='rbf', probability=True, C=1.0)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model = VotingClassifier(
            estimators=[('knn', knn), ('svm', svm), ('rf', rf)],
            voting='soft'
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Create pipeline with sampling and scaling
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    if sampler:
        steps.append(('sampler', sampler))
    steps.append(('model', model))
    
    pipeline = ImbPipeline(steps=steps)
    
    # Train with cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    st.write(f"Cross-validation F1 scores: {cv_scores}")
    st.write(f"Mean CV F1 score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    # Compile results
    results = {
        'pipeline': pipeline,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_extraction': feature_extraction,
        'model_type': model_type,
        'cv_scores': cv_scores
    }
    
    # Store feature names if using RandomForest
    if model_type == "RF" or model_type == "Ensemble":
        if feature_extraction == 'pca':
            results['feature_names'] = feature_names
        else:
            results['feature_names'] = X.columns.tolist()
            
    if feature_extraction == 'pca':
        results['pca'] = pca
        
    return results

def evaluate_model(results):
    """Evaluate model performance with multiple metrics"""
    pipeline = results['pipeline']
    X_test = results['X_test']
    y_test = results['y_test']
    y_pred = results['y_pred']
    model_type = results['model_type']
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    
    # Display classification report
    st.write("### Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))
    
    # Display confusion matrix
    st.write("### Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues", 
                xticklabels=['Non-Exoplanet', 'Exoplanet'], 
                yticklabels=['Non-Exoplanet', 'Exoplanet'])
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    # Create ROC curve if model supports predict_proba
    if hasattr(pipeline, 'predict_proba'):
        st.write("### ROC Curve:")
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        st.pyplot(fig)
    
    # Show feature importance for RandomForest
    if model_type == "RF" or model_type == "Ensemble":
        if model_type == "RF":
            model = pipeline.named_steps['model']
            importances = model.feature_importances_
        elif model_type == "Ensemble" and 'rf' in pipeline.named_steps['model'].named_estimators_:
            model = pipeline.named_steps['model'].named_estimators_['rf']
            importances = model.feature_importances_
        else:
            return
            
        if 'feature_names' in results:
            feature_names = results['feature_names']
            
            # Only show top 20 features to avoid cluttered chart
            if len(feature_names) > 20:
                indices = np.argsort(importances)[-20:]
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
            else:
                indices = np.argsort(importances)
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
                
            st.write("### Feature Importance (Top 20):")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_importances)), top_importances, align='center')
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)

def hunt_exoplanets(pipeline, scaler, hunt_data, expected_flux_count=3197, confidence_threshold=0.7):
    """Process new data to hunt for exoplanets"""
    # 1. Validate required columns
    if 'Star Name' not in hunt_data.columns:
        st.error("❌ Dataset must contain 'Star Name' column")
        return
    
    # 2. Create expected column names and check for FLUX columns
    flux_cols = [col for col in hunt_data.columns if col.startswith('FLUX')]
    
    if not flux_cols:
        st.error("❌ No FLUX columns found in the data")
        return
        
    # Check if we have enough flux columns
    if len(flux_cols) < expected_flux_count:
        st.error(f"❌ Expected {expected_flux_count} flux columns, but found only {len(flux_cols)}")
        return
    
    # 3. Prepare features
    try:
        X_hunt = hunt_data[flux_cols]
        
        # Use the same PCA transformation if applicable
        if hasattr(pipeline, 'pca'):
            X_hunt = pipeline.pca.transform(X_hunt)
        
        # 4. Predict
        y_pred = pipeline.predict(X_hunt)
        
        # Some pipelines might not have predict_proba
        if hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(X_hunt)[:, 1]
        else:
            # If predict_proba not available, use a placeholder confidence
            y_proba = np.ones_like(y_pred) * 0.99 * y_pred
        
        # 5. Display results
        results = pd.DataFrame({
            'Star Name': hunt_data['Star Name'],
            'Confidence': y_proba,
            'Detection': ['EXOPLANET' if p == 1 else 'Normal' for p in y_pred]
        })
        
        # Filter high-confidence detections
        exoplanets = results[results['Confidence'] > confidence_threshold].sort_values('Confidence', ascending=False)
        
        if not exoplanets.empty:
            st.success(f"🌌 Found {len(exoplanets)} exoplanet candidates!")
            
            # Display interactive table
            st.dataframe(
                exoplanets.style.format({'Confidence': '{:.1%}'})
                .bar(subset=['Confidence'], color='#5fba7d')
            )
            
            # Plot top candidates
            st.subheader("🔭 Top Exoplanet Candidates")
            
            # Limit to top 3 for clarity
            top_candidates = exoplanets.head(min(3, len(exoplanets)))
            
            cols = st.columns(len(top_candidates))
            
            for i, (idx, candidate) in enumerate(top_candidates.iterrows()):
                star_name = candidate['Star Name']
                confidence = candidate['Confidence']
                
                with cols[i]:
                    st.write(f"**{star_name}**")
                    st.write(f"Confidence: {confidence:.1%}")
                    
                    # Plot light curve
                    fig, ax = plt.subplots(figsize=(8, 4))
                    flux_data = X_hunt.loc[idx] if isinstance(X_hunt, pd.DataFrame) else X_hunt[idx]
                    ax.plot(flux_data, color='blue', alpha=0.8)
                    ax.set_title(f"Light curve: {star_name}")
                    ax.set_xlabel("Time index")
                    ax.set_ylabel("Flux")
                    
                    # Add potential transit markers if confidence is high
                    if confidence > 0.85:
                        # This is a simple way to highlight potential transits
                        # In a real application, you'd use a more sophisticated transit detection
                        flux_data_array = np.array(flux_data)
                        mean_flux = np.mean(flux_data_array)
                        std_flux = np.std(flux_data_array)
                        threshold = mean_flux - 1.5 * std_flux
                        
                        # Find potential transits (consecutive points below threshold)
                        transit_markers = np.where(flux_data_array < threshold)[0]
                        ax.scatter(transit_markers, flux_data_array[transit_markers], 
                                 color='red', marker='o', label='Potential transit')
                        ax.legend()
                    
                    st.pyplot(fig)
        else:
            st.warning("No clear exoplanet signals detected (confidence < 70%)")
            
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.write("Common issues:")
        st.write("- Column count mismatch")
        st.write("- Non-numeric values in flux columns")
        st.write("- Data shape doesn't match training data")
        
def plot_light_curve(flux_data, title="Light Curve", detect_transits=True):
    """Plot a light curve with optional transit detection"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(flux_data, 'b-', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time (observation index)")
    ax.set_ylabel("Stellar Flux")
    
    if detect_transits:
        # Simple transit detection by finding dips below threshold
        flux_array = np.array(flux_data)
        mean_flux = np.mean(flux_array)
        std_flux = np.std(flux_array)
        threshold = mean_flux - 1.5 * std_flux
        
        # Find potential transits (points below threshold)
        transit_markers = np.where(flux_array < threshold)[0]
        
        # Only highlight if we find potential transits
        if len(transit_markers) > 0:
            ax.scatter(transit_markers, flux_array[transit_markers], 
                      color='red', marker='o', label='Potential transit')
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                      label='Transit threshold')
            ax.legend()
    
    return fig

def save_model(pipeline, filename='exoplanet_model.joblib'):
    """Save trained model to disk"""
    joblib.dump(pipeline, filename)
    
    # Create download link
    with open(filename, 'rb') as f:
        bytes_data = f.read()
    
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:file/joblib;base64,{b64}" download="{filename}">Download trained model</a>'
    st.markdown(href, unsafe_allow_html=True)

def load_model(uploaded_file):
    """Load trained model from upload"""
    bytes_data = uploaded_file.read()
    with open('temp_model.joblib', 'wb') as f:
        f.write(bytes_data)
    
    try:
        pipeline = joblib.load('temp_model.joblib')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def data_exploration(data):
    """Explore and visualize the dataset"""
    st.write("### Dataset Information")
    
    # Basic info
    st.write(f"**Rows:** {data.shape[0]}")
    st.write(f"**Columns:** {data.shape[1]}")
    
    # Class distribution
    if 'LABEL' in data.columns:
        st.write("### Class Distribution")
        class_counts = data['LABEL'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        
        # Create a more interpretable class label
        class_counts['Class'] = class_counts['Class'].map({
            1: 'Non-Exoplanet', 
            2: 'Exoplanet',
            0: 'Non-Exoplanet (0)', 
            3: 'Unknown (3)'
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Class', y='Count', data=class_counts, ax=ax)
        ax.set_title('Class Distribution')
        st.pyplot(fig)
        
        # Calculate class imbalance ratio
        if len(class_counts) >= 2:
            minority_count = class_counts['Count'].min()
            majority_count = class_counts['Count'].max()
            imbalance_ratio = majority_count / minority_count
            st.write(f"**Class Imbalance Ratio:** {imbalance_ratio:.2f}:1")
    
    # Sample light curves
    st.write("### Sample Light Curves")
    flux_cols = [col for col in data.columns if col.startswith('FLUX')]
    
    if flux_cols:
        # Select sample rows
        if 'LABEL' in data.columns:
            # Try to get one sample from each class if available
            sample_indices = []
            for label in data['LABEL'].unique():
                class_samples = data[data['LABEL'] == label].index.tolist()
                if class_samples:
                    sample_indices.append(class_samples[0])
        else:
            # Or just take first row
            sample_indices = [0]
        
        # Plot each sample
        for idx in sample_indices:
            row = data.iloc[idx]
            label = row.get('LABEL', 'Unknown')
            
            # Map label to readable class
            if label == 1:
                class_name = "Non-Exoplanet"
            elif label == 2:
                class_name = "Exoplanet"
            elif label == 0:
                class_name = "Non-Exoplanet (0)"
            else:
                class_name = f"Class {label}"
            
            flux_data = row[flux_cols]
            fig = plot_light_curve(flux_data, title=f"Light Curve (Class: {class_name})")
            st.pyplot(fig)
            
    # Check for missing values
    st.write("### Data Quality Check")
    missing_vals = data.isnull().sum().sum()
    if missing_vals > 0:
        st.warning(f"Dataset contains {missing_vals} missing values")
    else:
        st.success("No missing values detected")
    
    # Statistical summary of FLUX values
    if flux_cols:
        st.write("### FLUX Values Summary")
        flux_data = data[flux_cols]
        flux_stats = pd.DataFrame({
            'Min': flux_data.min().min(),
            'Max': flux_data.max().max(),
            'Mean': flux_data.mean().mean(),
            'Std Dev': flux_data.std().mean()
        }, index=['Flux'])
        st.dataframe(flux_stats)

# Main App Logic
if options == "Introduction":
    st.header("Welcome to ExoHunter 🔭")
   
    st.write("""
    This application helps you detect exoplanets using machine learning techniques applied to stellar light curves.
    
    ### What are Exoplanets?
    Exoplanets are planets outside our solar system that orbit stars other than the Sun. 
    They are detected through various methods, with transit photometry being one of the most successful.
    """)
    
    st.write("### How Transit Detection Works")
    
    # Create two columns for the transit explanation
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.write("""
        When an exoplanet passes in front of its host star (as seen from Earth), it blocks a small 
        portion of the star's light. This causes a periodic dip in the star's brightness, which can 
        be detected in the star's light curve.
        
        This subtle dimming - often less than 1% of the star's total brightness - creates a 
        characteristic pattern that our machine learning models are trained to recognize.
        """)
    
    with col2:
        # Display the transit animation in the right column
        st.image("transit.gif", caption="Exoplanet Transit Method", use_container_width=True)
    
    st.write("""
    ### Using This App
    1. **Data Exploration**: Analyze your dataset
    2. **Train Model**: Train machine learning models on transit data
    3. **Test Model**: Evaluate model performance
    4. **HUNT**: Search for new exoplanets in your data
    
    Let's begin exploring the cosmos! 🚀
    """)
    
    # Show sample light curve with transit
    st.subheader("Example: Exoplanet Transit Light Curve")
    # Generate sample data - ideally this would be a real example
    time = np.linspace(0, 10, 1000)
    flux = np.ones_like(time) - 0.02 * np.exp(-((time-3)/0.1)**2) - 0.02 * np.exp(-((time-7)/0.1)**2)
    flux += np.random.normal(0, 0.005, size=len(time))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, flux)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Example Light Curve with Exoplanet Transits")
    
    # Highlight transits
    ax.axvspan(2.8, 3.2, alpha=0.2, color='red')
    ax.axvspan(6.8, 7.2, alpha=0.2, color='red')
    ax.text(3, 0.965, "Transit", color='red')
    ax.text(7, 0.965, "Transit", color='red')
    
    st.pyplot(fig)

elif options == "Data Exploration":
    st.header("Data Exploration")
    
    # File uploader for data
    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load and explore data
        data = load_data(uploaded_file)
        data_exploration(data)

elif options == "Train Model":
    st.header("Train Model")
    
    # File uploader for training data
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    
    if train_file is not None:
        # Check file size
        file_details = {"FileName": train_file.name, "FileType": train_file.type}
        st.write(file_details)
        
        if train_file.size > 1024 * 1024 * 1024:  # 1GB
            st.error("File size exceeds 1GB. Please upload a smaller file.")
        else:
            # Load and preprocess data
            with st.spinner("Loading and preprocessing data..."):
                train_data = load_data(train_file)
                X, y = preprocess_data(train_data)
                
                st.write(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
                
                # Class distribution
                class_dist = pd.DataFrame(y.value_counts()).reset_index()
                class_dist.columns = ['Class', 'Count']
                
                # Update class labels for better readability
                class_dist['Class'] = class_dist['Class'].map({
                    0: 'Non-Exoplanet',
                    1: 'Exoplanet'
                })
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Class', y='Count', data=class_dist, ax=ax)
                ax.set_title('Class Distribution in Training Data')
                st.pyplot(fig)
            
            # Model Selection
            st.subheader("Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Select Model", 
                    ["KNN", "SVM", "RF", "Ensemble"],
                    help="KNN: K-Nearest Neighbors, SVM: Support Vector Machine, RF: Random Forest, Ensemble: Combination of models"
                )
                
                class_balancing = st.selectbox(
                    "Class Imbalance Handling",
                    ["smote", "random", "none"],
                    help="SMOTE: Synthetic Minority Over-sampling Technique, Random: Random Over-sampling, None: No resampling"
                )
                
                cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
            
            with col2:
                feature_method = st.selectbox(
                    "Feature Processing",
                    ["raw", "pca"],
                    help="Raw: Use all flux features, PCA: Reduce dimensions with Principal Component Analysis"
                )
                
                if feature_method == 'pca':
                    n_components = st.slider(
                        "Number of PCA components", 
                        min_value=2, 
                        max_value=min(100, X.shape[0], X.shape[1]),
                        value=min(50, X.shape[0], X.shape[1])
                    )
                else:
                    n_components = None
                    
                scaler_type = st.selectbox(
                    "Scaler Type",
                    ["standard", "robust"],
                    help="Standard: StandardScaler, Robust: RobustScaler (less sensitive to outliers)"
                )
            
            # Model Parameters
            st.subheader("Model Parameters")
            
            params = {}
            if model_type == "KNN":
                params['n_neighbors'] = st.slider("Number of neighbors (k)", 1, 20, 5)
                params['weights'] = st.selectbox("Weight function", ["uniform", "distance"])
                
            elif model_type == "SVM":
                params['C'] = st.select_slider("Regularization parameter (C)", options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
                params['kernel'] = st.selectbox("Kernel type", ["rbf", "linear", "poly", "sigmoid"])
                if params['kernel'] == 'rbf' or params['kernel'] == 'poly' or params['kernel'] == 'sigmoid':
                    params['gamma'] = st.select_slider("Kernel coefficient (gamma)", options=['scale', 'auto', 0.001, 0.01, 0.1, 1.0])
                
            elif model_type == "RF":
                params['n_estimators'] = st.slider("Number of trees", 10, 500, 100)
                params['max_depth'] = st.slider("Maximum tree depth", 2, 30, 10)
                params['min_samples_split'] = st.slider("Minimum samples to split", 2, 10, 2)
                params['min_samples_leaf'] = st.slider("Minimum samples per leaf", 1, 10, 1)
                
            # Train button
            if st.button("Train Model"):
                with st.spinner(f"Training {model_type} model..."):
                    try:
                        # Convert gamma to float if it's a number
                        if model_type == "SVM" and 'gamma' in params and params['gamma'] not in ['scale', 'auto']:
                            params['gamma'] = float(params['gamma'])
                            
                        # Training with selected parameters
                        results = train_model(
                            X, y, model_type, 
                            params=params,
                            feature_extraction=feature_method,
                            sampling_strategy=class_balancing,
                            cv=cv_folds,
                            n_components=n_components,
                            use_scaler=scaler_type
                        )
                        
                        # Store model in session state
                        st.session_state['model_results'] = results
                        st.session_state['model_pipeline'] = results['pipeline']
                        
                        # Display evaluation metrics
                        st.subheader("Model Evaluation")
                        evaluate_model(results)
                        
                        # Option to save model
                        st.subheader("Save Trained Model")
                        model_name = st.text_input("Model filename", "exoplanet_model.joblib")
                        if st.button("Save Model"):
                            save_model(results['pipeline'], filename=model_name)
                            
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        st.exception(e)

elif options == "Test Model":
    st.header("Test Model")
    
    # Check if we have a model
    if 'model_pipeline' not in st.session_state:
        st.warning("No trained model found. Please train a model first or upload a saved model.")
        
        # Option to upload a saved model
        uploaded_model = st.file_uploader("Upload a trained model (.joblib)", type=['joblib'])
        if uploaded_model:
            pipeline = load_model(uploaded_model)
            if pipeline:
                st.session_state['model_pipeline'] = pipeline
                st.success("Model loaded successfully!")
            
    # If we have a model now, allow testing
    if 'model_pipeline' in st.session_state:
        test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
        
        if test_file is not None:
            # Check file size
            if test_file.size > 1024 * 1024 * 1024:  # 1GB
                st.error("File size exceeds 1GB. Please upload a smaller file.")
            else:
                # Load and preprocess test data
                with st.spinner("Loading and preprocessing test data..."):
                    test_data = load_data(test_file)
                    X_test, y_test = preprocess_data(test_data)
                    
                    # Check for NaN values
                    if X_test.isnull().any().any() or y_test.isnull().any():
                        st.error("The test data contains NaN values. Please clean your data.")
                    else:
                        # Run Button for Testing
                        if st.button("Test Model"):
                            with st.spinner("Evaluating model on test data..."):
                                pipeline = st.session_state['model_pipeline']
                                
                                # Apply same transformations if needed
                                if 'model_results' in st.session_state and st.session_state['model_results'].get('feature_extraction') == 'pca':
                                    pca = st.session_state['model_results'].get('pca')
                                    if pca:
                                        X_test_transformed = pca.transform(X_test)
                                        feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]
                                        X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
                                
                                # Make predictions
                                y_pred = pipeline.predict(X_test)
                                
                                # Construct results dict for evaluation
                                results = {
                                    'pipeline': pipeline,
                                    'X_test': X_test,
                                    'y_test': y_test,
                                    'y_pred': y_pred,
                                    'model_type': st.session_state.get('model_results', {}).get('model_type', 'Unknown')
                                }
                                
                                # Add feature names if available
                                if 'model_results' in st.session_state and 'feature_names' in st.session_state['model_results']:
                                    results['feature_names'] = st.session_state['model_results']['feature_names']
                                
                                # Evaluate
                                evaluate_model(results)

elif options == "HUNT":
    st.header("HUNT for Exoplanets 🔭")
    
    # Check if we have a model
    if 'model_pipeline' not in st.session_state:
        st.warning("No trained model found. Please train a model first or upload a saved model.")
        
        # Option to upload a saved model
        uploaded_model = st.file_uploader("Upload a trained model (.joblib)", type=['joblib'])
        if uploaded_model:
            pipeline = load_model(uploaded_model)
            if pipeline:
                st.session_state['model_pipeline'] = pipeline
                st.success("Model loaded successfully!")
    
    # If we have a model now, allow hunting
    if 'model_pipeline' in st.session_state:
        hunt_file = st.file_uploader("Upload HUNT Data (CSV)", type=["csv"])
        
        if hunt_file is not None:
            # Check file size
            if hunt_file.size > 1024 * 1024 * 1024:  # 1GB
                st.error("File size exceeds 1GB. Please upload a smaller file.")
            else:
                # Optional parameters
                confidence = st.slider(
                    "Detection confidence threshold", 
                    min_value=0.5, 
                    max_value=0.95, 
                    value=0.7,
                    step=0.05,
                    help="Minimum confidence level to report an exoplanet detection"
                )
                
                # HUNT button
                if st.button("HUNT for Exoplanets 🔍"):
                    with st.spinner("Analyzing stellar data..."):
                        # Load hunt data
                        hunt_data = load_data(hunt_file)
                        
                        # Check if we need to add a star name column
                        if 'Star Name' not in hunt_data.columns:
                            # Create Star Name column using index
                            hunt_data['Star Name'] = [f"Star-{i}" for i in range(len(hunt_data))]
                        
                        # Run the hunt function
                        pipeline = st.session_state['model_pipeline']
                        hunt_exoplanets(pipeline, None, hunt_data, confidence_threshold=confidence)

elif options == "About":
    st.header("About ExoHunter")
    st.write("""
### Project Information
ExoHunter is an advanced tool for detecting exoplanets using machine learning techniques
applied to stellar light curve data.
    
### How Exoplanets Are Detected
The primary method used here is the transit method, where an exoplanet passes in front of its host star
from our vantage point, causing a small but detectable dip in the star's brightness.
             
### Machine Learning Approach
This application uses several algorithms to classify light curves:
- **[K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)**: Simple but effective for pattern recognition
- **[Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)**: Powerful for binary classification tasks
- **[Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)**: Ensemble method that builds multiple decision trees
- **[Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)**: Combining multiple models for improved performance
    
    ### Data Processing
    The app employs various techniques:
    - Feature extraction and dimensionality reduction (PCA)
    - Handling class imbalance (SMOTE, RandomOverSampling)
    - Cross-validation for robust evaluation
    - Feature importance analysis
    
    ### Credits
    Developed as part of a thesis project in Masters of Science in Computer Science by Aswin Lohani at University of Texas Permian Basin.
    """)
    
    st.write("### Exoplanet Resources")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)")
        st.write("[Kepler & K2 Missions](https://www.nasa.gov/mission_pages/kepler/main/index.html)")
    
    with col2:
        st.write("[TESS Mission](https://tess.mit.edu/)")
        st.write("[Exoplanet Transit Database](http://var2.astro.cz/ETD/)")
