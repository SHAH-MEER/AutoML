import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_covtype
from utils.automl import AutoMLClassifier
from io import BytesIO
import joblib
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Classification - AutoML Suite", page_icon="📊", layout="wide")

st.title("📊 Classification Analysis")

# Initialize session state
if 'classification_data_loaded' not in st.session_state:
    st.session_state.classification_data_loaded = False
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data loading options
    st.subheader("Data Source")
    data_source = st.radio("Choose data source", ["Built-in Dataset", "Upload Custom Data"])
    
    if data_source == "Built-in Dataset":
        DATASETS = {
            "Breast Cancer": load_breast_cancer,
            "Iris": load_iris,
            "Wine": load_wine,
            "Digits": load_digits,
            "Forest Covertypes": fetch_covtype
        }
        dataset_choice = st.selectbox("Choose dataset", list(DATASETS.keys()))
        if st.button("Load Dataset"):
            try:
                loader = DATASETS[dataset_choice]()
                if dataset_choice == "Forest Covertypes":
                    X = pd.DataFrame(loader.data[:5000], columns=loader.feature_names)
                    y = pd.Series(loader.target[:5000] - 1)
                else:
                    X = pd.DataFrame(loader.data, columns=loader.feature_names)
                    y = pd.Series(loader.target)
                
                if hasattr(loader, 'target_names'):
                    y = y.map(dict(enumerate(loader.target_names)))
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.classification_data_loaded = True
                st.session_state.dataset_description = loader.DESCR.split('\n')[0] if hasattr(loader, 'DESCR') else ""
                st.success(f"{dataset_choice} dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    else:  # Custom data upload
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                target_col = st.selectbox("Select target column", df.columns)
                if st.button("Load Custom Data"):
                    st.session_state.X = df.drop(columns=[target_col])
                    st.session_state.y = df[target_col]
                    st.session_state.classification_data_loaded = True
                    st.session_state.dataset_description = f"Custom dataset with {df.shape[1] - 1} features"
                    st.success("Custom data loaded!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Model configuration
    st.subheader("Model Settings")
    test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
    random_state = st.number_input("Random State", 42)
    handle_imbalance = st.checkbox("Handle Class Imbalance", False)
    
    # Feature handling
    st.subheader("Feature Settings")
    max_cardinality = st.slider("Max Cardinality", 10, 1000, 50)
    max_correlation = st.slider("Max Correlation Threshold", 0.7, 1.0, 0.95)
    
    # Model selection
    st.subheader("Select Models")
    model_options = {
        'Random Forest': st.checkbox("Random Forest", True),
        'Gradient Boosting': st.checkbox("Gradient Boosting", True),
        'Logistic Regression': st.checkbox("Logistic Regression", False),
        'SVM': st.checkbox("SVM", False),
        'KNN': st.checkbox("K-Nearest Neighbors", False),
        'Decision Tree': st.checkbox("Decision Tree", False),
        'AdaBoost': st.checkbox("AdaBoost", False),
        'Naive Bayes': st.checkbox("Naive Bayes", False)
    }

# Main content area
if st.session_state.classification_data_loaded:
    X = st.session_state.X
    y = st.session_state.y
    
    # Data preview
    st.header("Data Preview")
    st.markdown(f"**Dataset Description:** {st.session_state.dataset_description}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Features (X)")
        st.dataframe(X.head())
    
    with col2:
        st.subheader("Target (y)")
        # Removed the plot from here

    # Re-add the plot below the columns
    st.subheader("Class Distribution") # Adding the header here
    st.bar_chart(y.value_counts())
    
    # Run analysis button
    if st.button("Run Classification Analysis"):
        try:
            with st.spinner("Training models..."):
                # Initialize models based on selection
                models = []
                model_constructors = {
                    'Random Forest': lambda rs: RandomForestClassifier(random_state=rs),
                    'Gradient Boosting': lambda rs: GradientBoostingClassifier(random_state=rs),
                    'Logistic Regression': lambda rs: LogisticRegression(random_state=rs),
                    'SVM': lambda rs: SVC(probability=True, random_state=rs),
                    'KNN': lambda _: KNeighborsClassifier(),
                    'Decision Tree': lambda rs: DecisionTreeClassifier(random_state=rs),
                    'AdaBoost': lambda rs: AdaBoostClassifier(random_state=rs),
                    'Naive Bayes': lambda _: GaussianNB()
                }
                
                for name, checked in model_options.items():
                    if checked:
                        models.append((name, model_constructors[name](random_state)))
                
                # Initialize and run AutoML
                automl = AutoMLClassifier(
                    models=models,
                    test_size=test_size,
                    random_state=random_state,
                    max_cardinality=max_cardinality,
                    max_correlation=max_correlation,
                    handle_imbalance=handle_imbalance
                )
                
                automl.fit(X, y)
                st.session_state.classification_results = automl
                st.success("Analysis completed!")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    # Display results if available
    if st.session_state.classification_results:
        automl = st.session_state.classification_results
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Best Model Details", "Feature Analysis", "Model Persistence"])
        
        with tab1:
            st.subheader("Model Comparison")
            results_df = pd.DataFrame({
                'Model': list(automl.results.keys()),
                'Accuracy': [res['accuracy'] for res in automl.results.values()]
            }).sort_values('Accuracy', ascending=False)
            st.dataframe(results_df.style.format({'Accuracy': "{:.2%}"}))
        
        with tab2:
            st.subheader("Best Model Details")
            st.markdown(f"**Best Model:** {automl.best_model} - **Accuracy:** {automl.best_score:.2%}")
            
            # Confusion Matrix
            st.write("Confusion Matrix:")
            cm_data = automl.results[automl.best_model]['confusion_matrix']
            st.dataframe(pd.DataFrame(
                cm_data,
                index=[f"True {i}" for i in range(cm_data.shape[0])],
                columns=[f"Pred {i}" for i in range(cm_data.shape[1])]
            ))
            
            # Classification Report
            st.write("Classification Report:")
            st.json(automl.results[automl.best_model]['classification_report'])
        
        with tab3:
            st.subheader("Feature Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**High Cardinality Features Removed**")
                st.write(automl.feature_report['high_cardinality'] or "None")
            with col2:
                st.write("**Potential Leaky Features Removed**")
                st.write(automl.feature_report['leaky_features'] or "None")
            with col3:
                st.write("**Date-like Features Detected**")
                st.write(automl.feature_report['datetime_features'] or "None")
            
            # Feature Importance
            if hasattr(automl.final_model.named_steps['classifier'], 'feature_importances_'):
                st.subheader("Feature Importance")
                importances = automl.final_model.named_steps['classifier'].feature_importances_
                try:
                    preprocessor = automl.final_model.named_steps['preprocessor']
                    feature_names = preprocessor.get_feature_names_out()
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(20)
                    st.bar_chart(importance_df.set_index('Feature'))
                except Exception as e:
                    st.warning(f"Feature importance visualization limited: {str(e)}")
        
        with tab4:
            st.subheader("Model Persistence")
            col1, col2 = st.columns(2)
            with col1:
                buffer = BytesIO()
                automl.save_model(buffer)
                st.download_button(
                    label="Download Model (.pkl)",
                    data=buffer.getvalue(),
                    file_name="classification_model.pkl",
                    mime="application/octet-stream"
                )
            with col2:
                save_path = st.text_input("Save path:", "classification_model.pkl")
                if st.button("Save Model"):
                    try:
                        automl.save_model(save_path)
                        st.success(f"Model saved to {save_path}")
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")
            
            st.subheader("Load Existing Model")
            uploaded_model = st.file_uploader("Upload model file", type=["pkl"])
            if uploaded_model:
                try:
                    loaded_data = joblib.load(uploaded_model)
                    loaded_model = loaded_data['model']
                    sample_input = st.session_state.X.iloc[:1]
                    prediction = loaded_model.predict(sample_input)
                    st.write("Sample prediction:", prediction)
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
else:
    st.info("👈 Please load a dataset from the sidebar to begin analysis.") 