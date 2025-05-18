import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from utils.automl import AutoMLRegressor
from io import BytesIO
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Regression - AutoML Suite", page_icon="📈", layout="wide")

st.title("📈 Regression Analysis")

# Initialize session state
if 'regression_data_loaded' not in st.session_state:
    st.session_state.regression_data_loaded = False
if 'regression_results' not in st.session_state:
    st.session_state.regression_results = None

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data loading options
    st.subheader("Data Source")
    data_source = st.radio("Choose data source", ["Built-in Dataset", "Upload Custom Data"])
    
    if data_source == "Built-in Dataset":
        DATASETS = {
            "Diabetes": load_diabetes,
            "California Housing": fetch_california_housing
        }
        dataset_choice = st.selectbox("Choose dataset", list(DATASETS.keys()))
        if st.button("Load Dataset"):
            try:
                loader = DATASETS[dataset_choice]()
                X = pd.DataFrame(loader.data, columns=loader.feature_names)
                y = pd.Series(loader.target)
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.regression_data_loaded = True
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
                    st.session_state.regression_data_loaded = True
                    st.session_state.dataset_description = f"Custom dataset with {df.shape[1] - 1} features"
                    st.success("Custom data loaded!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Model configuration
    st.subheader("Model Settings")
    test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
    random_state = st.number_input("Random State", 42)
    
    # Feature handling
    st.subheader("Feature Settings")
    max_cardinality = st.slider("Max Cardinality", 10, 1000, 50)
    max_correlation = st.slider("Max Correlation Threshold", 0.7, 1.0, 0.95)
    
    # Model selection
    st.subheader("Select Models")
    model_options = {
        'Random Forest': st.checkbox("Random Forest", True),
        'Gradient Boosting': st.checkbox("Gradient Boosting", True),
        'Linear Regression': st.checkbox("Linear Regression", False),
        'SVR': st.checkbox("Support Vector Regression", False),
        'KNN': st.checkbox("K-Nearest Neighbors", False),
        'Decision Tree': st.checkbox("Decision Tree", False),
        'AdaBoost': st.checkbox("AdaBoost", False),
        'Elastic Net': st.checkbox("Elastic Net", False)
    }

# Main content area
if st.session_state.regression_data_loaded:
    X = st.session_state.X
    y = st.session_state.y
    
    # Data preview
    st.header("Data Preview")
    st.markdown(f"**Dataset Description:** {st.session_state.dataset_description}")
    

    st.subheader("Features (X)")
    st.dataframe(X.head())
    
        
    # Re-add the plot below the columns
    st.subheader("Target (y)")
    st.subheader("Target Distribution")
    fig = px.histogram(y, title="Target Distribution")
    st.plotly_chart(fig)
    
    # Run analysis button
    if st.button("Run Regression Analysis"):
        try:
            with st.spinner("Training models..."):
                # Initialize models based on selection
                models = []
                model_constructors = {
                    'Random Forest': lambda rs: RandomForestRegressor(random_state=rs),
                    'Gradient Boosting': lambda rs: GradientBoostingRegressor(random_state=rs),
                    'Linear Regression': lambda _: LinearRegression(),
                    'SVR': lambda _: SVR(),
                    'KNN': lambda _: KNeighborsRegressor(),
                    'Decision Tree': lambda rs: DecisionTreeRegressor(random_state=rs),
                    'AdaBoost': lambda rs: AdaBoostRegressor(random_state=rs),
                    'Elastic Net': lambda rs: ElasticNet(random_state=rs)
                }
                
                for name, checked in model_options.items():
                    if checked:
                        models.append((name, model_constructors[name](random_state)))
                
                # Initialize and run AutoML
                automl = AutoMLRegressor(
                    models=models,
                    test_size=test_size,
                    random_state=random_state,
                    max_cardinality=max_cardinality,
                    max_correlation=max_correlation
                )
                
                automl.fit(X, y)
                st.session_state.regression_results = automl
                st.success("Analysis completed!")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    # Display results if available
    if st.session_state.regression_results:
        automl = st.session_state.regression_results
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Best Model Details", "Feature Analysis", "Model Persistence"])
        
        with tab1:
            st.subheader("Model Comparison")
            results_df = pd.DataFrame({
                'Model': list(automl.results.keys()),
                'R² Score': [res['r2_score'] for res in automl.results.values()],
                'MSE': [res['mse'] for res in automl.results.values()],
                'MAE': [res['mae'] for res in automl.results.values()]
            }).sort_values('R² Score', ascending=False)
            st.dataframe(results_df.style.format({
                'R² Score': "{:.3f}",
                'MSE': "{:.3f}",
                'MAE': "{:.3f}"
            }))
        
        with tab2:
            st.subheader("Best Model Details")
            st.markdown(f"**Best Model:** {automl.best_model}")
            st.markdown(f"**R² Score:** {automl.best_score:.3f}")
            
            # Residuals Plot
            st.write("Residuals Plot:")
            residuals = automl.results[automl.best_model]['residuals']
            fig = px.scatter(
                x=automl.results[automl.best_model]['y_test'],
                y=residuals,
                title="Residuals Plot",
                labels={'x': 'Actual Values', 'y': 'Residuals'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig)
            
            # Prediction vs Actual Plot
            st.write("Prediction vs Actual Plot:")
            fig = px.scatter(
                x=automl.results[automl.best_model]['y_test'],
                y=automl.results[automl.best_model]['y_pred'],
                title="Prediction vs Actual",
                labels={'x': 'Actual Values', 'y': 'Predicted Values'}
            )
            fig.add_trace(px.line(x=[min(y), max(y)], y=[min(y), max(y)]).data[0])
            st.plotly_chart(fig)
        
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
            if hasattr(automl.final_model.named_steps['regressor'], 'feature_importances_'):
                st.subheader("Feature Importance")
                importances = automl.final_model.named_steps['regressor'].feature_importances_
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
                    file_name="regression_model.pkl",
                    mime="application/octet-stream"
                )
            with col2:
                save_path = st.text_input("Save path:", "regression_model.pkl")
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