import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
# We will add more imports later as we implement algorithms
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.manifold import TSNE
# from utils.automl import AutoMLUnsupervised # We will create this class later

st.set_page_config(page_title="Unsupervised Learning - AutoML Web App", page_icon="🧠", layout="wide")

st.title("🧠 Unsupervised Learning Analysis")

# Initialize session state
if 'unsupervised_data_loaded' not in st.session_state:
    st.session_state.unsupervised_data_loaded = False
if 'unsupervised_results' not in st.session_state:
    st.session_state.unsupervised_results = None

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data loading options
    st.subheader("Data Source")
    data_source = st.radio("Choose data source", ["Built-in Dataset", "Upload Custom Data"])
    
    if data_source == "Built-in Dataset":
        DATASETS = {
            "Digits": load_digits,
            # Add other built-in unsupervised datasets here if needed
        }
        dataset_choice = st.selectbox("Choose dataset", list(DATASETS.keys()))
        if st.button("Load Dataset"):
            try:
                loader = DATASETS[dataset_choice]()
                X = pd.DataFrame(loader.data, columns=loader.feature_names)
                # For unsupervised learning, there is no target variable 'y' initially
                # However, some datasets might have labels useful for evaluation (e.g., digits labels for clustering)
                y = pd.Series(loader.target) if hasattr(loader, 'target') else None
                
                st.session_state.X_unsupervised = X
                st.session_state.y_unsupervised = y # Store labels if available for evaluation/visualization
                st.session_state.unsupervised_data_loaded = True
                st.session_state.dataset_description_unsupervised = loader.DESCR.split('\n')[0] if hasattr(loader, 'DESCR') else ""
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
                
                # For custom data, users might upload data with or without labels.
                # We can provide an option to specify if a column contains labels.
                has_labels = st.checkbox("Does your dataset contain a column for labels?")
                label_col = None
                if has_labels:
                    label_col = st.selectbox("Select the column containing labels", df.columns)
                
                if st.button("Load Custom Data"):
                    if has_labels and label_col:
                         X = df.drop(columns=[label_col])
                         y = df[label_col]
                    else:
                         X = df
                         y = None

                    st.session_state.X_unsupervised = X
                    st.session_state.y_unsupervised = y
                    st.session_state.unsupervised_data_loaded = True
                    st.session_state.dataset_description_unsupervised = f"Custom dataset with {df.shape[1] - (1 if has_labels else 0)} features"
                    st.success("Custom data loaded!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # --- Add configuration options for unsupervised models here later ---
    # st.subheader("Unsupervised Model Settings")
    # model_options_unsupervised = {
    #    'KMeans': st.checkbox("KMeans", True),
    #    'DBSCAN': st.checkbox("DBSCAN", False),
    #    # Add more models
    # }

# Main content area
if st.session_state.unsupervised_data_loaded:
    X = st.session_state.X_unsupervised
    y = st.session_state.y_unsupervised
    
    # Data preview
    st.header("Data Preview")
    st.markdown(f"**Dataset Description:** {st.session_state.dataset_description_unsupervised}")
    
    st.subheader("Features (X)")
    st.dataframe(X.head())

    if y is not None:
        st.subheader("Labels")
        # Display label distribution if labels are available
        if pd.api.types.is_numeric_dtype(y) or pd.api.types.is_categorical_dtype(y) or y.nunique() < 50:
             st.write("Label Distribution:")
             st.bar_chart(y.value_counts())
        else:
             st.write("Label distribution not shown for high cardinality or non-numeric labels.")


    # --- Add button to run analysis and display results here later ---
    # if st.button("Run Unsupervised Analysis"):
    #    try:
    #        with st.spinner("Performing unsupervised analysis..."):
    #            # Initialize and run AutoML Unsupervised
    #            automl_unsupervised = AutoMLUnsupervised(
    #                models=selected_unsupervised_models,
    #                # Add relevant parameters
    #            )
    #            # Results could be a dict of results per algorithm
    #            unsupervised_results = automl_unsupervised.run(X, y) # Pass labels for evaluation if available
    #            st.session_state.unsupervised_results = unsupervised_results
    #            st.success("Unsupervised analysis completed!")
    #    except Exception as e:
    #        st.error(f"Analysis failed: {str(e)}")
    
    # --- Display results if available here later ---
    # if st.session_state.unsupervised_results:
    #     st.header("Analysis Results")
    #     # Display results based on the algorithms used
    #     # e.g., show cluster plots, dimensionality reduction plots, metrics

else:
    st.info("👈 Please load a dataset from the sidebar to begin unsupervised analysis.") 