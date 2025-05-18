import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from utils.automl import AutoMLUnsupervised # Import the unsupervised class

# Import unsupervised specific modules for parameter controls and constructors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.express as px
from sklearn.model_selection import ParameterGrid

st.set_page_config(page_title="Unsupervised Learning - AutoML Web App", page_icon="🧠", layout="wide")

st.title("🧠 Unsupervised Learning Analysis")

# Initialize session state
if 'unsupervised_data_loaded' not in st.session_state:
    st.session_state.unsupervised_data_loaded = False
if 'unsupervised_results' not in st.session_state:
    st.session_state.unsupervised_results = None
if 'X_unsupervised' not in st.session_state:
    st.session_state.X_unsupervised = None
if 'y_unsupervised' not in st.session_state:
    st.session_state.y_unsupervised = None
if 'dataset_description_unsupervised' not in st.session_state:
    st.session_state.dataset_description_unsupervised = ""

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
                st.session_state.unsupervised_results = None # Clear previous results
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
                    st.session_state.unsupervised_results = None # Clear previous results
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Unsupervised Model Settings - only show if data is loaded
    if st.session_state.unsupervised_data_loaded:
        st.subheader("Algorithm Selection")
        
        # Clustering Algorithms
        st.markdown("**Clustering Algorithms**")
        clustering_options = {
           'KMeans': st.checkbox("KMeans", True),
           'DBSCAN': st.checkbox("DBSCAN", False)
        }

        # Dimensionality Reduction Algorithms
        st.markdown("**Dimensionality Reduction**")
        reduction_options = {
           'PCA': st.checkbox("PCA", True),
           'TSNE': st.checkbox("t-SNE", False)
        }

        # Add t-SNE disclaimer
        if reduction_options['TSNE']:
            with st.expander("ℹ️ Important Note about t-SNE"):
                st.markdown("""
                **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is a powerful visualization technique, but please note:
                
                - ⏱️ **Time Complexity**: t-SNE can be computationally expensive, especially with large datasets
                - 📊 **Best for Visualization**: t-SNE is primarily used for visualization, not feature reduction
                - 🔢 **Dataset Size**: Recommended for datasets with less than 10,000 samples
                - 🎯 **Perplexity**: The perplexity parameter significantly affects runtime
                
                If your dataset is large, consider using PCA first to reduce dimensions before applying t-SNE.
                """)

        selected_algorithms = []
        selected_algorithms.extend([(name, True) for name, selected in clustering_options.items() if selected])
        selected_algorithms.extend([(name, True) for name, selected in reduction_options.items() if selected])

        # Algorithm Parameters (only show if algorithm is selected)
        algorithm_params = {}
        st.subheader("Algorithm Parameters")
        
        # Enable/Disable Auto-tuning
        auto_tune = st.checkbox("Enable Automatic Hyperparameter Tuning", value=False)
        
        # Clustering Parameters
        if any(clustering_options.values()):
            st.markdown("**Clustering Parameters**")
            if clustering_options['KMeans']:
                if st.session_state.y_unsupervised is not None and len(np.unique(st.session_state.y_unsupervised)) > 1:
                    n_unique_labels = len(np.unique(st.session_state.y_unsupervised))
                    st.info(f"Labels provided: Setting KMeans clusters to {n_unique_labels} (number of unique labels)")
                    algorithm_params['kmeans_n_clusters'] = n_unique_labels
                elif auto_tune:
                    st.info("KMeans: Will automatically find optimal number of clusters (2-10) using silhouette score")
                    algorithm_params['kmeans_n_clusters'] = 'auto'
                else:
                    algorithm_params['kmeans_n_clusters'] = st.number_input("Number of clusters (KMeans)", min_value=1, value=8)
            
            if clustering_options['DBSCAN']:
                if auto_tune:
                    st.info("DBSCAN: Will automatically find optimal eps and min_samples using silhouette score")
                    algorithm_params['dbscan_auto_tune'] = True
                    algorithm_params['dbscan_eps_range'] = np.linspace(0.1, 2.0, 10)
                    algorithm_params['dbscan_min_samples_range'] = range(2, 11)
                else:
                    algorithm_params['dbscan_eps'] = st.number_input("Epsilon (DBSCAN)", min_value=0.1, value=0.5, step=0.1)
                    algorithm_params['dbscan_min_samples'] = st.number_input("Minimum samples (DBSCAN)", min_value=1, value=5)
        
        # Dimensionality Reduction Parameters
        if any(reduction_options.values()):
            st.markdown("**Dimensionality Reduction Parameters**")
            if reduction_options['PCA']:
                if auto_tune:
                    st.info("PCA: Will automatically find optimal number of components to explain 95% variance")
                    algorithm_params['pca_n_components'] = 'auto'
                else:
                    pca_n_components_option = st.radio("Number of components (PCA)", ['Auto', 2, 3, 5])
                    if pca_n_components_option == 'Auto':
                        algorithm_params['pca_n_components'] = None
                    else:
                        algorithm_params['pca_n_components'] = pca_n_components_option
            
            if reduction_options['TSNE']:
                if auto_tune:
                    st.info("t-SNE: Will automatically find optimal perplexity (5-50) using reconstruction error")
                    algorithm_params['tsne_auto_tune'] = True
                    algorithm_params['tsne_perplexity_range'] = range(5, 51, 5)
                else:
                    algorithm_params['tsne_n_components'] = st.radio("Number of components (t-SNE)", [2, 3])
                    algorithm_params['tsne_perplexity'] = st.slider("Perplexity (t-SNE)", min_value=5, max_value=50, value=30)


# Main content area
if st.session_state.unsupervised_data_loaded and selected_algorithms:
    X = st.session_state.X_unsupervised
    y = st.session_state.y_unsupervised
    
    # Data preview
    st.header("Data Preview")
    st.markdown(f"**Dataset Description:** {st.session_state.dataset_description_unsupervised}")
    
    st.subheader("Features (X)")
    st.dataframe(X.head())

    if y is not None:
        st.subheader("Labels (if provided)")
        # Display label distribution if labels are available
        if pd.api.types.is_numeric_dtype(y) or pd.api.types.is_categorical_dtype(y) or y.nunique() < 50:
             st.write("Label Distribution:")
             st.bar_chart(y.value_counts())
        else:
             st.write("Label distribution not shown for high cardinality or non-numeric labels.")


    # Button to run analysis
    if st.button("Run Unsupervised Analysis"):
        if not selected_algorithms:
             st.warning("Please select at least one unsupervised algorithm to run.")
        else:
            try:
                with st.spinner("Performing unsupervised analysis..."):
                    # Initialize and run AutoML Unsupervised
                    automl_unsupervised = AutoMLUnsupervised(
                        algorithms=[(name, globals()[name]()) for name, _ in selected_algorithms],
                        random_state=42,
                        **algorithm_params
                    )
                    
                    # Results could be a dict of results per algorithm
                    unsupervised_results = automl_unsupervised.run(X, y) # Pass labels for evaluation if available
                    st.session_state.unsupervised_results = unsupervised_results
                    st.success("Unsupervised analysis completed!")
                    
            except ValueError as e:
                 st.error(f"Analysis failed: Data or preprocessing error - {str(e)}")
            except Exception as e:
                st.error(f"Analysis failed: An unexpected error occurred - {str(e)}")
    
    # Display results if available
    if st.session_state.unsupervised_results:
        st.header("Analysis Results")
        
        results = st.session_state.unsupervised_results

        if 'error' in results:
             st.error(f"An error occurred during the analysis: {results['error']}")
        else:
            # Display Clustering Results
            clustering_results = {name: results[name] for name in clustering_options.keys() if name in results}
            if clustering_results:
                st.subheader("Clustering Results")
                for algo_name, algo_result in clustering_results.items():
                    st.markdown(f"### {algo_name} Results")
                    if 'error' in algo_result:
                        st.error(f"Error running {algo_name}: {algo_result['error']}")
                        continue

                    if 'labels' in algo_result:
                        labels = algo_result['labels']
                        st.write("Cluster Labels:", labels[:10])
                        st.write(f"Number of clusters found (excluding noise if any): {len(np.unique(labels)) - (1 if -1 in labels else 0)}")

                        if 'silhouette_score' in algo_result:
                            score = algo_result['silhouette_score']
                            if isinstance(score, float):
                                st.write(f"Silhouette Score: {score:.3f}")
                            else:
                                st.write(f"Silhouette Score: {score}")

                        if 'tuning_results' in algo_result:
                            st.markdown("**Hyperparameter Tuning Results**")
                            tuning_df = pd.DataFrame(algo_result['tuning_results'])
                            st.dataframe(tuning_df)
                            
                            # Plot tuning results
                            if 'param' in tuning_df.columns and 'score' in tuning_df.columns:
                                fig = px.line(tuning_df, x='param', y='score', 
                                            title=f'{algo_name} Hyperparameter Tuning Results')
                                st.plotly_chart(fig)

                        # Visualization code for clustering results
                        if X.shape[1] > 2 and ('PCA' not in results or 'TSNE' not in results):
                            st.info("Data has more than 2 dimensions. Consider running PCA or t-SNE for visualization.")
                        else:
                            # Use PCA or t-SNE results for visualization if available
                            if 'PCA' in results and 'transformed_data' in results['PCA']:
                                pca_data = np.array(results['PCA']['transformed_data'])
                                if pca_data.shape[1] >= 2:
                                    plot_df = pd.DataFrame(pca_data[:, :2], columns=['PC1', 'PC2'])
                                    plot_df['Cluster'] = [str(l) for l in labels]
                                    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster', 
                                                   title=f'{algo_name} Clusters (PCA Reduced)')
                                    st.plotly_chart(fig)
                                if pca_data.shape[1] >= 3:
                                    plot_df = pd.DataFrame(pca_data[:, :3], columns=['PC1', 'PC2', 'PC3'])
                                    plot_df['Cluster'] = [str(l) for l in labels]
                                    fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                                                      title=f'{algo_name} Clusters (PCA Reduced)')
                                    st.plotly_chart(fig)
                            elif 'TSNE' in results and 'transformed_data' in results['TSNE']:
                                tsne_data = np.array(results['TSNE']['transformed_data'])
                                if tsne_data.shape[1] >= 2:
                                    plot_df = pd.DataFrame(tsne_data[:, :2], columns=['TSNE1', 'TSNE2'])
                                    plot_df['Cluster'] = [str(l) for l in labels]
                                    fig = px.scatter(plot_df, x='TSNE1', y='TSNE2', color='Cluster',
                                                   title=f'{algo_name} Clusters (t-SNE Reduced)')
                                    st.plotly_chart(fig)
                                if tsne_data.shape[1] >= 3:
                                    plot_df = pd.DataFrame(tsne_data[:, :3], columns=['TSNE1', 'TSNE2', 'TSNE3'])
                                    plot_df['Cluster'] = [str(l) for l in labels]
                                    fig = px.scatter_3d(plot_df, x='TSNE1', y='TSNE2', z='TSNE3', color='Cluster',
                                                      title=f'{algo_name} Clusters (t-SNE Reduced)')
                                    st.plotly_chart(fig)

                        # Compare with original labels if available
                        if y is not None:
                            st.subheader(f"{algo_name} Cluster vs Original Labels")
                            comparison_df = pd.DataFrame({'Cluster': labels, 'Original Label': y.tolist()})
                            st.dataframe(pd.crosstab(comparison_df['Original Label'], comparison_df['Cluster']))

            # Display Dimensionality Reduction Results
            reduction_results = {name: results[name] for name in reduction_options.keys() if name in results}
            if reduction_results:
                st.subheader("Dimensionality Reduction Results")
                for algo_name, algo_result in reduction_results.items():
                    st.markdown(f"### {algo_name} Results")
                    if 'error' in algo_result:
                        st.error(f"Error running {algo_name}: {algo_result['error']}")
                        continue

                    if 'transformed_data' in algo_result:
                        transformed_data = np.array(algo_result['transformed_data'])
                        st.write("Transformed Data (first 5 rows):", transformed_data[:5])

                        if 'explained_variance_ratio' in algo_result:
                            st.write("Explained Variance Ratio:", algo_result['explained_variance_ratio'])
                            st.write(f"Total Explained Variance: {sum(algo_result['explained_variance_ratio']):.3f}")

                            if algo_name == 'PCA':
                                explained_variance_df = pd.DataFrame({
                                    'Principal Component': range(1, len(algo_result['explained_variance_ratio']) + 1),
                                    'Explained Variance Ratio': algo_result['explained_variance_ratio']
                                })
                                fig = px.bar(explained_variance_df, x='Principal Component', y='Explained Variance Ratio',
                                            title='PCA Explained Variance Ratio')
                                st.plotly_chart(fig)

                        if 'tuning_results' in algo_result:
                            st.markdown("**Hyperparameter Tuning Results**")
                            tuning_df = pd.DataFrame(algo_result['tuning_results'])
                            st.dataframe(tuning_df)
                            
                            # Plot tuning results
                            if 'param' in tuning_df.columns and 'score' in tuning_df.columns:
                                fig = px.line(tuning_df, x='param', y='score', 
                                            title=f'{algo_name} Hyperparameter Tuning Results')
                                st.plotly_chart(fig)

                        # Visualize the reduced data
                        if transformed_data.shape[1] >= 2:
                            plot_df = pd.DataFrame(transformed_data[:, :2], columns=['Component 1', 'Component 2'])
                            if y is not None:
                                plot_df['Original Label'] = y.tolist()
                                color_col = 'Original Label'
                            else:
                                color_col = None

                            fig = px.scatter(plot_df, x='Component 1', y='Component 2', color=color_col,
                                            title=f'{algo_name} Reduced Data (2 Components)')
                            st.plotly_chart(fig)

                        if transformed_data.shape[1] >= 3:
                            plot_df = pd.DataFrame(transformed_data[:, :3], columns=['Component 1', 'Component 2', 'Component 3'])
                            if y is not None:
                                plot_df['Original Label'] = y.tolist()
                                color_col = 'Original Label'
                            fig = px.scatter_3d(plot_df, x='Component 1', y='Component 2', z='Component 3', color=color_col,
                                                title=f'{algo_name} Reduced Data (3 Components)')
                            st.plotly_chart(fig)



else:
    st.info("👈 Please load a dataset from the sidebar and select at least one algorithm to begin unsupervised analysis.") 
    st.info("👈 Please load a dataset from the sidebar and select at least one algorithm to begin unsupervised analysis.") 