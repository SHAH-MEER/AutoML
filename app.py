import streamlit as st

st.set_page_config(
    page_title="AutoML Web App",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
        """
        <style>
        .social-container {
            display: flex;
            justify-content: flex-end;
            gap: 15px;
            padding: 10px;
        }
        .social-icon {
            font-size: 24px;
            color: white;
        }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <div class="social-container">
            <a href="https://github.com/SHAH-MEER" target="_blank">
                <i class="fab fa-github social-icon"></i>
            </a>
            <a href="https://www.linkedin.com/in/shahmeer-shahzad-790b67356/" target="_blank">
                <i class="fab fa-linkedin social-icon"></i>
            </a>
            <a href="mailto:shahmeershahzad67@gmail.com" target="_blank">
                <i class="fas fa-envelope social-icon"></i>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("🤖 AutoML Web App")
st.markdown("""
Welcome to AutoML Web App - Your one-stop solution for automated machine learning!

This application provides a streamlined workflow for performing **Classification**, **Regression**, and **Unsupervised Learning** tasks.

Use the sidebar on the left to navigate to the specific task you want to perform:

- **Classification Tasks** 📊
- **Regression Tasks** 📈
- **Unsupervised Learning** 🧠

Below is a general guide on how to use the application and its key features:

### How to Use the App

1.  **Select Your Task:** Choose either "Classification", "Regression", or "Unsupervised Learning" from the sidebar navigation.

2.  **Load Your Data:**
    *   **Built-in Datasets:** Select from a list of popular datasets provided by scikit-learn (available on the task-specific pages).
    *   **Upload Custom Data:** Upload your own dataset in CSV or Excel format. For supervised tasks (Classification and Regression), you will need to select the target column. For unsupervised tasks, you may optionally provide a column containing labels if available for evaluation/visualization.

3.  **Configure Analysis Parameters:** Adjust various settings using the controls in the sidebar, such as:
    *   **Test Size:** Determine the proportion of data to be used for testing the trained models (supervised tasks only).
    *   **Random State:** Set a seed for reproducibility of results.
    *   **Feature Handling:** Configure options like maximum cardinality for categorical features and a correlation threshold for identifying potential leaky features.
    *   **Model/Algorithm Selection:** Choose which algorithms you want to include in the automated process for the selected task.
    *   *Classification specific:* Option to handle class imbalance.
    *   *Unsupervised specific:* Parameters for selected algorithms (e.g., number of clusters for KMeans, epsilon and min_samples for DBSCAN, number of components for PCA).

4.  **Run the AutoML Analysis:** Once the data is loaded and parameters are set, click the "Run [Task Type] Analysis" button on the main content area.

5.  **Explore the Results:** After the analysis is complete, the results will be displayed, typically including:
    *   **Supervised Tasks (Classification/Regression):** Model comparison, best model details, feature analysis, performance metrics, and various plots (Confusion Matrices, ROC Curves, Feature Importance, Residuals Plots, Prediction vs Actual Plots).
    *   **Unsupervised Tasks:** Visualizations of clusters or dimensionality reduction, and relevant metrics (e.g., Silhouette score for clustering).

6.  **Model Persistence:** Use the provided options to download a trained supervised model or upload a previously saved supervised model for inference.

### What the App Does (Capabilities)

-   **Automated Data Preprocessing:** Handles missing values, scales numerical features, and encodes categorical features.
-   **Automated Model/Algorithm Application and Tuning:** Trains/applies multiple selected algorithms and performs basic parameter tuning where applicable.
-   **Feature Handling:** Identifies and allows removal of features with high cardinality or high correlation with the target.
-   **Class Imbalance Handling (Classification):** Provides an option to apply techniques like SMOTE to address imbalanced datasets.
-   **Comprehensive Evaluation:** Evaluates supervised models using appropriate metrics and provides detailed reports and visualizations.
-   **Results Visualization:** Generates informative plots for both supervised and unsupervised analyses.
-   **Model Saving and Loading (Supervised):** Allows users to easily save their trained supervised models and load them back later for making predictions without retraining.

Navigate to the Classification, Regression, or Unsupervised Learning page in the sidebar to get started with your analysis.

For more technical details and setup instructions, please refer to the `README.md` file in the project repository.
""")

st.sidebar.success("Select a task type from above.")