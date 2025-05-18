# AutoML Suite

AutoML Suite is a Streamlit web application that provides automated machine learning capabilities for both classification and regression tasks.

## Features

- **Classification Analysis**: Perform automated classification on built-in or custom datasets.
- **Regression Analysis**: Perform automated regression on built-in or custom datasets.

- Automated data preprocessing and feature handling.
- Model training, evaluation, and comparison.
- Interactive visualizations (Confusion Matrix, Classification Report, Feature Importance, Residuals Plot, Prediction vs Actual Plot).
- Model persistence (saving and loading trained models).

## Setup

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd AutoML
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit application from the project root directory:

    ```bash
    streamlit run app.py
    ```

3.  The application will open in your web browser.

## Project Structure

-   `app.py`: The main entry point and home page (documentation only).
-   `pages/`: Contains the individual Streamlit pages.
    -   `1_Classification.py`: Handles classification tasks.
    -   `2_Regression.py`: Handles regression tasks.
-   `utils/`: Contains helper classes and functions.
    -   `automl.py`: Base AutoML class and task-specific subclasses.
-   `requirements.txt`: Lists the project dependencies.

## Usage

-   Navigate between Classification and Regression tasks using the sidebar.
-   On each task page, select a built-in dataset or upload your own.
-   Configure the model settings and feature handling options.
-   Click the "Run Analysis" button to train and evaluate models.
-   Explore the results, visualizations, and download/load trained models.

## Dependencies

See `requirements.txt` for a full list of dependencies.

## Contributing

(Optional: Add guidelines for contributing if this is an open-source project)

## License

(Optional: Add license information) 