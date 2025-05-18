# 🤖 AutoML Web App 📊📈

A powerful and intuitive Streamlit web application for automated machine learning, supporting both **Classification** and **Regression** tasks. ✨

## ✨ Features ✨

- 📊 **Classification Analysis**: Automated classification on built-in or custom datasets.
- 📈 **Regression Analysis**: Automated regression on built-in or custom datasets.
- 📁 **Built-in Datasets**: Load popular scikit-learn datasets for both tasks with a single click 🖱️ (Iris, Wine, Breast Cancer, Digits, Forest Covertypes for Classification; Diabetes, California Housing for Regression).
- 📑 **Custom Data Upload**: Support for CSV and Excel file formats.
- 🧠 **Multiple Algorithms**: Access a range of algorithms for each task type (e.g., Random Forest, Gradient Boosting, Logistic Regression, SVM, KNN, Decision Tree, AdaBoost, Naive Bayes for Classification; Linear Regression, SVR, Elastic Net, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, KNeighborsRegressor, DecisionTreeRegressor for Regression).
- 🔍 **Feature Analysis**: Automatic detection of high cardinality features, potential leaky features, and datetime features 🕒.
- ⚖️ **Class Imbalance Handling**: SMOTE and other techniques available for classification tasks 📊.
- 📊📈 **Comprehensive Visualization**: Model comparison, confusion matrices, ROC curves, feature importance for classification; R² Score, MSE, MAE, Residuals Plot, Prediction vs Actual Plot, Feature Importance for regression.
- 💾 **Model Persistence**: Save and load trained models for future use 🔄.
- ⚙️ **Customizable Parameters**: Adjust test size, random state, cross-validation folds (for classification), and more.

## 🚀 Quick Start 🚀

1. Clone the repository 📥
   ```bash
   git clone <repository_url>
   cd AutoML
   ```
   (Replace `<repository_url>` with your actual repository URL if you have one.)

2. Install dependencies 🔧
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app 🏃‍♂️
   ```bash
   streamlit run app.py
   ```

4. Access the app in your web browser at: `http://localhost:8501` 🌐

## 📋 Requirements 📋

- Python 3.7+ 🐍
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `plotly`
- `joblib`
- `imbalanced-learn` (for classification imbalance handling)
- `openpyxl` (for Excel file support)

See `requirements.txt` for specific versions.

## 🛠️ Usage 🛠️

1. Launch the app using the command in the Quick Start section.
2. Navigate to either the "Classification" or "Regression" page using the sidebar on the left.
3. On the chosen task page, select a built-in dataset or upload your own data.
4. Configure parameters in the sidebar, including data split, model settings, and feature handling.
5. Select the models you wish to train for the chosen task.
6. Click the "Run Analysis" button (e.g., "Run Classification Analysis" or "Run Regression Analysis") to train and evaluate the selected models.
7. Explore the results through tables and interactive visualizations.
8. Use the Model Persistence section to download or load trained models.

## 📂 Project Structure 📂

-   `app.py`: The main entry point and home page, providing an overview and directing users to task pages.
-   `pages/`: Directory containing the Streamlit pages for specific tasks.
    -   `1_Classification.py`: Code for the Classification analysis page.
    -   `2_Regression.py`: Code for the Regression analysis page.
-   `utils/`: Directory for shared helper code.
    -   `automl.py`: Contains the `BaseAutoML` class and task-specific subclasses (`AutoMLClassifier`, `AutoMLRegressor`).
-   `requirements.txt`: Lists the project dependencies.
-   `README.md`: Project documentation.

## 📬 Contact 📬

[![Email](https://img.shields.io/badge/Email-shahmeershahzad67%40gmail.com-blue?style=flat-square&logo=gmail)](mailto:shahmeershahzad67@gmail.com) 📧
--------- 
[![GitHub](https://img.shields.io/badge/GitHub-SHAH--MEER-black?style=flat-square&logo=github)](https://github.com/SHAH-MEER) 🐙
----------- 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shahmeer%20Shahzad-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shahmeer-shahzad-790b67356/) 👔

## 📜 License 📜

This project is licensed under the MIT License - see the LICENSE file for details (if you have one). ⚖️

## 🤝 Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🙌

1. Fork the repository 🍴
2. Create your feature branch (`git checkout -b feature/your-feature-name`) 🌿
3. Commit your changes (`git commit -m 'Add your feature'`) 💬
4. Push to the branch (`git push origin feature/your-feature-name`) 🚀
5. Open a Pull Request on GitHub 📝

## 🙏 Acknowledgements 🙏

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms 🧠.
- [Streamlit](https://streamlit.io/) for the web application framework 🖥️.
- [imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance (Classification) ⚖️.
- [Plotly](https://plotly.com/) for interactive visualizations 📊📈.
- [Pandas](https://pandas.pydata.org/) for data manipulation 🐼.

## 🌟 What's Next 🌟

- Adding more advanced models for both tasks.
- Implementing feature selection and engineering options.
- Improving visualization capabilities.
- Exploring time series and NLP task support.

---

🌟 Star this repository if you find it useful! ⭐
