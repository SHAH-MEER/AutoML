# 🤖 AutoML Classification Web App 🔬

A powerful and intuitive web application for automated machine learning classification built with Streamlit and scikit-learn. ✨

## ✨ Features ✨

- 📊 **Built-in Datasets**: Load popular scikit-learn datasets with a single click 🖱️
- 📁 **Custom Data Upload**: Support for CSV and Excel file formats 📑
- 🧠 **Multiple Algorithms**: Random Forest 🌲, Gradient Boosting 📈, Logistic Regression 📉, SVM 🔄, KNN 🔍, Decision Tree 🌳, AdaBoost 🚀, and Naive Bayes 🎲
- 🔍 **Feature Analysis**: Automatic detection of high cardinality features, leaky features, and datetime features 🕒
- ⚖️ **Class Imbalance Handling**: SMOTE, ADASYN, and UnderSampling techniques 📊
- 📈 **Comprehensive Visualization**: Model comparison 📊, confusion matrices 🔢, ROC curves 📉, and feature importance 🏆
- 💾 **Model Persistence**: Save and load trained models for future use 🔄
- ⚙️ **Customizable Parameters**: Adjust test size, random state, cross-validation folds, and more ⚙️
- 🚀 **Fast Mode**: Quick training option for rapid prototyping ⚡
- 🔄 **Cross-validation**: Configurable k-fold cross-validation for robust evaluation 🎯

## 🚀 Quick Start 🚀

1. Clone the repository 📥
   ```
   git clone https://github.com/SHAH-MEER/automl-classification-app.git
   cd automl-classification-app
   ```
2. Install dependencies 🔧
   ```
   pip install -r requirements.txt
   ```
3. Run the app 🏃‍♂️
   ```
   streamlit run app.py
   ```
4. Access the app in your browser at: `http://localhost:8501` 🌐

## 📋 Requirements 📋

- Python 3.7+ 🐍
- pandas 🐼
- numpy 🔢
- matplotlib 📊
- seaborn 📈
- plotly 📉
- scikit-learn 🧠
- imbalanced-learn ⚖️
- streamlit 🖥️
- joblib 💾

## 🛠️ Usage 🛠️

1. Choose a built-in dataset or upload your own data 📁
2. Configure parameters in the sidebar ⚙️
3. Select models to train 🧠
4. Click "Run AutoML Analysis" 🚀
5. Explore results and download the best model 📊

## 🔧 Advanced Configuration 🔧

- **Test Size** 📏: Adjust the train/test split ratio
- **Random State** 🎲: Set seed for reproducible results
- **Class Imbalance** ⚖️: Toggle handling of imbalanced datasets
- **Max Cardinality** 🔢: Configure threshold for high cardinality features
- **Max Correlation** 📊: Set threshold for identifying leaky features
- **CV Folds** 📂: Customize number of cross-validation folds
- **Max Training Time** ⏱️: Limit training time per model
- **Hyperparameter Iterations** 🔄: Set number of search iterations
- **Feature Selection** 🔍: Enable automatic feature selection

## 📊 Output 📊

- Model performance comparison 📈
- Feature importance analysis 🏆
- Confusion matrices 🔢
- ROC curves 📉
- Classification reports 📝
- Downloadable trained model 💾
- Performance metrics visualization 📊

## 💻 Example Code 💻

```python
# Load a dataset 📁
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# Train models 🧠
automl = AutoMLClassifier(
    models=[('Random Forest', RandomForestClassifier())],
    test_size=0.2,
    random_state=42
)
best_model = automl.fit(X, y)

# Make predictions 🔮
predictions = best_model.predict(X_test)

# Save the model 💾
automl.save_model("iris_classifier.pkl")
```

## 📬 Contact 📬

[![Email](https://img.shields.io/badge/Email-shahmeershahzad67%40gmail.com-blue?style=flat-square&logo=gmail)](mailto:shahmeershahzad67@gmail.com) 📧
[![GitHub](https://img.shields.io/badge/GitHub-SHAH--MEER-black?style=flat-square&logo=github)](https://github.com/SHAH-MEER) 🐙
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shahmeer%20Shahzad-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shahmeer-shahzad-790b67356/) 👔

## 📜 License 📜

This project is licensed under the MIT License - ⚖️

## 🤝 Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🙌

1. Fork the repository 🍴
2. Create your feature branch (`git checkout -b feature/amazing-feature`) 🌿
3. Commit your changes (`git commit -m 'Add some amazing feature'`) 💬
4. Push to the branch (`git push origin feature/amazing-feature`) 🚀
5. Open a Pull Request 📝

## 🙏 Acknowledgements 🙏

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms 🧠
- [Streamlit](https://streamlit.io/) for the web application framework 🖥️
- [imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance ⚖️
- [Plotly](https://plotly.com/) for interactive visualizations 📊
- [Pandas](https://pandas.pydata.org/) for data manipulation 🐼

## 🌟 What's Next 🌟

- Time series forecasting support 📅
- Natural language processing features 📝
- Integration with more data sources 🌐
- Automated report generation 📄
- Model interpretability tools 🔍

---

🌟 Star this repository if you find it useful! ⭐
