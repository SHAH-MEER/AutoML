import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import joblib
import time
import datetime
from io import BytesIO, StringIO
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_covtype
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, RocCurveDisplay, f1_score
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

class AutoMLClassifier:
    DEFAULT_PARAMS = {
        RandomForestClassifier: {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        },
        GradientBoostingClassifier: {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        },
        LogisticRegression: {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2', 'none'],
            'classifier__solver': ['lbfgs', 'saga'],
            'classifier__multi_class': ['auto', 'multinomial']
        },
        SVC: {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale']
        },
        KNeighborsClassifier: {
            'classifier__n_neighbors': [3, 5, 7, 10],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        DecisionTreeClassifier: {
            'classifier__max_depth': [None, 5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        },
        AdaBoostClassifier: {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 1.0]
        },
        GaussianNB: {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    }

    def __init__(self, models=None, test_size=0.2, max_cardinality=50,
                 max_correlation=0.95, random_state=42, n_iter=20, handle_imbalance=False, cv_folds=5):
        self.random_state = random_state
        self.test_size = test_size
        self.models = models or [
            ('Random Forest', RandomForestClassifier(random_state=random_state)),
            ('Gradient Boosting', GradientBoostingClassifier(random_state=random_state)),
            ('Logistic Regression', LogisticRegression(random_state=random_state)),
            ('SVM', SVC(probability=True, random_state=random_state)),
            ('KNN', KNeighborsClassifier()),
            ('Decision Tree', DecisionTreeClassifier(random_state=random_state)),
            ('AdaBoost', AdaBoostClassifier(random_state=random_state)),
            ('Naive Bayes', GaussianNB())
        ]
        self.best_model = None
        self.best_score = 0
        self.results = {}
        self.final_model = None
        self.n_iter = n_iter
        self.label_encoder = None
        self.max_cardinality = max_cardinality
        self.max_correlation = max_correlation
        self.handle_imbalance = handle_imbalance
        self.cv_folds = cv_folds

    def _feature_analysis(self, X, y):
        self.feature_report = {
            'high_cardinality': [],
            'leaky_features': [],
            'datetime_features': [],
            'mixed_type_features': []
        }

        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        categorical_cols = X.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            if X[col].nunique() > self.max_cardinality:
                self.feature_report['high_cardinality'].append(col)
        X = X.drop(columns=self.feature_report['high_cardinality'])

        for col in X.columns:
            try:
                pd.to_numeric(X[col])
            except ValueError:
                self.feature_report['mixed_type_features'].append(col)
        X = X.drop(columns=self.feature_report['mixed_type_features'])

        if pd.api.types.is_numeric_dtype(y):
            correlations = X.corrwith(y, method='pearson').abs()
        else:
            mi = mutual_info_classif(X, y, random_state=self.random_state)
            correlations = pd.Series(mi, index=X.columns)
        leaky_features = correlations[correlations > self.max_correlation].index.tolist()
        self.feature_report['leaky_features'] = leaky_features
        X = X.drop(columns=leaky_features)

        datetime_cols = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
        self.feature_report['datetime_features'] = datetime_cols

        return X

    def preprocess_data(self, X, y):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        X = self._feature_analysis(X, y)

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le

        if len(np.unique(y)) < 2:
            raise ValueError("Target variable must have at least two classes")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def create_preprocessing_pipeline(self, X):
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        st.session_state.feature_types = {
            'numeric': numeric_features,
            'categorical': categorical_features
        }

        return preprocessor

    def hyperparameter_tuning(self, X_train, y_train, model, param_distributions):
        preprocessor = self.create_preprocessing_pipeline(X_train)
        pipeline_steps = [('preprocessor', preprocessor)]

        if self.handle_imbalance:
            pipeline_steps.append(('smote', SMOTE(random_state=self.random_state)))

        pipeline_steps.append(('classifier', model))
        full_pipeline = Pipeline(pipeline_steps)

        random_search = RandomizedSearchCV(
            full_pipeline,
            param_distributions=param_distributions or self.DEFAULT_PARAMS.get(type(model), {}),
            n_iter=self.n_iter,
            cv=5,
            scoring='accuracy',
            random_state=self.random_state,
            n_jobs=-1,
            pre_dispatch='2*n_jobs'
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve': None
        }

        n_classes = len(np.unique(y_test))
        if y_proba is not None:
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                results['roc_curve'] = (fpr, tpr, roc_auc)
            else:
                results['roc_curve'] = self._multiclass_roc(y_test, y_proba, n_classes)

        results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        return results

    def _multiclass_roc(self, y_test, y_proba, n_classes):
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        return (fpr, tpr, roc_auc)

    def visualize_results(self):
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [res['accuracy'] for res in self.results.values()]
        })
        st.plotly_chart(px.bar(results_df, x='Model', y='Accuracy', title='Model Performance Comparison'))

        st.subheader(f"Best Model Performance: {self.best_model}")
        cm_data = self.results[self.best_model]['confusion_matrix']
        cm_array = np.array(cm_data).astype(int)
        classes = self._get_class_labels(cm_array.shape[0])
        self._render_confusion_matrix(cm_array, classes)

    def _get_class_labels(self, num_classes):
        try:
            return self.label_encoder.classes_.tolist()
        except AttributeError:
            return [str(i) for i in range(num_classes)]

    def _render_confusion_matrix(self, cm_array, classes):
        cm_df = pd.DataFrame(
            [[int(val) for val in row] for row in cm_array],
            index=[f"True {label}" for label in classes],
            columns=[f"Pred {label}" for label in classes]
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        st.pyplot(fig)

        if self.results[self.best_model]['roc_curve']:
            self._plot_roc_curve()

    def _plot_roc_curve(self):
        roc_data = self.results[self.best_model]['roc_curve']
        if isinstance(roc_data, tuple) and len(roc_data) == 3:
            if isinstance(roc_data[0], dict):
                fig = px.line(title='Multiclass ROC Curves')
                avg_auc = 0
                for i in roc_data[2]:
                    avg_auc += roc_data[2][i]
                    fig.add_scatter(
                        x=roc_data[0][i],
                        y=roc_data[1][i],
                        name=f'Class {i} (AUC = {roc_data[2][i]:.2f})',
                        mode='lines'
                    )
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True
                )
                avg_auc /= len(roc_data[2])
                st.plotly_chart(fig)
                st.markdown(f"**Average AUC:** {avg_auc:.2f}")
            else:
                fpr, tpr, roc_auc = roc_data
                fig = px.area(
                    x=fpr,
                    y=tpr,
                    title=f'ROC Curve (AUC = {roc_auc:.2f})',
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                )
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig)

    def show_interpretation(self, model):
        try:
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()

            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                df = df.sort_values('Importance', ascending=False).head(20)
                fig = px.bar(df, x='Importance', y='Feature',
                             title='Feature Importance', orientation='h')
                st.plotly_chart(fig)

            elif hasattr(classifier, 'coef_'):
                coefs = classifier.coef_[0]
                df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
                df['Absolute'] = np.abs(df['Coefficient'])
                df = df.sort_values('Absolute', ascending=False).head(20)
                fig = px.bar(df, x='Coefficient', y='Feature',
                             title='Feature Coefficients', orientation='h',
                             color='Coefficient', color_continuous_scale='RdBu')
                st.plotly_chart(fig)

            else:
                st.warning("Feature interpretation not available for this model type")

        except Exception as e:
            st.warning(f"Interpretation failed: {str(e)}")

    def fit(self, X, y):
        try:
            self._validate_data(X, y)
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

            for name, model in self.models:
                with st.spinner(f"Training {name}..."):
                    try:
                        tuned_model = self.hyperparameter_tuning(X_train, y_train, clone(model),
                                                                 self.DEFAULT_PARAMS.get(type(model), {}))
                        results = self.evaluate_model(tuned_model, X_test, y_test)
                        self.results[name] = results

                        if results['accuracy'] > self.best_score:
                            self.best_model = name
                            self.best_score = results['accuracy']
                            self.final_model = tuned_model
                    except Exception as e:
                        st.error(f"Error training {name}: {str(e)}")
            return self.final_model
        except ValueError as e:
            st.error(f"Data validation error: {str(e)}")
            raise

    def _validate_data(self, X, y):
        if y.isnull().any():
            raise ValueError("Target variable contains missing values")

        numeric_cols = X.select_dtypes(include=np.number).columns
        low_variance = X[numeric_cols].var() < 1e-6
        if low_variance.any():
            st.warning(f"Low variance features detected: {list(low_variance.index[low_variance])}")

    def save_model(self, output_path):
        model_metadata = {
            'feature_names': list(st.session_state.X.columns),
            'target_names': self.label_encoder.classes_ if self.label_encoder else None,
            'preprocessor': self.final_model.named_steps['preprocessor']
        }
        joblib.dump({'model': self.final_model, 'metadata': model_metadata}, output_path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)

DATASETS = {
    "Breast Cancer": load_breast_cancer,
    "Iris": load_iris,
    "Wine": load_wine,
    "Digits": load_digits,
    "Forest Covertypes": fetch_covtype
}

def load_dataset(dataset_name):
    loader = DATASETS[dataset_name]()

    if dataset_name == "Forest Covertypes":
        X = pd.DataFrame(loader.data[:5000], columns=loader.feature_names)
        y = pd.Series(loader.target[:5000] - 1)
    else:
        X = pd.DataFrame(loader.data, columns=loader.feature_names)
        y = pd.Series(loader.target)

    if hasattr(loader, 'target_names'):
        y = y.map(dict(enumerate(loader.target_names)))

    return X, y, loader.DESCR.split('\n')[0] if hasattr(loader, 'DESCR') else ""

def main():
    st.set_page_config(page_title="AutoML Classification", page_icon="🤖", layout="wide")

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

    st.title("AutoML Classification Web App")

    session_defaults = {
        'data_loaded': False,
        'dataset_description': "",
        'show_results': False
    }
    for key, val in session_defaults.items():
        st.session_state.setdefault(key, val)

    with st.sidebar:
        st.header("Configuration")
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
        random_state = st.number_input("Random State", 42)
        handle_imbalance = st.checkbox("Handle Class Imbalance", False)

        st.subheader("Feature Handling")
        max_cardinality = st.slider("Max Cardinality", 10, 1000, 50)
        max_correlation = st.slider("Max Correlation Threshold", 0.7, 1.0, 0.95)

        st.subheader("Advanced Options")
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        timeout = st.number_input("Max Training Time per Model (minutes)", 1, 60, 10)

        st.subheader("Class Imbalance Handling")
        imbalance_method = st.selectbox("Imbalance Technique",
                                        ["None", "SMOTE", "ADASYN", "UnderSampling"])

        st.subheader("Model Selection")
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

        st.subheader("Performance Settings")
        fast_mode = st.checkbox("Fast Mode", True)
        max_rows = st.number_input("Max Rows", 1000, 100000, 5000)
        n_iter = st.slider("HP Iterations", 5, 50, 10)

        st.markdown("---")
        st.subheader("Built-in Datasets")
        dataset_choice = st.selectbox("Choose dataset", list(DATASETS.keys()))
        if st.button(f"Load {dataset_choice} Dataset"):
            try:
                X, y, description = load_dataset(dataset_choice)
                st.session_state.update({
                    'X': X,
                    'y': y,
                    'data_loaded': True,
                    'dataset_description': description
                })
                st.success(f"{dataset_choice} dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")

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

    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Custom Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if len(df) > max_rows:
                    df = df.sample(max_rows, random_state=random_state)
                    st.warning(f"Using {max_rows} random rows")

                target_col = st.selectbox("Select target column", df.columns)
                if st.button("Load Custom Data"):
                    st.session_state.update({
                        'X': df.drop(columns=[target_col]),
                        'y': df[target_col],
                        'data_loaded': True,
                        'dataset_description': f"Custom dataset with {df.shape[1] - 1} features"
                    })
                    st.success("Custom data loaded!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    if st.session_state.data_loaded:
        X = st.session_state.X
        y = st.session_state.y

        with col2:
            st.header("Data Preview")
            st.markdown(f"**Dataset Description:** {st.session_state.dataset_description}")

            st.subheader("Features (X)")
            numeric_cols = X.select_dtypes(include=['float', 'int']).columns
            if numeric_cols.empty:
                st.dataframe(X.head())
            else:
                st.dataframe(X.head().style.format("{:.2f}", subset=numeric_cols))

            st.subheader("Target (y)")
            st.write("Class Distribution:")
            st.bar_chart(y.value_counts())
            st.dataframe(y.head().to_frame())

            if st.button("Run AutoML Analysis"):
                try:
                    with st.spinner("Training models..."):
                        if fast_mode:
                            for model in models:
                                if model[0] == 'Random Forest':
                                    model[1].set_params(n_estimators=50, max_depth=10)
                                elif model[0] == 'Gradient Boosting':
                                    model[1].set_params(n_estimators=50, learning_rate=0.1)

                        automl = AutoMLClassifier(
                            models=models,
                            test_size=test_size,
                            random_state=random_state,
                            n_iter=n_iter,
                            max_cardinality=max_cardinality,
                            max_correlation=max_correlation,
                            handle_imbalance=handle_imbalance
                        )
                        automl.fit(X, y)
                        st.session_state.update({
                            'results': automl,
                            'show_results': True
                        })
                        st.success("Analysis completed!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    if st.session_state.get('show_results') and st.session_state.results:
        automl = st.session_state.results

        st.subheader("Feature Analysis Report")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.write("**High Cardinality Features Removed**")
            st.write(automl.feature_report['high_cardinality'] or "None")
        with col_f2:
            st.write("**Potential Leaky Features Removed**")
            st.write(automl.feature_report['leaky_features'] or "None")
        with col_f3:
            st.write("**Date-like Features Detected**")
            st.write(automl.feature_report['datetime_features'] or "None")

        st.subheader("Best Model")
        st.markdown(f"""- **Model**: {automl.best_model} - **Accuracy**: {automl.best_score:.2%}""")

        st.subheader("Model Comparison")
        results_df = pd.DataFrame({
            'Model': automl.results.keys(),
            'Accuracy': [res['accuracy'] for res in automl.results.values()]
        }).sort_values('Accuracy', ascending=False)
        st.dataframe(results_df.style.format({'Accuracy': "{:.2%}"}))

        st.subheader("Detailed Report")
        selected_model = st.selectbox("Choose model to inspect", list(automl.results.keys()))
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("Classification Report:")
            st.json(automl.results[selected_model]['classification_report'])
        with col_d2:
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(
                automl.results[selected_model]['confusion_matrix'],
                index=[f"True {i}" for i in range(automl.results[selected_model]['confusion_matrix'].shape[0])],
                columns=[f"Pred {i}" for i in range(automl.results[selected_model]['confusion_matrix'].shape[1])]
            ))

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

        if 'roc_curve' in automl.results[automl.best_model]:
            st.subheader("ROC Curve for Best Model")
            roc_data = automl.results[automl.best_model]['roc_curve']
            if isinstance(roc_data[2], dict):
                fig = px.line(title='Multiclass ROC Curves')
                avg_auc = 0
                for i in roc_data[2]:
                    avg_auc += roc_data[2][i]
                    fig.add_scatter(
                        x=roc_data[0][i],
                        y=roc_data[1][i],
                        name=f'Class {i} (AUC = {roc_data[2][i]:.2f})',
                        mode='lines'
                    )
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True
                )
                avg_auc /= len(roc_data[2])
                st.plotly_chart(fig)
                st.markdown(f"**Average AUC:** {avg_auc:.2f}")
            else:
                fpr, tpr, roc_auc = roc_data
                fig = px.area(
                    x=fpr,
                    y=tpr,
                    title=f'ROC Curve (AUC = {roc_auc:.2f})',
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                )
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig)

        st.subheader("Model Persistence")
        col1, col2 = st.columns(2)
        with col1:
            buffer = BytesIO()
            automl.save_model(buffer)
            st.download_button(
                label="Download Model (.pkl)",
                data=buffer.getvalue(),
                file_name="automl_model.pkl",
                mime="application/octet-stream"
            )
        with col2:
            save_path = st.text_input("Save path:", "automl_model.pkl")
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

if __name__ == "__main__":
    main()