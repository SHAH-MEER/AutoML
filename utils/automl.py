import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import joblib
from io import BytesIO

class BaseAutoML:
    def __init__(self, models=None, test_size=0.2, max_cardinality=50,
                 max_correlation=0.95, random_state=42):
        self.random_state = random_state
        self.test_size = test_size
        self.models = models
        self.best_model = None
        self.best_score = float('-inf')
        self.results = {}
        self.final_model = None
        self.max_cardinality = max_cardinality
        self.max_correlation = max_correlation
        self.feature_report = {
            'high_cardinality': [],
            'leaky_features': [],
            'datetime_features': [],
            'mixed_type_features': []
        }

    def _feature_analysis(self, X, y):
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

        return preprocessor

    def _validate_data(self, X, y):
        if y.isnull().any():
            raise ValueError("Target variable contains missing values")

        numeric_cols = X.select_dtypes(include=np.number).columns
        low_variance = X[numeric_cols].var() < 1e-6
        if low_variance.any():
            print(f"Warning: Low variance features detected: {list(low_variance.index[low_variance])}")

    def save_model(self, output_path):
        if isinstance(output_path, BytesIO):
            joblib.dump(self.final_model, output_path)
        else:
            joblib.dump(self.final_model, output_path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)

class AutoMLClassifier(BaseAutoML):
    def __init__(self, models=None, test_size=0.2, max_cardinality=50,
                 max_correlation=0.95, random_state=42, handle_imbalance=False):
        super().__init__(models, test_size, max_cardinality, max_correlation, random_state)
        self.handle_imbalance = handle_imbalance
        self.label_encoder = None

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

    def fit(self, X, y):
        try:
            self._validate_data(X, y)
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

            for name, model in self.models:
                try:
                    preprocessor = self.create_preprocessing_pipeline(X_train)
                    pipeline_steps = [('preprocessor', preprocessor)]

                    if self.handle_imbalance:
                        from imblearn.over_sampling import SMOTE
                        pipeline_steps.append(('smote', SMOTE(random_state=self.random_state)))

                    pipeline_steps.append(('classifier', model))
                    full_pipeline = Pipeline(pipeline_steps)

                    full_pipeline.fit(X_train, y_train)
                    y_pred = full_pipeline.predict(X_test)
                    y_proba = full_pipeline.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                    results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_proba': y_proba
                    }

                    self.results[name] = results

                    if results['accuracy'] > self.best_score:
                        self.best_model = name
                        self.best_score = results['accuracy']
                        self.final_model = full_pipeline

                except Exception as e:
                    print(f"Error training {name}: {str(e)}")

            return self.final_model

        except ValueError as e:
            print(f"Data validation error: {str(e)}")
            raise

class AutoMLRegressor(BaseAutoML):
    def __init__(self, models=None, test_size=0.2, max_cardinality=50,
                 max_correlation=0.95, random_state=42):
        super().__init__(models, test_size, max_cardinality, max_correlation, random_state)

    def preprocess_data(self, X, y):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        X = self._feature_analysis(X, y)

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target variable must be numeric for regression")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        try:
            self._validate_data(X, y)
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

            for name, model in self.models:
                try:
                    preprocessor = self.create_preprocessing_pipeline(X_train)
                    pipeline_steps = [
                        ('preprocessor', preprocessor),
                        ('regressor', model)
                    ]
                    full_pipeline = Pipeline(pipeline_steps)

                    full_pipeline.fit(X_train, y_train)
                    y_pred = full_pipeline.predict(X_test)

                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    results = {
                        'r2_score': r2_score(y_test, y_pred),
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'residuals': y_test - y_pred
                    }

                    self.results[name] = results

                    if results['r2_score'] > self.best_score:
                        self.best_model = name
                        self.best_score = results['r2_score']
                        self.final_model = full_pipeline

                except Exception as e:
                    print(f"Error training {name}: {str(e)}")

            return self.final_model

        except ValueError as e:
            print(f"Data validation error: {str(e)}")
            raise 