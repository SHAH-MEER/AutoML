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

# Import unsupervised specific modules
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score # A common metric for clustering evaluation

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
            # For unsupervised, y might be None or labels, so use mutual_info_classif if y is not None
            if y is not None:
                 mi = mutual_info_classif(X, y, random_state=self.random_state)
                 correlations = pd.Series(mi, index=X.columns)
            else:
                 # If no target, cannot calculate correlation with target, skip leaky feature detection
                 correlations = pd.Series(0, index=X.columns) # Assign 0 correlation if no target

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

        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
             transformers.append(('cat', categorical_transformer, categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        return preprocessor

    def _validate_data(self, X, y):
        if y is not None and y.isnull().any():
            # For unsupervised, allow None y, but check for nulls if y is provided
            raise ValueError("Target variable contains missing values")

        numeric_cols = X.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            low_variance = X[numeric_cols].var() < 1e-6
            if low_variance.any():
                print(f"Warning: Low variance features detected: {list(low_variance.index[low_variance])}")

    def save_model(self, output_path):
        # This method is primarily for supervised models. Unsupervised results might be different.
        # We can adjust this or create a separate save method for unsupervised results.
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
            raise ValueError("Target variable must have at least two classes for classification")

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

                    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
                    results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
                    }

                    self.results[name] = results

                    # For classification, best score is typically accuracy or F1
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
        # For regression, best score is typically R2, higher is better
        self.best_score = float('-inf')

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

                    # For regression, best score is R2
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

class AutoMLUnsupervised(BaseAutoML):
    def __init__(self, algorithms=None, max_cardinality=50,
                 max_correlation=0.95, random_state=42):
        # Unsupervised learning typically doesn't use test_size or the same kind of 'models'
        super().__init__(models=None, test_size=0, max_cardinality=max_cardinality, max_correlation=max_correlation, random_state=random_state)
        self.algorithms = algorithms or [
            # Default unsupervised algorithms - we can add parameters later on the Streamlit page
            ('KMeans', KMeans(random_state=random_state, n_init=10)),
            ('DBSCAN', DBSCAN()),
            ('PCA', PCA(random_state=random_state)),
            ('TSNE', TSNE(random_state=random_state, n_components=2, learning_rate='auto', init='random'))
        ]
        # For unsupervised, we store results per algorithm run
        self.results = {}
        # There isn't a single 'best_model' in the same way as supervised learning
        self.best_model = None # or maybe track algorithm with best silhouette score if clustering
        self.best_score = None
        self.final_model = None # Could store a pipeline if needed

    def preprocess_data(self, X, y=None):
        # Unsupervised preprocessing might be simpler, often just scaling
        X = X.copy()
        # Feature analysis can still be useful
        X = self._feature_analysis(X, y) # Pass y so leaky feature analysis can be skipped if y is None

        # Create preprocessing pipeline - similar to supervised but without target handling
        preprocessor = self.create_preprocessing_pipeline(X)

        return preprocessor.fit_transform(X)

    def run_algorithms(self, X, y=None):
        # X here is the preprocessed data
        self.results = {}
        for name, algorithm in self.algorithms:
            try:
                print(f"Running {name}...")
                # Apply the algorithm
                if name == 'KMeans':
                    # KMeans expects n_clusters, let's add a placeholder or default
                    # The actual parameter control will be on the Streamlit page
                    algorithm.n_clusters = getattr(self, 'kmeans_n_clusters', 8) # Example default
                    labels = algorithm.fit_predict(X)
                    score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
                    self.results[name] = {'labels': labels, 'silhouette_score': score}
                elif name == 'DBSCAN':
                     # DBSCAN parameters need to be exposed on the Streamlit page
                     # Example defaults:
                     algorithm.eps = getattr(self, 'dbscan_eps', 0.5)
                     algorithm.min_samples = getattr(self, 'dbscan_min_samples', 5)
                     labels = algorithm.fit_predict(X)
                     # Silhouette score for DBSCAN can be tricky with noise points (-1)
                     core_samples_mask = np.zeros_like(labels, dtype=bool)
                     if hasattr(algorithm, 'core_sample_indices_'):
                        core_samples_mask[algorithm.core_sample_indices_] = True
                     unique_labels = set(labels)
                     if -1 in unique_labels: # Don't include noise points in silhouette calculation
                         unique_labels.remove(-1)

                     if len(unique_labels) > 1:
                          score = silhouette_score(X[labels != -1], labels[labels != -1])
                     else:
                          score = None
                     self.results[name] = {'labels': labels, 'silhouette_score': score}

                elif name == 'PCA':
                    # PCA expects n_components, exposed on the Streamlit page
                    algorithm.n_components = getattr(self, 'pca_n_components', min(X.shape[0], X.shape[1]))
                    transformed_data = algorithm.fit_transform(X)
                    explained_variance = algorithm.explained_variance_ratio_
                    self.results[name] = {'transformed_data': transformed_data, 'explained_variance_ratio': explained_variance}
                elif name == 'TSNE':
                     # t-SNE is primarily for visualization, typically 2 or 3 components
                     # Parameters like n_components and perplexity on the Streamlit page
                     # t-SNE does not have a fit_predict method for clustering labels directly
                     # It learns a low-dimensional representation
                     algorithm.n_components = getattr(self, 'tsne_n_components', 2) # Default to 2 for easy visualization
                     algorithm.perplexity = getattr(self, 'tsne_perplexity', 30)
                     algorithm.init = getattr(self, 'tsne_init', 'random')

                     # TSNE does not handle missing values or non-finite values
                     # The preprocessor should handle this, but double check

                     transformed_data = algorithm.fit_transform(X)
                     self.results[name] = {'transformed_data': transformed_data}

                # Add other unsupervised algorithms here

            except Exception as e:
                print(f"Error running {name}: {str(e)}")
                self.results[name] = {'error': str(e)}

        return self.results

    def run(self, X, y=None):
        # This is the main method to call from the Streamlit page
        try:
            self._validate_data(X, y)
            X_processed = self.preprocess_data(X, y)
            unsupervised_results = self.run_algorithms(X_processed, y)
            return unsupervised_results

        except ValueError as e:
            print(f"Data validation error: {str(e)}")
            raise 