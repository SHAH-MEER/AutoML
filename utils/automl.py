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
        # Initialize best_score based on task type default or to -inf for maximization
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

        if y is not None and pd.api.types.is_numeric_dtype(y):
            correlations = X.corrwith(y, method='pearson').abs()
        elif y is not None:
             # For classification target, use mutual info
             mi = mutual_info_classif(X, y, random_state=self.random_state)
             correlations = pd.Series(mi, index=X.columns)
        else:
             # If no target (unsupervised), cannot calculate correlation with target
             correlations = pd.Series(0, index=X.columns) # Assign 0 correlation if no target


        leaky_features = correlations[correlations > self.max_correlation].index.tolist()
        # For unsupervised tasks where y is None, this list will be empty, which is desired.
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

        # Handle case where there are no features left after analysis
        if not transformers:
             # Create a dummy preprocessor that does nothing, or handle this case upstream
             # For now, let's return None or raise an error if no features are left.
             # Let's return a minimal transformer that just passes through if no features require processing
            return ColumnTransformer(transformers=[]) # Empty transformer passes through

        preprocessor = ColumnTransformer(transformers=transformers)

        return preprocessor

    def _validate_data(self, X, y):
        if y is not None and y.isnull().any():
            # For unsupervised, allow None y, but check for nulls if y is provided
            raise ValueError("Target variable contains missing values")

        # Check for empty dataframe after feature analysis
        # Note: Feature analysis happens in preprocess_data, not here
        # But we can add a check on X after loading/before processing if needed.

        numeric_cols = X.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            low_variance = X[numeric_cols].var() < 1e-6
            if low_variance.any():
                print(f"Warning: Low variance features detected: {list(low_variance.index[low_variance])}")

    def save_model(self, output_path):
        # This method is primarily for supervised models.
        if self.final_model:
            if isinstance(output_path, BytesIO):
                joblib.dump(self.final_model, output_path)
            else:
                joblib.dump(self.final_model, output_path)
        else:
            print("No model trained or loaded to save.")

    @staticmethod
    def load_model(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

class AutoMLClassifier(BaseAutoML):
    def __init__(self, models=None, test_size=0.2, max_cardinality=50,
                 max_correlation=0.95, random_state=42, handle_imbalance=False):
        super().__init__(models, test_size, max_cardinality, max_correlation, random_state)
        self.handle_imbalance = handle_imbalance
        self.label_encoder = None
        # Classification best score is accuracy or F1, initialize to -inf
        self.best_score = float('-inf')

    def preprocess_data(self, X, y):
        X = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        X = self._feature_analysis(X, y) # Pass y for correlation analysis

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoder = le

        if len(np.unique(y)) < 2:
            raise ValueError("Target variable must have at least two classes for classification")

        # Check if there are any features left after analysis before splitting
        if X.empty:
             raise ValueError("No features remaining after preprocessing steps.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        self.results = {} # Reset results for a new fit
        self.best_model = None
        self.best_score = float('-inf')
        self.final_model = None

        try:
            self._validate_data(X, y) # Validate original data before preprocessing
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

            # Check if there are features left in X_train after preprocessing
            if X_train.shape[1] == 0:
                 raise ValueError("No features remaining after preprocessing for training.")

            for name, model in self.models:
                try:
                    preprocessor = self.create_preprocessing_pipeline(X_train) # Create pipeline on training data
                    pipeline_steps = [('preprocessor', preprocessor)]

                    if self.handle_imbalance:
                        from imblearn.over_sampling import SMOTE
                        # Ensure SMOTE is only applied to training data
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
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(), # Convert to list for easier handling in Streamlit
                        'y_test': y_test.tolist(), # Convert series to list
                        'y_pred': y_pred.tolist(), # Convert array to list
                        'y_proba': y_proba.tolist() if y_proba is not None else None, # Convert array to list
                        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
                    }

                    self.results[name] = results

                    # For classification, best score is typically accuracy or F1. Let's use accuracy for now.
                    if results['accuracy'] > self.best_score:
                        self.best_model = name
                        self.best_score = results['accuracy']
                        self.final_model = full_pipeline

                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    self.results[name] = {'error': str(e)}

            # After trying all models, check if a best model was found
            if self.best_model is None:
                 raise RuntimeError("None of the selected models could be trained successfully.")

            return self.final_model

        except ValueError as e:
            print(f"Data validation error: {str(e)}")
            raise
        except RuntimeError as e:
             print(f"Analysis error: {str(e)}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred during fit: {str(e)}")
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

        X = self._feature_analysis(X, y) # Pass y for correlation analysis

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target variable must be numeric for regression")

        # Check if there are any features left after analysis before splitting
        if X.empty:
             raise ValueError("No features remaining after preprocessing steps.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        self.results = {} # Reset results for a new fit
        self.best_model = None
        self.best_score = float('-inf')
        self.final_model = None

        try:
            self._validate_data(X, y) # Validate original data before preprocessing
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

            # Check if there are features left in X_train after preprocessing
            if X_train.shape[1] == 0:
                 raise ValueError("No features remaining after preprocessing for training.")

            for name, model in self.models:
                try:
                    preprocessor = self.create_preprocessing_pipeline(X_train) # Create pipeline on training data
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
                        'y_test': y_test.tolist(), # Convert series to list
                        'y_pred': y_pred.tolist(), # Convert array to list
                        'residuals': (y_test - y_pred).tolist() # Convert array to list
                    }

                    self.results[name] = results

                    # For regression, best score is R2
                    if results['r2_score'] > self.best_score:
                        self.best_model = name
                        self.best_score = results['r2_score']
                        self.final_model = full_pipeline

                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    self.results[name] = {'error': str(e)}

            # After trying all models, check if a best model was found
            if self.best_model is None:
                 raise RuntimeError("None of the selected models could be trained successfully.")

            return self.final_model

        except ValueError as e:
            print(f"Data validation error: {str(e)}")
            raise
        except RuntimeError as e:
             print(f"Analysis error: {str(e)}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred during fit: {str(e)}")
            raise

class AutoMLUnsupervised(BaseAutoML):
    def __init__(self, algorithms, random_state=42, **kwargs):
        super().__init__(random_state=random_state)
        self.algorithms = algorithms
        self.kwargs = kwargs
        self.results = {}

    def _tune_kmeans(self, X):
        """Find optimal number of clusters for KMeans using silhouette score."""
        best_score = -1
        best_n_clusters = 2
        tuning_results = []

        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            score = silhouette_score(X, labels)
            tuning_results.append({
                'param': n_clusters,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

        return best_n_clusters, tuning_results

    def _tune_dbscan(self, X):
        """Find optimal eps and min_samples for DBSCAN using silhouette score."""
        best_score = -1
        best_params = {'eps': 0.5, 'min_samples': 5}
        tuning_results = []

        eps_range = self.kwargs.get('dbscan_eps_range', np.linspace(0.1, 2.0, 10))
        min_samples_range = self.kwargs.get('dbscan_min_samples_range', range(2, 11))

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                # Skip if all points are noise or only one cluster
                if len(np.unique(labels)) < 2 or -1 in labels:
                    continue
                
                # Calculate silhouette score
                score = silhouette_score(X, labels)
                tuning_results.append({
                    'param': f'eps={eps:.2f}, min_samples={min_samples}',
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}

        return best_params, tuning_results

    def _tune_pca(self, X):
        """Find optimal number of components for PCA to explain 95% variance."""
        pca = PCA()
        pca.fit(X)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        tuning_results = [{
            'param': i + 1,
            'score': cumulative_variance[i]
        } for i in range(len(cumulative_variance))]
        
        return n_components, tuning_results

    def _tune_tsne(self, X):
        """Find optimal perplexity for t-SNE using reconstruction error."""
        best_score = float('inf')
        best_perplexity = 30
        tuning_results = []
        
        perplexity_range = self.kwargs.get('tsne_perplexity_range', range(5, 51, 5))
        n_components = self.kwargs.get('tsne_n_components', 2)
        
        for perplexity in perplexity_range:
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=self.random_state)
            try:
                X_embedded = tsne.fit_transform(X)
                # Calculate reconstruction error (MSE between original and reconstructed distances)
                score = np.mean((X - tsne.inverse_transform(X_embedded)) ** 2)
                
                tuning_results.append({
                    'param': perplexity,
                    'score': score
                })
                
                if score < best_score:
                    best_score = score
                    best_perplexity = perplexity
            except:
                continue
        
        return best_perplexity, tuning_results

    def preprocess_data(self, X, y=None):
        # Unsupervised preprocessing might be simpler, often just scaling
        X = X.copy()
        # Feature analysis can still be useful, but leaky feature detection needs y
        X = self._feature_analysis(X, y) # Pass y so leaky feature analysis can be skipped if y is None

        # Check if there are any features left after analysis
        if X.empty:
             raise ValueError("No features remaining after preprocessing steps.")

        # Create preprocessing pipeline - similar to supervised but without target handling
        # Fit and transform the data here
        preprocessor = self.create_preprocessing_pipeline(X)
        X_processed = preprocessor.fit_transform(X)

        # Store the preprocessor if needed for later (e.g., saving/loading)
        self.preprocessor = preprocessor

        return X_processed

    def run(self, X, y=None):
        """Run unsupervised learning algorithms with optional hyperparameter tuning."""
        X = self.preprocess_data(X)
        
        for name, algorithm in self.algorithms:
            try:
                if name == 'KMeans':
                    if self.kwargs.get('kmeans_n_clusters') == 'auto':
                        n_clusters, tuning_results = self._tune_kmeans(X)
                        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                        labels = kmeans.fit_predict(X)
                        self.results[name] = {
                            'labels': labels,
                            'silhouette_score': silhouette_score(X, labels),
                            'tuning_results': tuning_results
                        }
                    else:
                        kmeans = KMeans(n_clusters=self.kwargs.get('kmeans_n_clusters', 8), 
                                      random_state=self.random_state)
                        labels = kmeans.fit_predict(X)
                        self.results[name] = {
                            'labels': labels,
                            'silhouette_score': silhouette_score(X, labels)
                        }
                
                elif name == 'DBSCAN':
                    if self.kwargs.get('dbscan_auto_tune'):
                        best_params, tuning_results = self._tune_dbscan(X)
                        dbscan = DBSCAN(**best_params)
                        labels = dbscan.fit_predict(X)
                        self.results[name] = {
                            'labels': labels,
                            'silhouette_score': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None,
                            'tuning_results': tuning_results
                        }
                    else:
                        dbscan = DBSCAN(
                            eps=self.kwargs.get('dbscan_eps', 0.5),
                            min_samples=self.kwargs.get('dbscan_min_samples', 5)
                        )
                        labels = dbscan.fit_predict(X)
                        self.results[name] = {
                            'labels': labels,
                            'silhouette_score': silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
                        }
                
                elif name == 'PCA':
                    if self.kwargs.get('pca_n_components') == 'auto':
                        n_components, tuning_results = self._tune_pca(X)
                        pca = PCA(n_components=n_components)
                    else:
                        pca = PCA(n_components=self.kwargs.get('pca_n_components'))
                    
                    transformed_data = pca.fit_transform(X)
                    self.results[name] = {
                        'transformed_data': transformed_data,
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                    }
                    if self.kwargs.get('pca_n_components') == 'auto':
                        self.results[name]['tuning_results'] = tuning_results
                
                elif name == 'TSNE':
                    if self.kwargs.get('tsne_auto_tune'):
                        best_perplexity, tuning_results = self._tune_tsne(X)
                        tsne = TSNE(
                            n_components=self.kwargs.get('tsne_n_components', 2),
                            perplexity=best_perplexity,
                            random_state=self.random_state
                        )
                        transformed_data = tsne.fit_transform(X)
                        self.results[name] = {
                            'transformed_data': transformed_data,
                            'tuning_results': tuning_results
                        }
                    else:
                        tsne = TSNE(
                            n_components=self.kwargs.get('tsne_n_components', 2),
                            perplexity=self.kwargs.get('tsne_perplexity', 30),
                            random_state=self.random_state
                        )
                        transformed_data = tsne.fit_transform(X)
                        self.results[name] = {
                            'transformed_data': transformed_data
                        }
            
            except Exception as e:
                self.results[name] = {'error': str(e)}
        
        return self.results 