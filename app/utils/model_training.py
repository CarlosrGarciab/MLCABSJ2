"""
Módulo para entrenamiento de modelos de machine learning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import traceback
from flask import current_app
from .data_validation import convert_numpy_types
from .training_utils import clean_column_names

def get_available_models():
    """Retorna los modelos disponibles para entrenamiento"""
    return {
        'decision_tree': {
            'name': 'Árbol de Decisión',
            'description': 'Modelo interpretable que toma decisiones usando reglas simples',
            'params': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Conjunto de árboles de decisión para mayor precisión',
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'logistic_regression': {
            'name': 'Regresión Logística',
            'description': 'Modelo lineal rápido y eficiente para clasificación',
            'params': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Modelo potente para separar clases complejas',
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'description': 'Clasifica basándose en los vecinos más cercanos',
            'params': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        },
        'naive_bayes': {
            'name': 'Naive Bayes',
            'description': 'Modelo probabilístico rápido y eficiente para clasificación',
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
    }

def train_model(X, y, model_type, test_size=0.2, random_state=42, **params):
    """Entrena un modelo de clasificación"""
    try:
        # Validaciones iniciales de tamaño del dataset
        if len(X) == 0:
            raise ValueError("El dataset está vacío (0 muestras)")
        
        if len(X) < 5:
            raise ValueError(f"Dataset muy pequeño ({len(X)} muestras). Se necesitan al menos 5 muestras para entrenamiento")
        
        # Asegurar que y sea una pandas Series para usar value_counts()
        if not isinstance(y, pd.Series):
            y = pd.Series(y) if hasattr(y, '__iter__') else pd.Series([y])
        
        # Para datasets muy pequeños, usar un test_size menor o fijo
        if len(X) < 10:
            test_size = 1  # Solo una muestra para test
            current_app.logger.warning(f"Dataset pequeño ({len(X)} muestras), usando test_size=1")
        elif len(X) < 25:
            test_size = max(1, int(len(X) * 0.1))  # 10% o mínimo 1
            current_app.logger.warning(f"Dataset pequeño ({len(X)} muestras), usando test_size={test_size}")
        
        # Verificar que cada clase tenga al menos 2 muestras para stratify
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        original_classes = y.nunique()
        original_samples = len(y)
        classes_removed = []
        stratify_used = True
        
        # Si alguna clase tiene menos de 2 muestras, no usar stratify
        if min_class_count < 2:
            current_app.logger.warning(f"Clase con solo {min_class_count} muestra(s) detectada. No se usará stratify.")
            current_app.logger.info(f"Distribución de clases: {dict(class_counts)}")
            
            # Opción 1: Remover clases con muy pocas muestras
            classes_to_remove = class_counts[class_counts < 2].index.tolist()
            if len(classes_to_remove) > 0:
                current_app.logger.info(f"Removiendo clases con pocas muestras: {classes_to_remove}")
                classes_removed = classes_to_remove
                mask = ~y.isin(classes_to_remove)
                
                # Filtrar X y y usando la máscara
                if isinstance(X, pd.DataFrame):
                    X = X[mask]
                else:
                    X = X[mask.values] if hasattr(mask, 'values') else X[mask]
                
                y = y[mask]
                # Mantener y como pandas Series después del filtrado
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)
                current_app.logger.info(f"Datos después de filtrar: {len(X)} filas, {y.nunique()} clases")
                
                # Verificar si después del filtrado podemos usar stratify
                remaining_class_counts = y.value_counts()
                min_remaining_count = remaining_class_counts.min()
                
                if min_remaining_count >= 2:
                    # Ahora sí podemos usar stratify
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    stratify_used = True
                else:
                    # Aún no podemos usar stratify
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    stratify_used = False
            else:
                # No usar stratify si no se pueden remover clases
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                stratify_used = False
        else:
            # Caso normal: todas las clases tienen al menos 2 muestras
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            stratify_used = True
        
        # Crear el modelo según el tipo
        if model_type == 'decision_tree':
            model = DecisionTreeClassifier(random_state=random_state, **params)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=random_state, **params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=random_state, max_iter=1000, **params)
        elif model_type == 'svm':
            model = SVC(random_state=random_state, **params)
        elif model_type == 'knn':
            model = KNeighborsClassifier(**params)
        elif model_type == 'naive_bayes':
            model = GaussianNB(**params)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Métricas adicionales detalladas
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Calcular métricas macro y micro para multiclase
        test_precision_macro = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_precision_micro = precision_score(y_test, y_pred_test, average='micro', zero_division=0)
        test_recall_macro = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_recall_micro = recall_score(y_test, y_pred_test, average='micro', zero_division=0)
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_f1_micro = f1_score(y_test, y_pred_test, average='micro', zero_division=0)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        
        # Resultados del entrenamiento
        training_results = {
            'model_type': model_type,
            'model_params': params,
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'test_precision_macro': float(test_precision_macro),
            'test_precision_micro': float(test_precision_micro),
            'test_recall_macro': float(test_recall_macro),
            'test_recall_micro': float(test_recall_micro),
            'test_f1_macro': float(test_f1_macro),
            'test_f1_micro': float(test_f1_micro),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'classification_report': convert_numpy_types(class_report),
            'confusion_matrix': convert_numpy_types(conf_matrix.tolist()),
            'feature_names': X.columns.tolist(),
            'n_classes': len(np.unique(y)),
            'original_samples': original_samples,
            'original_classes': original_classes,
            'classes_removed': classes_removed,
            'stratify_used': stratify_used
        }
        
        # Importancia de características (si está disponible)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            training_results['feature_importance'] = convert_numpy_types(feature_importance)
        
        return model, training_results
        
    except Exception as e:
        raise Exception(f"Error al entrenar modelo: {str(e)}")

def save_model(model, encoders, target_encoder, model_info, save_path):
    """Guarda el modelo entrenado y sus encoders"""
    try:
        model_data = {
            'model': model,
            'encoders': encoders,
            'target_encoder': target_encoder,
            'model_info': model_info
        }
        
        joblib.dump(model_data, save_path)
        return True
        
    except Exception as e:
        raise Exception(f"Error al guardar modelo: {str(e)}")

def load_model(model_path):
    """Carga un modelo guardado"""
    try:
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['encoders'], model_data['target_encoder'], model_data['model_info']
        
    except Exception as e:
        raise Exception(f"Error al cargar modelo: {str(e)}")

def get_available_regression_models():
    """Retorna los modelos de regresión disponibles para entrenamiento"""
    return {
        'linear_regression': {
            'name': 'Regresión Lineal',
            'description': 'Modelo lineal simple y rápido para predicción de valores continuos',
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'ridge_regression': {
            'name': 'Ridge Regression',
            'description': 'Regresión lineal con regularización L2 para evitar sobreajuste',
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False]
            }
        },
        'lasso_regression': {
            'name': 'Lasso Regression',
            'description': 'Regresión lineal con regularización L1 para selección de características',
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False]
            }
        },
        'elastic_net': {
            'name': 'ElasticNet',
            'description': 'Combina regularización L1 y L2 para balance entre Ridge y Lasso',
            'params': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9],
                'fit_intercept': [True, False]
            }
        },
        'decision_tree_regressor': {
            'name': 'Árbol de Decisión (Regresión)',
            'description': 'Modelo interpretable para regresión usando reglas de decisión',
            'params': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        },
        'random_forest_regressor': {
            'name': 'Random Forest (Regresión)',
            'description': 'Conjunto de árboles de decisión para regresión con mayor precisión',
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'svr': {
            'name': 'Support Vector Regression',
            'description': 'Regresión con vectores de soporte para relaciones no lineales',
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'knn_regressor': {
            'name': 'K-Nearest Neighbors (Regresión)',
            'description': 'Predice basándose en el promedio de los vecinos más cercanos',
            'params': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        }
    }

def prepare_data_for_training(df, target_column, predictor_columns, task_type='classification'):
    """Prepara los datos para entrenamiento (clasificación o regresión)
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna target
        predictor_columns: Lista de nombres de columnas predictoras
        task_type: 'classification' o 'regression'
    
    Returns:
        X_encoded, y_encoded, encoders, target_encoder, preparation_info
    """
    try:
        # Limpiar nombres de columnas usando la función estándar
        df.columns = clean_column_names(df.columns.tolist())
        
        # También limpiar y normalizar las columnas predictoras y target que vienen como parámetros
        if isinstance(target_column, str):
            target_column = clean_column_names([target_column])[0]
        
        predictor_columns = clean_column_names(predictor_columns)

        # Validar que las columnas seleccionadas existen en el DataFrame
        missing_columns = [col for col in predictor_columns + [target_column] if col not in df.columns]
        if missing_columns:
            current_app.logger.error(f"Missing columns in DataFrame: {missing_columns}")
            raise Exception(f"Las siguientes columnas seleccionadas no existen en el archivo CSV: {missing_columns}. Por favor, selecciona nuevamente las columnas.")

        # Seleccionar solo las columnas necesarias
        data = df[predictor_columns + [target_column]].copy()
        
        # Separar características y target
        X = data[predictor_columns]
        y = data[target_column]
        
        # Información inicial sobre los datos
        initial_rows = len(data)
        nan_counts_before = int(X.isnull().sum().sum() + y.isnull().sum())

        # Verificar que la variable target es numérica para regresión
        if task_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
            # Intentar convertir a numérico
            try:
                y = pd.to_numeric(y, errors='coerce')
            except (ValueError, TypeError) as e:
                raise ValueError(f"La variable target '{target_column}' no es numérica y no se puede convertir para regresión: {str(e)}")
        
        # MANEJAR VALORES FALTANTES EN LA VARIABLE TARGET
        if y.isnull().any():
            # Eliminar filas donde target es NaN (no se puede imputar el target)
            mask_target_not_null = y.notnull()
            X = X[mask_target_not_null]
            y = y[mask_target_not_null]
            current_app.logger.info(f"Eliminadas {(~mask_target_not_null).sum()} filas con target faltante")
        
        # MANEJAR VALORES FALTANTES EN LAS CARACTERÍSTICAS
        # Identificar columnas numéricas y categóricas
        numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Imputar valores faltantes en columnas numéricas
        if numeric_columns and len(numeric_columns) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X_numeric = X[numeric_columns].copy()
            X_numeric_imputed = pd.DataFrame(
                numeric_imputer.fit_transform(X_numeric),
                columns=numeric_columns,
                index=X.index
            )
            X[numeric_columns] = X_numeric_imputed
        
        # Manejar valores faltantes en columnas categóricas
        if categorical_columns and len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent', fill_value='Unknown')
            X_categorical = X[categorical_columns].copy()
            X_categorical_imputed = pd.DataFrame(
                categorical_imputer.fit_transform(X_categorical),
                columns=categorical_columns,
                index=X.index
            )
            X[categorical_columns] = X_categorical_imputed

        # Información sobre la preparación
        final_rows = len(X)
        nan_counts_after = int(X.isnull().sum().sum() + y.isnull().sum())
        
        # Calcular distribución del target según el tipo de tarea
        if task_type == 'regression':
            # Para regresión, crear bins para distribución tipo histograma
            n_bins = min(20, int(y.nunique() / 2) + 1) if y.nunique() > 10 else y.nunique()
            if n_bins > 1:
                counts, bins = np.histogram(y.dropna(), bins=n_bins)
                # Crear etiquetas de bins más legibles
                bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
                target_distribution = {label: int(count) for label, count in zip(bin_labels, counts)}
                total_samples = sum(counts)
                target_distribution_percentages = {label: round((count/total_samples)*100, 2) 
                                                 for label, count in target_distribution.items()}
            else:
                target_distribution = {}
                target_distribution_percentages = {}
        else:
            # Para clasificación, usar distribución de clases
            target_distribution = y.value_counts().to_dict()
            target_distribution_percentages = y.value_counts(normalize=True).mul(100).round(2).to_dict()
        
        preparation_info = {
            'original_rows': initial_rows,
            'rows_after_cleaning': final_rows,
            'dropped_rows': initial_rows - final_rows,
            'features_count': len(predictor_columns),
            'target_unique_values': int(y.nunique()),
            'target_distribution': target_distribution,
            'target_distribution_percentages': target_distribution_percentages,
            'feature_types': {},
            'nan_values_before': nan_counts_before,
            'nan_values_after': nan_counts_after,
            'imputation_applied': bool(nan_counts_before > 0)
        }
        
        # Para regresión, agregar estadísticas adicionales
        if task_type == 'regression':
            preparation_info.update({
                'target_min': float(y.min()),
                'target_max': float(y.max()),
                'target_mean': float(y.mean()),
                'target_std': float(y.std())
            })

        # Codificar variables categóricas si es necesario
        encoders = {}
        X_encoded = X.copy()
        
        for col in X.columns:
            try:
                if X[col].dtype == 'object':  # Variable categórica
                    current_app.logger.info(f"Encoding categorical column: {col}")
                    le = LabelEncoder()
                    # Asegurar que no hay NaN después de la imputación
                    col_values = X[col].astype(str)
                    current_app.logger.info(f"Column {col} unique values before encoding: {col_values.nunique()}")
                    X_encoded[col] = le.fit_transform(col_values)
                    encoders[col] = le
                    preparation_info['feature_types'][col] = f'categorical (encoded: {len(le.classes_)} categories)'
                    current_app.logger.info(f"Successfully encoded column {col}")
                else:
                    # Para columnas numéricas, verificar que realmente son numéricas
                    current_app.logger.info(f"Processing numeric column: {col}, dtype: {X[col].dtype}")
                    # Intentar convertir a float si no es claramente numérico
                    if X[col].dtype == 'object':
                        try:
                            X_encoded[col] = pd.to_numeric(X[col], errors='raise')
                        except (ValueError, TypeError):
                            # Si falla, tratarlo como categórico
                            current_app.logger.warning(f"Column {col} appeared numeric but has non-numeric values, treating as categorical")
                            le = LabelEncoder()
                            X_encoded[col] = le.fit_transform(X[col].astype(str))
                            encoders[col] = le
                            preparation_info['feature_types'][col] = f'categorical (encoded: {len(le.classes_)} categories)'
                    else:
                        preparation_info['feature_types'][col] = f'numeric ({X[col].dtype})'
            except Exception as e:
                current_app.logger.error(f"Error processing column {col}: {e}")
                current_app.logger.error(f"Column {col} dtype: {X[col].dtype}, sample values: {X[col].head().tolist()}")
                raise Exception(f"Error procesando columna {col}: {str(e)}")
        
        # Codificar variable target según el tipo de tarea
        target_encoder = None
        y_encoded = y.copy()
        
        if task_type == 'classification':
            # Para clasificación, siempre codificar como categórica
            target_encoder = LabelEncoder()
            y_encoded_array = target_encoder.fit_transform(y.astype(str))
            # Mantener como pandas Series con el mismo índice
            y_encoded = pd.Series(y_encoded_array, index=y.index)
            preparation_info['target_type'] = f'categorical (encoded: {len(target_encoder.classes_)} classes)'
            preparation_info['target_classes'] = target_encoder.classes_.tolist()
        else:  # regression
            preparation_info['target_type'] = f'numeric ({y.dtype})'
        
        # Verificación final: asegurar que no hay NaN
        y_has_nan = y_encoded.isnull().any() if hasattr(y_encoded, 'isnull') else pd.isnull(y_encoded).any()
        if X_encoded.isnull().any().any() or y_has_nan:
            raise Exception("Aún hay valores NaN después de la limpieza. Revisa los datos.")
        
        return X_encoded, y_encoded, encoders, target_encoder, preparation_info
        
    except Exception as e:
        # Logging más detallado del error
        current_app.logger.error(f"Error en prepare_data_for_training: {str(e)}")
        current_app.logger.error(f"Error type: {type(e)}")
        
        # Log información sobre los datos problemáticos
        try:
            if 'df' in locals():
                current_app.logger.error(f"DataFrame shape: {df.shape}")
                current_app.logger.error(f"DataFrame columns: {df.columns.tolist()}")
                current_app.logger.error(f"DataFrame dtypes: {df.dtypes.to_dict()}")
                
                if 'target_column' in locals() and target_column in df.columns:
                    current_app.logger.error(f"Target column '{target_column}' type: {df[target_column].dtype}")
                    current_app.logger.error(f"Target column sample values: {df[target_column].head().tolist()}")
                    
                if 'predictor_columns' in locals():
                    for col in predictor_columns[:3]:  # Solo las primeras 3 para no llenar el log
                        if col in df.columns:
                            current_app.logger.error(f"Feature '{col}' type: {df[col].dtype}, sample: {df[col].head().tolist()}")
        except Exception as log_error:
            current_app.logger.error(f"Could not log DataFrame details: {log_error}")
        
        import traceback
        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Error al preparar datos: {str(e)}")

# Alias para compatibilidad hacia atrás
prepare_data_for_regression = prepare_data_for_training

def train_regression_model(X, y, model_type, test_size=0.2, random_state=42, **params):
    """Entrena un modelo de regresión"""
    try:
        # Validaciones iniciales de tamaño del dataset
        if len(X) == 0:
            raise ValueError("El dataset está vacío (0 muestras)")
        
        if len(X) < 5:
            raise ValueError(f"Dataset muy pequeño ({len(X)} muestras). Se necesitan al menos 5 muestras para entrenamiento de regresión")
        
        # Asegurar que y sea una pandas Series para métodos compatibles
        if not isinstance(y, pd.Series):
            y = pd.Series(y) if hasattr(y, '__iter__') else pd.Series([y])
        
        # Para datasets muy pequeños, usar un test_size menor o fijo
        if len(X) < 10:
            test_size = 1  # Solo una muestra para test
            current_app.logger.warning(f"Dataset pequeño ({len(X)} muestras), usando test_size=1")
        elif len(X) < 25:
            test_size = max(1, int(len(X) * 0.1))  # 10% o mínimo 1
            current_app.logger.warning(f"Dataset pequeño ({len(X)} muestras), usando test_size={test_size}")
            
        # Para regresión no validamos clases, pero sí verificamos que haya datos suficientes
        if len(y) < 4:  # Necesitamos al menos 4 muestras para split y validación
            raise ValueError("Se necesitan al menos 4 muestras para entrenar un modelo de regresión.")

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Crear el modelo según el tipo
        if model_type == 'linear_regression':
            model = LinearRegression(**params)
        elif model_type == 'ridge_regression':
            model = Ridge(random_state=random_state, **params)
        elif model_type == 'lasso_regression':
            model = Lasso(random_state=random_state, **params)
        elif model_type == 'elastic_net':
            model = ElasticNet(random_state=random_state, **params)
        elif model_type == 'decision_tree_regressor':
            model = DecisionTreeRegressor(random_state=random_state, **params)
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(random_state=random_state, **params)
        elif model_type == 'svr':
            model = SVR(**params)
        elif model_type == 'knn_regressor':
            model = KNeighborsRegressor(**params)
        else:
            raise ValueError(f"Tipo de modelo de regresión no soportado: {model_type}")
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas de regresión
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Resultados del entrenamiento
        training_results = {
            'model_type': model_type,
            'model_params': params.copy(),
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_names': X.columns.tolist(),
            'predictions_sample': {
                'y_true_sample': y_test[:10].tolist(),
                'y_pred_sample': y_pred_test[:10].tolist()
            }
        }
        # Guardar coeficientes e intercepto si el modelo es lineal
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            training_results['model_params']['coef_'] = model.coef_.tolist()
            training_results['model_params']['intercept_'] = float(model.intercept_)
            # Para compatibilidad con el template, si el nombre es 'Ridge Regression' o 'Lasso Regression', también guardar bajo 'Ridge' y 'Lasso'
            if model_type == 'ridge_regression':
                training_results['model_params']['ridge_coef_'] = model.coef_.tolist()
                training_results['model_params']['ridge_intercept_'] = float(model.intercept_)
            if model_type == 'lasso_regression':
                training_results['model_params']['lasso_coef_'] = model.coef_.tolist()
                training_results['model_params']['lasso_intercept_'] = float(model.intercept_)
        
        # Importancia de características (si está disponible)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            training_results['feature_importance'] = convert_numpy_types(feature_importance)
        elif hasattr(model, 'coef_'):
            # Para modelos lineales, usar coeficientes como importancia
            feature_importance = dict(zip(X.columns, model.coef_))
            training_results['feature_importance'] = convert_numpy_types(feature_importance)
        
        return model, training_results
        
    except Exception as e:
        raise Exception(f"Error al entrenar modelo de regresión: {str(e)}")
