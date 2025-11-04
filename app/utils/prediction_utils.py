"""
Utilidades para realizar predicciones con modelos guardados
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import current_app
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from .training_utils import clean_column_names, get_training_file
from .data_validation import convert_numpy_types

def get_available_models():
    """Obtiene lista de modelos disponibles con sus metadatos"""
    models_dir = current_app.config['MODELS_FOLDER']
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    
    # Buscar archivos de metadatos JSON
    for filename in os.listdir(models_dir):
        if filename.endswith('_metadata.json'):
            metadata_path = os.path.join(models_dir, filename)
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Verificar que el archivo del modelo existe
                model_file = metadata.get('model_file')
                if model_file and os.path.exists(model_file):
                    # Obtener información adicional
                    model_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                    metadata['model_size_mb'] = round(model_size, 2)
                    
                    # Formatear fecha
                    if 'created_at' in metadata:
                        try:
                            created_date = datetime.fromisoformat(metadata['created_at'])
                            metadata['created_at_formatted'] = created_date.strftime('%d/%m/%Y %H:%M')
                        except (ValueError, TypeError, KeyError):
                            metadata['created_at_formatted'] = metadata.get('created_at', 'Desconocida')
                    
                    available_models.append(metadata)
                    
            except Exception as e:
                current_app.logger.warning(f"Error loading metadata for {filename}: {e}")
                continue
    
    # Ordenar por fecha de creación (más recientes primero)
    available_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return available_models

def load_model_and_metadata(model_name):
    """Carga un modelo y sus metadatos"""
    models_dir = current_app.config['MODELS_FOLDER']
    
    # Cargar metadatos
    metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadatos no encontrados para el modelo {model_name}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Cargar modelo
    model_file = metadata.get('model_file')
    if not model_file or not os.path.exists(model_file):
        # Intentar con el path alternativo
        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_file}")
    
    try:
        model_data = joblib.load(model_file)
        
        # Verificar si model_data es un diccionario (formato nuevo) o el modelo directamente (formato antiguo)
        if isinstance(model_data, dict) and 'model' in model_data:
            # Formato nuevo: diccionario con modelo, encoders, etc.
            current_app.logger.info(f"Modelo cargado exitosamente (formato nuevo): {model_name}")
            return model_data, metadata
        else:
            # Formato antiguo: modelo directamente
            current_app.logger.info(f"Modelo cargado (formato antiguo), adaptando estructura: {model_name}")
            # Convertir a formato nuevo
            adapted_model_data = {
                'model': model_data,
                'encoders': {},
                'target_encoder': None,
                'model_info': {}
            }
            return adapted_model_data, metadata
    except Exception as e:
        raise Exception(f"Error cargando el modelo: {str(e)}")

def prepare_prediction_data(df, feature_columns, encoders=None):
    """Prepara datos para predicción usando el mismo procesamiento que el entrenamiento"""
    try:
        # Limpiar nombres de columnas
        df.columns = clean_column_names(df.columns.tolist())
        
        # Verificar que todas las columnas requeridas están presentes
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes en el archivo: {missing_columns}")
        
        # Seleccionar solo las columnas necesarias
        X = df[feature_columns].copy()
        
        # Manejar valores faltantes usando la misma estrategia que en entrenamiento
        from sklearn.impute import SimpleImputer
        
        # Columnas numéricas
        numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy='median')
            X_numeric_imputed = pd.DataFrame(
                numeric_imputer.fit_transform(X[numeric_columns]),
                columns=numeric_columns,
                index=X.index
            )
            X[numeric_columns] = X_numeric_imputed
        
        # Columnas categóricas
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent', fill_value='Unknown')
            X_categorical_imputed = pd.DataFrame(
                categorical_imputer.fit_transform(X[categorical_columns]),
                columns=categorical_columns,
                index=X.index
            )
            X[categorical_columns] = X_categorical_imputed
        
        # Aplicar encoders si están disponibles
        X_encoded = X.copy()
        
        if encoders:
            for col in feature_columns:
                if col in encoders and col in X_encoded.columns:
                    if X_encoded[col].dtype == 'object':
                        encoder = encoders[col]
                        
                        # Manejar categorías no vistas durante el entrenamiento
                        unknown_categories = set(X_encoded[col].unique()) - set(encoder.classes_)
                        if unknown_categories:
                            current_app.logger.warning(f"Categorías no vistas en {col}: {unknown_categories}")
                            # Reemplazar categorías desconocidas con la más frecuente
                            most_frequent = encoder.classes_[0]  # Primera clase como fallback
                            X_encoded[col] = X_encoded[col].apply(
                                lambda x: most_frequent if x not in encoder.classes_ else x
                            )
                        
                        X_encoded[col] = encoder.transform(X_encoded[col])
        else:
            # Si no hay encoders, aplicar encoding básico
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        return X_encoded
        
    except Exception as e:
        current_app.logger.error(f"Error preparando datos para predicción: {str(e)}")
        raise Exception(f"Error preparando datos: {str(e)}")

def make_predictions(model_data, metadata, df):
    """Realiza predicciones usando el modelo cargado"""
    try:
        # Debug logging
        current_app.logger.info(f"Type of model_data: {type(model_data)}")
        current_app.logger.info(f"model_data keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
        
        # Verificar que model_data es un diccionario
        if not isinstance(model_data, dict):
            raise ValueError(f"model_data debe ser un diccionario, pero es: {type(model_data)}")
        
        # Extraer componentes del modelo
        model = model_data.get('model')
        encoders = model_data.get('encoders', {})
        target_encoder = model_data.get('target_encoder')
        
        if model is None:
            raise ValueError("No se encontró el modelo en los datos cargados")
        
        # Obtener información del tipo de tarea
        task_type = metadata.get('task_type', 'classification')
        current_app.logger.info(f"Task type: {task_type}")
        
        # Obtener columnas de features del metadata
        feature_columns = metadata.get('feature_columns', [])
        if not feature_columns:
            # Intentar obtenerlas de preparation_info si no están directamente
            prep_info = metadata.get('preparation_info', {})
            feature_types = prep_info.get('feature_types', {})
            if feature_types:
                feature_columns = list(feature_types.keys())
            
            if not feature_columns:
                raise ValueError("No se encontraron las columnas de features en los metadatos")
        
        current_app.logger.info(f"Feature columns found: {feature_columns}")
        
        # Preparar datos
        X_prepared = prepare_prediction_data(df, feature_columns, encoders)
        
        # Realizar predicciones
        predictions = model.predict(X_prepared)
        
        # Crear DataFrame con resultados
        results_df = df.copy()
        
        # Manejar según el tipo de tarea
        if task_type.lower() == 'regression':
            # Para regresión, las predicciones son valores numéricos directos
            results_df['Prediccion'] = predictions
            
            # Agregar intervalos de confianza si el modelo lo soporta
            confidence_intervals = None
            if hasattr(model, 'predict') and hasattr(model, 'score'):
                try:
                    # Para algunos modelos de regresión, podemos calcular intervalos
                    # Por ahora, usar desviación estándar como proxy de confianza
                    std_pred = np.std(predictions)
                    confidence_lower = predictions - std_pred
                    confidence_upper = predictions + std_pred
                    results_df['Confianza_Inferior'] = confidence_lower
                    results_df['Confianza_Superior'] = confidence_upper
                    confidence_intervals = True
                except Exception as e:
                    current_app.logger.warning(f"No se pudieron calcular intervalos de confianza: {e}")
            
            # Estadísticas para regresión
            prediction_stats = {
                'total_predictions': len(predictions),
                'model_type': metadata.get('model_type', 'Desconocido'),
                'task_type': task_type,
                'target_column': metadata.get('target_column', ''),
                'features_used': len(feature_columns),
                'prediction_mean': float(np.mean(predictions)),
                'prediction_std': float(np.std(predictions)),
                'prediction_min': float(np.min(predictions)),
                'prediction_max': float(np.max(predictions)),
                'has_confidence_intervals': confidence_intervals is not None,
                'target_type': 'numeric'
            }
            
        else:
            # Para clasificación, manejar como antes
            # Obtener probabilidades si está disponible
            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_proba = model.predict_proba(X_prepared)
                except Exception as e:
                    current_app.logger.warning(f"No se pudieron obtener probabilidades: {e}")
            
            # Decodificar predicciones usando target_encoder
            predictions_decoded = predictions.copy()
            predictions_text = None
            
            if target_encoder and hasattr(target_encoder, 'inverse_transform'):
                try:
                    predictions_text = target_encoder.inverse_transform(predictions)
                    current_app.logger.info(f"Predicciones decodificadas exitosamente usando target_encoder")
                except Exception as e:
                    current_app.logger.warning(f"Error decodificando predicciones: {e}")
                    # Fallback: usar las clases desde metadata si están disponibles
                    target_classes = metadata.get('preparation_info', {}).get('target_classes', [])
                    if target_classes:
                        try:
                            predictions_text = [target_classes[int(p)] for p in predictions]
                            current_app.logger.info(f"Usando clases desde metadata: {target_classes}")
                        except (IndexError, ValueError) as e2:
                            current_app.logger.warning(f"Error usando clases desde metadata: {e2}")
                            predictions_text = None
            else:
                # Si no hay target_encoder, intentar usar clases desde metadata
                target_classes = metadata.get('preparation_info', {}).get('target_classes', [])
                if target_classes:
                    try:
                        predictions_text = [target_classes[int(p)] for p in predictions]
                        current_app.logger.info(f"Usando clases desde metadata (sin encoder): {target_classes}")
                    except (IndexError, ValueError) as e:
                        current_app.logger.warning(f"Error usando clases desde metadata: {e}")
                        predictions_text = None
            
            # Agregar tanto la predicción numérica como el texto (si está disponible)
            results_df['Prediccion_Numerica'] = predictions_decoded
            if predictions_text is not None:
                results_df['Prediccion'] = predictions_text
            else:
                # Si no hay texto, intentar dar contexto con metadatos
                target_classes = metadata.get('preparation_info', {}).get('target_classes', [])
                
                if target_classes:
                    # Mapear índices a clases reales
                    prediction_labels = []
                    for pred in predictions_decoded:
                        if int(pred) < len(target_classes):
                            prediction_labels.append(target_classes[int(pred)])
                        else:
                            prediction_labels.append(f'Clase_{int(pred)}')
                    
                    results_df['Prediccion'] = prediction_labels
                else:
                    results_df['Prediccion'] = predictions_decoded
            
            # Agregar columnas de confianza si hay probabilidades
            if prediction_proba is not None:
                confidence_scores = np.max(prediction_proba, axis=1)
                results_df['Confianza'] = np.round(confidence_scores * 100, 2)
                
                # Agregar probabilidades por clase usando nombres reales
                if predictions_text is not None:
                    # Usar target_encoder si está disponible
                    if target_encoder and hasattr(target_encoder, 'classes_'):
                        class_names = target_encoder.classes_
                    else:
                        # Usar clases desde metadata
                        class_names = metadata.get('preparation_info', {}).get('target_classes', [])
                    
                    if class_names and len(class_names) <= 10:  # Solo si hay pocas clases
                        for i, class_name in enumerate(class_names):
                            if i < prediction_proba.shape[1]:  # Verificar que el índice existe
                                results_df[f'Prob_{class_name}'] = np.round(prediction_proba[:, i] * 100, 2)
                else:
                    # Si no hay texto, usar índices
                    n_classes = prediction_proba.shape[1]
                    if n_classes <= 10:
                        for i in range(n_classes):
                            results_df[f'Prob_Clase_{i}'] = np.round(prediction_proba[:, i] * 100, 2)
            
            # Estadísticas para clasificación
            prediction_stats = {
                'total_predictions': len(predictions),
                'unique_predictions': len(np.unique(predictions_decoded)),
                'model_type': metadata.get('model_type', 'Desconocido'),
                'task_type': task_type,
                'target_column': metadata.get('target_column', ''),
                'features_used': len(feature_columns),
                'avg_confidence': np.round(np.mean(confidence_scores) * 100, 2) if prediction_proba is not None else None,
                'prediction_distribution': {},
                'class_mapping': {},
                'target_type': 'categorical'
            }
            
            # Agregar distribución de predicciones y mapeo de clases
            if predictions_text is not None:
                # Contar predicciones por texto
                unique_text, counts_text = np.unique(predictions_text, return_counts=True)
                prediction_stats['prediction_distribution'] = {
                    text: int(count) for text, count in zip(unique_text, counts_text)
                }
                
                # Crear mapeo de números a texto
                target_classes = metadata.get('preparation_info', {}).get('target_classes', [])
                if target_classes:
                    prediction_stats['class_mapping'] = {
                        i: class_name for i, class_name in enumerate(target_classes)
                    }
            else:
                # Solo números - crear mapeo básico
                unique_nums, counts_nums = np.unique(predictions_decoded, return_counts=True)
                
                # Intentar usar clases desde metadata
                target_classes = metadata.get('preparation_info', {}).get('target_classes', [])
                if target_classes:
                    # Crear mapeo de índices a clases reales
                    class_mapping = {}
                    prediction_distribution = {}
                    
                    for num, count in zip(unique_nums, counts_nums):
                        if int(num) < len(target_classes):
                            class_name = target_classes[int(num)]
                            class_mapping[int(num)] = class_name
                            prediction_distribution[class_name] = int(count)
                        else:
                            class_mapping[int(num)] = f'Clase_{int(num)}'
                            prediction_distribution[f'Clase_{int(num)}'] = int(count)
                    
                    prediction_stats['class_mapping'] = class_mapping
                    prediction_stats['prediction_distribution'] = prediction_distribution
                else:
                    # Fallback: usar números directamente
                    prediction_stats['prediction_distribution'] = {
                        f'Clase_{int(num)}': int(count) for num, count in zip(unique_nums, counts_nums)
                    }
                    prediction_stats['class_mapping'] = {
                        int(num): f'Clase_{int(num)}' for num in unique_nums
                    }
        
        # Convertir tipos numpy para JSON
        prediction_stats = convert_numpy_types(prediction_stats)
        
        return results_df, prediction_stats
        
    except Exception as e:
        current_app.logger.error(f"Error realizando predicciones: {str(e)}")
        raise Exception(f"Error en predicción: {str(e)}")

def validate_prediction_compatibility(metadata, df):
    """Valida si un archivo CSV es compatible con un modelo guardado"""
    try:
        # Limpiar nombres de columnas del DataFrame
        df.columns = clean_column_names(df.columns.tolist())
        
        required_features = metadata.get('feature_columns', [])
        available_columns = df.columns.tolist()
        
        # Verificar columnas faltantes
        missing_columns = [col for col in required_features if col not in available_columns]
        
        # Verificar tipos de datos (básico)
        type_warnings = []
        if 'preparation_info' in metadata and 'feature_types' in metadata['preparation_info']:
            feature_types = metadata['preparation_info']['feature_types']
            
            for col in required_features:
                if col in available_columns:
                    expected_type = feature_types.get(col, '')
                    actual_dtype = str(df[col].dtype)
                    
                    # Verificaciones básicas de compatibilidad
                    if 'numeric' in expected_type and df[col].dtype == 'object':
                        type_warnings.append(f"'{col}' debería ser numérica pero es categórica")
                    elif 'categorical' in expected_type and df[col].dtype in ['int64', 'float64']:
                        type_warnings.append(f"'{col}' debería ser categórica pero es numérica")
        
        # Verificar tamaño mínimo del dataset
        min_rows_warning = None
        if len(df) < 5:
            min_rows_warning = f"El archivo tiene solo {len(df)} filas. Se recomiendan al menos 5 para predicciones confiables."
        
        compatibility_result = {
            'compatible': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'available_columns': available_columns,
            'required_columns': required_features,
            'type_warnings': type_warnings,
            'min_rows_warning': min_rows_warning,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        
        return compatibility_result
        
    except Exception as e:
        current_app.logger.error(f"Error validando compatibilidad: {str(e)}")
        return {
            'compatible': False,
            'error': str(e),
            'missing_columns': [],
            'available_columns': [],
            'required_columns': [],
            'type_warnings': [],
            'min_rows_warning': None
        }