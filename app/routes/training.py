import json
import hashlib
import joblib
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, send_file
import pandas as pd
import os
from datetime import datetime, timedelta
from ..utils.csv_reader import CSVReader
from ..utils.model_training import train_model, train_regression_model, prepare_data_for_training
from ..utils.training_utils import get_training_file, match_column_names, clean_column_names
from ..utils.target_analysis import analyze_target_compatibility, get_model_availability_message
from ..services.session_service import SessionService

training_bp = Blueprint('training', __name__)

# Función movida al sistema centralizado de limpieza (system_cleanup.py)

def save_training_results(results_data):
    """Guarda los resultados de entrenamiento en un archivo temporal"""
    # Crear un hash único basado en el timestamp y datos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = hashlib.md5(f"{timestamp}_{results_data['model_type']}".encode()).hexdigest()[:8]
    
    # Crear directorio para resultados temporales
    temp_dir = current_app.config['TEMP_RESULTS_FOLDER']
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Guardar en archivo JSON
    filename = f"training_results_{unique_id}.json"
    filepath = os.path.join(temp_dir, filename)
    
    # Convertir numpy arrays a listas para JSON
    json_data = {}
    for key, value in results_data.items():
        if hasattr(value, 'tolist'):
            json_data[key] = value.tolist()
        elif isinstance(value, dict):
            json_data[key] = {}
            for k, v in value.items():
                if hasattr(v, 'tolist'):
                    json_data[key][k] = v.tolist()
                else:
                    json_data[key][k] = v
        else:
            json_data[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    current_app.logger.info(f"Results saved to: {filepath}")
    return unique_id

def load_training_results(unique_id):
    """Carga los resultados de entrenamiento desde archivo temporal"""
    temp_dir = current_app.config['TEMP_RESULTS_FOLDER']
    filename = f"training_results_{unique_id}.json"
    filepath = os.path.join(temp_dir, filename)
    
    if not os.path.exists(filepath):
        current_app.logger.error(f"Results file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        current_app.logger.error(f"Error loading results: {e}")
        return None

@training_bp.route('/training')
def training():
    """Página principal de entrenamiento de modelos."""
    
    # Determinar qué archivo está siendo usado actualmente
    filename = 'N/A'
    data_shape = [0, 0]
    
    try:
        # Obtener columnas de la sesión si están disponibles
        target_column = session.get('target_column')
        predictor_columns = session.get('predictor_columns', [])
        
        if target_column and predictor_columns:
            # Si hay columnas seleccionadas, usar la función utilitaria para encontrar el archivo correcto
            from ..utils.training_utils import get_training_file
            # Limpiar columnas antes de llamar a get_training_file
            from ..utils.training_utils import clean_column_names
            target_column_clean = clean_column_names([target_column])[0] if isinstance(target_column, str) else target_column
            predictor_columns_clean = clean_column_names(predictor_columns)
            file_to_use, df, success = get_training_file(target_column_clean, predictor_columns_clean)
            
            if success and df is not None:
                filename = os.path.basename(file_to_use)
                data_shape = [len(df), len(df.columns)]
        else:
            # Si no hay columnas seleccionadas, usar el archivo original
            if 'uploaded_file' in session and os.path.exists(session['uploaded_file']):
                file_to_use = session['uploaded_file']
                filename = os.path.basename(file_to_use)
                
                # Cargar el archivo para obtener el shape usando utilidad centralizada
                df, read_info = CSVReader.read_csv_robust(file_to_use)
                
                if read_info['success']:
                    data_shape = [read_info['rows'], read_info['columns']]
                    
    except Exception as e:
        current_app.logger.warning(f"Could not determine current file info: {e}")
        # Usar valores por defecto si hay error
    
    return render_template('model_training_hub.html', 
                         filename=filename,
                         data_shape=data_shape)

@training_bp.route('/sklearn_regression')
def sklearn_regression():
    """Página para entrenar modelos de regresión con sklearn."""
    current_app.logger.info("Accessed sklearn regression page")
    
    # Verificar si hay archivo cargado
    if 'uploaded_file' not in session:
        flash('Debe cargar un archivo primero', 'error')
        return redirect(url_for('main.index'))
    
    try:
        # Obtener columnas seleccionadas de la sesión
        target_column = session.get('target_column')
        predictor_columns = session.get('predictor_columns', [])
        
        # Usar la función utilitaria para encontrar el archivo correcto si hay columnas seleccionadas
        file_to_use = None
        df = None
        
        if target_column and predictor_columns:
            file_to_use, df, success = get_training_file(target_column, predictor_columns)
            
            if not success:
                current_app.logger.warning("Selected columns not found in any available file, redirecting to column selection")
                flash('Las columnas seleccionadas ya no existen en el archivo actual. Por favor, vuelve a seleccionar las columnas.', 'error')
                return redirect(url_for('data.select_columns'))
            
            # Validación adicional: verificar que TODAS las columnas existan en el DataFrame
            if df is not None:
                # Limpiar columnas del DataFrame usando la función estándar
                df.columns = clean_column_names(df.columns.tolist())
                
                # Limpiar también las columnas de sesión usando la misma función
                target_clean = clean_column_names([target_column])[0] if isinstance(target_column, str) else target_column
                predictors_clean = clean_column_names(predictor_columns)
                
                missing_columns = []
                if target_clean not in df.columns:
                    missing_columns.append(target_clean)
                for col in predictors_clean:
                    if col not in df.columns:
                        missing_columns.append(col)
                
                if missing_columns:
                    current_app.logger.warning(f"Missing columns in current file: {missing_columns}")
                    flash(f'Las siguientes columnas no existen en el archivo actual: {missing_columns[:5]}... Por favor, vuelve a seleccionar las columnas.', 'error')
                    return redirect(url_for('data.select_columns'))
        else:
            # Si no hay columnas seleccionadas, usar el archivo original
            file_to_use = session['uploaded_file']
            
            if not os.path.exists(file_to_use):
                current_app.logger.error(f"File not found: {file_to_use}")
                flash(f'El archivo no existe: {file_to_use}', 'error')
                return redirect(url_for('main.index'))
            
            # Cargar el archivo con manejo de encoding
            df, read_info = CSVReader.read_csv_robust(file_to_use)
            if not read_info['success']:
                flash(f'Error leyendo archivo: {read_info["error"]}', 'error')
                return redirect(url_for('data.select_columns'))
        
        columns = df.columns.tolist()
        current_app.logger.info(f"Available columns in regression: {columns}")
        current_app.logger.info(f"Target column in session: {target_column}")
        current_app.logger.info(f"Predictor columns in session: {predictor_columns}")
        
        # Si no hay columnas seleccionadas, redirigir a selección de columnas
        if not target_column or not predictor_columns:
            flash('Por favor, selecciona primero las columnas target y predictoras.', 'info')
            return redirect(url_for('data.select_columns'))
        
        # VERIFICAR COMPATIBILIDAD CON REGRESIÓN
        target_analysis = analyze_target_compatibility(df, target_column)
        
        if not target_analysis.get('regression_compatible', False):
            availability_messages = get_model_availability_message(target_analysis)
            regression_msg = availability_messages['regression']
            
            flash(f'La variable target seleccionada no es compatible con modelos de regresión. {regression_msg["message"]}', 'warning')
            
            # Si es compatible con clasificación, sugerir esa opción
            if target_analysis.get('classification_compatible', False):
                flash('Esta variable es compatible con modelos de clasificación. Te redirigimos allí.', 'info')
                return redirect(url_for('training.sklearn_classification'))
            else:
                flash('Por favor, selecciona una variable target diferente.', 'info')
                return redirect(url_for('data.select_columns'))
        
        # Si hay una advertencia pero es compatible, mostrarla
        availability_messages = get_model_availability_message(target_analysis)
        if availability_messages['regression']['status'] == 'available':
            flash(f"Nota: {availability_messages['regression']['message']}", 'info')
        
        # Definir modelos disponibles para regresión
        models_info = {
            'linear_regression': {
                'name': 'Regresión Lineal',
                'description': 'Modelo lineal básico',
                'params': {}
            },
            'ridge_regression': {
                'name': 'Ridge Regression',
                'description': 'Regresión lineal con regularización L2',
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'lasso_regression': {
                'name': 'Lasso Regression',
                'description': 'Regresión lineal con regularización L1',
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'elastic_net': {
                'name': 'Elastic Net',
                'description': 'Regresión con regularización L1 y L2',
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            },
            'decision_tree_regressor': {
                'name': 'Árbol de Decisión',
                'description': 'Regresor basado en árboles',
                'params': {
                    'max_depth': ['Sin límite', 3, 5, 10, 15, 20]
                }
            },
            'random_forest_regressor': {
                'name': 'Random Forest',
                'description': 'Conjunto de árboles de regresión',
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': ['Sin límite', 3, 5, 10, 15, 20]
                }
            },
            'svr': {
                'name': 'Support Vector Regression',
                'description': 'Regresión con vectores de soporte',
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                }
            },
            'knn_regressor': {
                'name': 'K-Nearest Neighbors',
                'description': 'Regresor basado en vecinos cercanos',
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15]
                }
            }
        }
        
        return render_template('sklearn_regression.html', 
                             columns=columns, 
                             models_info=models_info,
                             filename=os.path.basename(file_to_use),
                             target_column=target_column,
                             predictor_columns=predictor_columns,
                             data_shape=[len(df), len(df.columns)])
        
    except Exception as e:
        current_app.logger.error(f"Error in sklearn_regression: {str(e)}")
        flash(f'Error al cargar el archivo: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@training_bp.route('/train_regression', methods=['POST'])
def train_regression():
    """Entrenar modelo de regresión."""
    try:
        model_type = request.form.get('model_type')
        target_column = request.form.get('target_column')
        feature_columns = request.form.getlist('feature_columns')
        
        current_app.logger.info(f"=== TRAIN REGRESSION DEBUG ===")
        current_app.logger.info(f"model_type: {model_type}")
        current_app.logger.info(f"target_column: {target_column}")
        current_app.logger.info(f"feature_columns: {feature_columns}")
        current_app.logger.info(f"All form data: {dict(request.form)}")
        
        if not model_type or not target_column or not feature_columns:
            flash('Debe seleccionar el tipo de modelo, la variable objetivo y las características', 'error')
            return redirect(url_for('training.sklearn_regression'))
        
        # Usar la función utilitaria para encontrar el archivo correcto
        file_to_use, df, success = get_training_file(target_column, feature_columns)
        
        if not success:
            flash(f'Las columnas seleccionadas ya no existen en ningún archivo disponible', 'error')
            return redirect(url_for('training.sklearn_regression'))
        
        # Preparar los datos para entrenamiento
        try:
            X_processed, y_processed, encoders, target_encoder, preparation_info = prepare_data_for_training(
                df, target_column, feature_columns, task_type='regression'
            )
            current_app.logger.info(f"Data prepared successfully: {preparation_info}")
        except Exception as e:
            current_app.logger.error(f"Error preparing data: {e}")
            flash(f'Error preparando los datos: {str(e)}', 'error')
            return redirect(url_for('training.sklearn_regression'))
        
        model_result = None
        
        try:
            # Importar la función de entrenamiento de regresión
            from ..utils.model_training import train_regression_model
            
            current_app.logger.info(f"Starting training for model_type: {model_type}")
            
            if model_type == 'linear_regression':
                current_app.logger.info("Training Linear Regression model...")
                model_result = train_regression_model(X_processed, y_processed, 'linear_regression')
            elif model_type == 'ridge_regression':
                alpha_param = request.form.get('alpha', '1.0')
                current_app.logger.info(f"Training Ridge Regression with alpha: '{alpha_param}'")
                alpha = float(alpha_param) if alpha_param and alpha_param not in ['None', ''] else 1.0
                model_result = train_regression_model(X_processed, y_processed, 'ridge_regression', alpha=alpha)
            elif model_type == 'lasso_regression':
                alpha_param = request.form.get('alpha', '1.0')
                current_app.logger.info(f"Training Lasso Regression with alpha: '{alpha_param}'")
                alpha = float(alpha_param) if alpha_param and alpha_param not in ['None', ''] else 1.0
                model_result = train_regression_model(X_processed, y_processed, 'lasso_regression', alpha=alpha)
            elif model_type == 'elastic_net':
                alpha_param = request.form.get('alpha', '1.0')
                l1_ratio_param = request.form.get('l1_ratio', '0.5')
                current_app.logger.info(f"ElasticNet alpha: '{alpha_param}', l1_ratio: '{l1_ratio_param}'")
                alpha = float(alpha_param) if alpha_param and alpha_param not in ['None', ''] else 1.0
                l1_ratio = float(l1_ratio_param) if l1_ratio_param and l1_ratio_param not in ['None', ''] else 0.5
                model_result = train_regression_model(X_processed, y_processed, 'elastic_net', alpha=alpha, l1_ratio=l1_ratio)
            elif model_type == 'decision_tree_regressor':
                max_depth = request.form.get('max_depth')
                current_app.logger.info(f"Decision Tree Regressor max_depth from form: '{max_depth}'")
                max_depth = None if max_depth in ['Sin límite', 'None', None, ''] else int(max_depth)
                current_app.logger.info(f"Decision Tree Regressor max_depth processed: {max_depth}")
                model_result = train_regression_model(X_processed, y_processed, 'decision_tree_regressor', max_depth=max_depth)
            elif model_type == 'random_forest_regressor':
                n_estimators_param = request.form.get('n_estimators', '100')
                max_depth = request.form.get('max_depth')
                current_app.logger.info(f"Random Forest Regressor n_estimators: '{n_estimators_param}', max_depth: '{max_depth}'")
                n_estimators = int(n_estimators_param) if n_estimators_param and n_estimators_param not in ['None', ''] else 100
                max_depth = None if max_depth in ['Sin límite', 'None', None, ''] else int(max_depth)
                model_result = train_regression_model(X_processed, y_processed, 'random_forest_regressor', n_estimators=n_estimators, max_depth=max_depth)
            elif model_type == 'knn_regressor':
                n_neighbors_param = request.form.get('n_neighbors', '5')
                current_app.logger.info(f"KNN Regressor n_neighbors param: '{n_neighbors_param}'")
                n_neighbors = int(n_neighbors_param) if n_neighbors_param and n_neighbors_param not in ['None', ''] else 5
                model_result = train_regression_model(X_processed, y_processed, 'knn_regressor', n_neighbors=n_neighbors)
            elif model_type == 'svr':
                C_param = request.form.get('C', '1.0')
                kernel_param = request.form.get('kernel', 'rbf')
                current_app.logger.info(f"SVR C: '{C_param}', kernel: '{kernel_param}'")
                C = float(C_param) if C_param and C_param not in ['None', ''] else 1.0
                model_result = train_regression_model(X_processed, y_processed, 'svr', C=C, kernel=kernel_param)
            else:
                current_app.logger.error(f"Unknown model_type: {model_type}")
                flash('Tipo de modelo no válido', 'error')
                return redirect(url_for('training.sklearn_regression'))
                
            current_app.logger.info(f"Model training call completed. Result type: {type(model_result)}")
            if model_result:
                current_app.logger.info(f"Model result length: {len(model_result) if isinstance(model_result, (list, tuple)) else 'Not a sequence'}")
            
        except ValueError as ve:
            current_app.logger.error(f"Error converting parameters: {str(ve)}")
            flash(f'Error en los parámetros del modelo: {str(ve)}. Verifica que todos los valores numéricos sean válidos.', 'error')
            return redirect(url_for('training.sklearn_regression'))
        except Exception as pe:
            current_app.logger.error(f"Error preparing model parameters: {str(pe)}")
            flash(f'Error preparando el modelo: {str(pe)}', 'error')
            return redirect(url_for('training.sklearn_regression'))
        
        current_app.logger.info(f"Model training completed. Result: {type(model_result)}")
        current_app.logger.info(f"Model result is None: {model_result is None}")
        
        if model_result:
            current_app.logger.info("Unpacking model result...")
            model, results = model_result
            current_app.logger.info(f"Model: {type(model)}, Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
            
            # Limpiar datos temporales anteriores antes de guardar el nuevo modelo
            old_temp_path = session.get('temp_model_path')
            if old_temp_path and os.path.exists(old_temp_path):
                try:
                    os.remove(old_temp_path)
                    current_app.logger.info(f"Cleaned up previous temporary model: {old_temp_path}")
                except Exception as e:
                    current_app.logger.warning(f"Could not clean up previous temp file: {e}")
            
            # Limpiar resultados anteriores de la sesión para evitar conflictos
            SessionService.clear_large_results('regression_results')
            
            # Guardar el modelo temporalmente usando joblib para evitar problemas de serialización JSON
            temp_model_id = str(uuid.uuid4())[:8]
            temp_model_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'temp_model_{temp_model_id}.joblib')
            
            try:
                # Asegurar que el directorio temp_results existe
                os.makedirs(current_app.config['TEMP_RESULTS_FOLDER'], exist_ok=True)
                joblib.dump(model, temp_model_path)
                
                # Actualizar la sesión con el nuevo modelo
                session['temp_model_id'] = temp_model_id
                session['temp_model_path'] = temp_model_path
                session['temp_model_timestamp'] = datetime.now().isoformat()
                session['temp_model_type'] = 'regression'
                current_app.logger.info(f"Trained regression model stored temporarily: {temp_model_path}")
            except Exception as e:
                current_app.logger.error(f"Error storing temporary model: {e}")
                session.pop('temp_model_id', None)
                session.pop('temp_model_path', None)
                session.pop('temp_model_timestamp', None)
                session.pop('temp_model_type', None)
            
            # Preparar información adicional para modelos lineales
            model_coefficients = None
            model_intercept = None
            linear_models = ['linear_regression', 'ridge_regression', 'lasso_regression', 'elastic_net']
            
            if model_type in linear_models and hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                model_coefficients = model.coef_.tolist()
                model_intercept = float(model.intercept_)
            
            # Guardar los resultados usando el sistema de almacenamiento de archivos
            # para evitar el límite de tamaño de cookies del navegador
            regression_results = {
                'model_info': {
                    'model_type': model_type,
                    'file_used': os.path.basename(file_to_use),
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'n_features': len(feature_columns)
                },
                'main_metrics': {
                    'train_mse': results.get('train_mse', 0),
                    'test_mse': results.get('test_mse', 0),
                    'train_r2': results.get('train_r2', 0),
                    'test_r2': results.get('test_r2', 0),
                    'train_mae': results.get('train_mae', 0),
                    'test_mae': results.get('test_mae', 0)
                },
                'preparation_info': {
                    **preparation_info,
                    'coef_': model_coefficients,
                    'intercept': model_intercept
                },
                'feature_importance': results.get('feature_importance', None)
            }
            
            # Usar el nuevo sistema de almacenamiento para resultados grandes
            storage_success = SessionService.store_large_results('regression_results', regression_results)
            
            current_app.logger.info("Regression results prepared for session:")
            current_app.logger.info(f"- Model info: {regression_results['model_info']}")
            current_app.logger.info(f"- Main metrics: {regression_results['main_metrics']}")
            
            if storage_success:
                current_app.logger.info("Regression results saved successfully using file storage")
                return redirect(url_for('training.regression_results'))
            else:
                current_app.logger.error("Failed to save regression results")
                flash('Error al guardar los resultados del entrenamiento', 'error')
                return redirect(url_for('training.sklearn_regression'))
        else:
            current_app.logger.warning("model_result is None or False")
            flash('Error al entrenar el modelo', 'error')
            return redirect(url_for('training.sklearn_regression'))
        
    except Exception as e:
        flash(f'Error al entrenar el modelo: {str(e)}', 'error')
        return redirect(url_for('training.sklearn_regression'))

# Ruta temporal para debug
@training_bp.route('/debug_regression')
def debug_regression():
    """Crear resultados de regresión para debug."""
    session['regression_results'] = {
        'model_info': {
            'model_type': 'debug_model',
            'file_used': 'debug_file.csv', 
            'target_column': 'debug_target',
            'feature_columns': ['debug_feature1', 'debug_feature2'],
            'n_features': 2
        },
        'main_metrics': {
            'train_mse': 0.5678,
            'test_mse': 0.6789,
            'train_r2': 0.9123,
            'test_r2': 0.8456,
            'train_mae': 0.3456,
            'test_mae': 0.4567
        },
        'preparation_info': {
            'original_rows': 500,
            'rows_after_cleaning': 480,
            'dropped_rows': 20,
            'features_count': 2,
            'target_type': 'float64'
        }
    }
    
    flash('Resultados de debug creados directamente', 'info')
    return redirect(url_for('training.regression_results'))

@training_bp.route('/sklearn_classification')
def sklearn_classification():
    """Página para entrenar modelos de clasificación con sklearn."""
    if 'uploaded_file' not in session:
        flash('Debe cargar un archivo primero', 'error')
        return redirect(url_for('main.index'))
    
    try:
        # Obtener configuración de columnas desde la sesión
        target_column = session.get('target_column')
        predictor_columns = session.get('predictor_columns', [])
        
        current_app.logger.info(f"=== SKLEARN CLASSIFICATION DEBUG ===")
        current_app.logger.info(f"target_column from session: {target_column}")
        current_app.logger.info(f"predictor_columns from session: {predictor_columns}")
        
        # Usar la nueva función utilitaria para encontrar el archivo correcto
        file_to_use, df, success = get_training_file(target_column, predictor_columns)
        
        if not success:
            flash(f'La columna target "{target_column}" ya no existe en ningún archivo. Por favor, vuelve a seleccionar las columnas.', 'warning')
            session.pop('target_column', None)
            session.pop('predictor_columns', None)
            return redirect(url_for('data.select_columns'))
        
        # ANÁLISIS DETALLADO DE COLUMNAS DISPONIBLES
        columns = df.columns.tolist()
        
        # Usar funciones de matching mejoradas en lugar de comparación directa
        column_matches = match_column_names(predictor_columns, columns)
        available_predictors = [col for col in predictor_columns if col in column_matches]
        missing_predictors = [col for col in predictor_columns if col not in column_matches]
        
        current_app.logger.info(f"=== COLUMN AVAILABILITY ANALYSIS ===")
        current_app.logger.info(f"Original predictor columns: {len(predictor_columns)}")
        current_app.logger.info(f"Available predictor columns: {len(available_predictors)}")
        current_app.logger.info(f"Missing predictor columns: {len(missing_predictors)}")
        current_app.logger.info(f"Available: {available_predictors}")
        current_app.logger.info(f"Missing: {missing_predictors}")
        
        # Mostrar información detallada al usuario
        if missing_predictors:
            missing_count = len(missing_predictors)
            total_count = len(predictor_columns)
            flash(f'Atención: {missing_count}/{total_count} columnas predictoras no están disponibles en el archivo actual. Usando {len(available_predictors)} columnas disponibles.', 'warning')
            flash(f'DEBUG: Las primeras 5 columnas de tu selección: {predictor_columns[:5]}', 'info')
            flash(f'DEBUG: Las primeras 5 columnas que faltan: {missing_predictors[:5]}', 'info')
            if missing_count <= 5:
                flash(f'Columnas no disponibles: {", ".join(missing_predictors)}', 'info')
            else:
                flash(f'Algunas columnas no disponibles: {", ".join(missing_predictors[:5])}... y {missing_count-5} más.', 'info')
        
        # Actualizar predictor_columns con solo las disponibles
        predictor_columns = available_predictors
        
        # Si no hay columnas válidas, redirigir a selección de columnas
        if not target_column or not predictor_columns:
            flash('Por favor, selecciona primero las columnas target y predictoras.', 'info')
            return redirect(url_for('data.select_columns'))
        
        # VERIFICAR COMPATIBILIDAD CON CLASIFICACIÓN
        target_analysis = analyze_target_compatibility(df, target_column)
        
        if not target_analysis.get('classification_compatible', False):
            availability_messages = get_model_availability_message(target_analysis)
            classification_msg = availability_messages['classification']
            
            flash(f'La variable target seleccionada no es compatible con modelos de clasificación. {classification_msg["message"]}', 'warning')
            
            # Si es compatible con regresión, sugerir esa opción
            if target_analysis.get('regression_compatible', False):
                flash('Esta variable es compatible con modelos de regresión. Te redirigimos allí.', 'info')
                return redirect(url_for('training.sklearn_regression'))
            else:
                flash('Por favor, selecciona una variable target diferente.', 'info')
                return redirect(url_for('data.select_columns'))
        
        # Si hay una advertencia pero es compatible, mostrarla
        availability_messages = get_model_availability_message(target_analysis)
        if availability_messages['classification']['status'] == 'available':
            flash(f"Nota: {availability_messages['classification']['message']}", 'info')
        
        # Generar información de preparación para el template
        preparation_info = {
            'original_rows': len(df),
            'rows_after_cleaning': len(df),
            'dropped_rows': 0,
            'features_count': len(predictor_columns),
            'target_unique_values': df[target_column].nunique() if target_column else 0,
            'feature_types': {},
            'target_type': 'Categórico' if target_column and df[target_column].dtype == 'object' else 'Numérico'
        }
        
        # Analizar tipos de características
        for col in predictor_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    preparation_info['feature_types'][col] = f'categorical ({unique_count} categorías únicas)'
                else:
                    preparation_info['feature_types'][col] = f'numeric ({df[col].dtype})'
        
        # Definir modelos disponibles para clasificación
        available_models = {
            'decision_tree': {
                'name': 'Árbol de Decisión',
                'description': 'Modelo basado en reglas de decisión',
                'params': {
                    'max_depth': ['Sin límite', 3, 5, 10, 15, 20]
                }
            },
            'random_forest': {
                'name': 'Random Forest', 
                'description': 'Conjunto de árboles de decisión',
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': ['Sin límite', 3, 5, 10, 15, 20]
                }
            },
            'svm': {
                'name': 'Support Vector Machine',
                'description': 'Clasificador basado en vectores de soporte',
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                }
            },
            'knn': {
                'name': 'K-Nearest Neighbors',
                'description': 'Clasificador basado en vecinos cercanos',
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15]
                }
            },
            'logistic_regression': {
                'name': 'Regresión Logística',
                'description': 'Modelo lineal para clasificación',
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'naive_bayes': {
                'name': 'Naive Bayes',
                'description': 'Clasificador probabilístico',
                'params': {}
            }
        }
        
        return render_template('sklearn_classification.html', 
                             columns=columns, 
                             preparation_info=preparation_info,
                             available_models=available_models,
                             filename=os.path.basename(file_to_use),
                             target_column=target_column,
                             predictor_columns=predictor_columns,
                             columns_info={
                                 'original_count': len(session.get('predictor_columns', [])),
                                 'available_count': len(available_predictors),
                                 'missing_count': len(missing_predictors),
                                 'missing_columns': missing_predictors[:10]  # Solo primeras 10
                             })
    except Exception as e:
        current_app.logger.error(f"Error in sklearn_classification: {str(e)}")
        flash(f'Error al cargar el archivo: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@training_bp.route('/train_classification', methods=['POST'])
def train_classification():
    """Entrenar modelo de clasificación."""
    try:
        model_type = request.form.get('model_type')
        target_column = request.form.get('target_column')
        feature_columns = request.form.getlist('feature_columns')
        
        current_app.logger.info(f"=== TRAIN CLASSIFICATION DEBUG ===")
        current_app.logger.info(f"model_type: {model_type}")
        current_app.logger.info(f"target_column: {target_column}")
        current_app.logger.info(f"feature_columns: {feature_columns}")
        current_app.logger.info(f"All form data: {dict(request.form)}")
        
        if not model_type or not target_column or not feature_columns:
            flash('Debe seleccionar el tipo de modelo, la variable objetivo y las características', 'error')
            return redirect(url_for('training.sklearn_classification'))
        
        # Usar la función utilitaria para encontrar el archivo correcto
        file_to_use, df, success = get_training_file(target_column, feature_columns)
        
        if not success:
            flash(f'Las columnas seleccionadas ya no existen en ningún archivo disponible', 'error')
            return redirect(url_for('training.sklearn_classification'))
        
        # Mapear los nombres de columnas seleccionadas a los nombres reales en el archivo
        all_selected_columns = [target_column] + feature_columns
        column_matches = match_column_names(all_selected_columns, df.columns.tolist())
        
        # Mapear target_column y feature_columns a nombres reales
        real_target_column = column_matches.get(target_column, target_column)
        real_feature_columns = [column_matches.get(col, col) for col in feature_columns if col in column_matches]
        
        current_app.logger.info(f"Mapped target_column: {target_column} -> {real_target_column}")
        current_app.logger.info(f"Mapped feature_columns: {len(real_feature_columns)} columns")
        
        # Preparar los datos para entrenamiento (incluyendo codificación de categóricas)
        try:
            X_processed, y_processed, encoders, target_encoder, preparation_info = prepare_data_for_training(
                df, real_target_column, real_feature_columns, task_type='classification'
            )
            current_app.logger.info(f"Data prepared successfully: {preparation_info}")
            current_app.logger.info(f"X_processed shape: {X_processed.shape if X_processed is not None else 'None'}")
            current_app.logger.info(f"y_processed shape: {y_processed.shape if y_processed is not None else 'None'}")
            current_app.logger.info(f"real_target_column used: {real_target_column}")
            current_app.logger.info(f"real_feature_columns count: {len(real_feature_columns)}")
        except Exception as e:
            current_app.logger.error(f"Error preparing data: {e}")
            flash(f'Error preparando los datos: {str(e)}', 'error')
            return redirect(url_for('training.sklearn_classification'))
        
        models_dir = current_app.config['MODELS_FOLDER']
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_to_use)
        
        model_result = None
        
        try:
            if model_type == 'decision_tree':
                max_depth = request.form.get('max_depth')
                current_app.logger.info(f"Decision Tree max_depth from form: '{max_depth}' (type: {type(max_depth)})")
                max_depth = None if max_depth in ['Sin límite', 'None', None, ''] else int(max_depth)
                current_app.logger.info(f"Decision Tree max_depth processed: {max_depth}")
                current_app.logger.info(f"About to train decision tree with X shape: {X_processed.shape}, y shape: {y_processed.shape}")
                model_result = train_model(X_processed, y_processed, 'decision_tree', max_depth=max_depth)
            elif model_type == 'random_forest':
                n_estimators_param = request.form.get('n_estimators', '100')
                max_depth = request.form.get('max_depth')
                current_app.logger.info(f"Random Forest n_estimators: '{n_estimators_param}', max_depth: '{max_depth}'")
                n_estimators = int(n_estimators_param) if n_estimators_param and n_estimators_param not in ['None', ''] else 100
                max_depth = None if max_depth in ['Sin límite', 'None', None, ''] else int(max_depth)
                current_app.logger.info(f"Random Forest processed - n_estimators: {n_estimators}, max_depth: {max_depth}")
                model_result = train_model(X_processed, y_processed, 'random_forest', n_estimators=n_estimators, max_depth=max_depth)
            elif model_type == 'logistic_regression':
                C_param = request.form.get('C', '1.0')
                current_app.logger.info(f"Logistic Regression C param: '{C_param}'")
                C = float(C_param) if C_param and C_param not in ['None', ''] else 1.0
                model_result = train_model(X_processed, y_processed, 'logistic_regression', C=C)
            elif model_type == 'naive_bayes':
                model_result = train_model(X_processed, y_processed, 'naive_bayes')
            elif model_type == 'knn':
                n_neighbors_param = request.form.get('n_neighbors', '5')
                current_app.logger.info(f"KNN n_neighbors param: '{n_neighbors_param}'")
                n_neighbors = int(n_neighbors_param) if n_neighbors_param and n_neighbors_param not in ['None', ''] else 5
                model_result = train_model(X_processed, y_processed, 'knn', n_neighbors=n_neighbors)
            elif model_type == 'svm':
                C_param = request.form.get('C', '1.0')
                kernel = request.form.get('kernel', 'rbf')
                current_app.logger.info(f"SVM C param: '{C_param}', kernel: '{kernel}'")
                C = float(C_param) if C_param and C_param not in ['None', ''] else 1.0
                model_result = train_model(X_processed, y_processed, 'svm', C=C, kernel=kernel)
            else:
                flash('Tipo de modelo no válido', 'error')
                return redirect(url_for('training.sklearn_classification'))
        except ValueError as ve:
            current_app.logger.error(f"Error converting parameters: {str(ve)}")
            flash(f'Error en los parámetros del modelo: {str(ve)}. Verifica que todos los valores numéricos sean válidos.', 'error')
            return redirect(url_for('training.sklearn_classification'))
        except Exception as pe:
            current_app.logger.error(f"Error preparing model parameters: {str(pe)}")
            flash(f'Error preparando el modelo: {str(pe)}', 'error')
            return redirect(url_for('training.sklearn_classification'))
        
        if model_result:
            model, results = model_result
            
            current_app.logger.info(f"=== TRAINING SUCCESSFUL ===")
            current_app.logger.info(f"Model: {model}")
            current_app.logger.info(f"Results keys: {list(results.keys()) if results else 'None'}")
            
            # Limpiar datos temporales anteriores antes de guardar el nuevo modelo
            old_temp_path = session.get('temp_model_path')
            if old_temp_path and os.path.exists(old_temp_path):
                try:
                    os.remove(old_temp_path)
                    current_app.logger.info(f"Cleaned up previous temporary model: {old_temp_path}")
                except Exception as e:
                    current_app.logger.warning(f"Could not clean up previous temp file: {e}")
            
            # Limpiar resultados anteriores de la sesión para evitar conflictos
            SessionService.clear_large_results('classification_results')
            
            # Guardar el modelo temporalmente usando joblib para evitar problemas de serialización JSON
            temp_model_id = str(uuid.uuid4())[:8]
            temp_model_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'temp_model_{temp_model_id}.joblib')
            
            try:
                # Asegurar que el directorio temp_results existe
                os.makedirs(current_app.config['TEMP_RESULTS_FOLDER'], exist_ok=True)
                joblib.dump(model, temp_model_path)
                
                # Actualizar la sesión con el nuevo modelo
                session['temp_model_id'] = temp_model_id
                session['temp_model_path'] = temp_model_path
                session['temp_model_timestamp'] = datetime.now().isoformat()
                session['temp_model_type'] = 'classification'
                current_app.logger.info(f"Trained classification model stored temporarily: {temp_model_path}")
            except Exception as e:
                current_app.logger.error(f"Error storing temporary model: {e}")
                session.pop('temp_model_id', None)
                session.pop('temp_model_path', None)
                session.pop('temp_model_timestamp', None)
                session.pop('temp_model_type', None)
            
            # Organizar métricas principales para compatibilidad con save_model
            main_metrics = {
                'test_accuracy': results.get('test_accuracy', 0),
                'test_precision_macro': results.get('test_precision_macro', 0),
                'test_recall_macro': results.get('test_recall_macro', 0),
                'test_f1_macro': results.get('test_f1_macro', 0),
                'train_accuracy': results.get('train_accuracy', 0)
            }
            
            # Preparar datos para guardar usando el mismo formato que regresión
            classification_results = {
                'model_info': {
                    'model_type': model_type,
                    'file_used': os.path.basename(file_to_use),
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'n_features': len(feature_columns),
                    'n_classes': results.get('n_classes', 0)
                },
                'main_metrics': main_metrics,
                'preparation_info': preparation_info,
                'results': results  # Mantener results completo para la vista
            }
            
            # Guardar usando SessionService (para save_model) Y sistema de archivos temporales (para results page)
            storage_success = SessionService.store_large_results('classification_results', classification_results)
            
            # También guardar en formato antiguo para compatibilidad con classification_results page
            training_results = {
                'model_type': model_type,
                'results': results,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'preparation_info': preparation_info,
                'file_used': os.path.basename(file_to_use)
            }
            unique_id = save_training_results(training_results)
            
            current_app.logger.info(f"=== RESULTS SAVED ===")
            current_app.logger.info(f"SessionService storage success: {storage_success}")
            current_app.logger.info(f"Temp file storage ID: {unique_id}")
            current_app.logger.info(f"Main metrics stored: {main_metrics}")
            
            if storage_success:
                # Redirigir con el ID único
                return redirect(url_for('training.classification_results', results_id=unique_id))
            else:
                current_app.logger.error("Failed to save classification results to SessionService")
                flash('Error al guardar los resultados del entrenamiento', 'error')
                return redirect(url_for('training.sklearn_classification'))
        else:
            current_app.logger.error(f"=== TRAINING FAILED ===")
            current_app.logger.error(f"model_result is: {model_result}")
            flash('Error al entrenar el modelo', 'error')
            return redirect(url_for('training.sklearn_classification'))
        
    except Exception as e:
        flash(f'Error al entrenar el modelo: {str(e)}', 'error')
        return redirect(url_for('training.sklearn_classification'))

@training_bp.route('/classification_results')
def classification_results():
    """Página para mostrar los resultados del entrenamiento de clasificación"""
    # Obtener el ID de resultados desde la URL o la sesión
    results_id = request.args.get('results_id')
    
    training_data = None
    
    if results_id:
        # Cargar desde archivo temporal
        training_data = load_training_results(results_id)
        current_app.logger.info(f"Loaded results from file with ID: {results_id}")
    
    # Si no se puede cargar desde archivo, intentar desde sesión (fallback)
    if not training_data and 'last_training_results' in session:
        training_data = session['last_training_results']
        current_app.logger.info("Loaded results from session (fallback)")
    
    # Si no hay datos disponibles
    if not training_data:
        flash('No hay resultados de entrenamiento para mostrar', 'info')
        current_app.logger.error("No training results available")
        return redirect(url_for('training.sklearn_classification'))
    
    # Formatear los resultados para mostrar
    results = training_data['results']
    
    # Organizar métricas principales
    main_metrics = {
        'train_accuracy': results.get('train_accuracy', 0),
        'test_accuracy': results.get('test_accuracy', 0),
        'test_precision_macro': results.get('test_precision_macro', 0),
        'test_recall_macro': results.get('test_recall_macro', 0),
        'test_f1_macro': results.get('test_f1_macro', 0)
    }
    
    # Información del modelo
    model_info = {
        'model_type': training_data['model_type'],
        'target_column': training_data['target_column'],
        'feature_columns': training_data['feature_columns'],
        'file_used': training_data['file_used'],
        'n_classes': results.get('n_classes', 0)
    }
    
    # Matriz de confusión si existe
    confusion_matrix = results.get('confusion_matrix')
    if confusion_matrix is not None:
        confusion_matrix = confusion_matrix.tolist() if hasattr(confusion_matrix, 'tolist') else confusion_matrix
    
    # Reporte de clasificación detallado
    classification_report = results.get('classification_report', {})
    
    context = {
        'main_metrics': main_metrics,
        'model_info': model_info,
        'confusion_matrix': confusion_matrix,
        'classification_report': classification_report,
        'preparation_info': training_data.get('preparation_info', {})
    }
    
    return render_template('classification_results.html', **context)

@training_bp.route('/regression_results')
def regression_results():
    """Mostrar resultados detallados del entrenamiento de regresión."""
    current_app.logger.info("Accessing regression results page")
    
    # Usar el nuevo sistema de almacenamiento para recuperar resultados
    training_data = SessionService.get_large_results('regression_results')
    
    if not training_data:
        current_app.logger.warning("No regression results found in session or file storage")
        flash('No hay resultados de regresión para mostrar.', 'warning')
        return redirect(url_for('training.sklearn_regression'))
    
    current_app.logger.info("Regression results found, rendering template")
    
    # Preparar contexto para el template
    context = {
        'model_info': training_data.get('model_info', {}),
        'main_metrics': training_data.get('main_metrics', {}),
        'preparation_info': training_data.get('preparation_info', {}),
        'feature_importance': training_data.get('feature_importance', None)
    }
    
    current_app.logger.info(f"Rendering regression results with context keys: {context.keys()}")
    return render_template('regression_results.html', **context)

@training_bp.route('/save_model', methods=['POST'])
def save_model():
    """Guardar el modelo entrenado con joblib"""
    try:
        # Limpiar archivos temporales antiguos usando sistema centralizado
        from app.utils.system_cleanup import cleanup_manager
        cleanup_manager.manual_cleanup_now(max_age_hours=1)
        
        # Obtener información del modelo desde el formulario
        model_type = request.form.get('model_type')
        model_name = request.form.get('model_name', '')
        task_type = request.form.get('task_type')  # 'classification' o 'regression'
        results_id = request.form.get('results_id')
        
        current_app.logger.info(f"=== SAVE MODEL DEBUG ===")
        current_app.logger.info(f"model_type: {model_type}")
        current_app.logger.info(f"model_name: {model_name}")
        current_app.logger.info(f"task_type: {task_type}")
        current_app.logger.info(f"results_id: {results_id}")
        
        if not model_type or not task_type:
            flash('Información del modelo incompleta', 'error')
            return redirect(request.referrer or url_for('training.training'))
        
        # Generar nombre automático si no se proporciona
        if not model_name.strip():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{task_type}_{timestamp}"
        else:
            # Limpiar el nombre del modelo
            model_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-')).strip()
            if not model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{model_type}_{task_type}_{timestamp}"
        
        # Crear directorios necesarios
        models_dir = current_app.config['MODELS_FOLDER']
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Intentar obtener el modelo entrenado desde archivo temporal
        temp_model_id = session.get('temp_model_id')
        temp_model_path = session.get('temp_model_path')
        temp_model_timestamp = session.get('temp_model_timestamp')
        temp_model_type = session.get('temp_model_type')
        model = None
        
        current_app.logger.info(f"Temporary model info - ID: {temp_model_id}, Path: {temp_model_path}, Type: {temp_model_type}, Timestamp: {temp_model_timestamp}")
        
        # Validar que el modelo temporal corresponde al tipo solicitado
        if temp_model_id and temp_model_path and os.path.exists(temp_model_path):
            # Verificar que el tipo de modelo coincide
            if temp_model_type and temp_model_type != task_type:
                current_app.logger.warning(f"Model type mismatch: expected {task_type}, found {temp_model_type}")
                # Limpiar referencias inconsistentes
                session.pop('temp_model_id', None)
                session.pop('temp_model_path', None)
                session.pop('temp_model_timestamp', None)
                session.pop('temp_model_type', None)
            else:
                # Verificar que el modelo no sea muy antiguo (más de 30 minutos)
                if temp_model_timestamp:
                    try:
                        model_time = datetime.fromisoformat(temp_model_timestamp)
                        if datetime.now() - model_time > timedelta(minutes=30):
                            current_app.logger.warning(f"Temporary model is too old: {temp_model_timestamp}")
                            # Limpiar modelo antiguo
                            try:
                                os.remove(temp_model_path)
                                current_app.logger.info(f"Removed old temporary model: {temp_model_path}")
                            except (OSError, FileNotFoundError):
                                pass
                            session.pop('temp_model_id', None)
                            session.pop('temp_model_path', None)
                            session.pop('temp_model_timestamp', None)
                            session.pop('temp_model_type', None)
                        else:
                            try:
                                model = joblib.load(temp_model_path)
                                current_app.logger.info(f"Loaded valid temporary model: {temp_model_path}")
                            except Exception as e:
                                current_app.logger.error(f"Error loading temporary model: {e}")
                                model = None
                    except Exception as e:
                        current_app.logger.warning(f"Could not parse timestamp: {e}")
                        try:
                            model = joblib.load(temp_model_path)
                            current_app.logger.info(f"Loaded temporary model despite timestamp issue: {temp_model_path}")
                        except Exception as load_e:
                            current_app.logger.error(f"Error loading temporary model: {load_e}")
                            model = None
                else:
                    try:
                        model = joblib.load(temp_model_path)
                        current_app.logger.info(f"Loaded temporary model without timestamp: {temp_model_path}")
                    except Exception as e:
                        current_app.logger.error(f"Error loading temporary model: {e}")
                        model = None
        
        training_data = None
        
        # Si no hay modelo temporal, intentar cargar desde resultados guardados
        if not model and results_id:
            training_data = load_training_results(results_id)
            if training_data:
                current_app.logger.info("Loaded training data from file, but no model object available")
        
        # También intentar desde los resultados almacenados en sesión
        if not model and task_type == 'classification':
            classification_results = SessionService.get_large_results('classification_results')
            if classification_results:
                training_data = classification_results
                current_app.logger.info("Loaded classification results from session storage")
        
        if not model and task_type == 'regression':
            regression_results = SessionService.get_large_results('regression_results')
            if regression_results:
                training_data = regression_results
                current_app.logger.info("Loaded regression results from session storage")
        
        # Si no hay modelo objeto, mostrar mensaje informativo
        if not model:
            current_app.logger.warning("No trained model object available in session")
            flash('El modelo ya no está disponible en la sesión actual. Los modelos solo se pueden guardar inmediatamente después del entrenamiento.', 'warning')
            
            # Aún así, guardar la información del modelo sin el objeto
            if training_data:
                metadata = {
                    'model_name': model_name,
                    'model_type': model_type,
                    'task_type': task_type,
                    'created_at': datetime.now().isoformat(),
                    'model_file': None,  # No hay archivo del modelo
                    'status': 'metadata_only',
                    'message': 'Solo metadatos guardados - modelo no disponible',
                    **training_data.get('model_info', {}),
                    'metrics': training_data.get('main_metrics', {}),
                    'preparation_info': training_data.get('preparation_info', {})
                }
                
                # Guardar solo metadatos
                metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                flash(f'Información del modelo guardada como "{model_name}" (solo metadatos)', 'info')
                flash('Para guardar el modelo completo, use el botón de guardar inmediatamente después del entrenamiento.', 'info')
            else:
                flash('No se encontraron datos del modelo para guardar', 'error')
            
            return redirect(request.referrer or url_for('prediction.saved_models'))
        
        # Guardar el modelo completo con joblib
        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        
        try:
            joblib.dump(model, model_file)
            current_app.logger.info(f"Model saved successfully to: {model_file}")
        except Exception as e:
            current_app.logger.error(f"Error saving model with joblib: {e}")
            flash(f'Error al guardar el modelo: {str(e)}', 'error')
            return redirect(request.referrer or url_for('training.training'))
        
        # Preparar metadatos del modelo
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'task_type': task_type,
            'created_at': datetime.now().isoformat(),
            'model_file': model_file,
            'model_size_mb': round(os.path.getsize(model_file) / (1024 * 1024), 2),
            'status': 'complete'
        }
        
        # Agregar información adicional según el tipo de tarea
        if training_data:
            metadata.update({
                **training_data.get('model_info', {}),
                'metrics': training_data.get('main_metrics', {}),
                'preparation_info': training_data.get('preparation_info', {})
            })
        elif task_type == 'classification':
            classification_results = SessionService.get_large_results('classification_results')
            if classification_results:
                metadata.update({
                    **classification_results.get('model_info', {}),
                    'metrics': classification_results.get('main_metrics', {}),
                    'preparation_info': classification_results.get('preparation_info', {})
                })
        elif task_type == 'regression':
            regression_results = SessionService.get_large_results('regression_results')
            if regression_results:
                metadata.update({
                    **regression_results.get('model_info', {}),
                    'metrics': regression_results.get('main_metrics', {}),
                    'preparation_info': regression_results.get('preparation_info', {})
                })
        
        # Guardar metadatos
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Limpiar archivo temporal si existe
        if temp_model_path and os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
                current_app.logger.info(f"Cleaned up temporary model file: {temp_model_path}")
            except Exception as e:
                current_app.logger.warning(f"Could not clean up temporary file: {e}")
        
        # Limpiar todas las variables de sesión relacionadas con el modelo temporal
        session.pop('temp_model_id', None)
        session.pop('temp_model_path', None)
        session.pop('temp_model_timestamp', None)
        session.pop('temp_model_type', None)
        
        # Limpiar también los resultados de sesión para evitar conflictos futuros
        SessionService.clear_large_results('classification_results')
        SessionService.clear_large_results('regression_results')
        
        flash(f'Modelo guardado exitosamente como "{model_name}"', 'success')
        current_app.logger.info(f"Model and metadata saved: {model_file}, {metadata_file}")
        current_app.logger.info("All temporary data and session variables cleared")
        
        return redirect(url_for('prediction.saved_models'))
        
    except Exception as e:
        current_app.logger.error(f"Error in save_model: {str(e)}")
        flash(f'Error al guardar el modelo: {str(e)}', 'error')
        return redirect(request.referrer or url_for('training.training'))

@training_bp.route('/saved_models')
def saved_models():
    """Redirige a la nueva página de modelos guardados con funcionalidad de predicción"""
    return redirect(url_for('prediction.saved_models'))

@training_bp.route('/download_model/<model_name>')
def download_model(model_name):
    """Descargar un modelo guardado"""
    try:
        models_dir = current_app.config['MODELS_FOLDER']
        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        
        if not os.path.exists(model_file):
            flash(f'El modelo "{model_name}" no existe', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        return send_file(model_file, as_attachment=True, download_name=f"{model_name}.joblib")
        
    except Exception as e:
        current_app.logger.error(f"Error downloading model {model_name}: {e}")
        flash(f'Error al descargar el modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@training_bp.route('/delete_model/<model_name>', methods=['POST'])
def delete_model(model_name):
    """Eliminar un modelo guardado"""
    try:
        models_dir = current_app.config['MODELS_FOLDER']
        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        deleted_files = []
        
        # Eliminar archivo del modelo
        if os.path.exists(model_file):
            os.remove(model_file)
            deleted_files.append("modelo")
        
        # Eliminar metadatos
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            deleted_files.append("metadatos")
        
        if deleted_files:
            flash(f'Modelo "{model_name}" eliminado ({", ".join(deleted_files)})', 'success')
        else:
            flash(f'El modelo "{model_name}" no se encontró', 'warning')
        
        return redirect(url_for('prediction.saved_models'))
        
    except Exception as e:
        current_app.logger.error(f"Error deleting model {model_name}: {e}")
        flash(f'Error al eliminar el modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@training_bp.route('/update_model_metadata/<model_name>')
def update_model_metadata(model_name):
    """Actualizar metadatos de un modelo sin métricas para mostrar mensaje informativo"""
    try:
        models_dir = current_app.config['MODELS_FOLDER']
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_file):
            flash(f'El modelo "{model_name}" no existe', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        # Cargar metadatos existentes
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Si ya tiene métricas completas, no hacer nada
        if metadata.get('metrics') and any(v for v in metadata['metrics'].values() if v is not None):
            flash(f'El modelo "{model_name}" ya tiene métricas', 'info')
            return redirect(url_for('prediction.saved_models'))
        
        # Agregar información básica para modelos sin métricas
        if metadata.get('task_type') == 'classification':
            # Agregar métricas vacías pero correctamente estructuradas para clasificación
            metadata['metrics'] = {
                'test_accuracy': None,
                'test_precision_macro': None,
                'test_recall_macro': None,
                'test_f1_macro': None
            }
            if 'model_info' not in metadata:
                metadata['model_info'] = {}
            if 'preparation_info' not in metadata:
                metadata['preparation_info'] = {
                    'message': 'Métricas no disponibles - modelo guardado desde sesión expirada'
                }
        elif metadata.get('task_type') == 'regression':
            # Agregar métricas vacías pero correctamente estructuradas para regresión
            metadata['metrics'] = {
                'test_r2': None,
                'test_mse': None,
                'test_mae': None
            }
            if 'model_info' not in metadata:
                metadata['model_info'] = {}
            if 'preparation_info' not in metadata:
                metadata['preparation_info'] = {
                    'message': 'Métricas no disponibles - modelo guardado desde sesión expirada'
                }
        
        # Guardar metadatos actualizados
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        flash(f'Metadatos actualizados para "{model_name}"', 'success')
        return redirect(url_for('prediction.saved_models'))
        
    except Exception as e:
        current_app.logger.error(f"Error updating model metadata {model_name}: {e}")
        flash(f'Error al actualizar metadatos: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@training_bp.route('/add_records/<model_name>')
def add_records(model_name):
    """Página para agregar nuevos registros a un modelo y reentrenarlo"""
    try:
        models_dir = current_app.config['MODELS_FOLDER']
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_file):
            flash(f'El modelo "{model_name}" no existe', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        # Cargar metadatos del modelo
        with open(metadata_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Verificar que el modelo tiene la información necesaria
        required_fields = ['target_column', 'feature_columns', 'model_type', 'task_type']
        missing_fields = [field for field in required_fields if not model_info.get(field)]
        
        if missing_fields:
            flash(f'El modelo no tiene información completa. Campos faltantes: {", ".join(missing_fields)}', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        # Obtener información sobre tipos de variables del dataset original
        file_used = model_info.get('file_used', '')
        target_column = model_info.get('target_column')
        feature_columns = model_info.get('feature_columns', [])
        
        # Inicializar variables
        column_types = {}
        sample_data = None
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        original_df = None
        
        # Primero, intentar usar la información ya guardada en preparation_info
        prep_info = model_info.get('preparation_info', {})
        if prep_info and 'feature_types' in prep_info:
            feature_types = prep_info['feature_types']
            target_type = prep_info.get('target_type', '')
            
            # Convertir los tipos guardados a formato legible
            for feature in feature_columns:
                if feature in feature_types:
                    type_info = feature_types[feature]
                    if 'categorical' in type_info.lower():
                        if 'encoded:' in type_info:
                            cats = type_info.split('encoded: ')[1].split(' ')[0]
                            column_types[feature] = f'Categórica ({cats} categorías)'
                        else:
                            column_types[feature] = 'Categórica'
                    elif 'numeric' in type_info.lower():
                        column_types[feature] = f'Numérica ({type_info.split("(")[1].split(")")[0]})'
                    else:
                        column_types[feature] = type_info
                else:
                    column_types[feature] = 'Tipo no determinado'
            
            # Tipo del target
            if target_column:
                if 'categorical' in target_type.lower():
                    unique_vals = prep_info.get('target_unique_values', 'N/A')
                    column_types[target_column] = f'Categórica ({unique_vals} valores únicos)'
                elif 'numeric' in target_type.lower():
                    column_types[target_column] = f'Numérica ({target_type.split("(")[1].split(")")[0]})'
                else:
                    column_types[target_column] = target_type
        
        # Cargar el dataset para obtener muestra y tipos adicionales si es necesario
        if file_used:
            current_app.logger.info(f"Looking for dataset file: {file_used}")
            possible_files = []
            
            # Buscar el archivo exacto
            file_path = os.path.join(uploads_dir, file_used)
            if os.path.exists(file_path):
                possible_files.append(file_path)
                current_app.logger.info(f"Found exact file: {file_path}")
            
            # Buscar versión limpia del archivo
            base_name = file_used.replace('.csv', '')
            cleaned_versions = [
                f"cleaned_{file_used}",
                f"{base_name}_cleaned.csv"
            ]
            
            for cleaned_file in cleaned_versions:
                cleaned_path = os.path.join(uploads_dir, cleaned_file)
                if os.path.exists(cleaned_path):
                    possible_files.append(cleaned_path)
                    current_app.logger.info(f"Found cleaned version: {cleaned_path}")
            
            # También buscar otras variaciones del nombre
            try:
                base_search = base_name.replace('_cleaned', '').replace('cleaned_', '')
                current_app.logger.info(f"Searching for variations of: {base_search}")
                
                for filename in os.listdir(uploads_dir):
                    if (base_search.lower() in filename.lower() and 
                        filename.endswith('.csv') and
                        filename not in [os.path.basename(f) for f in possible_files]):
                        full_path = os.path.join(uploads_dir, filename)
                        possible_files.append(full_path)
                        current_app.logger.info(f"Found variation: {full_path}")
            except Exception as e:
                current_app.logger.error(f"Error listing uploads directory: {e}")
            
            current_app.logger.info(f"Total possible files found: {len(possible_files)}")
            
            # Cargar el dataset para análisis
            for file_path in possible_files:
                try:
                    current_app.logger.info(f"Attempting to load: {file_path}")
                    
                    # Intentar diferentes configuraciones de lectura
                    # Probar primero con separadores más comunes
                    load_attempts = [
                        {'encoding': 'utf-8', 'sep': ','},
                        {'encoding': 'utf-8', 'sep': ';'},
                        {'encoding': 'latin-1', 'sep': ','},
                        {'encoding': 'latin-1', 'sep': ';'},
                        {'encoding': 'cp1252', 'sep': ','},
                        {'encoding': 'cp1252', 'sep': ';'},
                        {'encoding': 'utf-8', 'sep': '\t'},  # Tab separado
                        {'encoding': 'iso-8859-1', 'sep': ','},
                        {'encoding': 'iso-8859-1', 'sep': ';'},
                        {'encoding': 'utf-8', 'sep': ',', 'on_bad_lines': 'skip'},
                        {'encoding': 'utf-8', 'sep': ';', 'on_bad_lines': 'skip'},
                        {'encoding': 'latin-1', 'sep': ',', 'on_bad_lines': 'skip'},
                        {'encoding': 'latin-1', 'sep': ';', 'on_bad_lines': 'skip'},
                        {'encoding': 'utf-8', 'sep': ',', 'quoting': 1},  # QUOTE_ALL
                        {'encoding': 'utf-8', 'sep': ';', 'quoting': 1},
                        {'encoding': 'latin-1', 'sep': ',', 'quoting': 1},
                        {'encoding': 'latin-1', 'sep': ';', 'quoting': 1},
                    ]
                    
                    for attempt in load_attempts:
                        try:
                            original_df = pd.read_csv(file_path, **attempt)
                            
                            # Verificar que el DataFrame tenga sentido
                            if len(original_df.columns) > 0 and len(original_df) > 0:
                                # Verificar que las columnas esperadas existen
                                expected_columns = [target_column] + feature_columns
                                available_expected = [col for col in expected_columns if col in original_df.columns]
                                
                                if len(available_expected) >= len(expected_columns) * 0.5:  # Al menos 50% de las columnas esperadas
                                    current_app.logger.info(f"Successfully loaded dataset: {file_path} (shape: {original_df.shape}) with config: {attempt}")
                                    break
                                else:
                                    current_app.logger.info(f"Loaded {file_path} but missing expected columns. Found: {list(original_df.columns)}")
                                    original_df = None
                                    continue
                            else:
                                original_df = None
                                continue
                                
                        except (pd.errors.ParserError, UnicodeDecodeError, FileNotFoundError) as e:
                            current_app.logger.debug(f"Failed to load {file_path} with {attempt}: {type(e).__name__} - {str(e)}")
                            continue
                        except Exception as e:
                            current_app.logger.debug(f"Unexpected error loading {file_path} with {attempt}: {str(e)}")
                            continue
                    
                    if original_df is not None:
                        break
                        
                except (FileNotFoundError, PermissionError) as e:
                    current_app.logger.warning(f"File access error for {file_path}: {type(e).__name__} - {str(e)}")
                    continue
                except Exception as e:
                    current_app.logger.warning(f"Could not load {file_path} with any configuration: {e}")
                    continue
            
            if original_df is None:
                current_app.logger.warning(f"Could not load any dataset file for: {file_used}")
                # Crear un mensaje informativo para el usuario
                error_message = f'No se pudo cargar el archivo "{file_used}" para la vista previa.'
                
                if possible_files:
                    error_message += f' Se intentaron {len(possible_files)} archivo(s) con múltiples configuraciones de encoding y separadores.'
                else:
                    error_message += ' No se encontró el archivo en el directorio uploads.'
                
                sample_data = {
                    'error': True,
                    'message': error_message,
                    'files_tried': [os.path.basename(f) for f in possible_files] if possible_files else [],
                    'suggestions': 'Verifica que el archivo tenga el formato CSV correcto y las columnas esperadas.'
                }
            
            # Analizar tipos de datos si tenemos el dataset y no tenemos información previa
            if original_df is not None and not column_types:
                # Analizar variable target
                if target_column and target_column in original_df.columns:
                    col = original_df[target_column]
                    if col.dtype == 'object' or col.dtype.name == 'category':
                        unique_vals = col.nunique()
                        column_types[target_column] = f'Categórica ({unique_vals} valores únicos)'
                    elif col.dtype in ['int64', 'int32', 'float64', 'float32']:
                        col_clean = col.dropna()
                        if len(col_clean) > 0:
                            try:
                                if col.nunique() <= 10 and all(col_clean == col_clean.astype(int)):
                                    column_types[target_column] = f'Numérica discreta ({col.min()}-{col.max()})'
                                else:
                                    column_types[target_column] = f'Numérica continua ({col.min():.2f}-{col.max():.2f})'
                            except (ValueError, TypeError):
                                column_types[target_column] = f'Numérica ({col.min():.2f}-{col.max():.2f})'
                        else:
                            column_types[target_column] = 'Sin datos válidos'
                    else:
                        column_types[target_column] = 'Mixto'
                
                # Analizar características
                for feature in feature_columns:
                    if feature in original_df.columns:
                        col = original_df[feature]
                        if col.dtype == 'object' or col.dtype.name == 'category':
                            unique_vals = col.nunique()
                            column_types[feature] = f'Categórica ({unique_vals} categorías)'
                        elif col.dtype in ['int64', 'int32', 'float64', 'float32']:
                            col_clean = col.dropna()
                            if len(col_clean) > 0:
                                try:
                                    if col.nunique() <= 10 and all(col_clean == col_clean.astype(int)):
                                        column_types[feature] = f'Numérica discreta ({col.min()}-{col.max()})'
                                    else:
                                        column_types[feature] = f'Numérica continua'
                                except (ValueError, TypeError):
                                    column_types[feature] = 'Numérica'
                            else:
                                column_types[feature] = 'Sin datos válidos'
                                column_types[feature] = f'Numérica continua ({col.min():.2f}-{col.max():.2f})'
                        else:
                            column_types[feature] = 'Mixto'
                    else:
                        column_types[feature] = 'No disponible'
        
        # Obtener muestra de las primeras 5 filas del dataset
        if original_df is not None:
            # Seleccionar solo las columnas relevantes (target + features) en ese orden
            all_columns = [target_column] + feature_columns if target_column else feature_columns
            available_columns = [col for col in all_columns if col in original_df.columns]
            current_app.logger.info(f"Available columns for sample: {available_columns}")
            current_app.logger.info(f"Original DataFrame shape: {original_df.shape}")
            if available_columns:
                sample_df = original_df[available_columns].head(5)
                # Convertir a formato que se pueda mostrar en HTML
                sample_data = {
                    'columns': available_columns,
                    'rows': sample_df.values.tolist()
                }
                current_app.logger.info(f"Sample data created with {len(sample_data['rows'])} rows and {len(sample_data['columns'])} columns")
            else:
                current_app.logger.warning(f"No available columns found. Target: {target_column}, Features: {feature_columns}")
        else:
            current_app.logger.warning(f"Could not load dataset for file: {file_used}")
        
        # Si no pudimos obtener tipos, usar valores por defecto
        if not column_types:
            if target_column:
                column_types[target_column] = 'Tipo no determinado'
            for feature in feature_columns:
                column_types[feature] = 'Tipo no determinado'
        
        # Preparar información del modelo para el template
        model_info['model_name'] = model_name
        model_info['column_types'] = column_types
        model_info['sample_data'] = sample_data
        
        current_app.logger.info(f"Sending to template - sample_data: {sample_data is not None}")
        if sample_data:
            current_app.logger.info(f"Sample data structure: columns={len(sample_data.get('columns', []))}, rows={len(sample_data.get('rows', []))}")
        
        return render_template('add_records.html', model_info=model_info)
        
    except Exception as e:
        current_app.logger.error(f"Error accessing add_records for model {model_name}: {e}")
        flash(f'Error al acceder al modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@training_bp.route('/retrain_model_with_records/<model_name>', methods=['POST'])
def retrain_model_with_records(model_name):
    """Reentrenar un modelo con nuevos registros agregados"""
    try:
        # Obtener datos del formulario
        new_records_data = request.form.get('new_records_data')
        new_model_name = request.form.get('new_model_name', '').strip()
        
        if not new_records_data:
            flash('No se proporcionaron nuevos registros', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Parsear los nuevos registros
        try:
            new_records = json.loads(new_records_data)
        except json.JSONDecodeError:
            flash('Error al procesar los nuevos registros', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        if not new_records:
            flash('No se proporcionaron registros válidos', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Cargar metadatos del modelo original
        models_dir = current_app.config['MODELS_FOLDER']
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_file):
            flash(f'El modelo "{model_name}" no existe', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            original_model_info = json.load(f)
        
        # Generar nombre para el nuevo modelo si no se proporciona
        if not new_model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_name = f"{model_name}_retrained_{timestamp}"
        else:
            new_model_name = "".join(c for c in new_model_name if c.isalnum() or c in ('_', '-')).strip()
            if not new_model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_model_name = f"{model_name}_retrained_{timestamp}"
        
        # Buscar el dataset original
        file_used = original_model_info.get('file_used', '')
        target_column = original_model_info.get('target_column')
        feature_columns = original_model_info.get('feature_columns', [])
        
        # Intentar encontrar el archivo de datos original
        original_df = None
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        possible_files = []
        
        if file_used:
            # Buscar el archivo original
            for ext in ['', '.csv', '_cleaned.csv']:
                test_file = os.path.join(uploads_dir, file_used + ext)
                if os.path.exists(test_file):
                    possible_files.append(test_file)
            
            # También buscar sin extensión
            for filename in os.listdir(uploads_dir):
                if file_used.lower() in filename.lower() and filename.endswith('.csv'):
                    possible_files.append(os.path.join(uploads_dir, filename))
        
        # Cargar el dataset original con detección de separador
        for file_path in possible_files:
            # Intentar diferentes combinaciones de encoding y separador
            read_attempts = [
                {'encoding': 'utf-8', 'sep': ','},
                {'encoding': 'utf-8', 'sep': ';'},
                {'encoding': 'latin-1', 'sep': ','},
                {'encoding': 'latin-1', 'sep': ';'},
                {'encoding': 'cp1252', 'sep': ','},
                {'encoding': 'cp1252', 'sep': ';'},
                {'encoding': 'iso-8859-1', 'sep': ','},
                {'encoding': 'iso-8859-1', 'sep': ';'},
            ]
            
            for attempt in read_attempts:
                try:
                    original_df = pd.read_csv(file_path, **attempt)
                    # Verificar que tiene contenido válido
                    if len(original_df.columns) > 1 and len(original_df) > 0:
                        current_app.logger.info(f"Successfully loaded original dataset: {file_path} with {attempt}")
                        break
                except (UnicodeDecodeError, pd.errors.ParserError, Exception) as e:
                    continue
            
            # Si se cargó exitosamente, salir del loop de archivos
            if original_df is not None and len(original_df.columns) > 1:
                break
        
        if original_df is None:
            flash(f'No se pudo encontrar el dataset original. Archivos buscados: {possible_files}', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Verificar que las columnas necesarias existen en el dataset original
        missing_columns = []
        if target_column not in original_df.columns:
            missing_columns.append(target_column)
        for col in feature_columns:
            if col not in original_df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            flash(f'Columnas faltantes en el dataset original: {missing_columns}', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Convertir nuevos registros a DataFrame
        new_records_df_data = []
        for record in new_records:
            row = {}
            # Mapear target_value al nombre de la columna target
            if 'target_value' in record:
                row[target_column] = record['target_value']
            
            # Mapear las características
            for feature in feature_columns:
                if feature in record:
                    row[feature] = record[feature]
            
            new_records_df_data.append(row)
        
        new_records_df = pd.DataFrame(new_records_df_data)
        
        # Validar y normalizar tipos de datos antes de combinar
        try:
            # Analizar tipos de datos en el dataset original
            original_types = {}
            for col in [target_column] + feature_columns:
                if col in original_df.columns:
                    col_data = original_df[col].dropna()
                    if len(col_data) > 0:
                        # Determinar si es numérico o categórico
                        try:
                            # Intentar convertir a float para ver si es numérico
                            pd.to_numeric(col_data, errors='raise')
                            original_types[col] = 'numeric'
                        except (ValueError, TypeError):
                            original_types[col] = 'categorical'
                    else:
                        original_types[col] = 'unknown'
            
            current_app.logger.info(f"Original data types detected: {original_types}")
            
            # Convertir nuevos registros para que coincidan con los tipos originales
            for col in new_records_df.columns:
                if col in original_types:
                    if original_types[col] == 'numeric':
                        # Convertir a numérico, manejando errores
                        try:
                            new_records_df[col] = pd.to_numeric(new_records_df[col], errors='coerce')
                            current_app.logger.info(f"Converted column {col} to numeric")
                        except Exception as e:
                            current_app.logger.warning(f"Could not convert {col} to numeric: {e}")
                    elif original_types[col] == 'categorical':
                        # Asegurar que sea string
                        new_records_df[col] = new_records_df[col].astype(str)
                        current_app.logger.info(f"Converted column {col} to categorical (string)")
            
            # Verificar que los nuevos registros no tienen NaN críticos después de la conversión
            critical_nans = new_records_df.isnull().sum()
            if critical_nans.any():
                current_app.logger.warning(f"NaN values found in new records after conversion: {critical_nans[critical_nans > 0].to_dict()}")
        
        except Exception as e:
            current_app.logger.error(f"Error during data type validation: {e}")
            flash(f'Error validando tipos de datos: {str(e)}', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Combinar datasets
        try:
            combined_df = pd.concat([original_df, new_records_df], ignore_index=True)
            current_app.logger.info(f"Combined dataset shape: {combined_df.shape}")
            current_app.logger.info(f"Original: {len(original_df)}, New: {len(new_records_df)}, Combined: {len(combined_df)}")
            
            # Validar que el dataset combinado no esté vacío
            if len(combined_df) == 0:
                raise ValueError("El dataset combinado está vacío")
            
            # Verificar que tenemos las columnas necesarias
            required_columns = [target_column] + feature_columns
            missing_cols = [col for col in required_columns if col not in combined_df.columns]
            if missing_cols:
                raise ValueError(f"Columnas faltantes en el dataset combinado: {missing_cols}")
            
            # Verificar tipos finales en el dataset combinado
            for col in [target_column] + feature_columns:
                if col in combined_df.columns:
                    current_app.logger.info(f"Final type for {col}: {combined_df[col].dtype}, unique values: {combined_df[col].nunique()}")
                    
        except Exception as e:
            current_app.logger.error(f"Error combining datasets: {e}")
            flash(f'Error combinando datasets: {str(e)}', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Preparar los datos para entrenamiento
        try:
            # Usar el task_type del modelo original
            original_task_type = original_model_info.get('task_type', 'regression')
            current_app.logger.info(f"Using original task_type: {original_task_type}")
            
            X_processed, y_processed, encoders, target_encoder, preparation_info = prepare_data_for_training(
                combined_df, target_column, feature_columns, task_type=original_task_type
            )
            
            current_app.logger.info(f"Processed data shapes: X={X_processed.shape}, y={y_processed.shape}")
            
            # Verificar que tenemos suficientes datos
            if len(X_processed) == 0:
                raise ValueError("No quedan datos después del procesamiento")
            
            if len(X_processed) < 2:
                raise ValueError(f"Insuficientes datos para entrenamiento. Solo se tienen {len(X_processed)} registros. Se necesitan al menos 2.")
                
        except Exception as e:
            current_app.logger.error(f"Error preparing combined data: {e}")
            flash(f'Error preparando los datos combinados: {str(e)}', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        # Reentrenar el modelo usando los mismos parámetros que el original
        model_type = original_model_info.get('model_type')
        task_type = original_model_info.get('task_type')
        
        model_result = None
        
        try:
            if task_type == 'classification':
                # Usar parámetros por defecto para clasificación
                if model_type == 'decision_tree':
                    model_result = train_model(X_processed, y_processed, 'decision_tree')
                elif model_type == 'random_forest':
                    model_result = train_model(X_processed, y_processed, 'random_forest')
                elif model_type == 'logistic_regression':
                    model_result = train_model(X_processed, y_processed, 'logistic_regression')
                elif model_type == 'naive_bayes':
                    model_result = train_model(X_processed, y_processed, 'naive_bayes')
                elif model_type == 'knn':
                    model_result = train_model(X_processed, y_processed, 'knn')
                elif model_type == 'svm':
                    model_result = train_model(X_processed, y_processed, 'svm')
                else:
                    flash(f'Tipo de modelo de clasificación no soportado: {model_type}', 'error')
                    return redirect(url_for('training.add_records', model_name=model_name))
            
            elif task_type == 'regression':
                # Usar parámetros por defecto para regresión
                if model_type == 'linear_regression':
                    model_result = train_regression_model(X_processed, y_processed, 'linear_regression')
                elif model_type == 'ridge_regression':
                    model_result = train_regression_model(X_processed, y_processed, 'ridge_regression')
                elif model_type == 'lasso_regression':
                    model_result = train_regression_model(X_processed, y_processed, 'lasso_regression')
                elif model_type == 'elastic_net':
                    model_result = train_regression_model(X_processed, y_processed, 'elastic_net')
                elif model_type == 'decision_tree_regressor':
                    model_result = train_regression_model(X_processed, y_processed, 'decision_tree_regressor')
                elif model_type == 'random_forest_regressor':
                    model_result = train_regression_model(X_processed, y_processed, 'random_forest_regressor')
                elif model_type == 'knn_regressor':
                    model_result = train_regression_model(X_processed, y_processed, 'knn_regressor')
                elif model_type == 'svr':
                    model_result = train_regression_model(X_processed, y_processed, 'svr')
                else:
                    flash(f'Tipo de modelo de regresión no soportado: {model_type}', 'error')
                    return redirect(url_for('training.add_records', model_name=model_name))
            
            else:
                flash(f'Tipo de tarea no soportado: {task_type}', 'error')
                return redirect(url_for('training.add_records', model_name=model_name))
            
        except Exception as e:
            current_app.logger.error(f"Error during model retraining: {e}")
            flash(f'Error durante el reentrenamiento: {str(e)}', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        if not model_result:
            flash('Error al reentrenar el modelo', 'error')
            return redirect(url_for('training.add_records', model_name=model_name))
        
        model, results = model_result
        
        # Guardar el modelo temporalmente (no definitivamente)
        temp_model_id = str(uuid.uuid4())[:8]
        temp_model_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'retrained_model_{temp_model_id}.joblib')
        
        # Asegurar que el directorio temp_results existe
        os.makedirs(current_app.config['TEMP_RESULTS_FOLDER'], exist_ok=True)
        joblib.dump(model, temp_model_path)
        
        # Preparar métricas para el nuevo modelo
        if task_type == 'classification':
            new_metrics = {
                'test_accuracy': results.get('test_accuracy', 0),
                'test_precision_macro': results.get('test_precision_macro', 0),
                'test_recall_macro': results.get('test_recall_macro', 0),
                'test_f1_macro': results.get('test_f1_macro', 0),
                'train_accuracy': results.get('train_accuracy', 0)
            }
        else:  # regression
            new_metrics = {
                'test_r2': results.get('test_r2', 0),
                'test_mse': results.get('test_mse', 0),
                'test_mae': results.get('test_mae', 0),
                'train_r2': results.get('train_r2', 0),
                'train_mse': results.get('train_mse', 0),
                'train_mae': results.get('train_mae', 0)
            }
        
        # Preparar metadatos del nuevo modelo (temporal)
        new_metadata = {
            'model_name': new_model_name,
            'model_type': model_type,
            'task_type': task_type,
            'created_at': datetime.now().isoformat(),
            'temp_model_path': temp_model_path,
            'temp_model_id': temp_model_id,
            'status': 'pending_save',
            'target_column': target_column,
            'feature_columns': feature_columns,
            'file_used': file_used,
            'n_features': len(feature_columns),
            'metrics': new_metrics,
            'preparation_info': preparation_info,
            'retrained_from': model_name,
            'original_records': len(original_df),
            'new_records_added': len(new_records_df),
            'total_records': len(combined_df)
        }
        
        # Guardar metadatos temporales
        temp_metadata_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'retrained_metadata_{temp_model_id}.json')
        with open(temp_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)
        
        # Preparar información para la página de resultados
        retraining_info = {
            'original_records': len(original_df),
            'new_records_count': len(new_records_df),
            'total_records': len(combined_df),
            'retrain_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'new_records_preview': new_records[:5]  # Primeros 5 registros para vista previa
        }
        
        # Preparar información de los modelos para comparación
        original_model = {
            'model_name': model_name,
            'model_type': original_model_info.get('model_type'),
            'task_type': original_model_info.get('task_type'),
            'target_column': original_model_info.get('target_column'),
            'feature_columns': original_model_info.get('feature_columns'),
            'metrics': original_model_info.get('metrics', {})
        }
        
        new_model_info = {
            'model_name': new_model_name,
            'model_type': model_type,
            'task_type': task_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'metrics': new_metrics
        }
        
        current_app.logger.info(f"Model retrained successfully: {new_model_name}")
        current_app.logger.info(f"Original records: {len(original_df)}, New records: {len(new_records_df)}, Total: {len(combined_df)}")
        
        return render_template('retraining_results.html',
                             original_model=original_model,
                             new_model=new_model_info,
                             retraining_info=retraining_info,
                             temp_model_id=temp_model_id,
                             original_model_name=model_name)
        
    except Exception as e:
        current_app.logger.error(f"Error in retrain_model_with_records: {e}")
        flash(f'Error durante el reentrenamiento: {str(e)}', 'error')
        return redirect(url_for('training.add_records', model_name=model_name))


@training_bp.route('/save_retrained_model/<temp_model_id>', methods=['POST'])
def save_retrained_model(temp_model_id):
    """Guardar definitivamente un modelo reentrenado"""
    try:
        # Cargar metadatos temporales
        temp_metadata_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'retrained_metadata_{temp_model_id}.json')
        temp_model_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'retrained_model_{temp_model_id}.joblib')
        
        if not os.path.exists(temp_metadata_path) or not os.path.exists(temp_model_path):
            flash('Modelo temporal no encontrado o expirado', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        # Cargar metadatos temporales
        with open(temp_metadata_path, 'r', encoding='utf-8') as f:
            temp_metadata = json.load(f)
        
        model_name = temp_metadata['model_name']
        models_dir = current_app.config['MODELS_FOLDER']
        
        # Verificar si ya existe un modelo con ese nombre
        final_model_file = os.path.join(models_dir, f"{model_name}.joblib")
        final_metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        if os.path.exists(final_model_file):
            # Generar nombre único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_name}_{timestamp}"
            final_model_file = os.path.join(models_dir, f"{model_name}.joblib")
            final_metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        # Cargar y guardar el modelo definitivamente
        model = joblib.load(temp_model_path)
        joblib.dump(model, final_model_file)
        
        # Actualizar metadatos
        temp_metadata['model_name'] = model_name
        temp_metadata['model_file'] = final_model_file
        temp_metadata['model_size_mb'] = round(os.path.getsize(final_model_file) / (1024 * 1024), 2)
        temp_metadata['status'] = 'complete'
        temp_metadata['saved_at'] = datetime.now().isoformat()
        
        # Remover campos temporales
        temp_metadata.pop('temp_model_path', None)
        temp_metadata.pop('temp_model_id', None)
        
        # Guardar metadatos definitivos
        with open(final_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(temp_metadata, f, indent=2, ensure_ascii=False)
        
        # Limpiar archivos temporales
        try:
            os.remove(temp_model_path)
            os.remove(temp_metadata_path)
        except OSError:
            pass  # No es crítico si no se pueden eliminar
        
        current_app.logger.info(f"Retrained model saved permanently: {model_name}")
        flash(f'Modelo "{model_name}" guardado exitosamente', 'success')
        return redirect(url_for('prediction.saved_models'))
        
    except Exception as e:
        current_app.logger.error(f"Error saving retrained model: {e}")
        flash(f'Error al guardar el modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))


@training_bp.route('/discard_retrained_model/<temp_model_id>', methods=['POST'])
def discard_retrained_model(temp_model_id):
    """Descartar un modelo reentrenado temporal"""
    try:
        # Rutas de archivos temporales
        temp_metadata_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'retrained_metadata_{temp_model_id}.json')
        temp_model_path = os.path.join(current_app.config['TEMP_RESULTS_FOLDER'], f'retrained_model_{temp_model_id}.joblib')
        
        # Eliminar archivos temporales
        files_deleted = 0
        for file_path in [temp_metadata_path, temp_model_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    files_deleted += 1
                except OSError as e:
                    current_app.logger.warning(f"Could not delete temporary file {file_path}: {e}")
        
        if files_deleted > 0:
            current_app.logger.info(f"Temporary retrained model discarded: {temp_model_id}")
            flash('Modelo reentrenado descartado exitosamente', 'info')
        else:
            flash('El modelo temporal ya no existe o fue eliminado', 'warning')
            
        return redirect(url_for('prediction.saved_models'))
        
    except Exception as e:
        current_app.logger.error(f"Error discarding retrained model: {e}")
        flash(f'Error al descartar el modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))