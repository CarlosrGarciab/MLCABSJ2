"""
Rutas para manejo de predicciones con modelos guardados
"""
import os
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, send_file, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from ..utils.csv_reader import CSVReader
from ..utils.prediction_utils import (
    get_available_models, load_model_and_metadata, make_predictions, 
    validate_prediction_compatibility
)
from ..utils.file_handling import allowed_file, get_csv_preview, get_csv_columns
from ..forms import UploadForm

bp = Blueprint('prediction', __name__)

@bp.route('/saved_models')
def saved_models():
    """Página que muestra los modelos guardados disponibles"""
    try:
        models = get_available_models()
        return render_template('saved_models.html', models=models)
    except Exception as e:
        current_app.logger.error(f"Error loading saved models: {e}")
        flash(f'Error cargando modelos: {str(e)}', 'error')
        return render_template('saved_models.html', models=[])

@bp.route('/predict_with_model/<model_name>')
def predict_with_model(model_name):
    """Página para cargar archivo CSV y realizar predicciones con un modelo específico"""
    try:
        # Cargar metadatos del modelo para mostrar información
        _, metadata = load_model_and_metadata(model_name)
        
        # Crear formulario de upload
        form = UploadForm()
        
        return render_template('predict_with_model.html', 
                             model_name=model_name, 
                             metadata=metadata, 
                             form=form)
        
    except Exception as e:
        current_app.logger.error(f"Error loading model {model_name}: {e}")
        flash(f'Error cargando el modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@bp.route('/upload_for_prediction/<model_name>', methods=['POST'])
def upload_for_prediction(model_name):
    """Maneja la subida de archivo CSV para predicción"""
    try:
        form = UploadForm()
        
        if form.validate_on_submit():
            file = form.file.data
            if file and allowed_file(file.filename):
                # Guardar archivo
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"pred_{timestamp}_{filename}"
                
                if not os.path.exists(current_app.config['UPLOAD_FOLDER']):
                    os.makedirs(current_app.config['UPLOAD_FOLDER'])
                
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Cargar metadatos del modelo
                _, metadata = load_model_and_metadata(model_name)
                
                # Leer archivo CSV para validación usando utilidad centralizada
                df, read_info = CSVReader.read_csv_robust(file_path)
                
                if not read_info['success']:
                    # Limpiar archivo si hay error
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    flash(f'Error leyendo archivo CSV: {read_info["error"]}', 'error')
                    return redirect(url_for('prediction.upload_for_prediction', model_name=model_name))
                    
                # Validar compatibilidad
                compatibility = validate_prediction_compatibility(metadata, df)
                
                # Guardar información en sesión
                session['prediction_file'] = file_path
                session['prediction_model'] = model_name
                session['prediction_filename'] = filename
                session['compatibility_check'] = compatibility
                
                return redirect(url_for('prediction.validate_prediction'))
                    
            else:
                flash('Por favor, sube un archivo CSV válido', 'error')
        else:
            flash('Error en el formulario', 'error')
        
        # Si hay error, volver a la página del modelo
        return redirect(url_for('prediction.predict_with_model', model_name=model_name))
        
    except Exception as e:
        current_app.logger.error(f"Error uploading file for prediction: {e}")
        flash(f'Error subiendo archivo: {str(e)}', 'error')
        return redirect(url_for('prediction.predict_with_model', model_name=model_name))

@bp.route('/validate_prediction')
def validate_prediction():
    """Página para validar compatibilidad y confirmar predicción"""
    
    # Verificar que tenemos los datos necesarios en sesión
    if not all(key in session for key in ['prediction_file', 'prediction_model', 'compatibility_check']):
        flash('Información de predicción no encontrada. Por favor, vuelve a cargar el archivo.', 'error')
        return redirect(url_for('prediction.saved_models'))
    
    try:
        model_name = session['prediction_model']
        file_path = session['prediction_file']
        compatibility = session['compatibility_check']
        
        # Cargar metadatos del modelo
        _, metadata = load_model_and_metadata(model_name)
        
        # Obtener preview del archivo
        preview_data = get_csv_preview(file_path, rows=5)
        
        return render_template('validate_prediction.html',
                             model_name=model_name,
                             metadata=metadata,
                             compatibility=compatibility,
                             preview_data=preview_data,
                             filename=session.get('prediction_filename', 'archivo.csv'))
        
    except Exception as e:
        current_app.logger.error(f"Error validating prediction: {e}")
        flash(f'Error validando predicción: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@bp.route('/execute_prediction', methods=['POST'])
def execute_prediction():
    """Ejecuta la predicción"""
    
    # Verificar datos en sesión
    if not all(key in session for key in ['prediction_file', 'prediction_model']):
        flash('Información de predicción no encontrada.', 'error')
        return redirect(url_for('prediction.saved_models'))
    
    try:
        model_name = session['prediction_model']
        file_path = session['prediction_file']
        
        # Verificar compatibilidad
        compatibility = session.get('compatibility_check', {})
        if not compatibility.get('compatible', False):
            flash('El archivo no es compatible con el modelo seleccionado.', 'error')
            return redirect(url_for('prediction.validate_prediction'))
        
        # Cargar modelo y metadatos
        model_data, metadata = load_model_and_metadata(model_name)
        
        # Cargar archivo CSV usando utilidad centralizada
        df, read_info = CSVReader.read_csv_robust(file_path)
        
        if not read_info['success']:
            raise Exception(f"No se pudo leer el archivo CSV: {read_info['error']}")
        
        # Realizar predicción
        results_df, prediction_stats = make_predictions(model_data, metadata, df)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"predicciones_{model_name}_{timestamp}.csv"
        results_path = os.path.join(current_app.config['UPLOAD_FOLDER'], results_filename)
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # Guardar información en sesión
        session['prediction_results'] = {
            'results_file': results_path,
            'results_filename': results_filename,
            'stats': prediction_stats,
            'model_name': model_name,
            'original_filename': session.get('prediction_filename', 'archivo.csv'),
            'timestamp': timestamp
        }
        
        return redirect(url_for('prediction.prediction_results'))
        
    except Exception as e:
        current_app.logger.error(f"Error executing prediction: {e}")
        flash(f'Error ejecutando predicción: {str(e)}', 'error')
        return redirect(url_for('prediction.validate_prediction'))

@bp.route('/prediction_results')
def prediction_results():
    """Página que muestra los resultados de la predicción"""
    
    if 'prediction_results' not in session:
        flash('No hay resultados de predicción disponibles.', 'error')
        return redirect(url_for('prediction.saved_models'))
    
    try:
        results_info = session['prediction_results']
        results_file = results_info['results_file']
        
        # Verificar que el archivo existe
        if not os.path.exists(results_file):
            flash('Archivo de resultados no encontrado.', 'error')
            return redirect(url_for('prediction.saved_models'))
        
        # Leer todos los resultados para mostrar
        # Intentar con diferentes separadores
        results_df = None
        for sep in [',', ';', '\t']:
            try:
                results_df = pd.read_csv(results_file, sep=sep)
                if len(results_df.columns) > 1:  # Validar que tiene múltiples columnas
                    break
            except Exception:
                continue
        
        # Si no se pudo leer con múltiples columnas, usar lectura por defecto
        if results_df is None or len(results_df.columns) == 1:
            results_df = pd.read_csv(results_file)
        
        results_preview = {
            'columns': results_df.columns.tolist(),
            'data': results_df.values.tolist(),
            'total_rows': len(results_df)
        }
        
        return render_template('prediction_results.html',
                             results_info=results_info,
                             results_preview=results_preview)
        
    except Exception as e:
        current_app.logger.error(f"Error showing prediction results: {e}")
        flash(f'Error mostrando resultados: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@bp.route('/download_predictions')
def download_predictions():
    """Descarga el archivo de predicciones"""
    
    if 'prediction_results' not in session:
        flash('No hay resultados disponibles para descargar.', 'error')
        return redirect(url_for('prediction.saved_models'))
    
    try:
        results_info = session['prediction_results']
        results_file = results_info['results_file']
        download_name = results_info['results_filename']
        
        if not os.path.exists(results_file):
            flash('Archivo no encontrado.', 'error')
            return redirect(url_for('prediction.prediction_results'))
        
        return send_file(results_file, 
                        as_attachment=True, 
                        download_name=download_name,
                        mimetype='text/csv')
        
    except Exception as e:
        current_app.logger.error(f"Error downloading predictions: {e}")
        flash(f'Error descargando archivo: {str(e)}', 'error')
        return redirect(url_for('prediction.prediction_results'))

@bp.route('/model_details/<model_name>')
def model_details(model_name):
    """Página con detalles completos de un modelo"""
    try:
        model_data, metadata = load_model_and_metadata(model_name)
        
        return render_template('model_details.html',
                             model_name=model_name,
                             metadata=metadata)
        
    except Exception as e:
        current_app.logger.error(f"Error loading model details: {e}")
        flash(f'Error cargando detalles del modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@bp.route('/delete_model/<model_name>', methods=['POST'])
def delete_model(model_name):
    """Elimina un modelo guardado (con confirmación)"""
    try:
        models_dir = current_app.config['MODELS_FOLDER']
        
        # Archivos a eliminar
        model_file = os.path.join(models_dir, f"{model_name}.joblib")
        metadata_file = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        # Verificar confirmación
        if request.form.get('confirm') != 'yes':
            flash('Eliminación cancelada.', 'info')
            return redirect(url_for('prediction.saved_models'))
        
        # Eliminar archivos
        deleted_files = []
        if os.path.exists(model_file):
            os.remove(model_file)
            deleted_files.append('modelo')
        
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            deleted_files.append('metadatos')
        
        if deleted_files:
            flash(f'Modelo "{model_name}" eliminado exitosamente ({", ".join(deleted_files)}).', 'success')
        else:
            flash(f'No se encontraron archivos para el modelo "{model_name}".', 'warning')
        
        return redirect(url_for('prediction.saved_models'))
        
    except Exception as e:
        current_app.logger.error(f"Error deleting model {model_name}: {e}")
        flash(f'Error eliminando modelo: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))

@bp.route('/cleanup_predictions')
def cleanup_predictions():
    """Usa el sistema centralizado de limpieza"""
    try:
        from app.utils.system_cleanup import cleanup_manager
        cleanup_manager.manual_cleanup_now(max_age_hours=1)
        flash('Limpieza de predicciones completada usando el sistema centralizado.', 'success')
        return redirect(url_for('prediction.saved_models'))
    except Exception as e:
        current_app.logger.error(f"Error en limpieza centralizada: {e}")
        flash(f'Error durante la limpieza: {str(e)}', 'error')
        return redirect(url_for('prediction.saved_models'))