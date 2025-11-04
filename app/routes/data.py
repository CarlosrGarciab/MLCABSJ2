"""
Rutas para manejo y procesamiento de datos
"""
import os
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, jsonify
from ..forms import ColumnSelectionForm
from ..utils.file_handling import get_csv_columns, get_csv_preview
from ..utils.data_validation import validate_data_simple, convert_numpy_types
from ..utils.data_cleaning import analyze_cleaning_needs, clean_data_manual
from ..utils.csv_reader import CSVReader
from ..services.session_service import SessionService
from ..utils.training_utils import clean_column_names
from ..utils.target_analysis import analyze_target_compatibility, get_model_availability_message

bp = Blueprint('data', __name__)

@bp.route('/select_columns')
def select_columns():
    """Página para seleccionar columnas target y predictoras"""
    
    # Verificar que hay un archivo cargado
    if 'uploaded_file' not in session:
        flash('No hay archivo cargado', 'error')
        return redirect(url_for('main.index'))
    
    file_path = session['uploaded_file']
    
    # Debug: informar sobre el archivo que se está procesando
    current_app.logger.debug("=== SELECT_COLUMNS DEBUG ===")
    current_app.logger.info(f"File: {session.get('filename', 'unknown')}")
    
    try:
        if not os.path.exists(file_path):
            flash('El archivo ya no existe en el servidor', 'error')
            return redirect(url_for('main.index'))

        # Usar las funciones mejoradas para obtener columnas y preview
        columns = get_csv_columns(file_path)
        preview_data = get_csv_preview(file_path, rows=5)
        
        current_app.logger.info(f"Columns detected: {len(columns)}")
        current_app.logger.info(f"First 10 columns: {columns[:10]}")

        # Truncar celdas muy largas en el preview
        if preview_data.get('data'):
            for i, row in enumerate(preview_data['data']):
                preview_data['data'][i] = [
                    str(cell)[:50] + '...' if isinstance(cell, str) and len(str(cell)) > 50 else cell 
                    for cell in row
                ]

        # Obtener el separador y encoding detectados
        detected_separator = preview_data.get('separator', ';')
        detected_encoding = preview_data.get('encoding', 'utf-8')
        
        # Leer el CSV completo usando la utilidad centralizada
        df, read_info = CSVReader.read_csv_robust(
            file_path,
            separator=detected_separator,
            encoding=detected_encoding
        )
        if not read_info['success']:
            # Fallback: intentar sin encoding para probar todos los disponibles
            df, read_info = CSVReader.read_csv_robust(
                file_path,
                separator=detected_separator
            )
            if not read_info['success']:
                raise Exception(f"No se pudo cargar el archivo CSV: {read_info['error']}")
        # Actualizar sesión con la combinación exitosa
        session['csv_separator'] = read_info['separator_used']
        session['csv_encoding'] = read_info['encoding_used']
        current_app.logger.info(f"CSV cargado exitosamente: {read_info['encoding_used']} + '{read_info['separator_used']}' - {read_info['rows']} filas, {read_info['columns']} columnas")
        
        # Limpiar nombres de columnas del DataFrame
        df.columns = clean_column_names(df.columns.tolist())
        
        # Verificar si hay columnas duplicadas después de limpiar
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            current_app.logger.warning(f"Columnas duplicadas después de limpieza: {duplicated_cols}")
            # Hacer los nombres únicos agregando sufijos
            df.columns = pd.Series(df.columns).fillna('unnamed').astype(str)
            # Crear nombres únicos
            seen = {}
            new_cols = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols
            current_app.logger.info(f"Columnas renombradas para evitar duplicados")

        # Validar los datos
        validation_results = validate_data_simple(df)
        validation_results = convert_numpy_types(validation_results)

        form = ColumnSelectionForm()
        column_choices = [(col, col) for col in columns]
        form.target_column.choices = column_choices
        form.predictor_columns.choices = column_choices

        # Identificar columnas numéricas para el formulario
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Para el template, necesitamos un resumen más ligero
        validation_summary = {
            'total_rows': validation_results.get('total_rows', 0),
            'total_columns': validation_results.get('total_columns', 0),
            'duplicates': validation_results.get('duplicates', 0),
            'is_valid': validation_results.get('is_valid', True),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns),
        }

        return render_template('select_columns.html', 
                             form=form, 
                             columns=columns, 
                             preview_data=preview_data,
                             validation_results=validation_summary,
                             filename=session.get('filename', 'archivo.csv'),
                             numeric_columns=numeric_columns)
                             
    except Exception as e:
        flash(f'Error al procesar el archivo: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@bp.route('/analyze_target', methods=['POST'])
def analyze_target():
    """Analizar compatibilidad del target seleccionado con diferentes tipos de modelos"""
    try:
        target_column = request.json.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'No se proporcionó columna target'}), 400
            
        file_path = session.get('uploaded_file')
        if not file_path:
            return jsonify({'error': 'No hay archivo cargado'}), 400
            
        # Cargar el DataFrame con la configuración guardada en sesión
        encoding = session.get('csv_encoding', 'utf-8')
        separator = session.get('csv_separator', ';')
        
        df, read_info = CSVReader.read_csv_robust(file_path, separator=separator, encoding=encoding)
        if not read_info['success']:
            # Fallback: intentar sin encoding para probar todos los disponibles
            df, read_info = CSVReader.read_csv_robust(file_path, separator=separator)
            if not read_info['success']:
                return jsonify({'error': f'Error leyendo archivo: {read_info["error"]}'}), 400
        df.columns = clean_column_names(df.columns.tolist())
        
        # Analizar la compatibilidad del target
        analysis = analyze_target_compatibility(df, target_column)
        
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 400
            
        # Obtener mensajes de disponibilidad
        availability_messages = get_model_availability_message(analysis)
        
        # Preparar respuesta
        response = {
            'success': True,
            'target_info': {
                'name': target_column,
                'type': 'Numérica' if analysis['is_numeric'] else 'Categórica',
                'unique_values': analysis['unique_values'],
                'missing_values': analysis['missing_values'],
                'sample_values': analysis['sample_values'][:5]  # Solo primeros 5 valores
            },
            'model_compatibility': {
                'classification': availability_messages['classification'],
                'regression': availability_messages['regression']
            },
            'recommendation': {
                'primary': analysis['primary_recommendation'],
                'reason': analysis['recommendation_reason']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        current_app.logger.error(f"Error analyzing target: {str(e)}")
        return jsonify({'error': f'Error al analizar el target: {str(e)}'}), 500

@bp.route('/select_columns', methods=['POST'])
def select_columns_post():
    """Procesar la selección de columnas"""
    # Primero obtener las columnas disponibles para configurar el formulario
    file_path = session.get('uploaded_file')
    if not file_path:
        flash('No hay archivo cargado', 'error')
        return redirect(url_for('main.index'))
    
    try:
        columns = get_csv_columns(file_path)
        column_choices = [(col, col) for col in columns]
        
        # Crear y configurar el formulario
        form = ColumnSelectionForm()
        form.target_column.choices = column_choices
        form.predictor_columns.choices = column_choices
        
        if form.validate_on_submit():
            target_column = form.target_column.data
            predictor_columns = form.predictor_columns.data

            # Limpiar los nombres de columnas antes de cualquier validación o guardado
            target_column_clean = clean_column_names([target_column])[0]
            predictor_columns_clean = clean_column_names(predictor_columns)

            current_app.logger.debug("=== COLUMN SELECTION DEBUG ===")
            current_app.logger.info(f"Target column selected: '{target_column_clean}'")
            current_app.logger.info(f"Predictor columns selected: {predictor_columns_clean}")
            current_app.logger.info(f"Available columns in file: {columns[:10]}...")

            if not target_column_clean:
                flash('Debes seleccionar una columna target', 'warning')
            elif not predictor_columns_clean:
                flash('Debes seleccionar al menos una columna predictora', 'warning')
            elif target_column_clean in predictor_columns_clean:
                flash('La columna target no puede ser también una columna predictora', 'warning')
            else:
                # Guardar columnas seleccionadas LIMPIADAS en la sesión
                session['target_column'] = target_column_clean
                session['predictor_columns'] = predictor_columns_clean
                session['predictor_count'] = len(predictor_columns_clean)

                # Debug: confirmar que se guardó en sesión
                current_app.logger.info(f"Saved in session - target: '{session.get('target_column')}', predictors: {len(predictor_columns_clean)} columns")
                current_app.logger.info(f"First 5 predictors: {predictor_columns_clean[:5]}")
                current_app.logger.info(f"Last 5 predictors: {predictor_columns_clean[-5:]}")
                
                # Limpiar datos temporales innecesarios para reducir tamaño de sesión
                for key in ['association_analysis', 'association_temp_file', 'last_training_results', 'training_data_info']:
                    session.pop(key, None)

                # Ir directamente a opciones de limpieza manual
                return redirect(url_for('data.cleaning_options'))

        # Si hay errores en la validación o no se envió correctamente
        # Volver a mostrar el formulario con los datos actuales
        preview_data = get_csv_preview(file_path, rows=5)
        
        # Identificar columnas numéricas
        separator = session.get('csv_separator', ';')
        encoding = session.get('csv_encoding', 'utf-8')
        df, read_info = CSVReader.read_csv_robust(file_path, separator=separator, encoding=encoding, nrows=100)
        if not read_info['success']:
            # Fallback: intentar sin encoding para probar todos los disponibles
            df, read_info = CSVReader.read_csv_robust(file_path, separator=separator, nrows=100)
            if not read_info['success']:
                flash('Error leyendo archivo para análisis', 'error')
                return redirect(url_for('data.select_columns'))
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        return render_template('select_columns.html', 
                             form=form, 
                             columns=columns, 
                             preview_data=preview_data,
                             validation_results={},
                             filename=session.get('filename', 'archivo.csv'),
                             numeric_columns=numeric_columns)
                             
    except Exception as e:
        current_app.logger.error(f"Error in select_columns_post: {str(e)}")
        flash(f'Error al procesar el archivo: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@bp.route('/cleaning_options')
def cleaning_options():
    """Página para configurar opciones de limpieza de datos"""
    # Verificar que tenemos toda la información necesaria
    required_keys = ['uploaded_file', 'target_column']
    
    # Debug: imprimir lo que hay en la sesión
    current_app.logger.debug("=== CLEANING_OPTIONS DEBUG ===")
    current_app.logger.info(f"Session keys: {list(session.keys())}")
    for key in ['uploaded_file', 'target_column', 'predictor_columns', 'predictor_count']:
        if key in session:
            value = session[key]
            if isinstance(value, list) and len(value) > 10:
                current_app.logger.info(f"{key}: [showing first 10] {value[:10]}...")
            else:
                current_app.logger.info(f"{key}: {value}")
        else:
            current_app.logger.warning(f"MISSING: {key}")
    
    # Verificar configuración
    if not all(key in session for key in required_keys):
        flash('Configuración incompleta. Por favor, vuelve a configurar las columnas', 'warning')
        return redirect(url_for('data.select_columns'))
    
    # Verificar que tengamos columnas predictoras (de alguna manera)
    has_predictors = ('predictor_columns' in session and len(session['predictor_columns']) > 0) or \
                    ('predictor_count' in session and session['predictor_count'] > 0)
    
    if not has_predictors:
        flash('No se encontraron columnas predictoras. Por favor, vuelve a seleccionar las columnas', 'warning')
        return redirect(url_for('data.select_columns'))
    
    file_path = session['uploaded_file']
    
    try:
        # Leer el CSV usando el separador y encoding detectados
        separator = session.get('csv_separator', ';')
        encoding = session.get('csv_encoding', 'utf-8')
        df, read_info = CSVReader.read_csv_robust(file_path, separator=separator, encoding=encoding)
        if not read_info['success']:
            # Fallback: intentar sin encoding para probar todos los disponibles
            df, read_info = CSVReader.read_csv_robust(file_path, separator=separator)
            if not read_info['success']:
                flash(f'Error leyendo archivo: {read_info["error"]}', 'error')
                return redirect(url_for('data.select_columns'))
        
        # CRÍTICO: Limpiar nombres de columnas INMEDIATAMENTE después de leer
        df.columns = clean_column_names(df.columns.tolist())

        # Obtener las columnas predictoras (pueden estar truncadas en sesión)
        if session.get('all_predictors_truncated', False):
            # Leer columnas completas del archivo temporal
            temp_file = session.get('temp_predictors_file')
            if temp_file and os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    predictor_columns = [line.strip() for line in f.readlines()]
                # Limpiar los nombres leídos del archivo temporal
                predictor_columns = clean_column_names(predictor_columns)
            else:
                # Fallback: intentar obtener columnas del DataFrame
                target_column = session.get('target_column')
                predictor_columns = [col for col in df.columns if col != target_column]
        else:
            predictor_columns = session.get('predictor_columns', [])
            # Limpiar también las columnas de sesión por si tienen inconsistencias
            if predictor_columns:
                predictor_columns = clean_column_names(predictor_columns)
        
        target_column = session.get('target_column', None)
        # Limpiar el target_column también por consistencia
        if target_column:
            target_column = clean_column_names([target_column])[0]

        # Debug: mostrar información de las columnas
        current_app.logger.debug(f"=== COLUMN VALIDATION DEBUG ===")
        current_app.logger.info(f"Available columns in CSV: {list(df.columns)[:10]}...")
        current_app.logger.info(f"Target column from session: {target_column}")
        current_app.logger.info(f"Predictor columns from session/file (first 10): {predictor_columns[:10]}")
        current_app.logger.info(f"Total predictor columns: {len(predictor_columns)}")

        # Validar que las columnas existen en el DataFrame
        available_columns = list(df.columns)
        
        # Verificar target column
        if target_column not in available_columns:
            current_app.logger.error(f"Target column '{target_column}' not found in CSV. Available: {available_columns[:10]}")
            flash(f'La columna target "{target_column}" no se encontró en el archivo. Por favor, vuelve a seleccionar las columnas.', 'error')
            return redirect(url_for('data.select_columns'))
        
        # Filtrar columnas predictoras para solo incluir las que existen
        valid_predictor_columns = [col for col in predictor_columns if col in available_columns]
        invalid_predictor_columns = [col for col in predictor_columns if col not in available_columns]
        
        if invalid_predictor_columns:
            current_app.logger.warning(f"Some predictor columns not found: {invalid_predictor_columns[:5]}...")
            current_app.logger.info(f"Valid predictor columns: {len(valid_predictor_columns)}/{len(predictor_columns)}")
        
        if not valid_predictor_columns:
            flash('No se encontraron columnas predictoras válidas en el archivo. Por favor, vuelve a seleccionar las columnas.', 'error')
            return redirect(url_for('data.select_columns'))

        # Usar solo las columnas válidas
        all_columns = [target_column] + valid_predictor_columns
        df_subset = df[all_columns]

        current_app.logger.info(f"DataFrame subset created with {len(all_columns)} columns: {len(df_subset)} rows")


        # Refuerzo: guardar columnas limpias en sesión para consistencia en siguientes pasos
        session['target_column'] = target_column
        session['predictor_columns'] = valid_predictor_columns

        # Analizar necesidades de limpieza con manejo de errores
        try:
            cleaning_analysis = analyze_cleaning_needs(df_subset)
            current_app.logger.info(f"Cleaning analysis completed successfully")
        except Exception as e:
            current_app.logger.error(f"Error in analyze_cleaning_needs: {str(e)}")
            # Crear un análisis básico como fallback
            cleaning_analysis = {
                'total_rows': len(df_subset),
                'total_cols': len(df_subset.columns),
                'recommendations': [{
                    'type': 'basic',
                    'title': 'Análisis básico',
                    'description': f'Datos cargados: {len(df_subset)} filas, {len(df_subset.columns)} columnas',
                    'recommended': False,
                    'impact': 'Sin cambios recomendados'
                }],
                'error': str(e)
            }

        return render_template('cleaning_options.html',
                             cleaning_analysis=cleaning_analysis,
                             analysis=cleaning_analysis,  # Agregar también como 'analysis' para compatibilidad con template
                             target_column=target_column,
                             predictor_columns=valid_predictor_columns[:10],  # Mostrar solo primeras 10 en template
                             total_predictors=len(valid_predictor_columns),
                             filename=session.get('filename', 'archivo.csv'))

    except Exception as e:
        current_app.logger.error(f"Error in cleaning_options: {str(e)}")
        current_app.logger.error(f"Error type: {type(e).__name__}")
        import traceback
        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f'Error al analizar el archivo: {str(e)}', 'error')
        return redirect(url_for('data.select_columns'))

@bp.route('/cleaning_options', methods=['POST'])
def cleaning_options_post():
    """Procesar opciones de limpieza seleccionadas"""
    try:
        # Verificar si el usuario eligió omitir la limpieza
        if 'omit_cleaning' in request.form:
            # Usuario eligió omitir limpieza - usar archivo original
            file_path = session['uploaded_file']
            target_column = session.get('target_column')
            
            # Leer archivo original para obtener solo las columnas seleccionadas
            separator = session.get('csv_separator', ';')
            encoding = session.get('csv_encoding', 'utf-8')
            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            
            # IMPORTANTE: Limpiar nombres de columnas inmediatamente después de leer
            df.columns = clean_column_names(df.columns.tolist())
            
            # Limpiar nombres de columnas por consistencia
            if target_column:
                target_column = clean_column_names([target_column])[0]
            
            # Obtener columnas predictoras
            if session.get('all_predictors_truncated', False):
                temp_file = session.get('temp_predictors_file')
                if temp_file and os.path.exists(temp_file):
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        predictor_columns = [line.strip() for line in f.readlines()]
                    predictor_columns = clean_column_names(predictor_columns)
                else:
                    predictor_columns = session.get('predictor_columns', [])
            else:
                predictor_columns = session.get('predictor_columns', [])
                if predictor_columns:
                    predictor_columns = clean_column_names(predictor_columns)
            
            # Filtrar solo las columnas seleccionadas
            selected_columns = [target_column] + predictor_columns
            available_columns = list(df.columns)
            
            valid_columns = [col for col in selected_columns if col in available_columns]
            
            if not valid_columns:
                flash('No se encontraron las columnas seleccionadas en el archivo.', 'error')
                return redirect(url_for('data.select_columns'))
            
            df_final = df[valid_columns]

            # Refuerzo: guardar columnas limpias en sesión para consistencia en siguientes pasos
            session['target_column'] = target_column
            session['predictor_columns'] = predictor_columns
            
            # Guardar archivo "limpio" (que es realmente el original filtrado)
            cleaned_filename = f"cleaned_{session.get('filename', 'data.csv')}"
            cleaned_path = os.path.join(current_app.config['UPLOAD_FOLDER'], cleaned_filename)
            df_final.to_csv(cleaned_path, index=False, sep=separator, encoding=encoding)
            
            # Crear un reporte vacío indicando que no se aplicó limpieza
            cleaning_report = {
                'duplicates_removed': 0,
                'empty_rows_removed': 0,
                'rows_with_many_missing_removed': 0,
                'empty_cols_removed': 0,
                'text_cleaned': False,
                'normalized': False,
                'missing_handled': False,
                'omitted': True,  # Indicador de que se omitió la limpieza
                'original_shape': df.shape,
                'final_shape': df_final.shape
            }
            
            # Actualizar la sesión
            session['cleaned_file'] = cleaned_path
            session['data_file'] = cleaned_path
            session['data_shape'] = df_final.shape
            session['cleaning_report'] = cleaning_report
            session['original_data_shape'] = df.shape
            
            return redirect(url_for('data.cleaning_results'))
        
        # Si no se omitió la limpieza, proceder normalmente
        file_path = session['uploaded_file']
        target_column = session.get('target_column')
        # Limpiar el target_column también por consistencia
        if target_column:
            target_column = clean_column_names([target_column])[0]
        
        # Obtener columnas predictoras (manejando el caso truncado)
        if session.get('all_predictors_truncated', False):
            temp_file = session.get('temp_predictors_file')
            if temp_file and os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    predictor_columns = [line.strip() for line in f.readlines()]
                # Limpiar los nombres leídos del archivo temporal
                predictor_columns = clean_column_names(predictor_columns)
            else:
                predictor_columns = session.get('predictor_columns', [])
        else:
            predictor_columns = session.get('predictor_columns', [])
            # Limpiar también las columnas de sesión por si tienen inconsistencias
            if predictor_columns:
                predictor_columns = clean_column_names(predictor_columns)

        # Leer archivo con encoding detectado
        separator = session.get('csv_separator', ';')
        encoding = session.get('csv_encoding', 'utf-8')
        df = pd.read_csv(file_path, sep=separator, encoding=encoding)
        
        # IMPORTANTE: Limpiar nombres de columnas inmediatamente después de leer
        df.columns = clean_column_names(df.columns.tolist())

        # Guardar dimensiones originales para comparación
        session['original_data_shape'] = df.shape

        # Validar que las columnas existen en el DataFrame (igual que en GET)
        available_columns = list(df.columns)
        
        if target_column not in available_columns:
            flash(f'La columna target "{target_column}" no se encontró en el archivo.', 'error')
            return redirect(url_for('data.select_columns'))
        
        # Filtrar columnas predictoras para solo incluir las que existen
        valid_predictor_columns = [col for col in predictor_columns if col in available_columns]
        
        if not valid_predictor_columns:
            flash('No se encontraron columnas predictoras válidas en el archivo.', 'error')
            return redirect(url_for('data.select_columns'))

        # Obtener opciones de limpieza del formulario
        # Crear diccionario de opciones para la función de limpieza
        cleaning_options = {
            'remove_duplicates': 'remove_duplicates' in request.form,
            'remove_empty_rows': 'remove_empty_rows' in request.form,
            'remove_rows_many_missing': 'remove_rows_many_missing' in request.form,
            'remove_empty_cols': 'remove_empty_cols' in request.form,
            'clean_text_spaces': 'clean_text_spaces' in request.form,
            'normalize_data': 'normalize_data' in request.form,
            'normalization_method': request.form.get('normalization_method', 'standard'),
            'handle_missing': request.form.get('handle_missing', 'none')
        }
        
        # Columnas protegidas (target + predictores válidos)
        protected_columns = [target_column] + valid_predictor_columns
        
        current_app.logger.debug(f"=== CLEANING OPTIONS DEBUG ===")
        current_app.logger.info(f"Cleaning options: {cleaning_options}")
        current_app.logger.info(f"Protected columns: {len(protected_columns)} columns")
        
        # Aplicar limpieza con las opciones correctas
        df_cleaned, cleaning_report = clean_data_manual(
            df=df,
            options=cleaning_options,
            protected_columns=protected_columns
        )

        # Refuerzo: guardar columnas limpias en sesión para consistencia en siguientes pasos
        session['target_column'] = target_column
        session['predictor_columns'] = valid_predictor_columns
        
        # Filtrar el DataFrame limpio para incluir solo las columnas seleccionadas
        # Esto evita que aparezcan columnas no seleccionadas en los resultados
        df_cleaned = df_cleaned[protected_columns]
        
        current_app.logger.debug(f"=== CLEANING RESULTS DEBUG ===")
        current_app.logger.info(f"Original shape: {df.shape}")
        current_app.logger.info(f"Cleaned shape: {df_cleaned.shape}")
        current_app.logger.info(f"Cleaning report: {cleaning_report}")

        # Guardar el archivo limpio
        cleaned_filename = f"cleaned_{session.get('filename', 'data.csv')}"
        cleaned_path = os.path.join(current_app.config['UPLOAD_FOLDER'], cleaned_filename)
        df_cleaned.to_csv(cleaned_path, index=False, sep=separator, encoding=encoding)

        # Actualizar la sesión
        session['cleaned_file'] = cleaned_path
        session['data_file'] = cleaned_path  # Para compatibilidad con otras partes del sistema
        session['data_shape'] = df_cleaned.shape
        session['cleaning_report'] = cleaning_report  # Guardar el reporte para referencia

        return redirect(url_for('data.cleaning_results'))

    except Exception as e:
        flash(f'Error durante la limpieza de datos: {str(e)}', 'error')
        return redirect(url_for('data.cleaning_options'))


@bp.route('/cleaning_results')
def cleaning_results():
    """Mostrar los resultados detallados de la limpieza de datos"""
    
    if 'cleaned_file' not in session:
        flash('No hay datos limpiados disponibles', 'warning')
        return redirect(url_for('main.index'))
    
    # Cargar datos para analizar valores faltantes
    missing_values_info = None
    try:
        import pandas as pd
        import os
        
        # Cargar el archivo original para analizar valores faltantes
        original_file = session.get('uploaded_file')
        if original_file and os.path.exists(original_file):
            separator = session.get('csv_separator', ';')
            encoding = session.get('csv_encoding', 'utf-8')
            df_original = pd.read_csv(original_file, sep=separator, encoding=encoding)
            
            # IMPORTANTE: Limpiar nombres de columnas inmediatamente después de leer
            df_original.columns = clean_column_names(df_original.columns.tolist())
            
            # Calcular valores faltantes por columna
            missing_values_info = {}
            for col in df_original.columns:
                missing_count = df_original[col].isnull().sum()
                total_rows = len(df_original)
                missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0
                
                if missing_count > 0:  # Solo incluir columnas con valores faltantes
                    missing_values_info[col] = {
                        'count': int(missing_count),
                        'percentage': round(float(missing_percentage), 2),
                        'total_rows': total_rows
                    }
    except Exception as e:
        current_app.logger.error(f"Error calculando valores faltantes: {str(e)}")
    
    data_info = {
        'filename': session.get('filename', 'Desconocido'),
        'original_shape': session.get('original_data_shape', (0, 0)),
        'data_shape': session.get('data_shape', (0, 0)),
        'target_column': session.get('target_column'),
        'predictor_count': session.get('predictor_count', 0),
        'cleaning_report': session.get('cleaning_report', {}),
        'csv_encoding': session.get('csv_encoding', 'utf-8'),
        'csv_separator': session.get('csv_separator', ';'),
        'missing_values_info': missing_values_info
    }
    
    return render_template('cleaning_results.html', data_info=data_info)