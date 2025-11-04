"""
Rutas para análisis de reglas de asociación
"""
import os
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, Response
from ..utils.csv_reader import CSVReader
from ..utils.association_rules import (analyze_association_rules, get_top_rules,
                                     save_association_analysis, get_available_algorithms)
from ..utils.training_utils import clean_column_names

bp = Blueprint('association', __name__)

@bp.route('/association_rules', methods=['GET', 'POST'])
def association_rules_page():
    """Página para análisis de reglas de asociación"""
    
    # Verificar que hay un archivo de datos disponible
    # No requerimos columnas seleccionadas para GET, pero sí para POST
    data_file = None
    if 'cleaned_file' in session:
        data_file = session['cleaned_file']
    elif 'uploaded_file' in session:
        data_file = session['uploaded_file']
    elif 'filename' in session:
        # Construir path desde filename
        filename = session['filename']
        data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    if not data_file or not os.path.exists(data_file):
        flash('No hay datos disponibles. Por favor, sube y procesa un archivo primero.', 'warning')
        return redirect(url_for('main.index'))
    
    try:
        # Cargar datos usando utilidad centralizada
        separator = session.get('csv_separator', ';')
        encoding = session.get('csv_encoding', 'utf-8')
        
        df, read_info = CSVReader.read_csv_robust(data_file, separator=separator, encoding=encoding)
        if not read_info['success']:
            # Fallback: intentar sin encoding para probar todos los disponibles
            df, read_info = CSVReader.read_csv_robust(data_file, separator=separator)
            if not read_info['success']:
                flash(f'Error leyendo archivo de datos: {read_info["error"]}', 'error')
                return redirect(url_for('main.index'))
        
        # Limpiar nombres de columnas del DataFrame completo
        df.columns = clean_column_names(df.columns.tolist())
        
        # Guardar todas las columnas del DataFrame para mostrar en el formulario
        all_columns = df.columns.tolist()
        
        if request.method == 'POST':
            # Para POST: Usar todas las columnas del DataFrame (no filtrar)
            # El análisis de asociación puede usar todas las columnas disponibles
            df_for_analysis = df
            current_app.logger.info(f"Columns used for analysis: {len(df.columns)} columns")
            
            # Obtener parámetros del formulario
            analysis_type = request.form.get('analysis_type', 'basket')
            min_support = float(request.form.get('min_support', 0.1))
            min_confidence = float(request.form.get('min_confidence', 0.6))
            max_itemset_length = request.form.get('max_itemset_length')
            max_itemset_length = int(max_itemset_length) if max_itemset_length else None
            
            # Validación de seguridad: establecer límite máximo si no se especifica
            if max_itemset_length is None:
                num_columns = len(df.columns)
                if num_columns > 50:
                    max_itemset_length = 3
                elif num_columns > 20:
                    max_itemset_length = 4
                else:
                    max_itemset_length = 5
                flash(f'Advertencia: Se estableció automáticamente longitud máxima de {max_itemset_length} para evitar cálculos excesivos.', 'info')
            elif max_itemset_length > 8:
                max_itemset_length = 8
                flash('Advertencia: Se limitó la longitud máxima a 8 para evitar cálculos excesivos.', 'warning')
                
            max_rules = request.form.get('max_rules')
            max_rules = int(max_rules) if max_rules else 100
            target_variable = request.form.get('target_variable')
            target_variable = target_variable if target_variable else None
            
            # Obtener algoritmo seleccionado
            algorithm = request.form.get('algorithm', 'apriori')
            
            # Configurar columnas según el tipo de análisis
            transaction_column = None
            item_columns = None
            
            if analysis_type == 'transaction':
                transaction_column = request.form.get('transaction_column')
                # Validar que la columna existe en el DataFrame
                if transaction_column and transaction_column not in df_for_analysis.columns:
                    flash(f'La columna de transacción "{transaction_column}" no existe en el dataset.', 'error')
                    return redirect(url_for('training.training'))
            elif analysis_type == 'basket':
                item_columns = request.form.getlist('item_columns')
                if not item_columns:
                    flash('Selecciona al menos 2 columnas para análisis de canasta.', 'error')
                    return redirect(url_for('training.training'))
                # Validar que todas las columnas existen
                missing_cols = [col for col in item_columns if col not in df_for_analysis.columns]
                if missing_cols:
                    flash(f'Las siguientes columnas no existen en el dataset: {", ".join(missing_cols)}', 'error')
                    return redirect(url_for('training.training'))
            
            # Ejecutar análisis
            analysis_results, basket_df, frequent_itemsets, rules = analyze_association_rules(
                df=df_for_analysis,
                transaction_column=transaction_column,
                item_columns=item_columns,
                min_support=min_support,
                min_confidence=min_confidence,
                max_itemset_length=max_itemset_length,
                max_rules=max_rules,
                target_variable=target_variable,
                algorithm=algorithm
            )
            
            if analysis_results['success']:
                # Obtener top reglas
                top_rules_confidence = get_top_rules(rules, 'confidence', 10) if rules is not None and not rules.empty else []
                top_rules_lift = get_top_rules(rules, 'lift', 10) if rules is not None and not rules.empty else []
                
                # Guardar solo un resumen ligero en la sesión para evitar headers demasiado grandes
                session['association_analysis'] = {
                    'success': True,
                    'summary': {
                        'total_rules': analysis_results['association_rules']['info'].get('total_rules', 0),
                        'total_itemsets': analysis_results['frequent_itemsets']['info'].get('total_itemsets', 0),
                        'parameters': analysis_results['parameters']
                    }
                }
                
                # Guardar análisis completo en archivo temporal
                import tempfile
                import joblib
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, f'association_temp_{session.get("filename", "data")}.joblib')
                
                try:
                    joblib.dump({
                        'analysis_results': analysis_results,
                        'basket_df': basket_df,
                        'frequent_itemsets': frequent_itemsets,
                        'rules': rules,
                        'top_rules_confidence': top_rules_confidence,
                        'top_rules_lift': top_rules_lift
                    }, temp_file)
                    session['association_temp_file'] = temp_file
                except Exception as save_error:
                    current_app.logger.error(f"Error guardando en temporal: {save_error}")
                    # Continuar sin archivo temporal si falla
                
                # Redirigir a la página de resultados
                return redirect(url_for('association.association_results'))
            else:
                flash(f'Error en el análisis: {analysis_results.get("error", "Error desconocido")}', 'error')
                return redirect(url_for('training.training'))
        
        # GET request - mostrar formulario
        available_algorithms = get_available_algorithms()
        return render_template('association_rules.html', 
                             columns=all_columns, 
                             available_algorithms=available_algorithms,
                             df_sample=df.head().to_dict('records'))
        
    except Exception as e:
        import traceback
        error_details = f'Error en análisis de reglas de asociación: {str(e)}\n\nDetalles técnicos:\n{traceback.format_exc()}'
        current_app.logger.error(error_details)
        flash(f'Error en análisis de reglas de asociación: {str(e)}', 'error')
        return redirect(url_for('training.training'))

@bp.route('/association_results')
def association_results():
    """Página para mostrar los resultados del análisis de reglas de asociación"""
    if 'association_analysis' not in session:
        flash('No hay resultados de análisis disponibles. Ejecuta primero un análisis.', 'warning')
        return redirect(url_for('association.association_rules_page'))
    
    # Cargar datos desde archivo temporal
    if 'association_temp_file' in session and os.path.exists(session['association_temp_file']):
        try:
            import joblib
            temp_data = joblib.load(session['association_temp_file'])
            analysis_results = temp_data.get('analysis_results')
            top_rules_confidence = temp_data.get('top_rules_confidence', [])
            top_rules_lift = temp_data.get('top_rules_lift', [])
            
            return render_template('association_results.html',
                                 analysis_results=analysis_results,
                                 top_rules_confidence=top_rules_confidence,
                                 top_rules_lift=top_rules_lift,
                                 filename=session.get('filename'))
        except Exception as e:
            flash(f'Error cargando resultados: {str(e)}', 'error')
            return redirect(url_for('association.association_rules_page'))
    else:
        flash('No se pudieron cargar los resultados del análisis.', 'error')
        return redirect(url_for('association.association_rules_page'))

@bp.route('/save_association_analysis')
def save_association_analysis_route():
    """Guarda el análisis de reglas de asociación"""
    if 'association_analysis' not in session:
        flash('No hay análisis para guardar.', 'warning')
        return redirect(url_for('main.index'))
    
    try:
        # Crear directorio de modelos si no existe
        models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Generar nombre de archivo
        filename = session.get('filename', 'unknown')
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_filename = f'association_analysis_{filename}_{timestamp}.joblib'
        save_path = os.path.join(models_dir, save_filename)
        
        # Cargar datos para guardar
        separator = session.get('csv_separator', ',')
        encoding = session.get('csv_encoding', 'utf-8')
        df = pd.read_csv(session['cleaned_file'], sep=separator, encoding=encoding)
        
        # IMPORTANTE: Limpiar nombres de columnas inmediatamente después de leer
        df.columns = clean_column_names(df.columns.tolist())
        
        analysis_results = session['association_analysis']
        
        # Re-ejecutar análisis para obtener objetos completos
        # (esto es necesario porque algunos objetos no se pueden serializar en session)
        analysis_results_full, basket_df, frequent_itemsets, rules = analyze_association_rules(
            df=df,
            **analysis_results['parameters']
        )
        
        # Guardar análisis
        success = save_association_analysis(
            analysis_results_full, basket_df, frequent_itemsets, rules, save_path
        )
        
        if success:
            flash(f'Análisis guardado como: {save_filename}', 'success')
        else:
            flash('Error al guardar el análisis.', 'error')
            
    except Exception as e:
        flash(f'Error al guardar análisis: {str(e)}', 'error')
    
    return redirect(request.referrer or url_for('main.index'))

@bp.route('/export_association_results')
def export_association_results():
    """Exportar resultados de reglas de asociación a CSV"""
    if 'association_analysis' not in session:
        flash('No hay resultados de análisis disponibles para exportar.', 'warning')
        return redirect(url_for('association.association_rules_page'))
    
    try:
        # Cargar datos desde archivo temporal
        if 'association_temp_file' in session and os.path.exists(session['association_temp_file']):
            import joblib
            temp_data = joblib.load(session['association_temp_file'])
            analysis_results = temp_data.get('analysis_results')
            
            if not analysis_results or 'association_rules' not in analysis_results:
                flash('No se encontraron reglas de asociación para exportar.', 'error')
                return redirect(url_for('association.association_results'))
            
            # Crear DataFrame con las reglas
            rules_data = []
            for i, rule in enumerate(analysis_results['association_rules']['data'], 1):
                rules_data.append({
                    'Regla_ID': i,
                    'Antecedente': ', '.join(rule['antecedents']),
                    'Consecuente': ', '.join(rule['consequents']),
                    'Soporte': round(rule['support'], 4),
                    'Confianza': round(rule['confidence'], 4),
                    'Lift': round(rule['lift'], 4),
                    'Leverage': round(rule.get('leverage', 0), 4),
                    'Conviction': round(rule.get('conviction', 0), 4)
                })
            
            # Crear DataFrame
            df_rules = pd.DataFrame(rules_data)
            
            # Generar nombre de archivo
            filename = session.get('filename', 'association_analysis')
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            export_filename = f'reglas_asociacion_{filename}_{timestamp}.csv'
            
            # Crear archivo CSV en memoria
            from io import StringIO
            import csv
            
            output = StringIO()
            df_rules.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            
            # Crear respuesta de descarga
            from flask import Response
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename={export_filename}',
                    'Content-Type': 'text/csv; charset=utf-8'
                }
            )
            
        else:
            flash('No se pudieron cargar los datos para exportar.', 'error')
            return redirect(url_for('association.association_results'))
            
    except Exception as e:
        current_app.logger.error(f"Error exporting association results: {str(e)}")
        flash(f'Error al exportar resultados: {str(e)}', 'error')
        return redirect(url_for('association.association_results'))