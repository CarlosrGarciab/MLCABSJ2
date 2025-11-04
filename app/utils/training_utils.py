"""
Utilidades para manejo de archivos en entrenamiento de modelos
"""
import os
import pandas as pd
from flask import session, current_app
import re

def clean_column_names(columns):
    """
    Limpia y estandariza nombres de columnas para mejor compatibilidad
    """
    cleaned = []
    # Expresión regular para eliminar TODOS los espacios y caracteres invisibles Unicode al inicio y final
    unicode_space_re = re.compile(r'^[\s\u00A0\u2000-\u200B\u2028\u2029]+|[\s\u00A0\u2000-\u200B\u2028\u2029]+$')
    for col in columns:
        clean_col = str(col)
        # Remover espacios y caracteres invisibles Unicode al inicio y final
        clean_col = unicode_space_re.sub('', clean_col)
        # Remover espacios múltiples internos
        clean_col = re.sub(r'\s+', ' ', clean_col)
        # Segunda pasada para asegurar limpieza completa
        clean_col = unicode_space_re.sub('', clean_col)
        cleaned.append(clean_col)
    return cleaned

def match_column_names(required_cols, available_cols):
    """
    Intenta hacer match entre columnas requeridas y disponibles
    usando diferentes estrategias de matching mejoradas
    """
    matches = {}
    # Usar la misma expresión regular que clean_column_names
    unicode_space_re = re.compile(r'^[\s\u00A0\u2000-\u200B\u2028\u2029]+|[\s\u00A0\u2000-\u200B\u2028\u2029]+$')
    # Limpiar y normalizar todas las columnas disponibles
    available_clean = {}
    for col in available_cols:
        clean_key = str(col)
        clean_key = unicode_space_re.sub('', clean_key)
        clean_key = re.sub(r'\s+', ' ', clean_key)
        clean_key = unicode_space_re.sub('', clean_key)
        clean_key = clean_key.lower()
        available_clean[clean_key] = col

    for req_col in required_cols:
        req_clean = str(req_col)
        req_clean = unicode_space_re.sub('', req_clean)
        req_clean = re.sub(r'\s+', ' ', req_clean)
        req_clean = unicode_space_re.sub('', req_clean)
        req_clean = req_clean.lower()

        # 1. Match exacto con limpieza agresiva
        if req_clean in available_clean:
            matches[req_col] = available_clean[req_clean]
            continue

        # 2. Match especial para casos como "IDIOMAS I" vs "IDIOMAS  I"
        for clean_key, original_col in available_clean.items():
            req_normalized = re.sub(r'\s+', ' ', req_clean.strip())
            available_normalized = re.sub(r'\s+', ' ', clean_key.strip())
            if req_normalized == available_normalized:
                matches[req_col] = original_col
                break

        if req_col in matches:
            continue

        # 3. Match parcial - buscar si la columna requerida está contenida en alguna disponible
        partial_matches = []
        req_words = req_clean.split()
        for clean_key, original_col in available_clean.items():
            available_words = clean_key.split()
            if all(word in available_words for word in req_words):
                partial_matches.append(original_col)
        if partial_matches:
            matches[req_col] = partial_matches[0]
            continue
    return matches

def get_training_file(target_column=None, feature_columns=None):
    """
    Determina qué archivo usar para entrenamiento basado en disponibilidad de columnas.
    
    Args:
        target_column: Nombre de la columna target
        feature_columns: Lista de columnas predictoras
    
    Returns:
        tuple: (file_path, dataframe, success)
    """
    current_app.logger.info("=== get_training_file DEBUG ===")
    current_app.logger.info(f"target_column: {target_column}")
    current_app.logger.info(f"feature_columns ({len(feature_columns) if feature_columns else 0}): {feature_columns}")
    current_app.logger.info(f"uploaded_file: {session.get('uploaded_file')}")
    current_app.logger.info(f"cleaned_file: {session.get('cleaned_file')}")
    current_app.logger.info(f"data_file: {session.get('data_file')}")
    
    # Lista de archivos a intentar en orden de preferencia (invertido para priorizar original)
    files_to_try = []
    
    # CAMBIO: Primero intentar con archivos que contengan más datos (113 columnas)
    # Estos archivos que empiezan con "datos_" o "archivo_" suelen tener más columnas
    data_file = session.get('data_file')
    if data_file:
        files_to_try.append(('data_file', data_file))
        
    uploaded_file = session.get('uploaded_file')
    if uploaded_file:
        # Priorizar archivos que probablemente tengan más columnas
        if 'datos_' in uploaded_file or 'archivo_' in uploaded_file:
            files_to_try.insert(0, ('uploaded_file_priority', uploaded_file))
        else:
            files_to_try.append(('uploaded_file', uploaded_file))
    
    # Luego cleaned_file si existe
    cleaned_file = session.get('cleaned_file')
    if cleaned_file:
        files_to_try.append(('cleaned_file', cleaned_file))
        # Agregar data_file de nuevo si existe
        if data_file:
            files_to_try.append(('data_file', data_file))
    
    required_columns = []
    if target_column:
        required_columns.append(target_column)
    if feature_columns:
        required_columns.extend(feature_columns)
    
    current_app.logger.info(f"Required columns: {required_columns}")
    current_app.logger.info(f"Files to try: {[f[0] for f in files_to_try]}")
    
    # Intentar cada archivo en orden
    for file_type, file_path in files_to_try:
        if not file_path:
            current_app.logger.warning(f"File {file_type} is None")
            continue
            
        if not os.path.exists(file_path):
            current_app.logger.warning(f"File {file_type} does not exist: {file_path}")
            continue
        
        try:
            current_app.logger.info(f"Trying {file_type}: {file_path}")
            
            # Intentar diferentes codificaciones y separadores (MEJORADO)
            df = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            separators = [';', ',', '\t', '|']
            
            successful_load = None
            best_column_count = 0
            
            for encoding in encodings:
                for separator in separators:
                    try:
                        temp_df = pd.read_csv(file_path, encoding=encoding, sep=separator, nrows=5)  # Solo leer primeras 5 filas para test
                        
                        # Evaluar la calidad del parseo
                        column_count = len(temp_df.columns)
                        row_count = len(temp_df)
                        
                        current_app.logger.info(f"Test {file_type} with {encoding} + '{separator}': {column_count} cols, {row_count} rows")
                        
                        # Preferir la combinación que genere más columnas (mejor parseo)
                        if column_count > best_column_count and column_count > 1 and row_count > 0:
                            best_column_count = column_count
                            successful_load = (encoding, separator)
                            current_app.logger.info(f"New best combination for {file_type}: {encoding} + '{separator}' ({column_count} columns)")
                        
                    except Exception as e:
                        continue
            
            # Cargar con la mejor combinación encontrada
            if successful_load:
                encoding, separator = successful_load
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=separator)
                    current_app.logger.info(f"SUCCESS: Read {file_type} with {encoding} + '{separator}' - Final: {len(df.columns)} cols, {len(df)} rows")
                except Exception as final_error:
                    current_app.logger.error(f"Failed final load of {file_type}: {final_error}")
                    continue
            else:
                current_app.logger.warning(f"No successful load combination found for {file_type}")
                continue
            
            if df is None:
                current_app.logger.error(f"Could not read {file_type} with any encoding/separator combination")
                continue
                
            available_columns = df.columns.tolist()
            # Limpiar nombres de columnas (quitar espacios y estandarizar)
            available_columns = clean_column_names(available_columns)
            df.columns = available_columns
            
            current_app.logger.info(f"Available columns in {file_type}: {available_columns[:10]}...")  # Mostrar solo las primeras 10
            
            # Verificar si todas las columnas requeridas están disponibles
            if required_columns:
                # Limpiar también las columnas requeridas
                required_columns_clean = clean_column_names(required_columns)
                
                # Intentar matching inteligente de columnas
                column_matches = match_column_names(required_columns_clean, available_columns)
                missing_columns = [col for col in required_columns_clean if col not in column_matches]
                found_columns = list(column_matches.keys())
                
                current_app.logger.info(f"=== COLUMN ANALYSIS FOR {file_type.upper()} ===")
                current_app.logger.info(f"Required columns (clean): {required_columns_clean[:5]}... (total: {len(required_columns_clean)})")
                current_app.logger.info(f"Available columns count: {len(available_columns)}")
                current_app.logger.info(f"Column matches found: {len(column_matches)}/{len(required_columns_clean)}")
                current_app.logger.info(f"Found columns: {found_columns[:5]}... (total: {len(found_columns)})")
                current_app.logger.info(f"Missing columns: {missing_columns[:5]}... (total: {len(missing_columns)})")
                
                # Si faltan columnas críticas, continuar con el siguiente archivo
                if missing_columns:
                    # Calcular porcentaje de columnas encontradas
                    match_percentage = len(column_matches) / len(required_columns_clean) * 100
                    current_app.logger.warning(f"Only {match_percentage:.1f}% of columns found in {file_type}")
                    
                    # CAMBIO: Ser más estricto - requerir al menos 80% de coincidencias
                    # para archivos críticos como IDIOMAS
                    if match_percentage < 80:
                        from flask import flash
                        # Mostrar columnas encontradas y faltantes en el mensaje
                        found_cols_str = ', '.join(found_columns[:10]) + ('...' if len(found_columns) > 10 else '')
                        missing_cols_str = ', '.join(missing_columns[:10]) + ('...' if len(missing_columns) > 10 else '')
                        flash(
                            f'Archivo {file_type}: Solo se encontraron {len(column_matches)}/{len(required_columns_clean)} columnas ({match_percentage:.1f}%). ' 
                            f'Encontradas: {found_cols_str}. Faltantes: {missing_cols_str}.',
                            'warning')
                        current_app.logger.warning(f"Too many missing columns in {file_type} ({match_percentage:.1f}%). Found: {found_cols_str}. Missing: {missing_cols_str}")
                        continue
                    else:
                        flash(f'Archivo {file_type}: Usando {len(column_matches)}/{len(required_columns_clean)} columnas disponibles ({match_percentage:.1f}%).', 'info')
                        current_app.logger.info(f"Acceptable match rate ({match_percentage:.1f}%) in {file_type}, proceeding")
                else:
                    current_app.logger.info(f"All required columns found in {file_type}")
            
            # Si llegamos aquí, el archivo es válido o aceptable
            current_app.logger.info(f"SUCCESS: Using {file_type}: {file_path}")
            return file_path, df, True
            
        except Exception as e:
            current_app.logger.error(f"Error reading {file_type}: {e}")
            continue
    
    # Si no se encontró ningún archivo válido
    current_app.logger.error("FAILURE: No valid file found with required columns")
    return None, None, False