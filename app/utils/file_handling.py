import os
import pandas as pd
from flask import current_app
from werkzeug.utils import secure_filename
from .training_utils import clean_column_names
from .csv_reader import CSVReader

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    if not os.path.exists(current_app.config['UPLOAD_FOLDER']):
        os.makedirs(current_app.config['UPLOAD_FOLDER'])
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return save_path

def get_csv_columns(file_path):
    """Obtiene las columnas de un archivo CSV usando CSVReader centralizado"""
    try:
        current_app.logger.info(f"=== GET_CSV_COLUMNS ===")
        current_app.logger.info(f"Processing file: {file_path}")
        
        # Usar CSVReader para lectura robusta
        df, read_info = CSVReader.read_csv_robust(file_path, nrows=1)
        
        if read_info['success'] and df is not None:
            columns = df.columns.tolist()
            current_app.logger.info(f"SUCCESS: {read_info['encoding_used']} + '{read_info['separator_used']}' - {len(columns)} columns")
            # Limpiar nombres de columnas antes de retornar
            return clean_column_names(columns)
        else:
            error_msg = read_info.get('error', 'Error desconocido')
            current_app.logger.error(f"Failed to read CSV columns: {error_msg}")
            raise Exception(f"No se pudo leer el archivo: {error_msg}")
    
    except Exception as e:
        current_app.logger.error(f"ERROR in get_csv_columns: {str(e)}")
        raise Exception(f"Error al leer el archivo CSV: {str(e)}")

def get_csv_preview(file_path, rows=5):
    """Obtiene una vista previa del archivo CSV usando CSVReader centralizado"""
    try:
        current_app.logger.info(f"=== GET_CSV_PREVIEW ===")
        current_app.logger.info(f"Processing file: {file_path} (rows={rows})")
        
        # Usar CSVReader para lectura robusta
        df, read_info = CSVReader.read_csv_robust(file_path, nrows=rows)
        
        if read_info['success'] and df is not None:
            current_app.logger.info(f"PREVIEW SUCCESS: {read_info['encoding_used']} + '{read_info['separator_used']}' - {len(df.columns)} columns, {len(df)} rows")
            return {
                'columns': clean_column_names(df.columns.tolist()),
                'data': df.values.tolist(),
                'separator': read_info['separator_used']
            }
        else:
            error_msg = read_info.get('error', 'Error desconocido')
            current_app.logger.error(f"Failed to get CSV preview: {error_msg}")
            raise Exception(f"No se pudo obtener vista previa: {error_msg}")
    
    except Exception as e:
        current_app.logger.error(f"ERROR in get_csv_preview: {str(e)}")
        raise Exception(f"Error al obtener vista previa: {str(e)}")