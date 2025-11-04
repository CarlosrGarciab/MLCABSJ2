"""
Utilidad centralizada para lectura de archivos CSV
Elimina duplicación de código y centraliza el manejo de encoding y separadores
"""
import pandas as pd
import os
from flask import current_app
from typing import Optional, Tuple, List, Dict, Any


class CSVReader:
    """Clase para manejar la lectura robusta de archivos CSV"""
    
    # Configuraciones por defecto
    DEFAULT_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    DEFAULT_SEPARATORS = [',', ';', '\t', '|']  # Comma first for better compatibility
    
    @staticmethod
    def read_csv_robust(
        file_path: str,
        separator: Optional[str] = None,
        encoding: Optional[str] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Lee un archivo CSV de manera robusta, probando diferentes encodings y separadores
        
        Args:
            file_path: Ruta al archivo CSV
            separator: Separador específico a usar (si se conoce)
            encoding: Encoding específico a usar (si se conoce)
            nrows: Número de filas a leer (para pruebas)
            **kwargs: Argumentos adicionales para pd.read_csv
            
        Returns:
            Tupla de (DataFrame, info_dict) donde info_dict contiene:
            - 'success': bool
            - 'encoding_used': str
            - 'separator_used': str
            - 'error': str (si hubo error)
            - 'attempts': int (número de intentos realizados)
        """
        
        if not os.path.exists(file_path):
            return None, {
                'success': False,
                'error': f'Archivo no encontrado: {file_path}',
                'attempts': 0
            }
        
        # Si se especifican encoding y separator, intentar primero con esos
        encodings = [encoding] if encoding else CSVReader.DEFAULT_ENCODINGS
        separators = [separator] if separator else CSVReader.DEFAULT_SEPARATORS
        
        attempts = 0
        last_error = None
        
        for enc in encodings:
            for sep in separators:
                try:
                    attempts += 1
                    
                    # Leer archivo CSV
                    df = pd.read_csv(
                        file_path, 
                        sep=sep, 
                        encoding=enc, 
                        nrows=nrows,
                        **kwargs
                    )
                    
                    # Validar que el DataFrame tiene contenido válido
                    # Preferir resultados con múltiples columnas
                    is_valid = False
                    
                    if len(df.columns) > 1:
                        # Multiple columns is always better
                        is_valid = True
                        current_app.logger.info(f"CSV leído exitosamente con {len(df.columns)} columnas: {file_path} (encoding={enc}, sep='{sep}')")
                    elif len(df.columns) == 1 and len(df) > 0:
                        # Single column - check if it might be improperly separated
                        first_cell = str(df.iloc[0, 0]) if len(df) > 0 else ""
                        if ',' in first_cell or ';' in first_cell or '\t' in first_cell:
                            # This single column likely contains unseparated data, skip it
                            current_app.logger.debug(f"Skipping single column with separators inside: {first_cell[:50]}...")
                            continue
                        else:
                            # Legitimate single column
                            is_valid = True
                            current_app.logger.info(f"CSV leído como columna única: {file_path} (encoding={enc}, sep='{sep}')")
                    
                    if is_valid:
                        return df, {
                            'success': True,
                            'encoding_used': enc,
                            'separator_used': sep,
                            'attempts': attempts,
                            'rows': len(df),
                            'columns': len(df.columns)
                        }
                        
                except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                    last_error = str(e)
                    continue
                except Exception as e:
                    last_error = str(e)
                    current_app.logger.warning(f"Error inesperado leyendo CSV: {e}")
                    continue
        
        # Si llegamos aquí, no se pudo leer el archivo
        return None, {
            'success': False,
            'error': f'No se pudo leer el archivo después de {attempts} intentos. Último error: {last_error}',
            'attempts': attempts
        }
    
    @staticmethod
    def detect_csv_properties(file_path: str) -> Dict[str, Any]:
        """
        Detecta las propiedades de un archivo CSV (encoding y separador)
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            Dict con 'encoding', 'separator', 'success', y detalles
        """
        
        # Intentar con una muestra pequeña para detectar propiedades
        df, info = CSVReader.read_csv_robust(file_path, nrows=5)
        
        if info['success']:
            return {
                'success': True,
                'encoding': info['encoding_used'],
                'separator': info['separator_used'],
                'sample_columns': df.columns.tolist() if df is not None else [],
                'sample_rows': len(df) if df is not None else 0
            }
        else:
            return {
                'success': False,
                'error': info.get('error', 'Error desconocido'),
                'encoding': None,
                'separator': None
            }
    
    @staticmethod
    def read_csv_with_fallbacks(
        file_paths: List[str],
        separator: Optional[str] = None,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Intenta leer múltiples archivos CSV en orden hasta encontrar uno válido
        
        Args:
            file_paths: Lista de rutas de archivos a intentar
            separator: Separador específico
            encoding: Encoding específico
            **kwargs: Argumentos adicionales para pd.read_csv
            
        Returns:
            Tupla de (DataFrame, info_dict)
        """
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                df, info = CSVReader.read_csv_robust(
                    file_path, 
                    separator=separator, 
                    encoding=encoding, 
                    **kwargs
                )
                
                if info['success']:
                    info['file_used'] = file_path
                    return df, info
                    
        return None, {
            'success': False,
            'error': f'No se pudo leer ninguno de los archivos: {file_paths}',
            'files_tried': file_paths
        }
