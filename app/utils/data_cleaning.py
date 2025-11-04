"""
Módulo para limpieza y análisis de datos
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .data_validation import convert_numpy_types

def analyze_cleaning_needs(df):
    """Analiza el DataFrame y recomienda operaciones de limpieza"""
    analysis = {
        'total_rows': int(len(df)),  # Convertir a int nativo
        'total_cols': int(len(df.columns)),  # Convertir a int nativo
        'recommendations': []
    }
    
    # 1. Analizar filas completamente vacías
    empty_rows = int(df.isnull().all(axis=1).sum())  # Convertir a int nativo
    analysis['empty_rows'] = empty_rows
    if empty_rows > 0:
        analysis['recommendations'].append({
            'type': 'remove_empty_rows',
            'title': 'Eliminar filas vacías',
            'description': f'Se encontraron {empty_rows} filas completamente vacías',
            'recommended': True,
            'impact': f'Se eliminarán {empty_rows} filas'
        })
    
    # 2. Analizar filas duplicadas
    duplicates = int(df.duplicated().sum())  # Convertir a int nativo
    analysis['duplicates'] = duplicates
    if duplicates > 0:
        analysis['recommendations'].append({
            'type': 'remove_duplicates',
            'title': 'Eliminar filas duplicadas',
            'description': f'Se encontraron {duplicates} filas duplicadas',
            'recommended': True,
            'impact': f'Se eliminarán {duplicates} filas duplicadas'
        })
    
    # 3. Analizar filas con excesivos valores faltantes
    threshold = 0.5  # 50% de las columnas
    missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
    rows_with_many_missing = int((missing_ratio > threshold).sum())  # Convertir a int
    analysis['rows_with_many_missing'] = rows_with_many_missing
    if rows_with_many_missing > 0:
        analysis['recommendations'].append({
            'type': 'remove_rows_many_missing',
            'title': 'Eliminar filas con muchos valores faltantes',
            'description': f'Se encontraron {rows_with_many_missing} filas con más del 50% de valores faltantes',
            'recommended': True,
            'impact': f'Se eliminarán {rows_with_many_missing} filas con datos insuficientes'
        })
    
    # 4. Analizar columnas completamente vacías
    empty_cols = int(df.isnull().all(axis=0).sum())  # Convertir a int
    empty_col_names = df.columns[df.isnull().all(axis=0)].tolist()
    analysis['empty_cols'] = empty_cols
    analysis['empty_col_names'] = empty_col_names
    if empty_cols > 0:
        analysis['recommendations'].append({
            'type': 'remove_empty_cols',
            'title': 'Eliminar columnas vacías',
            'description': f'Se encontraron {empty_cols} columnas completamente vacías: {", ".join(empty_col_names)}',
            'recommended': True,
            'impact': f'Se eliminarán {empty_cols} columnas'
        })
    
    # 5. Analizar espacios en columnas de texto
    text_cols_with_spaces = 0
    text_cols_names = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            # Verificar si hay espacios al inicio o final
            has_spaces = df[col].astype(str).str.strip().ne(df[col].astype(str)).any()
            if has_spaces:
                text_cols_with_spaces += 1
                text_cols_names.append(col)
    
    analysis['text_cols_with_spaces'] = text_cols_with_spaces
    analysis['text_cols_names'] = text_cols_names
    if text_cols_with_spaces > 0:
        analysis['recommendations'].append({
            'type': 'clean_text_spaces',
            'title': 'Limpiar espacios en texto',
            'description': f'Se encontraron espacios innecesarios en {text_cols_with_spaces} columnas de texto',
            'recommended': True,
            'impact': f'Se limpiarán espacios en: {", ".join(text_cols_names)}'
        })
    
    # 6. Analizar normalización/escalado de datos numéricos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_need_normalization = []
    normalization_info = {}
    
    for col in numeric_cols:
        if df[col].nunique() > 5:  # Solo para columnas con suficiente variabilidad
            col_range = df[col].max() - df[col].min()
            col_std = df[col].std()
            
            # Determinar si necesita normalización
            needs_norm = col_range > 100 or col_std > 10 or (df[col].min() < 0 and df[col].max() > 1000)
            
            if needs_norm:
                cols_need_normalization.append(col)
                
                # Recomendar método de normalización
                if col_range > 1000 or col_std > 100:
                    recommended_method = 'StandardScaler (Z-score)'
                elif df[col].min() >= 0:
                    recommended_method = 'MinMaxScaler (0-1)'
                else:
                    recommended_method = 'RobustScaler (robusto)'
                
                normalization_info[col] = {
                    'range': round(float(col_range), 2),
                    'std': round(float(col_std), 2),
                    'min': round(float(df[col].min()), 2),
                    'max': round(float(df[col].max()), 2),
                    'recommended_method': recommended_method
                }
    
    analysis['cols_need_normalization'] = cols_need_normalization
    analysis['normalization_info'] = normalization_info
    
    if cols_need_normalization:
        analysis['recommendations'].append({
            'type': 'normalize_data',
            'title': 'Normalizar/Escalar datos numéricos',
            'description': f'Se encontraron {len(cols_need_normalization)} columnas que se beneficiarían de normalización',
            'recommended': True,
            'impact': f'Se normalizarán: {", ".join(cols_need_normalization)}'
        })
    
    # Si no hay recomendaciones
    if not analysis['recommendations']:
        analysis['recommendations'].append({
            'type': 'no_cleaning',
            'title': 'No se requiere limpieza',
            'description': 'El dataset no presenta problemas obvios que requieran limpieza automática',
            'recommended': False,
            'impact': 'Ninguna operación necesaria'
        })
    
    # Convertir tipos de NumPy a tipos nativos de Python para serialización JSON
    def convert_numpy_types_recursive(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types_recursive(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    analysis = convert_numpy_types_recursive(analysis)
    
    return analysis

def clean_data_manual(df, options, protected_columns=None):
    """Aplica limpieza de datos según las opciones seleccionadas por el usuario
    
    Args:
        df: DataFrame a limpiar
        options: Diccionario con las opciones de limpieza
        protected_columns: Lista de columnas que no deben eliminarse
    """
    cleaning_report = []
    df_result = df.copy()
    
    # Asegurarse de que protected_columns es una lista
    if protected_columns is None:
        protected_columns = []
    
    # 1. Eliminar filas completamente vacías (si está seleccionado)
    if options.get('remove_empty_rows', False):
        rows_before = len(df_result)
        df_result = df_result.dropna(how='all')
        rows_after = len(df_result)
        if rows_before != rows_after:
            cleaning_report.append(f"Eliminadas {rows_before - rows_after} filas completamente vacías")
        else:
            cleaning_report.append("No se encontraron filas completamente vacías")
    
    # 2. Eliminar filas con excesivos valores faltantes (si está seleccionado)
    if options.get('remove_rows_many_missing', False):
        threshold = 0.5  # 50% de las columnas
        rows_before = len(df_result)
        missing_ratio = df_result.isnull().sum(axis=1) / len(df_result.columns)
        df_result = df_result[missing_ratio <= threshold]
        rows_after = len(df_result)
        if rows_before != rows_after:
            cleaning_report.append(f"Eliminadas {rows_before - rows_after} filas con más del 50% de valores faltantes")
        else:
            cleaning_report.append("No se encontraron filas con excesivos valores faltantes")
    
    # 3. Eliminar columnas completamente vacías (si está seleccionado)
    if options.get('remove_empty_cols', False):
        cols_before = len(df_result.columns)
        # Identificar columnas completamente vacías
        empty_cols = df_result.columns[df_result.isnull().all(axis=0)].tolist()
        # Filtrar las columnas protegidas (seleccionadas por el usuario)
        cols_to_remove = [col for col in empty_cols if col not in protected_columns]
        cols_protected = [col for col in empty_cols if col in protected_columns]
        
        if cols_to_remove:
            df_result = df_result.drop(columns=cols_to_remove)
            cleaning_report.append(f"Eliminadas {len(cols_to_remove)} columnas completamente vacías: {', '.join(cols_to_remove)}")
        
        if cols_protected:
            cleaning_report.append(f"Se mantuvieron {len(cols_protected)} columnas vacías seleccionadas: {', '.join(cols_protected)}")
        
        if not empty_cols:
            cleaning_report.append("No se encontraron columnas completamente vacías")
        
        cols_after = len(df_result.columns)
    
    # 4. Eliminar duplicados (si está seleccionado)
    if options.get('remove_duplicates', False):
        duplicates_before = df_result.duplicated().sum()
        df_result = df_result.drop_duplicates()
        if duplicates_before > 0:
            cleaning_report.append(f"Eliminadas {duplicates_before} filas duplicadas")
        else:
            cleaning_report.append("No se encontraron filas duplicadas")
    
    # 5. Limpiar espacios en columnas de texto (si está seleccionado)
    if options.get('clean_text_spaces', False):
        text_cols_cleaned = []
        for col in df_result.select_dtypes(include=['object']).columns:
            if df_result[col].dtype == 'object':
                original_values = df_result[col].astype(str)
                cleaned_values = original_values.str.strip()
                if not cleaned_values.equals(original_values):
                    df_result.loc[:, col] = cleaned_values
                    text_cols_cleaned.append(col)
        
        if text_cols_cleaned:
            cleaning_report.append(f"Limpiados espacios en {len(text_cols_cleaned)} columnas: {', '.join(text_cols_cleaned)}")
        else:
            cleaning_report.append("No se encontraron espacios innecesarios en columnas de texto")
    
    # 6. Normalizar/Escalar datos numéricos (si está seleccionado)
    if options.get('normalize_data', False):
        normalization_method = options.get('normalization_method', 'standard')
        # Solo obtener columnas numéricas que están en protected_columns (seleccionadas)
        all_numeric_cols = df_result.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in all_numeric_cols if col in protected_columns]
        normalized_cols = []
        
        for col in numeric_cols:
            if df_result[col].nunique() > 5:  # Solo normalizar columnas con suficiente variabilidad
                try:
                    col_data = df_result[[col]].dropna()
                    if len(col_data) > 0:
                        # Usar el mismo criterio que el análisis para determinar si necesita normalización
                        col_range = df_result[col].max() - df_result[col].min()
                        col_std = df_result[col].std()
                        needs_norm = col_range > 100 or col_std > 10 or (df_result[col].min() < 0 and df_result[col].max() > 1000)
                        
                        if needs_norm:  # Solo normalizar si cumple los criterios
                            if normalization_method == 'standard':
                                scaler = StandardScaler()
                                method_name = 'StandardScaler (Z-score)'
                            elif normalization_method == 'minmax':
                                scaler = MinMaxScaler()
                                method_name = 'MinMaxScaler (0-1)'
                            elif normalization_method == 'robust':
                                scaler = RobustScaler()
                                method_name = 'RobustScaler (robusto)'
                            else:
                                scaler = StandardScaler()
                                method_name = 'StandardScaler (Z-score)'
                            
                            # Aplicar normalización solo a valores no nulos
                            mask = df_result[col].notna()
                            df_result.loc[mask, col] = scaler.fit_transform(df_result.loc[mask, [col]]).flatten()
                            normalized_cols.append(col)
                        
                except Exception as e:
                    cleaning_report.append(f"Error al normalizar columna '{col}': {str(e)}")
        
        if normalized_cols:
            cleaning_report.append(f"Normalizadas {len(normalized_cols)} columnas con {method_name}: {', '.join(normalized_cols)}")
        else:
            cleaning_report.append("No se encontraron columnas numéricas que normalizar")
    
    # Si no se seleccionó ninguna opción
    if not any(options.values()):
        cleaning_report.append("No se seleccionó ninguna operación de limpieza")
    
    return df_result, cleaning_report

def clean_data_simple(df, protected_columns=None):
    """Realiza limpieza básica de datos de forma automática
    
    Args:
        df: DataFrame a limpiar
        protected_columns: Lista opcional de columnas que deben ser las únicas procesadas
    """
    cleaning_report = []
    
    # 1. Eliminar filas completamente vacías
    rows_before = len(df)
    df = df.dropna(how='all')
    rows_after = len(df)
    if rows_before != rows_after:
        cleaning_report.append(f"Eliminadas {rows_before - rows_after} filas completamente vacías")
    
    # 2. Eliminar columnas completamente vacías
    cols_before = len(df.columns)
    df = df.dropna(how='all', axis=1)
    cols_after = len(df.columns)
    if cols_before != cols_after:
        cleaning_report.append(f"Eliminadas {cols_before - cols_after} columnas completamente vacías")
    
    # 3. Eliminar duplicados
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    if duplicates_before > 0:
        cleaning_report.append(f"Eliminadas {duplicates_before} filas duplicadas")
    
    # 4. Eliminar filas con excesivos valores faltantes (>50% de columnas vacías)
    threshold = 0.5  # 50% de las columnas
    rows_before_missing = len(df)
    missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
    df = df[missing_ratio <= threshold]
    rows_after_missing = len(df)
    if rows_before_missing != rows_after_missing:
        cleaning_report.append(f"Eliminadas {rows_before_missing - rows_after_missing} filas con más del 50% de valores faltantes")
    
    # 5. Limpiar espacios en columnas de texto
    text_cols_cleaned = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            original_values = df[col].astype(str)
            cleaned_values = original_values.str.strip()
            if not cleaned_values.equals(original_values):
                df[col] = cleaned_values
                text_cols_cleaned.append(col)
    
    if text_cols_cleaned:
        cleaning_report.append(f"Limpiados espacios en {len(text_cols_cleaned)} columnas: {', '.join(text_cols_cleaned)}")
    
    # 6. Normalización básica de datos numéricos (StandardScaler para datos con alta variabilidad)
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Si se especifican columnas protegidas, solo normalizar esas
    if protected_columns:
        numeric_cols = [col for col in all_numeric_cols if col in protected_columns]
    else:
        numeric_cols = all_numeric_cols
    normalized_cols = []
    
    for col in numeric_cols:
        if df[col].nunique() > 5:  # Solo normalizar columnas con suficiente variabilidad
            col_range = df[col].max() - df[col].min()
            col_std = df[col].std()
            
            # Aplicar normalización si hay alta variabilidad
            if col_range > 100 or col_std > 10:
                try:
                    # Usar StandardScaler para normalización automática
                    scaler = StandardScaler()
                    mask = df[col].notna()
                    # Corregir warning de pandas: asegurar compatibilidad de tipos
                    scaled_values = scaler.fit_transform(df.loc[mask, [col]]).flatten()
                    df.loc[mask, col] = scaled_values.astype(df[col].dtype)
                    normalized_cols.append(col)
                except Exception:
                    pass  # Si falla la normalización, continuar sin ella
    
    if normalized_cols:
        cleaning_report.append(f"Normalizadas {len(normalized_cols)} columnas con StandardScaler: {', '.join(normalized_cols)}")
    
    return df, cleaning_report
