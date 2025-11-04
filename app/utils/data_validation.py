"""
Módulo para validación de datos
"""
import pandas as pd
import numpy as np

def convert_numpy_types(obj):
    """Convierte tipos de numpy/pandas a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, frozenset):
        return list(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def validate_data_simple(df):
    """Realiza validación básica de datos"""
    validation_report = {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'missing_values': {},
        'data_types': {},
        'duplicates': int(df.duplicated().sum()),
        'is_valid': True,
        'warnings': []
    }
    
    # Verificar valores faltantes por columna
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        validation_report['missing_values'][col] = {
            'count': int(missing_count),
            'percentage': round(float(missing_percentage), 2)
        }
        
        # Advertencia si hay muchos valores faltantes
        if missing_percentage > 50:
            validation_report['warnings'].append(f"Columna '{col}' tiene {missing_percentage:.1f}% de valores faltantes")
            validation_report['is_valid'] = False
    
    # Verificar tipos de datos
    for col in df.columns:
        validation_report['data_types'][col] = str(df[col].dtype)
    
    # Verificar si hay suficientes datos
    if len(df) < 10:
        validation_report['warnings'].append("El dataset tiene menos de 10 filas")
        validation_report['is_valid'] = False
    
    # Convertir todos los tipos numpy a tipos nativos de Python
    validation_report = convert_numpy_types(validation_report)
    
    return validation_report
