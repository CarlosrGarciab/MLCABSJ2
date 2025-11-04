"""
Utilidades para análisis de variables target y compatibilidad con modelos
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

def analyze_target_compatibility(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Analiza una columna target y determina qué tipos de modelos son compatibles
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna target
        
    Returns:
        Dict con información sobre compatibilidad y características del target
    """
    if target_column not in df.columns:
        return {
            'error': f'Columna "{target_column}" no encontrada',
            'classification_compatible': False,
            'regression_compatible': False
        }
    
    target_series = df[target_column].dropna()
    
    if len(target_series) == 0:
        return {
            'error': 'La columna target no tiene valores válidos',
            'classification_compatible': False,
            'regression_compatible': False
        }
    
    analysis = {
        'column_name': target_column,
        'total_values': len(df[target_column]),
        'valid_values': len(target_series),
        'missing_values': len(df[target_column]) - len(target_series),
        'data_type': str(target_series.dtype),
        'unique_values': target_series.nunique(),
        'sample_values': target_series.unique()[:10].tolist()  # Mostrar hasta 10 valores únicos
    }
    
    # Determinar si es numérico
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    # Determinar si es categórico/discreto
    is_categorical = not is_numeric or target_series.nunique() <= 20  # Máximo 20 categorías únicas
    
    # Si es numérico, verificar si podría ser categórico
    if is_numeric:
        # Verificar si son todos enteros y hay pocas categorías únicas
        is_integer_like = target_series.apply(lambda x: float(x).is_integer()).all()
        few_unique_values = target_series.nunique() <= 10
        
        if is_integer_like and few_unique_values:
            analysis['could_be_categorical'] = True
            analysis['numeric_but_discrete'] = True
        else:
            analysis['could_be_categorical'] = False
            analysis['numeric_but_discrete'] = False
    
    # Compatibilidad con clasificación
    classification_compatible = True
    classification_notes = []
    
    if not is_categorical:
        if target_series.nunique() > 50:
            classification_compatible = False
            classification_notes.append(f"Demasiadas categorías únicas ({target_series.nunique()}). Máximo recomendado: 20")
    
    if target_series.nunique() < 2:
        classification_compatible = False
        classification_notes.append("Necesita al menos 2 categorías diferentes")
    
    # Compatibilidad con regresión
    regression_compatible = True
    regression_notes = []
    
    if not is_numeric:
        regression_compatible = False
        regression_notes.append("La variable target debe ser numérica para regresión")
    else:
        # Verificar si todos los valores son iguales
        if target_series.nunique() == 1:
            regression_compatible = False
            regression_notes.append("Todos los valores son iguales, no hay variación para predecir")
        
        # Verificar si hay suficiente variación
        if is_numeric and target_series.nunique() < 5:
            regression_notes.append(f"Pocas variaciones únicas ({target_series.nunique()}). Considera usar clasificación")
    
    analysis.update({
        'is_numeric': is_numeric,
        'is_categorical': is_categorical,
        'classification_compatible': classification_compatible,
        'classification_notes': classification_notes,
        'regression_compatible': regression_compatible,
        'regression_notes': regression_notes
    })
    
    # Recomendación principal
    if classification_compatible and regression_compatible:
        if is_numeric and target_series.nunique() > 10:
            analysis['primary_recommendation'] = 'regression'
            analysis['recommendation_reason'] = 'Variable numérica continua con muchas variaciones'
        else:
            analysis['primary_recommendation'] = 'classification'
            analysis['recommendation_reason'] = 'Variable categórica o numérica discreta'
    elif classification_compatible:
        analysis['primary_recommendation'] = 'classification'
        analysis['recommendation_reason'] = 'Solo compatible con clasificación'
    elif regression_compatible:
        analysis['primary_recommendation'] = 'regression' 
        analysis['recommendation_reason'] = 'Solo compatible con regresión'
    else:
        analysis['primary_recommendation'] = 'none'
        analysis['recommendation_reason'] = 'Variable no compatible con ningún tipo de modelo'
    
    return analysis

def get_target_summary(df: pd.DataFrame, target_column: str) -> str:
    """
    Genera un resumen legible sobre el target seleccionado
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna target
        
    Returns:
        String con resumen del target
    """
    analysis = analyze_target_compatibility(df, target_column)
    
    if 'error' in analysis:
        return f"Error: {analysis['error']}"
    
    summary = f"Variable '{target_column}': "
    summary += f"{analysis['unique_values']} valores únicos "
    
    if analysis['is_numeric']:
        summary += f"(numérica - {analysis['data_type']})"
    else:
        summary += "(categórica/texto)"
    
    if analysis['missing_values'] > 0:
        summary += f". {analysis['missing_values']} valores faltantes"
    
    return summary

def get_model_availability_message(analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Genera mensajes sobre disponibilidad de modelos basado en el análisis del target
    
    Args:
        analysis: Resultado de analyze_target_compatibility
        
    Returns:
        Dict con mensajes para cada tipo de modelo
    """
    messages = {}
    
    # Mensaje para clasificación
    if analysis['classification_compatible']:
        if analysis['primary_recommendation'] == 'classification':
            messages['classification'] = {
                'status': 'recommended',
                'message': f'Recomendado para clasificación ({analysis["unique_values"]} categorías)',
                'class': 'alert-success'
            }
        else:
            messages['classification'] = {
                'status': 'available',
                'message': f'Compatible con clasificación ({analysis["unique_values"]} categorías)',
                'class': 'alert-info'
            }
    else:
        reason = '. '.join(analysis['classification_notes'])
        messages['classification'] = {
            'status': 'unavailable',
            'message': f'No compatible con clasificación: {reason}',
            'class': 'alert-danger'
        }
    
    # Mensaje para regresión
    if analysis['regression_compatible']:
        if analysis['primary_recommendation'] == 'regression':
            messages['regression'] = {
                'status': 'recommended', 
                'message': 'Recomendado para regresión (variable numérica continua)',
                'class': 'alert-success'
            }
        else:
            notes = '. '.join(analysis['regression_notes']) if analysis['regression_notes'] else ''
            if notes:
                message = f'Compatible con regresión pero se recomienda clasificación. {notes}'
            else:
                message = 'Compatible con regresión pero se recomienda clasificación para este tipo de variable'
            messages['regression'] = {
                'status': 'available',
                'message': message,
                'class': 'alert-warning'
            }
    else:
        reason = '. '.join(analysis['regression_notes'])
        messages['regression'] = {
            'status': 'unavailable',
            'message': f'No compatible con regresión: {reason}',
            'class': 'alert-danger'
        }
    
    return messages