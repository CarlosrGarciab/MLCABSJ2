"""
Módulo para análisis de reglas de asociación usando mlxtend
"""
import pandas as pd
import numpy as np
from flask import current_app
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from .data_validation import convert_numpy_types

def get_available_algorithms():
    """Retorna los algoritmos disponibles para reglas de asociación"""
    return {
        'apriori': {
            'name': 'Apriori',
            'description': 'Algoritmo clásico para minería de patrones frecuentes. Bueno para datasets pequeños a medianos.'
        },
        'fpgrowth': {
            'name': 'FP-Growth',
            'description': 'Algoritmo más eficiente que usa árboles FP. Ideal para datasets grandes.'
        }
    }

def prepare_data_for_apriori(df, transaction_column=None, item_columns=None):
    """Prepara los datos para análisis de reglas de asociación"""
    try:
        # Limpiar nombres de columnas del DataFrame al inicio
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        
        # Si item_columns fue especificado, también limpiar esos nombres
        if item_columns:
            item_columns = [str(col).strip() for col in item_columns]
        
        preparation_info = {
            'original_rows': len(df),
            'preparation_method': '',
            'transaction_count': 0,
            'unique_items': 0,
            'data_format': ''
        }
        
        # Método 1: Datos transaccionales (una columna con items separados)
        if transaction_column and transaction_column in df.columns:
            # Suponer que los items están separados por comas o espacios
            transactions = []
            for transaction in df[transaction_column].dropna():
                if isinstance(transaction, str):
                    # Separar por comas, punto y coma o espacios
                    items = [item.strip() for item in transaction.replace(';', ',').split(',')]
                    transactions.append([item for item in items if item])
                
            preparation_info['preparation_method'] = 'transaction_column'
            preparation_info['transaction_count'] = len(transactions)
            preparation_info['data_format'] = 'single_column_transactions'
            
            # Convertir a formato one-hot
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            basket_df = pd.DataFrame(te_ary, columns=te.columns_)
            
            preparation_info['unique_items'] = len(te.columns_)
            
            return basket_df, preparation_info, transactions
        
        # Método 2: Datos en formato canasta (columnas categóricas/nominales)
        elif item_columns:
            # Crear transacciones con formato columna=valor para cada fila
            transactions = []
            
            for _, row in df.iterrows():
                transaction = []
                for col in item_columns:
                    # Limpiar el nombre de la columna de espacios extra
                    clean_col = str(col).strip()
                    
                    # Verificar que el valor no sea nulo, vacío o cero
                    if pd.notna(row[col]) and str(row[col]).strip() not in ['', '0', 'nan', 'NaN']:
                        try:
                            # Formato: columna=valor, ambos limpiados
                            clean_value = str(row[col]).strip()
                            item = f"{clean_col}={clean_value}"
                            transaction.append(item)
                        except Exception as e:
                            # Si hay error con un item específico, continuar con los demás
                            current_app.logger.warning(f"Warning: Error procesando {col}={row[col]}: {e}")
                            continue
                
                if transaction:  # Solo agregar transacciones no vacías
                    transactions.append(transaction)
            
            # Convertir a formato one-hot usando TransactionEncoder
            if not transactions:
                raise ValueError("No se pudieron crear transacciones válidas. "
                               "Verifique que las columnas seleccionadas tengan valores válidos.")
            
            try:
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                basket_df = pd.DataFrame(te_ary, columns=te.columns_)
            except Exception as e:
                raise ValueError(f"Error al convertir transacciones a formato binario: {str(e)}. "
                               f"Se crearon {len(transactions)} transacciones.")
            
            preparation_info['preparation_method'] = 'item_columns_with_values'
            preparation_info['transaction_count'] = len(transactions)
            preparation_info['unique_items'] = len(te.columns_)
            preparation_info['data_format'] = 'categorical_to_basket_format'
            
            return basket_df, preparation_info, transactions
        
        # Método 3: Auto-detectar formato
        else:
            # Intentar detectar columnas categóricas/nominales
            categorical_columns = []
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                # Incluir columnas con valores categóricos razonables (no demasiados valores únicos)
                if len(unique_vals) <= 50 and len(unique_vals) >= 2:
                    categorical_columns.append(col)
            
            if len(categorical_columns) >= 2:
                # Usar columnas categóricas detectadas
                return prepare_data_for_apriori(df, item_columns=categorical_columns)
            else:
                raise ValueError("No se pudo detectar un formato adecuado para reglas de asociación. "
                               "Especifique 'transaction_column' o 'item_columns'. "
                               f"Se encontraron {len(categorical_columns)} columnas categóricas válidas "
                               "(se necesitan al menos 2).")
                
    except ValueError as ve:
        # Errores de validación de datos - más específicos
        raise Exception(f"Error de validación de datos: {str(ve)}")
    except Exception as e:
        # Otros errores - incluir más contexto
        error_msg = str(e)
        if "ADMINISTRACION Y MERCADOTECNIA" in error_msg or any(col for col in item_columns or [] if "ADMINISTRACION" in col):
            error_msg += ". Sugerencia: Algunos nombres de columnas pueden tener espacios extra o caracteres especiales."
        raise Exception(f"Error al preparar datos para Apriori: {error_msg}")

def find_frequent_itemsets(basket_df, min_support=0.1, max_len=None, algorithm='apriori'):
    """Encuentra itemsets frecuentes usando el algoritmo especificado"""
    try:
        import time
        start_time = time.time()
        
        # Establecer límite por defecto para evitar cálculos excesivos
        if max_len is None:
            # Calcular límite automático basado en número de columnas
            num_items = len(basket_df.columns)
            if num_items > 50:
                max_len = 3  # Para datasets grandes, limitar a 3 elementos
            elif num_items > 20:
                max_len = 4  # Para datasets medianos, limitar a 4 elementos
            else:
                max_len = 5  # Para datasets pequeños, hasta 5 elementos
            
            current_app.logger.warning(f"No se especificó max_len. Usando límite automático de {max_len} para {num_items} elementos únicos.")
        
        # Ejecutar algoritmo seleccionado
        current_app.logger.info(f"Ejecutando algoritmo {algorithm} con min_support={min_support}, max_len={max_len}")
        
        if algorithm == 'apriori':
            frequent_itemsets = apriori(
                basket_df, 
                min_support=min_support, 
                use_colnames=True,
                max_len=max_len
            )
        elif algorithm == 'fpgrowth':
            frequent_itemsets = fpgrowth(
                basket_df, 
                min_support=min_support, 
                use_colnames=True,
                max_len=max_len
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}. Algoritmos disponibles: apriori, fpgrowth")
        
        # Calcular tiempo de ejecución
        elapsed_time = time.time() - start_time
        current_app.logger.info(f"Algoritmo {algorithm} completado en {elapsed_time:.2f} segundos")
        
        if elapsed_time > 30:  # Si tarda más de 30 segundos
            current_app.logger.warning(f"El algoritmo tardó {elapsed_time:.2f} segundos. Considera reducir max_len o aumentar min_support.")
        
        if frequent_itemsets.empty:
            return pd.DataFrame(), {
                'total_itemsets': 0,
                'min_support_used': min_support,
                'algorithm_used': algorithm,
                'execution_time_seconds': elapsed_time,
                'warning': 'No se encontraron itemsets frecuentes con el soporte mínimo especificado'
            }
        
        # Agregar información adicional y convertir frozensets a listas
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        frequent_itemsets['items'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
        # Convertir frozensets a listas para serialización JSON
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: list(x))
        
        # Estadísticas
        itemset_info = {
            'total_itemsets': len(frequent_itemsets),
            'min_support_used': min_support,
            'algorithm_used': algorithm,
            'execution_time_seconds': elapsed_time,
            'max_support_found': float(frequent_itemsets['support'].max()),
            'avg_support': float(frequent_itemsets['support'].mean()),
            'itemsets_by_length': frequent_itemsets['length'].value_counts().to_dict()
        }
        
        return frequent_itemsets, itemset_info
        
    except Exception as e:
        raise Exception(f"Error al encontrar itemsets frecuentes: {str(e)}")

def generate_association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6, max_rules=None):
    """Genera reglas de asociación a partir de itemsets frecuentes de forma eficiente"""
    try:
        if frequent_itemsets.empty:
            return pd.DataFrame(), {
                'total_rules': 0,
                'metric_used': metric,
                'min_threshold_used': min_threshold,
                'warning': 'No hay itemsets frecuentes para generar reglas'
            }
        
        # Si max_rules es 0, devolver vacío inmediatamente
        if max_rules is not None and max_rules == 0:
            return pd.DataFrame(), {
                'total_rules': 0,
                'metric_used': metric,
                'min_threshold_used': min_threshold,
                'limited_to': 0,
                'optimization_note': 'Generación optimizada: se solicitaron 0 reglas'
            }
        
        # OPTIMIZACIÓN: Generar reglas de forma eficiente
        if max_rules is not None and max_rules > 0:
            # Estrategia optimizada: intentar con diferentes thresholds para obtener exactamente las reglas necesarias
            current_threshold = min_threshold
            best_rules = pd.DataFrame()
            attempts = 0
            max_attempts = 5
            
            current_app.logger.info(f"Optimización activada: buscando exactamente {max_rules} reglas de alta calidad")
            
            while attempts < max_attempts and (best_rules.empty or len(best_rules) < max_rules):
                try:
                    # Generar reglas con el threshold actual
                    rules = association_rules(
                        frequent_itemsets, 
                        metric=metric, 
                        min_threshold=current_threshold
                    )
                    
                    if not rules.empty:
                        # Ordenar por calidad (confianza y lift)
                        rules = rules.sort_values(['confidence', 'lift'], ascending=False)
                        
                        # Si tenemos más reglas de las necesarias, tomar solo las mejores
                        if len(rules) >= max_rules:
                            best_rules = rules.head(max_rules)
                            current_app.logger.info(f"Encontradas {max_rules} reglas de alta calidad con threshold {current_threshold:.3f}")
                            break
                        else:
                            # Si no tenemos suficientes, guardar las mejores encontradas y reducir threshold
                            best_rules = rules
                            current_app.logger.info(f"Encontradas {len(rules)} reglas con threshold {current_threshold:.3f}, reduciendo threshold...")
                    
                    # Reducir threshold para encontrar más reglas en la siguiente iteración
                    current_threshold = max(0.1, current_threshold - 0.1)
                    attempts += 1
                    
                except Exception as e:
                    current_app.logger.warning(f"Error en iteración optimizada {attempts}: {str(e)}")
                    break
            
            if not best_rules.empty:
                rules = best_rules
                rules_info = {
                    'total_rules': len(rules),
                    'metric_used': metric,
                    'min_threshold_used': min_threshold,
                    'final_threshold_used': current_threshold + 0.1,  # El último threshold que funcionó
                    'limited_to': max_rules,
                    'optimization_note': f'Generación optimizada: {len(rules)} reglas generadas de {max_rules} solicitadas en {attempts} iteraciones',
                    'generated_efficiently': True
                }
            else:
                # Fallback: generar todas las reglas si la optimización falla
                current_app.logger.warning("Optimización falló, generando todas las reglas...")
                rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
                if len(rules) > max_rules:
                    rules = rules.nlargest(max_rules, ['confidence', 'lift'])
                rules_info = {
                    'total_rules': len(rules),
                    'metric_used': metric,
                    'min_threshold_used': min_threshold,
                    'limited_to': max_rules,
                    'optimization_note': 'Fallback: optimización falló, se generaron todas las reglas',
                    'generated_efficiently': False
                }
        else:
            # Sin límite: generar todas las reglas (comportamiento original)
            rules = association_rules(
                frequent_itemsets, 
                metric=metric, 
                min_threshold=min_threshold
            )
            rules_info = {
                'total_rules': len(rules),
                'metric_used': metric,
                'min_threshold_used': min_threshold,
                'optimization_note': 'Sin límite: se generaron todas las reglas posibles'
            }
        
        if rules.empty:
            return pd.DataFrame(), {
                'total_rules': 0,
                'metric_used': metric,
                'min_threshold_used': min_threshold,
                'warning': f'No se encontraron reglas con {metric} >= {min_threshold}'
            }
        
        # Agregar representación legible de las reglas y convertir frozensets a listas
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['rule'] = rules['antecedents_str'] + ' => ' + rules['consequents_str']
        
        # Convertir frozensets a listas para serialización JSON
        rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
        
        # Agregar estadísticas adicionales si no se hizo en la optimización
        if 'avg_confidence' not in rules_info:
            rules_info.update({
                'avg_confidence': float(rules['confidence'].mean()),
                'avg_lift': float(rules['lift'].mean()),
                'avg_support': float(rules['support'].mean()),
                'max_confidence': float(rules['confidence'].max()),
                'max_lift': float(rules['lift'].max()),
                'rules_with_high_confidence': len(rules[rules['confidence'] >= 0.8]),
                'rules_with_high_lift': len(rules[rules['lift'] >= 1.5])
            })
        
        return rules, rules_info
        
    except Exception as e:
        raise Exception(f"Error al generar reglas de asociación: {str(e)}")

def analyze_association_rules(df, transaction_column=None, item_columns=None, 
                            min_support=0.1, min_confidence=0.6, 
                            max_itemset_length=None, max_rules=100, target_variable=None,
                            algorithm='apriori'):
    """Análisis completo de reglas de asociación"""
    try:
        # ADVERTENCIA Y AJUSTE AUTOMÁTICO DE PARÁMETROS
        num_cols = len(item_columns) if item_columns else len(df.columns)
        if num_cols > 20:
            if min_support < 0.2:
                min_support = 0.2
                current_app.logger.warning('Se aumentó el soporte mínimo a 0.2 para evitar cálculos excesivos.')
            if max_itemset_length is None or max_itemset_length > 3:
                max_itemset_length = 3
                current_app.logger.warning('Se redujo la longitud máxima de itemset a 3 para evitar cálculos excesivos.')
        # Si el número de filas es muy alto, advertir al usuario
        if len(df) > 10000:
            current_app.logger.warning('El dataset tiene más de 10,000 filas. Considera filtrar o reducir el tamaño para un análisis más rápido.')

        # Preparar datos
        basket_df, prep_info, transactions = prepare_data_for_apriori(
            df, transaction_column, item_columns
        )
        # Encontrar itemsets frecuentes
        frequent_itemsets, itemset_info = find_frequent_itemsets(
            basket_df, min_support, max_itemset_length, algorithm
        )
        # Generar reglas de asociación
        rules, rules_info = generate_association_rules(
            frequent_itemsets, 'confidence', min_confidence, max_rules
        )
        # Filtrar reglas por target si se especifica
        if not rules.empty and target_variable:
            # Filtrar reglas que tengan el target como consequent principal o único
            target_rules = []
            for idx, rule in rules.iterrows():
                # Verificar los consequents como lista
                consequents = rule['consequents'] if isinstance(rule['consequents'], list) else [rule['consequents']]
                
                # Buscar si hay items que empiecen con el target variable
                has_target = False
                for consequent_item in consequents:
                    if str(consequent_item).startswith(f"{target_variable}="):
                        has_target = True
                        break
                
                if has_target:
                    target_rules.append(rule.to_dict())
            
            if target_rules:
                rules = pd.DataFrame(target_rules)
                # Recalcular estadísticas
                rules_info['total_rules_before_target_filter'] = rules_info.get('total_rules', len(target_rules))
                rules_info['total_rules'] = len(rules)
                rules_info['filtered_by_target'] = target_variable
                
                # Recalcular estadísticas solo para las reglas filtradas
                if len(rules) > 0:
                    rules_info['avg_confidence'] = float(rules['confidence'].mean())
                    rules_info['avg_lift'] = float(rules['lift'].mean())
                    rules_info['avg_support'] = float(rules['support'].mean())
                    rules_info['max_confidence'] = float(rules['confidence'].max())
                    rules_info['max_lift'] = float(rules['lift'].max())
                    rules_info['rules_with_high_confidence'] = len(rules[rules['confidence'] >= 0.8])
                    rules_info['rules_with_high_lift'] = len(rules[rules['lift'] >= 1.5])
            else:
                # Si no hay reglas con el target, devolver DataFrame vacío con advertencia
                rules = pd.DataFrame()
                rules_info['target_filter_warning'] = f"No se encontraron reglas con {target_variable} como consecuente"
                rules_info['total_rules'] = 0
        
        # Análisis adicional - optimizado para evitar problemas de serialización
        analysis_results = convert_numpy_types({
            'data_preparation': prep_info,
            'frequent_itemsets': {
                'data': frequent_itemsets.head(50).to_dict('records') if not frequent_itemsets.empty else [],  # Solo primeros 50
                'info': itemset_info
            },
            'association_rules': {
                'data': rules.to_dict('records') if not rules.empty else [],  # Todas las reglas (ya limitadas)
                'info': rules_info
            },
            'parameters': {
                'min_support': min_support,
                'min_confidence': min_confidence,
                'max_itemset_length': max_itemset_length,
                'max_rules': max_rules,
                'target_variable': target_variable,
                'algorithm': algorithm
            },
            'sample_transactions': transactions[:5] if transactions else [],  # Solo 5 ejemplos
            'success': True
        })
        return analysis_results, basket_df, frequent_itemsets, rules
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'data_preparation': {'error': str(e)},
            'frequent_itemsets': {'data': [], 'info': {}},
            'association_rules': {'data': [], 'info': {}},
            'parameters': {
                'min_support': min_support,
                'min_confidence': min_confidence,
                'max_itemset_length': max_itemset_length,
                'max_rules': max_rules,
                'target_variable': target_variable
            }
        }, None, None, None

def get_top_rules(rules, metric='confidence', top_n=10):
    """Obtiene las mejores reglas según una métrica"""
    try:
        if rules.empty:
            return []
        
        top_rules = rules.nlargest(top_n, metric)
        
        result = []
        for _, rule in top_rules.iterrows():
            result.append({
                'rule': rule['rule'],
                'antecedents': rule['antecedents_str'],
                'consequents': rule['consequents_str'],
                'support': float(rule['support']),
                'confidence': float(rule['confidence']),
                'lift': float(rule['lift']),
                'conviction': float(rule['conviction']) if 'conviction' in rule else None
            })
        
        return result
        
    except Exception as e:
        raise Exception(f"Error al obtener top reglas: {str(e)}")

def save_association_analysis(analysis_results, basket_df, frequent_itemsets, rules, save_path):
    """Guarda el análisis de reglas de asociación"""
    try:
        import joblib
        
        analysis_data = {
            'analysis_results': analysis_results,
            'basket_data': basket_df,
            'frequent_itemsets': frequent_itemsets,
            'association_rules': rules,
            'analysis_type': 'association_rules'
        }
        
        joblib.dump(analysis_data, save_path)
        return True
        
    except Exception as e:
        raise Exception(f"Error al guardar análisis: {str(e)}")

def load_association_analysis(file_path):
    """Carga un análisis de reglas de asociación guardado"""
    try:
        import joblib
        
        analysis_data = joblib.load(file_path)
        return (analysis_data['analysis_results'], 
                analysis_data['basket_data'], 
                analysis_data['frequent_itemsets'], 
                analysis_data['association_rules'])
        
    except Exception as e:
        raise Exception(f"Error al cargar análisis: {str(e)}")
