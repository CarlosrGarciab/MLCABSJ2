"""
Servicio centralizado para manejo de sesiones de Flask
"""
import os
import json
import uuid
import tempfile
from datetime import datetime, timedelta
from flask import session, current_app


class SessionService:
    """Servicio para manejo centralizado de sesiones"""
    
    # Límite máximo de tamaño de sesión en bytes (4KB por defecto)
    MAX_SESSION_SIZE = 4096
    
    @staticmethod
    def check_and_cleanup_session():
        """Verifica el tamaño de la sesión y limpia si es necesario"""
        try:
            # Estimar tamaño de sesión
            session_str = json.dumps(dict(session))
            session_size = len(session_str.encode('utf-8'))
            
            if session_size > SessionService.MAX_SESSION_SIZE:
                current_app.logger.warning(f"Sesión excede límite: {session_size} bytes. Limpiando...")
                SessionService.clean_session_for_new_model()
                
                # Verificar nuevamente
                session_str = json.dumps(dict(session))
                new_size = len(session_str.encode('utf-8'))
                current_app.logger.info(f"Sesión después de limpieza: {new_size} bytes")
                
                return True
            return False
        except Exception as e:
            current_app.logger.error(f"Error verificando tamaño de sesión: {e}")
            return False
    
    @staticmethod
    def clean_session_for_new_model():
        """Limpia datos antiguos de modelos de la sesión para liberar espacio"""
        keys_to_clean = [
            'trained_model', 'model_results', 'regression_results', 
            'trained_regression_model', 'classification_results',
            'association_analysis', 'association_temp_file',
            'prediction_results'
        ]
        
        cleaned_keys = []
        for key in keys_to_clean:
            if key in session:
                session.pop(key, None)
                cleaned_keys.append(key)
        
        if cleaned_keys:
            current_app.logger.info(f"Sesión limpiada. Claves removidas: {cleaned_keys}")
        
        return cleaned_keys
    
    @staticmethod
    def optimize_results_for_session(results, max_features=10, max_samples=5):
        """Optimiza los resultados del modelo para reducir el tamaño en sesión"""
        if not results:
            return results
            
        optimized = {
            'model_type': results.get('model_type')
        }
        
        # Métricas de regresión
        regression_metrics = ['train_r2', 'test_r2', 'train_mse', 'test_mse', 
                             'train_rmse', 'test_rmse', 'train_mae', 'test_mae']
        for metric in regression_metrics:
            if metric in results:
                optimized[metric] = results[metric]
        
        # Métricas de clasificación
        classification_metrics = ['train_accuracy', 'test_accuracy', 'test_precision_macro', 
                                 'test_precision_micro', 'test_recall_macro', 'test_recall_micro',
                                 'test_f1_macro', 'test_f1_micro', 'n_classes']
        for metric in classification_metrics:
            if metric in results:
                optimized[metric] = results[metric]
        
        # Limitar feature importances si existen
        if 'feature_importance' in results:
            feature_importance = results['feature_importance']
            if isinstance(feature_importance, dict) and len(feature_importance) > max_features:
                # Mantener solo las más importantes
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                optimized['feature_importance'] = dict(sorted_features[:max_features])
            else:
                optimized['feature_importance'] = feature_importance
        
        # Limitar matrices de confusión si son muy grandes
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            if hasattr(cm, 'shape') and cm.shape[0] > 10:
                optimized['confusion_matrix_summary'] = "Matriz disponible en logs"
            else:
                optimized['confusion_matrix'] = cm.tolist() if hasattr(cm, 'tolist') else cm
        
        # Copiar otras métricas importantes
        other_important_keys = ['preparation_info', 'model_params', 'training_time', 
                               'cross_val_score', 'cross_val_std']
        for key in other_important_keys:
            if key in results:
                optimized[key] = results[key]
        
        return optimized
    
    @staticmethod
    def store_results(key, results, optimize=True):
        """Almacena resultados en sesión, opcionalmente optimizados"""
        if optimize:
            results = SessionService.optimize_results_for_session(results)
        
        session[key] = results
        current_app.logger.info(f"Resultados almacenados en sesión con clave: {key}")
    
    @staticmethod
    def get_results(key, default=None):
        """Obtiene resultados de la sesión"""
        results = session.get(key, default)
        if results:
            current_app.logger.info(f"Resultados recuperados de sesión con clave: {key}")
        return results
    
    @staticmethod
    def store_large_results(key, results, max_session_size=3000):
        """Almacena resultados grandes usando archivo temporal si excede el límite"""
        try:
            # Intentar almacenar directamente en sesión primero
            json_str = json.dumps(results)
            if len(json_str) < max_session_size:
                session[key] = results
                current_app.logger.info(f"Results stored directly in session for key: {key} ({len(json_str)} bytes)")
                return True
            
            # Si es muy grande, usar almacenamiento en archivo
            return SessionService._store_to_file(key, results)
            
        except Exception as e:
            current_app.logger.error(f"Error storing large results: {e}")
            return False
    
    @staticmethod
    def get_large_results(key, default=None):
        """Recupera resultados de sesión o archivo temporal"""
        try:
            # Verificar si está en sesión directamente
            if key in session:
                return session[key]
            
            # Verificar si hay una referencia de archivo
            file_key = f"{key}_file_id"
            if file_key in session:
                file_id = session[file_key]
                return SessionService._load_from_file(file_id)
            
            return default
            
        except Exception as e:
            current_app.logger.error(f"Error getting large results: {e}")
            return default
    
    @staticmethod
    def _store_to_file(key, results):
        """Almacena resultados en archivo temporal"""
        try:
            # Generar ID único para el archivo
            file_id = str(uuid.uuid4())
            
            # Usar el directorio temp_results existente
            temp_dir = os.path.join(current_app.root_path, '..', 'temp_results')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            file_path = os.path.join(temp_dir, f"session_results_{file_id}.json")
            
            # Guardar resultados con timestamp
            data_to_save = {
                'timestamp': datetime.now().isoformat(),
                'data': results
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            # Guardar referencia en sesión
            session[f"{key}_file_id"] = file_id
            
            current_app.logger.info(f"Large results stored in file: {file_path} for key: {key}")
            return True
            
        except Exception as e:
            current_app.logger.error(f"Error storing to file: {e}")
            return False
    
    @staticmethod
    def _load_from_file(file_id):
        """Carga resultados desde archivo temporal"""
        try:
            temp_dir = os.path.join(current_app.root_path, '..', 'temp_results')
            file_path = os.path.join(temp_dir, f"session_results_{file_id}.json")
            
            if not os.path.exists(file_path):
                current_app.logger.warning(f"Session file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verificar si el archivo es muy antiguo (más de 24 horas)
            timestamp = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - timestamp > timedelta(hours=24):
                current_app.logger.warning(f"Session file is too old, removing: {file_path}")
                os.remove(file_path)
                return None
            
            current_app.logger.info(f"Large results loaded from file: {file_path}")
            return data['data']
            
        except Exception as e:
            current_app.logger.error(f"Error loading from file: {e}")
            return None
    
    @staticmethod
    def cleanup_old_files(max_age_hours=24):
        """Limpia archivos temporales antiguos (session_results, training_results, temp_models)"""
        try:
            temp_dir = os.path.join(current_app.root_path, '..', 'temp_results')
            if not os.path.exists(temp_dir):
                return
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            files_removed = 0
            total_size_freed = 0
            
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                should_remove = False
                file_size = 0
                
                try:
                    file_size = os.path.getsize(file_path)
                    
                    # Para session_results_* usar timestamp interno
                    if filename.startswith('session_results_'):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            timestamp = datetime.fromisoformat(data['timestamp'])
                            should_remove = timestamp < cutoff_time
                        except Exception:
                            # Si no se puede leer el archivo o está corrupto, eliminarlo
                            should_remove = True
                    
                    # Para training_results_* y temp_model_* usar fecha de modificación del archivo
                    elif filename.startswith(('training_results_', 'temp_model_')):
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        should_remove = file_mtime < cutoff_time
                    
                    # Eliminar archivos que no siguen el patrón esperado
                    elif filename.endswith(('.json', '.joblib')):
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        should_remove = file_mtime < cutoff_time
                    
                    if should_remove:
                        os.remove(file_path)
                        files_removed += 1
                        total_size_freed += file_size
                        current_app.logger.debug(f"Removed old temp file: {filename} ({file_size} bytes)")
                        
                except Exception as e:
                    current_app.logger.warning(f"Error processing file {filename}: {e}")
                    try:
                        # Si hay error, intentar eliminar el archivo problemático
                        os.remove(file_path)
                        files_removed += 1
                        current_app.logger.info(f"Removed problematic file: {filename}")
                    except Exception:
                        pass
            
            if files_removed > 0:
                size_mb = total_size_freed / (1024 * 1024)
                current_app.logger.info(f"Cleaned up {files_removed} old temp files, freed {size_mb:.2f} MB")
            else:
                current_app.logger.debug("No old temp files to clean up")
                
        except Exception as e:
            current_app.logger.error(f"Error cleaning up old files: {e}")
    
    @staticmethod
    def clear_all():
        """Limpia completamente la sesión"""
        session.clear()
        current_app.logger.info("Sesión completamente limpiada")

    @staticmethod
    def cleanup_now(max_age_hours=168):  # 7 días por defecto para limpieza manual
        """Fuerza una limpieza inmediata con edad configurable"""
        current_app.logger.info(f"Starting manual cleanup of files older than {max_age_hours} hours")
        SessionService.cleanup_old_files(max_age_hours)
        
    @staticmethod
    def get_temp_files_info():
        """Obtiene información sobre los archivos temporales"""
        try:
            temp_dir = os.path.join(current_app.root_path, '..', 'temp_results')
            if not os.path.exists(temp_dir):
                return {"total_files": 0, "total_size": 0, "by_type": {}}
            
            info = {
                "total_files": 0,
                "total_size": 0,
                "by_type": {
                    "session_results": {"count": 0, "size": 0},
                    "training_results": {"count": 0, "size": 0},
                    "temp_models": {"count": 0, "size": 0},
                    "other": {"count": 0, "size": 0}
                }
            }
            
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    file_size = os.path.getsize(file_path)
                    info["total_files"] += 1
                    info["total_size"] += file_size
                    
                    if filename.startswith('session_results_'):
                        info["by_type"]["session_results"]["count"] += 1
                        info["by_type"]["session_results"]["size"] += file_size
                    elif filename.startswith('training_results_'):
                        info["by_type"]["training_results"]["count"] += 1
                        info["by_type"]["training_results"]["size"] += file_size
                    elif filename.startswith('temp_model_'):
                        info["by_type"]["temp_models"]["count"] += 1
                        info["by_type"]["temp_models"]["size"] += file_size
                    else:
                        info["by_type"]["other"]["count"] += 1
                        info["by_type"]["other"]["size"] += file_size
                        
                except Exception:
                    pass
            
            return info
            
        except Exception as e:
            current_app.logger.error(f"Error getting temp files info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def clear_large_results(key):
        """Limpia resultados de la sesión, tanto de memoria como de archivo"""
        try:
            # Limpiar de la sesión directa
            if key in session:
                session.pop(key, None)
                current_app.logger.info(f"Cleared session data for key: {key}")
            
            # Limpiar archivo temporal asociado
            file_key = f"{key}_file_id"
            if file_key in session:
                file_id = session[file_key]
                temp_dir = os.path.join(current_app.root_path, '..', 'temp_results')
                file_path = os.path.join(temp_dir, f"session_results_{file_id}.json")
                
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        current_app.logger.info(f"Removed session file: {file_path}")
                    except Exception as e:
                        current_app.logger.warning(f"Could not remove session file: {e}")
                
                session.pop(file_key, None)
                current_app.logger.info(f"Cleared file reference for key: {key}")
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Error clearing large results for key {key}: {e}")
            return False