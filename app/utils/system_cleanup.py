"""
Sistema centralizado de limpieza y mantenimiento del sistema
Reemplaza múltiples funciones de cleanup dispersas por un sistema unificado
"""
import os
import glob
import time
import threading
from datetime import datetime, timedelta
from flask import current_app


class SystemCleanupManager:
    """Gestor centralizado de limpieza del sistema"""
    
    def __init__(self):
        self.cleanup_interval = 3600  # 1 hora en segundos
        self.cleanup_thread = None
        self.is_running = False
    
    def init_app(self, app):
        """Inicializar el sistema de limpieza con la aplicación Flask"""
        with app.app_context():
            self.start_cleanup_service()
    
    def start_cleanup_service(self):
        """Iniciar el servicio de limpieza automática"""
        if not self.is_running:
            self.is_running = True
            self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
            self.cleanup_thread.start()
            current_app.logger.info("Sistema de limpieza iniciado")
    
    def stop_cleanup_service(self):
        """Detener el servicio de limpieza"""
        self.is_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        current_app.logger.info("Sistema de limpieza detenido")
    
    def _periodic_cleanup(self):
        """Ejecutar limpieza periódica"""
        while self.is_running:
            try:
                time.sleep(self.cleanup_interval)
                if self.is_running:  # Verificar nuevamente después del sleep
                    self.run_full_cleanup()
            except Exception as e:
                current_app.logger.error(f"Error en limpieza periódica: {e}")
    
    def run_full_cleanup(self, max_age_hours=24):
        """Ejecutar limpieza completa del sistema"""
        try:
            from app.utils.logging_system import log_activity
            
            # 1. Limpiar archivos de sesión antiguos
            self._cleanup_session_files(max_age_hours)
            
            # 2. Limpiar modelos temporales antiguos
            self._cleanup_temp_models(max_age_hours)
            
            # 2.1. Limpiar modelos de reentrenamiento temporales
            self._cleanup_temp_retrained_models(max_age_hours)
            
            # 3. Limpiar predicciones antiguas
            self._cleanup_old_predictions(max_age_hours)
            
            # 4. Limpiar archivos de uploads antiguos (opcional)
            self._cleanup_old_uploads(max_age_hours * 7)  # 7 días para uploads
            
            # 5. Rotar logs si son muy grandes
            self._rotate_large_logs()
            
            log_activity("SYSTEM_CLEANUP", details=f"Limpieza completa ejecutada (max_age: {max_age_hours}h)")
            current_app.logger.info(f"Limpieza del sistema completada (max_age: {max_age_hours}h)")
            
        except Exception as e:
            from app.utils.logging_system import log_error
            log_error(e, "Error en limpieza completa del sistema")
            current_app.logger.error(f"Error en limpieza del sistema: {e}")
    
    def _cleanup_session_files(self, max_age_hours):
        """Limpiar archivos de sesión antiguos"""
        try:
            from app.services.session_service import SessionService
            SessionService.cleanup_old_files(max_age_hours)
        except Exception as e:
            current_app.logger.error(f"Error limpiando archivos de sesión: {e}")
    
    def _cleanup_temp_models(self, max_age_hours):
        """Limpiar modelos temporales antiguos"""
        try:
            models_dir = os.path.join(current_app.root_path, '..', 'models')
            if not os.path.exists(models_dir):
                return
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            temp_files = glob.glob(os.path.join(models_dir, 'temp_*.joblib'))
            
            cleaned_count = 0
            for temp_file in temp_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(temp_file))
                    if file_time < cutoff_time:
                        os.remove(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    current_app.logger.warning(f"No se pudo eliminar {temp_file}: {e}")
            
            if cleaned_count > 0:
                current_app.logger.info(f"Eliminados {cleaned_count} modelos temporales antiguos")
                
        except Exception as e:
            current_app.logger.error(f"Error limpiando modelos temporales: {e}")
    
    def _cleanup_temp_retrained_models(self, max_age_hours):
        """Limpiar modelos de reentrenamiento temporales antiguos"""
        try:
            temp_results_dir = current_app.config['TEMP_RESULTS_FOLDER']
            if not os.path.exists(temp_results_dir):
                return
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Buscar archivos de modelos reentrenados temporales
            temp_model_files = glob.glob(os.path.join(temp_results_dir, 'retrained_model_*.joblib'))
            temp_metadata_files = glob.glob(os.path.join(temp_results_dir, 'retrained_metadata_*.json'))
            
            cleaned_count = 0
            for temp_file in temp_model_files + temp_metadata_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(temp_file))
                    if file_time < cutoff_time:
                        os.remove(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    current_app.logger.warning(f"No se pudo eliminar {temp_file}: {e}")
            
            if cleaned_count > 0:
                current_app.logger.info(f"Eliminados {cleaned_count} archivos de modelos reentrenados temporales")
                
        except Exception as e:
            current_app.logger.error(f"Error limpiando modelos reentrenados temporales: {e}")
    
    def _cleanup_old_predictions(self, max_age_hours):
        """Limpiar predicciones temporales antiguas"""
        try:
            temp_results_dir = os.path.join(current_app.root_path, '..', 'temp_results')
            if not os.path.exists(temp_results_dir):
                return
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            temp_files = glob.glob(os.path.join(temp_results_dir, '*'))
            
            cleaned_count = 0
            for temp_file in temp_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(temp_file))
                    if file_time < cutoff_time:
                        if os.path.isfile(temp_file):
                            os.remove(temp_file)
                            cleaned_count += 1
                except Exception as e:
                    current_app.logger.warning(f"No se pudo eliminar {temp_file}: {e}")
            
            if cleaned_count > 0:
                current_app.logger.info(f"Eliminados {cleaned_count} archivos de predicciones temporales")
                
        except Exception as e:
            current_app.logger.error(f"Error limpiando predicciones temporales: {e}")
    
    def _cleanup_old_uploads(self, max_age_hours):
        """Limpiar uploads muy antiguos (opcional y conservador)"""
        try:
            uploads_dir = os.path.join(current_app.root_path, '..', 'uploads')
            if not os.path.exists(uploads_dir):
                return
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            upload_files = glob.glob(os.path.join(uploads_dir, '*'))
            
            cleaned_count = 0
            for upload_file in upload_files:
                try:
                    # Solo eliminar archivos temporales, no archivos de ejemplo
                    if 'temp_' in os.path.basename(upload_file) or 'cleaned_' in os.path.basename(upload_file):
                        file_time = datetime.fromtimestamp(os.path.getmtime(upload_file))
                        if file_time < cutoff_time:
                            if os.path.isfile(upload_file):
                                os.remove(upload_file)
                                cleaned_count += 1
                except Exception as e:
                    current_app.logger.warning(f"No se pudo eliminar {upload_file}: {e}")
            
            if cleaned_count > 0:
                current_app.logger.info(f"Eliminados {cleaned_count} archivos de uploads temporales")
                
        except Exception as e:
            current_app.logger.error(f"Error limpiando uploads antiguos: {e}")
    
    def _rotate_large_logs(self, max_size_mb=50):
        """Rotar logs que sean muy grandes"""
        try:
            logs_dir = os.path.join(current_app.root_path, '..', 'logs')
            if not os.path.exists(logs_dir):
                return
            
            log_files = glob.glob(os.path.join(logs_dir, '*.log'))
            max_size_bytes = max_size_mb * 1024 * 1024
            
            for log_file in log_files:
                try:
                    if os.path.getsize(log_file) > max_size_bytes:
                        # Crear backup y truncar
                        backup_name = f"{log_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        os.rename(log_file, backup_name)
                        
                        # Crear nuevo archivo log vacío
                        with open(log_file, 'w', encoding='utf-8') as f:
                            f.write(f"# Log rotado automáticamente - {datetime.now().isoformat()}\n")
                        
                        current_app.logger.info(f"Log rotado: {log_file} -> {backup_name}")
                        
                except Exception as e:
                    current_app.logger.warning(f"No se pudo rotar {log_file}: {e}")
                    
        except Exception as e:
            current_app.logger.error(f"Error rotando logs: {e}")
    
    def manual_cleanup_now(self, max_age_hours=1):
        """Ejecutar limpieza manual inmediata"""
        current_app.logger.info("Iniciando limpieza manual del sistema")
        self.run_full_cleanup(max_age_hours)
    
    def get_cleanup_status(self):
        """Obtener estado del sistema de limpieza"""
        return {
            'is_running': self.is_running,
            'cleanup_interval': self.cleanup_interval,
            'thread_alive': self.cleanup_thread.is_alive() if self.cleanup_thread else False
        }

# Instancia global del gestor de limpieza
cleanup_manager = SystemCleanupManager()