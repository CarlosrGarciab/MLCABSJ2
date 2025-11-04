"""
Sistema de Logging y Manejo de Errores
Proporciona logging centralizado y manejo de errores para la aplicación Flask
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from functools import wraps
from flask import current_app, request, session, g
import json

class ActivityLogger:
    """Sistema de logging de actividades del usuario"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Inicializa el sistema de logging para la aplicación"""
        # Crear directorio de logs si no existe
        logs_dir = os.path.join(app.root_path, '..', 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Configurar logging de actividades
        activity_log_file = os.path.join(logs_dir, 'activity.log')
        error_log_file = os.path.join(logs_dir, 'error.log')
        
        # Logger de actividades
        activity_logger = logging.getLogger('activity')
        activity_handler = logging.handlers.RotatingFileHandler(
            activity_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        activity_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        activity_handler.setFormatter(activity_formatter)
        activity_logger.addHandler(activity_handler)
        activity_logger.setLevel(logging.INFO)
        
        # Logger de errores
        error_logger = logging.getLogger('errors')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        error_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(pathname)s:%(lineno)d | %(funcName)s | %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)
        
        # Configurar el logger principal de Flask
        if not app.debug:
            flask_handler = logging.handlers.RotatingFileHandler(
                os.path.join(logs_dir, 'flask.log'), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            flask_formatter = logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            )
            flask_handler.setFormatter(flask_formatter)
            flask_handler.setLevel(logging.INFO)
            app.logger.addHandler(flask_handler)
            app.logger.setLevel(logging.INFO)
    
    @staticmethod
    def _sanitize_message(message):
        """Sanitiza mensajes para evitar problemas de codificación"""
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        elif not isinstance(message, str):
            message = str(message)
        
        # Reemplazar caracteres problemáticos
        message = message.encode('utf-8', errors='replace').decode('utf-8')
        return message
    
    @staticmethod
    def clean_log_files(app):
        """Limpia archivos de logs existentes con problemas de codificación"""
        try:
            logs_dir = os.path.join(app.root_path, '..', 'logs')
            if not os.path.exists(logs_dir):
                return
            
            log_files = ['activity.log', 'error.log', 'flask.log']
            
            for log_file in log_files:
                filepath = os.path.join(logs_dir, log_file)
                if os.path.exists(filepath):
                    try:
                        # Intentar leer el archivo
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # Si hay problemas de codificación, reescribir el archivo
                        with open(filepath, 'rb') as f:
                            binary_content = f.read()
                        
                        # Decodificar con reemplazo de caracteres problemáticos
                        clean_content = binary_content.decode('utf-8', errors='replace')
                        
                        # Reescribir el archivo limpio
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(clean_content)
                        
                        # Usar logging básico de Python ya que esto ocurre antes de inicialización de Flask
                        logging.info(f"Archivo {log_file} limpiado de problemas de codificación")
                        
        except Exception as e:
            # Usar logging básico de Python ya que esto ocurre antes de inicialización de Flask
            logging.error(f"Error limpiando archivos de logs: {e}")
    
    @staticmethod
    def log_activity(action, details=None, user_id=None):
        """Registra una actividad del usuario"""
        try:
            logger = logging.getLogger('activity')
            
            # Obtener información del contexto
            ip_address = request.remote_addr if request else 'Unknown'
            user_agent = request.headers.get('User-Agent', 'Unknown') if request else 'Unknown'
            session_id = session.get('_id', 'No session') if session else 'No session'
            
            # Preparar mensaje de log
            log_data = {
                'action': action,
                'ip': ip_address,
                'session_id': session_id[:8] if session_id != 'No session' else session_id,
                'user_agent': user_agent[:100] if user_agent else 'Unknown',
                'timestamp': datetime.now().isoformat()
            }
            
            if details:
                log_data['details'] = details
            
            if user_id:
                log_data['user_id'] = user_id
            
            # Agregar información del archivo actual si existe
            current_filename = session.get('filename') if session else None
            if current_filename:
                log_data['current_file'] = current_filename
            
            message = f"{action} | IP: {ip_address} | Session: {log_data['session_id']}"
            if details:
                message += f" | Details: {details}"
            
            # Sanitizar el mensaje antes de loggearlo
            message = ActivityLogger._sanitize_message(message)
            logger.info(message)
            
        except Exception as e:
            # Si falla el logging de actividad, al menos registrar en el logger principal
            if current_app:
                current_app.logger.error(f"Error logging activity: {e}")
    
    @staticmethod
    def log_error(error, context=None, user_id=None):
        """Registra un error detallado"""
        try:
            logger = logging.getLogger('errors')
            
            # Obtener información del error
            error_type = type(error).__name__
            error_message = str(error)
            error_traceback = traceback.format_exc()
            
            # Obtener información del contexto
            request_info = {}
            if request:
                request_info = {
                    'method': request.method,
                    'url': request.url,
                    'endpoint': request.endpoint,
                    'remote_addr': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', '')[:200]
                }
            
            session_info = {}
            if session:
                # Solo información relevante, no datos sensibles
                session_info = {
                    'session_id': session.get('_id', 'Unknown')[:8],
                    'has_file': 'uploaded_file' in session,
                    'filename': session.get('filename', 'None')
                }
            
            # Preparar mensaje completo
            error_data = {
                'error_type': error_type,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat(),
                'request_info': request_info,
                'session_info': session_info,
                'traceback': error_traceback
            }
            
            if context:
                error_data['context'] = context
            
            if user_id:
                error_data['user_id'] = user_id
            
            # Log del error
            message = f"{error_type}: {error_message}"
            if request:
                message += f" | {request.method} {request.endpoint}"
            if context:
                message += f" | Context: {context}"
            
            # Sanitizar los mensajes antes de logearlos
            message = ActivityLogger._sanitize_message(message)
            error_traceback = ActivityLogger._sanitize_message(error_traceback)
            
            logger.error(message)
            logger.error(f"Full traceback: {error_traceback}")
            
        except Exception as e:
            # Si falla el logging de errores, usar el logger básico
            if current_app:
                current_app.logger.error(f"Error logging error: {e}")
                current_app.logger.error(f"Original error: {error}")

# Instancia global del logger
activity_logger = ActivityLogger()

def log_activity(action, details=None, user_id=None):
    """Función de conveniencia para logging de actividades"""
    ActivityLogger.log_activity(action, details, user_id)

def log_error(error, context=None, user_id=None):
    """Función de conveniencia para logging de errores"""
    ActivityLogger.log_error(error, context, user_id)

def activity_required(action_name):
    """Decorador para logging automático de actividades"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Log del inicio de la actividad
                log_activity(f"START_{action_name}")
                
                # Ejecutar función
                result = f(*args, **kwargs)
                
                # Log del éxito
                log_activity(f"SUCCESS_{action_name}")
                
                return result
                
            except Exception as e:
                # Log del error
                log_error(e, f"Error in {action_name}")
                log_activity(f"ERROR_{action_name}", details=str(e))
                raise
        
        return decorated_function
    return decorator

def safe_execute(func, *args, default=None, error_message="Error executing function", **kwargs):
    """Ejecuta una función de forma segura con logging automático"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(e, error_message)
        if current_app:
            current_app.logger.error(f"{error_message}: {e}")
        return default

def get_user_session_info():
    """Obtiene información segura de la sesión del usuario"""
    if not session:
        return {}
    
    return {
        'session_id': session.get('_id', 'Unknown')[:8],
        'has_uploaded_file': 'uploaded_file' in session,
        'filename': session.get('filename'),
        'target_column': session.get('target_column'),
        'predictor_count': session.get('predictor_count', 0)
    }

def create_error_context(additional_info=None):
    """Crea un contexto detallado para el logging de errores"""
    context = {
        'timestamp': datetime.now().isoformat(),
        'session_info': get_user_session_info()
    }
    
    if request:
        context['request_info'] = {
            'method': request.method,
            'endpoint': request.endpoint,
            'url': request.url,
            'remote_addr': request.remote_addr
        }
    
    if additional_info:
        context['additional_info'] = additional_info
    
    return context

class MLError(Exception):
    """Excepción personalizada para errores de ML"""
    def __init__(self, message, error_type='general', details=None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        
        # Log automático del error
        log_error(self, f"MLError - {error_type}")

class DataError(MLError):
    """Error relacionado con datos"""
    def __init__(self, message, details=None):
        super().__init__(message, 'data_error', details)

class ModelError(MLError):
    """Error relacionado con modelos"""
    def __init__(self, message, details=None):
        super().__init__(message, 'model_error', details)

class ValidationError(MLError):
    """Error de validación"""
    def __init__(self, message, details=None):
        super().__init__(message, 'validation_error', details)