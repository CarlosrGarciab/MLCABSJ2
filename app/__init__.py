from flask import Flask
from config import Config
import atexit
import threading
import time
import logging.handlers

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    config_class.init_app(app)
    
    # Inicializar sistema de logging
    from app.utils.logging_system import activity_logger, ActivityLogger
    activity_logger.init_app(app)
    
    # Limpiar archivos de logs con problemas de codificación
    ActivityLogger.clean_log_files(app)
    
    # Inicializar sistema de autenticación
    from app.utils.auth import auth
    auth.init_app(app)
    app.auth = auth  # Hacer accesible la instancia de auth
    
    # Registrar manejadores de errores
    from app.utils.error_handlers import register_error_handlers
    register_error_handlers(app)

    # Registrar blueprints usando la nueva estructura modular
    from app.routes import register_blueprints
    register_blueprints(app)
    
    # Registrar rutas de administración
    from app.routes.admin import bp as admin_bp
    app.register_blueprint(admin_bp)
    
    # Inicializar sistema centralizado de limpieza
    from app.utils.system_cleanup import cleanup_manager
    cleanup_manager.init_app(app)
    
    # Ejecutar limpieza inicial
    with app.app_context():
        try:
            from app.utils.logging_system import log_activity
            cleanup_manager.manual_cleanup_now(max_age_hours=24)
            log_activity("SYSTEM_START", details="Application started with centralized cleanup system")
        except Exception as e:
            app.logger.error(f"Error in initial cleanup: {e}")
            from app.utils.logging_system import log_error
            log_error(e, "Error in initial startup cleanup")

    return app