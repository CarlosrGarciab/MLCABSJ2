"""
Manejadores de errores personalizados para la aplicación Flask
Proporciona páginas de error personalizadas y logging automático
"""

from flask import render_template, request, current_app
from app.utils.logging_system import log_error, log_activity

def register_error_handlers(app):
    """Registra todos los manejadores de errores para la aplicación"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Página de error 404 - No encontrado"""
        log_activity("ERROR_404", details=f"URL: {request.url}")
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(403)
    def forbidden_error(error):
        """Página de error 403 - Prohibido"""
        log_error(error, "403 Forbidden access attempt")
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(500)
    def internal_error(error):
        """Página de error 500 - Error interno del servidor"""
        log_error(error, "500 Internal server error")
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(413)
    def too_large_error(error):
        """Error 413 - Archivo demasiado grande"""
        log_activity("ERROR_413", details="File too large uploaded")
        return render_template('errors/413.html'), 413
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Manejador genérico para excepciones no capturadas"""
        # Log del error con contexto completo
        log_error(error, f"Unhandled exception in {request.endpoint}")
        
        # Si es un error HTTP conocido, no interferir
        if hasattr(error, 'code'):
            return error
        
        # Para errores desconocidos, mostrar página genérica
        current_app.logger.error(f"Unhandled exception: {error}")
        return render_template('errors/500.html'), 500