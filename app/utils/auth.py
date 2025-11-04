"""
Sistema de Autenticación Básica
Proporciona protección para secciones sensibles de la aplicación
"""

import os
import hashlib
import secrets
from functools import wraps
from flask import request, jsonify, session, current_app, redirect, url_for, render_template, abort
from app.utils.logging_system import log_activity, log_error

class BasicAuth:
    """Sistema de autenticación básica para proteger rutas sensibles"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Inicializar autenticación con la aplicación Flask"""
        # Configuración por defecto
        app.config.setdefault('ADMIN_USERNAME', os.environ.get('ADMIN_USERNAME', 'admin'))
        app.config.setdefault('ADMIN_PASSWORD', os.environ.get('ADMIN_PASSWORD', 'admin123'))
        app.config.setdefault('SESSION_TIMEOUT', 3600)  # 1 hora por defecto
        app.config.setdefault('MAX_LOGIN_ATTEMPTS', 5)
        app.config.setdefault('LOGIN_RATE_LIMIT', 300)  # 5 minutos de bloqueo
        
        # Registrar rutas de autenticación
        app.add_url_rule('/admin/login', 'admin_login', self.login_view, methods=['GET', 'POST'])
        app.add_url_rule('/admin/logout', 'admin_logout', self.logout_view)
        
        # Middleware para verificar autenticación en rutas admin
        app.before_request(self._check_admin_auth)
    
    def _check_admin_auth(self):
        """Verificar autenticación antes de cada request"""
        # Solo verificar rutas que empiecen con /admin (excepto login/logout)
        if (request.endpoint and 
            request.endpoint.startswith('admin.') and 
            request.endpoint not in ['admin_login', 'admin_logout']):
            
            if not self.is_authenticated():
                log_activity("AUTH_REQUIRED", 
                           details=f"Authentication required for {request.endpoint}")
                
                # Si es una petición AJAX, devolver JSON
                if request.headers.get('Content-Type') == 'application/json':
                    return jsonify({'error': 'Authentication required', 'redirect': '/admin/login'}), 401
                
                return redirect(url_for('admin_login'))
    
    def is_authenticated(self):
        """Verificar si el usuario está autenticado"""
        if 'admin_authenticated' not in session:
            return False
        
        # Verificar timeout de sesión
        login_time = session.get('admin_login_time')
        if login_time:
            timeout = current_app.config['SESSION_TIMEOUT']
            
            if (login_time + timeout) < self._current_timestamp():
                self.logout()
                return False
        
        return session.get('admin_authenticated', False)
    
    def authenticate(self, username, password):
        """Autenticar usuario con credenciales"""
        # Verificar límite de intentos
        if not self._check_rate_limit():
            log_activity("AUTH_RATE_LIMITED", 
                       details=f"Rate limit exceeded for IP {request.remote_addr}")
            return False, "Demasiados intentos de login. Intenta más tarde."
        
        admin_username = current_app.config['ADMIN_USERNAME']
        admin_password = current_app.config['ADMIN_PASSWORD']
        
        # Verificar credenciales de forma segura
        username_valid = secrets.compare_digest(username, admin_username)
        password_valid = secrets.compare_digest(password, admin_password)
        
        if username_valid and password_valid:
            # Login exitoso
            session['admin_authenticated'] = True
            session['admin_login_time'] = self._current_timestamp()
            session['admin_username'] = username
            session.permanent = True
            
            # Limpiar intentos fallidos
            self._clear_failed_attempts()
            
            log_activity("AUTH_SUCCESS", 
                       details=f"Admin login successful for {username}")
            return True, "Login exitoso"
        else:
            # Login fallido
            self._record_failed_attempt()
            log_activity("AUTH_FAILED", 
                       details=f"Failed login attempt for {username}")
            return False, "Credenciales incorrectas"
    
    def logout(self):
        """Cerrar sesión de administrador"""
        username = session.get('admin_username', 'unknown')
        
        session.pop('admin_authenticated', None)
        session.pop('admin_login_time', None)
        session.pop('admin_username', None)
        
        log_activity("AUTH_LOGOUT", 
                   details=f"Admin logout for {username}")
    
    def login_view(self):
        """Vista de login para administradores"""
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            
            if not username or not password:
                return render_template('admin/login.html', 
                                     error="Usuario y contraseña son requeridos")
            
            success, message = self.authenticate(username, password)
            
            if success:
                next_url = request.args.get('next', '/admin/logs')
                return redirect(next_url)
            else:
                return render_template('admin/login.html', error=message)
        
        # GET request - mostrar formulario de login
        return render_template('admin/login.html')
    
    def logout_view(self):
        """Vista de logout"""
        self.logout()
        return redirect(url_for('admin_login'))
    
    def _current_timestamp(self):
        """Obtener timestamp actual"""
        import time
        return int(time.time())
    
    def _check_rate_limit(self):
        """Verificar límite de intentos de login"""
        ip = request.remote_addr
        key = f"failed_attempts_{ip}"
        
        failed_attempts = session.get(key, [])
        current_time = self._current_timestamp()
        rate_limit = current_app.config['LOGIN_RATE_LIMIT']
        
        # Limpiar intentos antiguos
        failed_attempts = [attempt for attempt in failed_attempts 
                          if (current_time - attempt) < rate_limit]
        
        session[key] = failed_attempts
        
        max_attempts = current_app.config['MAX_LOGIN_ATTEMPTS']
        return len(failed_attempts) < max_attempts
    
    def _record_failed_attempt(self):
        """Registrar intento fallido"""
        ip = request.remote_addr
        key = f"failed_attempts_{ip}"
        
        failed_attempts = session.get(key, [])
        failed_attempts.append(self._current_timestamp())
        session[key] = failed_attempts
    
    def _clear_failed_attempts(self):
        """Limpiar intentos fallidos después de login exitoso"""
        ip = request.remote_addr
        key = f"failed_attempts_{ip}"
        session.pop(key, None)


def require_admin_auth(f):
    """Decorador para requerir autenticación de administrador"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import current_app
        
        # Verificar si el sistema de auth está inicializado
        if not hasattr(current_app, 'auth'):
            log_error("Auth system not initialized")
            abort(500)
        
        if not current_app.auth.is_authenticated():
            log_activity("AUTH_REQUIRED", 
                       details=f"Authentication required for {f.__name__}")
            
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify({'error': 'Authentication required'}), 401
            
            return redirect(url_for('admin_login'))
        
        return f(*args, **kwargs)
    return decorated_function


def get_admin_info():
    """Obtener información del administrador actual"""
    if session.get('admin_authenticated'):
        return {
            'username': session.get('admin_username'),
            'login_time': session.get('admin_login_time'),
            'ip_address': request.remote_addr
        }
    return None


# Instancia global de autenticación
auth = BasicAuth()