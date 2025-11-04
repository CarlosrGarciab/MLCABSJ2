import os
from datetime import timedelta
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'una-clave-secreta-muy-segura'
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MODELS_FOLDER = os.path.join(basedir, 'models')
    TEMP_RESULTS_FOLDER = os.path.join(basedir, 'temp_results')
    LOGS_FOLDER = os.path.join(basedir, 'logs')
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    
    # Configuración de sesiones
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    SESSION_PERMANENT = True
    
    # Configuración de autenticación de administrador
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', 3600))  # 1 hora
    MAX_LOGIN_ATTEMPTS = int(os.environ.get('MAX_LOGIN_ATTEMPTS', 5))
    LOGIN_RATE_LIMIT = int(os.environ.get('LOGIN_RATE_LIMIT', 300))  # 5 minutos

    @staticmethod
    def init_app(app):
        # Crear directorios si no existen
        for folder in [Config.UPLOAD_FOLDER, Config.MODELS_FOLDER, Config.TEMP_RESULTS_FOLDER, Config.LOGS_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)