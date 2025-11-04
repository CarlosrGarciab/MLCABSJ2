"""
Módulo de rutas modular para la aplicación Flask
"""
from flask import Blueprint

def register_blueprints(app):
    """Registra todos los blueprints de la aplicación"""
    
    # Importar blueprints
    from .main import bp as main_bp
    from .data import bp as data_bp
    from .training import training_bp
    from .association import bp as association_bp
    from .prediction import bp as prediction_bp
    
    # Registrar blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(association_bp)
    app.register_blueprint(prediction_bp, url_prefix='/prediction')