"""
Rutas de administración para logs y monitoreo del sistema
Solo accessible para administradores del sistema
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils.logging_system import log_activity, log_error
import os
import json
from datetime import datetime, timedelta

bp = Blueprint('admin', __name__, url_prefix='/admin')

@bp.route('/logs')
def view_logs():
    """Ver logs de actividad del sistema"""
    log_activity("ADMIN_VIEW_LOGS", details="Admin accessed logs page")
    
    try:
        logs_dir = os.path.join(current_app.root_path, '..', 'logs')
        
        # Obtener lista de archivos de log
        log_files = []
        if os.path.exists(logs_dir):
            for filename in os.listdir(logs_dir):
                if filename.endswith('.log'):
                    filepath = os.path.join(logs_dir, filename)
                    stat_info = os.stat(filepath)
                    log_files.append({
                        'name': filename,
                        'size': stat_info.st_size,
                        'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })
        
        return render_template('admin/logs.html', log_files=log_files)
        
    except Exception as e:
        log_error(e, "Error accessing logs page")
        return jsonify({'error': str(e)}), 500

@bp.route('/logs/<log_file>')
def view_log_content(log_file):
    """Ver contenido de un archivo de log específico"""
    log_activity("ADMIN_VIEW_LOG_FILE", details=f"Viewing {log_file}")
    
    try:
        # Validar nombre de archivo por seguridad
        if not log_file.endswith('.log') or '..' in log_file:
            return jsonify({'error': 'Invalid log file name'}), 400
        
        logs_dir = os.path.join(current_app.root_path, '..', 'logs')
        filepath = os.path.join(logs_dir, log_file)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Log file not found'}), 404
        
        # Leer últimas líneas del archivo con manejo robusto de codificación
        lines = int(request.args.get('lines', 100))
        
        # Intentar múltiples codificaciones
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                    content = f.readlines()
                    encoding_used = encoding
                    break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            # Si todas las codificaciones fallan, leer como binario y convertir
            with open(filepath, 'rb') as f:
                binary_content = f.read()
                # Intentar decodificar con reemplazo de caracteres problemáticos
                content = binary_content.decode('utf-8', errors='replace').splitlines(True)
                encoding_used = 'utf-8 (with replacements)'
            
        # Obtener las últimas N líneas
        recent_lines = content[-lines:] if len(content) > lines else content
        
        return jsonify({
            'filename': log_file,
            'lines': recent_lines,
            'total_lines': len(content),
            'showing_lines': len(recent_lines),
            'encoding_used': encoding_used
        })
        
    except Exception as e:
        log_error(e, f"Error reading log file {log_file}")
        return jsonify({'error': str(e)}), 500

@bp.route('/system_status')
def system_status():
    """Información del estado del sistema"""
    log_activity("ADMIN_SYSTEM_STATUS", details="Admin checked system status")
    
    try:
        import psutil
        import platform
        
        # Información del sistema
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free
            }
        }
        
        # Información de la aplicación
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        models_dir = current_app.config['MODELS_FOLDER']
        
        app_info = {
            'upload_files': len(os.listdir(uploads_dir)) if os.path.exists(uploads_dir) else 0,
            'model_files': len([f for f in os.listdir(models_dir) if f.endswith('.joblib')]) if os.path.exists(models_dir) else 0,
            'debug_mode': current_app.debug,
            'config': {
                'max_content_length': current_app.config.get('MAX_CONTENT_LENGTH'),
                'upload_folder': current_app.config.get('UPLOAD_FOLDER')
            }
        }
        
        return render_template('admin/system_status.html', 
                             system_info=system_info, 
                             app_info=app_info)
                             
    except ImportError:
        # Si psutil no está disponible
        return render_template('admin/system_status.html', 
                             system_info={'error': 'psutil not available'}, 
                             app_info={})
    except Exception as e:
        log_error(e, "Error getting system status")
        return jsonify({'error': str(e)}), 500

@bp.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Limpiar logs antiguos"""
    log_activity("ADMIN_CLEAR_LOGS", details="Admin initiated log cleanup")
    
    try:
        days = int(request.form.get('days', 7))
        logs_dir = os.path.join(current_app.root_path, '..', 'logs')
        
        if not os.path.exists(logs_dir):
            return jsonify({'message': 'No logs directory found'})
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleared_files = []
        
        for filename in os.listdir(logs_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(logs_dir, filename)
                file_date = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_date < cutoff_date:
                    os.remove(filepath)
                    cleared_files.append(filename)
        
        log_activity("ADMIN_LOGS_CLEARED", details=f"Cleared {len(cleared_files)} log files older than {days} days")
        
        return jsonify({
            'message': f'Cleared {len(cleared_files)} log files',
            'files': cleared_files
        })
        
    except Exception as e:
        log_error(e, "Error clearing logs")
        return jsonify({'error': str(e)}), 500


@bp.route('/clear_current_logs', methods=['POST'])
def clear_current_logs():
    """Limpiar contenido de los logs actuales"""
    log_activity("ADMIN_CLEAR_CURRENT_LOGS", details="Admin initiated current logs content cleanup")
    
    try:
        logs_dir = os.path.join(current_app.root_path, '..', 'logs')
        
        if not os.path.exists(logs_dir):
            return jsonify({'message': 'No logs directory found'})
        
        cleared_files = []
        
        for filename in os.listdir(logs_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(logs_dir, filename)
                # Vaciar el contenido del archivo sin eliminarlo
                with open(filepath, 'w') as f:
                    f.write("")
                cleared_files.append(filename)
        
        log_activity("ADMIN_CURRENT_LOGS_CLEARED", details=f"Cleared content of {len(cleared_files)} log files")
        
        return jsonify({
            'message': f'Cleared content of {len(cleared_files)} log files',
            'files': cleared_files
        })
        
    except Exception as e:
        log_error(e, "Error clearing current logs content")
        return jsonify({'error': str(e)}), 500
