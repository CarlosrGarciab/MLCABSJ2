"""
Rutas principales de la aplicación Flask
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from ..forms import UploadForm
from ..utils.file_handling import allowed_file, save_uploaded_file
from ..services.session_service import SessionService

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
def index():
    """Página principal para subir archivos CSV"""
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = save_uploaded_file(file)
            
            # Guardar información del archivo en la sesión
            session['uploaded_file'] = save_path
            session['filename'] = filename
            
            # Limpiar selección de columnas y configuraciones previas
            session.pop('target_column', None)
            session.pop('predictor_columns', None)
            session.pop('cleaned_file', None)
            session.pop('cleaning_options', None)
            session.pop('applied_cleaning', None)
            session.pop('processed_file', None)
            session.pop('data_file', None)
            session.pop('csv_separator', None)
            
            return redirect(url_for('data.select_columns'))
        else:
            flash('Por favor sube un archivo CSV válido', 'danger')
    
    return render_template('index.html', form=form)