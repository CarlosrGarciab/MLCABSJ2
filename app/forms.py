from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, SelectField, SelectMultipleField, BooleanField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    file = FileField('Seleccionar archivo CSV', validators=[DataRequired()])
    submit = SubmitField('Subir Archivo')

class ColumnSelectionForm(FlaskForm):
    target_column = SelectField('Columna Target (Etiqueta)', validators=[DataRequired()])
    predictor_columns = SelectMultipleField('Columnas Predictoras')
    submit = SubmitField('Confirmar Selección')