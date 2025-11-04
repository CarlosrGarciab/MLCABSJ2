# MLCABSJ - Flask Machine Learning Platform

## Descripción

**Plataforma web completa para análisis de datos y machine learning** desarrollada con Flask. Permite cargar, procesar, visualizar datos y entrenar modelos de clasificación, regresión y análisis de asociación con una interfaz web intuitiva.

## Valor Académico

Este proyecto de pasantía demuestra competencias en:
- **Desarrollo Web Full-Stack**: Flask, Bootstrap, JavaScript
- **Ciencia de Datos**: Preprocesamiento, algoritmos ML, visualización
- **Seguridad Informática**: Autenticación, validación, logs
- **Ingeniería de Software**: Arquitectura modular

## Características Principales

- **Carga de archivos CSV** con validación automática
- **Limpieza inteligente de datos** (valores faltantes, duplicados, outliers)
- **Machine Learning**: Regresión, clasificación y análisis de asociación
- **Interfaz responsiva** con visualizaciones interactivas
- **Sistema de predicciones** con modelos guardados
- **Panel de administración** con logs y monitoreo

## Requisitos del Sistema

- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, Linux, macOS
- **RAM**: Mínimo 4GB (recomendado 8GB para datasets grandes)
- **Espacio en disco**: 500MB libres
- **Navegador**: Chrome, Firefox, Safari, Edge (versiones recientes)

## Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/CarlosrGarciab/MLCABSJ2.git
cd MLCABSJ2

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
python run.py

# 5. Abrir en navegador
http://localhost:5000
```

### Credenciales de Administrador
- **URL**: `/admin/login`
- **Usuario**: admin
- **Contraseña**: admin123

## Estructura del Proyecto

```
MLCABSJ/
├── app/                    # Aplicación Flask principal
│   ├── __init__.py         # Inicialización de la aplicación
│   ├── forms.py            # Formularios WTF
│   ├── routes/             # Blueprints de rutas
│   │   ├── __init__.py
│   │   ├── main.py         # Rutas principales
│   │   ├── data.py         # Carga y procesamiento de datos
│   │   ├── training.py     # Entrenamiento de modelos
│   │   ├── association.py  # Análisis de asociación
│   │   ├── prediction.py   # Sistema de predicciones
│   │   └── admin.py        # Panel de administración
│   ├── utils/              # Utilidades y herramientas
│   │   ├── data_cleaning.py    # Limpieza de datos
│   │   ├── data_validation.py  # Validación de datos
│   │   ├── file_handling.py    # Manejo de archivos
│   │   ├── model_training.py   # Entrenamiento de ML
│   │   ├── auth.py             # Sistema de autenticación
│   │   └── logging_system.py   # Sistema de logs
│   ├── templates/          # Plantillas HTML
│   │   ├── base.html       # Plantilla base
│   │   ├── index.html      # Página principal
│   │   ├── admin/          # Plantillas de administración
│   │   └── errors/         # Páginas de error
│   └── static/             # Archivos estáticos
│       ├── css/            # Hojas de estilo
│       └── js/             # JavaScript
├── models/                 # Modelos ML entrenados y guardados
├── uploads/                # Archivos CSV subidos por usuarios
├── temp_results/           # Resultados temporales de entrenamientos
├── logs/                   # Archivos de log del sistema
├── config.py               # Configuración de la aplicación
├── run.py                  # Punto de entrada principal
├── requirements.txt        # Dependencias Python
└── README.md               # Documentación del proyecto
```

## Guía de Uso

1. **Carga de Datos**: Subir archivo CSV y revisar vista previa
2. **Configuración**: Seleccionar variable target y predictoras
3. **Limpieza**: Elegir modo automático o manual para procesar datos
4. **Entrenamiento**: Seleccionar algoritmo y entrenar modelo
5. **Predicciones**: Usar modelos guardados para nuevas predicciones
6. **Administración**: Acceder a `/admin` para logs y monitoreo

## Tecnologías Utilizadas

### Backend
- **Flask 2.3+**: Framework web principal
- **Pandas**: Manipulación y análisis de datos
- **Scikit-learn**: Algoritmos de machine learning
- **NumPy**: Operaciones numéricas
- **Joblib**: Serialización de modelos ML
- **MLxtend**: Análisis de reglas de asociación

### Frontend
- **Bootstrap 5**: Framework CSS responsivo
- **JavaScript ES6**: Interactividad del cliente
- **Chart.js**: Visualizaciones de datos
- **jQuery**: Manipulación del DOM

### Seguridad y Logs
- **Flask-WTF**: Protección CSRF en formularios
- **Werkzeug**: Utilidades de seguridad
- **Sistema de logs personalizado**: Monitoreo de actividad

## Solución de Problemas

### Errores Comunes
- **"Archivo demasiado grande"**: Reducir tamaño o ajustar `MAX_CONTENT_LENGTH` en config.py
- **"Formato no válido"**: Verificar que sea CSV válido con encoding UTF-8
- **"No se puede acceder a /admin"**: Verificar credenciales admin/admin123
- **"Modelo no encontrado"**: Verificar archivos en directorio `/models`
- **"Puerto en uso"**: Cambiar puerto en run.py o cerrar aplicaciones que usen puerto 5000

### Logs de Depuración
Los logs están disponibles en `/admin` o directamente en:
- `logs/flask.log`: Logs generales de la aplicación
- `logs/error.log`: Errores del sistema
- `logs/activity.log`: Actividad de usuarios

## Autor

**Carlos García**
- GitHub: [@CarlosrGarciab](https://github.com/CarlosrGarciab)
- Proyecto: Pasantía Académica NIDTEC 2025
- Email: [carlosgarciaballadares@gmail.com]

## Agradecimientos

- **NIDTEC**: Por la oportunidad de pasantía
- **Comunidad de Flask**: Por la documentación y recursos
- **Scikit-learn**: Por las herramientas de machine learning