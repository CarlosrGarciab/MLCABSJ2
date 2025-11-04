// JavaScript para cleaning_options.html

// Variable global para almacenar los datos del análisis
let analysisData = null;

// Función para obtener los datos del análisis desde el DOM
function getAnalysisDataFromDOM() {
    // Los datos se pasan a través de un elemento script con tipo application/json
    const dataScript = document.getElementById('analysis-data');
    if (dataScript) {
        try {
            return JSON.parse(dataScript.textContent);
        } catch (error) {
            console.error('Error parsing analysis data:', error);
            return null;
        }
    }
    return null;
}

// Función para establecer los datos del análisis
function setAnalysisData(data) {
    analysisData = data;
    console.log('Analysis data loaded:', analysisData); // Debug
    
    // Ejecutar inicialización después de establecer los datos
    initializeCleaningOptions();
}

// Función para inicializar las opciones de limpieza
function initializeCleaningOptions() {
    if (!analysisData) return;
    
    // Deshabilitar checkboxes si no hay datos para limpiar
    disableUnavailableOptions();
    
    // Configurar el toggle de normalización
    setupNormalizationToggle();
    
    // Preseleccionar método recomendado
    preselectRecommendedMethod();
}

// Función para deshabilitar opciones no disponibles
function disableUnavailableOptions() {
    if (!analysisData) return;
    
    // Deshabilitar filas vacías si no las hay
    if (analysisData.empty_rows === 0) {
        const elem = document.getElementById('remove_empty_rows');
        if (elem) {
            elem.disabled = true;
            elem.closest('.form-check').style.opacity = '0.5';
        }
    }
    
    // Deshabilitar filas con muchos valores faltantes si no las hay
    if (analysisData.rows_with_many_missing === 0) {
        const elem = document.getElementById('remove_rows_many_missing');
        if (elem) {
            elem.disabled = true;
            elem.closest('.form-check').style.opacity = '0.5';
        }
    }
    
    // Deshabilitar columnas vacías si no las hay
    if (analysisData.empty_cols === 0) {
        const elem = document.getElementById('remove_empty_cols');
        if (elem) {
            elem.disabled = true;
            elem.closest('.form-check').style.opacity = '0.5';
        }
    }
    
    // Deshabilitar duplicados si no los hay
    if (analysisData.duplicates === 0) {
        const elem = document.getElementById('remove_duplicates');
        if (elem) {
            elem.disabled = true;
            elem.closest('.form-check').style.opacity = '0.5';
        }
    }
    
    // Deshabilitar limpieza de espacios si no hay columnas con espacios
    if (analysisData.text_cols_with_spaces === 0) {
        const elem = document.getElementById('clean_text_spaces');
        if (elem) {
            elem.disabled = true;
            elem.closest('.form-check').style.opacity = '0.5';
        }
    }
    
    // Deshabilitar normalización si no hay columnas que lo necesiten
    if (!analysisData.cols_need_normalization || analysisData.cols_need_normalization.length === 0) {
        const elem = document.getElementById('normalize_data');
        if (elem) {
            elem.disabled = true;
            elem.closest('.form-check').style.opacity = '0.5';
        }
    }
}

// Función para marcar todas las opciones recomendadas
function selectAllRecommended() {
    console.log('selectAllRecommended called'); // Debug
    
    if (!analysisData) {
        console.error('analysisData not available');
        return;
    }
    
    if (!analysisData.recommendations) {
        console.error('recommendations not available');
        return;
    }
    
    console.log('Processing recommendations:', analysisData.recommendations); // Debug
    
    // Marcar todas las opciones recomendadas
    analysisData.recommendations.forEach(function(rec) {
        console.log('Processing recommendation:', rec); // Debug
        
        if (rec.recommended && rec.type !== 'no_cleaning') {
            const element = document.getElementById(rec.type);
            console.log('Found element for', rec.type, ':', element); // Debug
            
            if (element && !element.disabled) {
                element.checked = true;
                console.log('Checked element:', rec.type); // Debug
            }
        }
    });
    
    // Mostrar opciones de normalización si está marcada
    const normalizeData = document.getElementById('normalize_data');
    if (normalizeData && normalizeData.checked) {
        toggleNormalizationMethod();
    }
    
    // Preseleccionar método recomendado de normalización
    preselectRecommendedMethod();
}

// Función para mostrar/ocultar opciones de método de normalización
function toggleNormalizationMethod() {
    const normalizeCheckbox = document.getElementById('normalize_data');
    const methodSection = document.getElementById('normalization_method_section');
    
    if (normalizeCheckbox && methodSection) {
        if (normalizeCheckbox.checked) {
            methodSection.style.display = 'block';
        } else {
            methodSection.style.display = 'none';
        }
    }
}

// Función para preseleccionar método recomendado
function preselectRecommendedMethod() {
    if (!analysisData || !analysisData.normalization_info) return;
    
    for (const [col, info] of Object.entries(analysisData.normalization_info)) {
        if (info.recommended_method === 'MinMaxScaler (0-1)') {
            const elem = document.getElementById('norm_minmax');
            if (elem) elem.checked = true;
            break;
        } else if (info.recommended_method === 'RobustScaler (robusto)') {
            const elem = document.getElementById('norm_robust');
            if (elem) elem.checked = true;
            break;
        }
    }
}

// Función para configurar el toggle de normalización
function setupNormalizationToggle() {
    const normalizeCheckbox = document.getElementById('normalize_data');
    if (normalizeCheckbox) {
        normalizeCheckbox.addEventListener('change', toggleNormalizationMethod);
        toggleNormalizationMethod(); // Ejecutar al cargar la página
    }
}

// Función principal que se ejecuta cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded'); // Debug
    
    // Intentar obtener datos del análisis desde el DOM primero
    const domData = getAnalysisDataFromDOM();
    if (domData) {
        setAnalysisData(domData);
    } else if (analysisData) {
        // Si ya tenemos datos, inicializar inmediatamente
        initializeCleaningOptions();
    }
    
    // Configurar event listener para el botón "Aplicar Todo Recomendado"
    const selectAllBtn = document.getElementById('selectAllRecommendedBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllRecommended);
    }
});
