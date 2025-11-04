// JavaScript para el Hub de Entrenamiento de Modelos

document.addEventListener('DOMContentLoaded', function() {
    // Verificar disponibilidad de motores - DISABLED
    // checkEngineAvailability();
    
    // Configurar tooltips
    initializeTooltips();
    
    // Configurar efectos de hover
    setupHoverEffects();
    
    // Configurar navegación inteligente
    setupSmartNavigation();
    
    // Mostrar estadísticas de datos
    displayDataStatistics();
});

// Inicializar tooltips
function initializeTooltips() {
    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Configurar efectos de hover
function setupHoverEffects() {
    const modelCards = document.querySelectorAll('.model-card');
    
    modelCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px)';
            this.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.2)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.15)';
        });
    });
    
    // Efectos para algoritmos
    const algorithmTags = document.querySelectorAll('.algorithm-tag');
    algorithmTags.forEach(tag => {
        tag.addEventListener('click', function() {
            showAlgorithmInfo(this.textContent);
        });
    });
}

// Mostrar información del algoritmo
function showAlgorithmInfo(algorithmName) {
    const algorithmInfo = {
        'Árbol de Decisión': 'Crea un modelo en forma de árbol que toma decisiones basadas en características.',
        'Random Forest': 'Ensemble de múltiples árboles de decisión para mayor precisión.',
        'Regresión Logística': 'Modelo lineal para clasificación binaria y multiclase.',
        'SVM': 'Support Vector Machine - encuentra el hiperplano óptimo de separación.',
        'Naive Bayes': 'Clasificador probabilístico basado en el teorema de Bayes.',
        'KNN': 'k-Nearest Neighbors - clasifica basándose en los k vecinos más cercanos.',
        'Regresión Lineal': 'Modelo que establece relación lineal entre variables.',
        'Ridge': 'Regresión lineal con regularización L2.',
        'Lasso': 'Regresión lineal con regularización L1.',
        'Apriori': 'Algoritmo para encontrar reglas de asociación frecuentes.',
        'FP-Growth': 'Algoritmo eficiente para minería de patrones frecuentes.'
    };
    
    const info = algorithmInfo[algorithmName] || 'Información no disponible';
    showNotification(`${algorithmName}: ${info}`, 'info');
}

// Configurar navegación inteligente
function setupSmartNavigation() {
    const buttons = document.querySelectorAll('.btn[href]');
    
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (this.disabled) {
                e.preventDefault();
                return;
            }
            
            // Mostrar loading
            showLoadingOverlay();
            
            // Guardar contexto de navegación
            sessionStorage.setItem('previous_page', 'model_training_hub');
            sessionStorage.setItem('navigation_time', new Date().toISOString());
        });
    });
}

// Mostrar estadísticas de datos
function displayDataStatistics() {
    // Obtener información de datos si está disponible
    const dataInfo = {
        filename: document.querySelector('[data-filename]')?.dataset.filename,
        rows: document.querySelector('[data-rows]')?.dataset.rows,
        columns: document.querySelector('[data-columns]')?.dataset.columns
    };
    
    if (dataInfo.rows && dataInfo.columns) {
        updateDataRecommendations(parseInt(dataInfo.rows), parseInt(dataInfo.columns));
    }
}

// Actualizar recomendaciones basadas en datos
function updateDataRecommendations(rows, columns) {
    let recommendations = [];
    
    if (rows < 100) {
        recommendations.push('Datos pequeños: considera usar modelos simples como Naive Bayes o KNN');
    } else if (rows > 10000) {
        recommendations.push('Datos grandes: Random Forest y SVM pueden dar buenos resultados');
    }
    
    if (columns > 50) {
        recommendations.push('Muchas características: considera usar técnicas de reducción de dimensionalidad');
    }
    
    if (recommendations.length > 0) {
        showRecommendations(recommendations);
    }
}

// Mostrar recomendaciones
function showRecommendations(recommendations) {
    const container = document.querySelector('.container');
    const recommendationDiv = document.createElement('div');
    recommendationDiv.className = 'alert alert-info';
    recommendationDiv.innerHTML = `
        <h6>Recomendaciones basadas en tus datos:</h6>
        <ul class="mb-0">
            ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
    `;
    
    // Insertar después del título
    const title = container.querySelector('h1');
    title.parentNode.insertBefore(recommendationDiv, title.nextSibling);
}

// Mostrar overlay de carga
function showLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="text-center text-white">
            <div class="loading-spinner mb-3"></div>
            <h5>Cargando...</h5>
            <p>Preparando el entorno de entrenamiento</p>
        </div>
    `;
    document.body.appendChild(overlay);
    
    // Remover después de máximo 10 segundos
    setTimeout(() => {
        if (overlay.parentNode) {
            overlay.remove();
        }
    }, 10000);
}

// Mostrar notificaciones
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        max-width: 500px;
    `;
    
    notification.innerHTML = `
        <strong>${type === 'error' ? 'Error' : type === 'success' ? 'Éxito' : 'Info'}:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remover después de 5 segundos
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 150);
        }
    }, 5000);
}

// Cleanup al salir
window.addEventListener('beforeunload', function() {
    // Limpiar overlays
    const overlays = document.querySelectorAll('.loading-overlay');
    overlays.forEach(overlay => overlay.remove());
});
