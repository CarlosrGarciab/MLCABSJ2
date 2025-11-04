// JavaScript para Análisis de Reglas de Asociación

// Consolidación de todas las inicializaciones en un solo DOMContentLoaded
document.addEventListener('DOMContentLoaded', function() {
    // Inicializar todas las funcionalidades
    initializeEvents();
    updateButtonStates();
    initializeFormValidation();
    initializeCheckboxEffects();
    initializeAnalysisType();
    initializeAlgorithmSelection();
    initializeSimulatedResults();
});

function initializeEvents() {
    // Escuchar cambios en los checkboxes
    const checkboxes = document.querySelectorAll('.item-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateButtonStates);
    });
    
    // Escuchar cambios en la selección de algoritmo
    const algorithmRadios = document.querySelectorAll('input[name="algorithm"]');
    algorithmRadios.forEach(radio => {
        radio.addEventListener('change', updateAlgorithmTitle);
    });
}

function initializeAlgorithmSelection() {
    // Configurar comportamiento inicial del selector de algoritmo
    updateAlgorithmTitle();
}

function updateAlgorithmTitle() {
    const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked');
    const titleElement = document.querySelector('.section-title h5');
    
    if (selectedAlgorithm && titleElement) {
        const algorithmName = selectedAlgorithm.value === 'apriori' ? 'Apriori' : 'FP-Growth';
        titleElement.textContent = `Parámetros del Algoritmo ${algorithmName}`;
    }
}

function selectAllItems() {
    const checkboxes = document.querySelectorAll('.item-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
    updateButtonStates();
}

function deselectAllItems() {
    const checkboxes = document.querySelectorAll('.item-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    updateButtonStates();
}

function updateButtonStates() {
    const checkboxes = document.querySelectorAll('.item-checkbox');
    const checkedBoxes = document.querySelectorAll('.item-checkbox:checked');
    
    const selectAllBtn = document.querySelector('.btn-select-all');
    const deselectAllBtn = document.querySelector('.btn-deselect-all');
    
    if (selectAllBtn && deselectAllBtn) {
        // Deshabilitar "Seleccionar Todo" si todos están seleccionados
        selectAllBtn.disabled = (checkedBoxes.length === checkboxes.length);
        
        // Deshabilitar "Deseleccionar Todo" si ninguno está seleccionado
        deselectAllBtn.disabled = (checkedBoxes.length === 0);
        
        // Actualizar estilos visuales
        if (selectAllBtn.disabled) {
            selectAllBtn.classList.add('disabled');
        } else {
            selectAllBtn.classList.remove('disabled');
        }
        
        if (deselectAllBtn.disabled) {
            deselectAllBtn.classList.add('disabled');
        } else {
            deselectAllBtn.classList.remove('disabled');
        }
    }
}

// Función para validar el formulario antes del envío
function validateForm() {
    const analysisType = document.getElementById('analysis_type').value;
    if (analysisType === 'basket') {
        const checkedBoxes = document.querySelectorAll('.item-checkbox:checked');
        if (checkedBoxes.length === 0) {
            showAlert('Por favor, selecciona al menos una columna para el análisis.', 'warning');
            return false;
        }
        if (checkedBoxes.length < 2) {
            showAlert('Se necesitan al menos 2 columnas para realizar el análisis de asociación.', 'warning');
            return false;
        }
    } else if (analysisType === 'transaction') {
        const transactionColumn = document.getElementById('transaction_column').value;
        if (!transactionColumn) {
            showAlert('Por favor, selecciona la columna transaccional para el análisis.', 'warning');
            return false;
        }
    }
    return true;
}

// Función para mostrar alertas bonitas
function showAlert(message, type = 'info') {
    // Remover alertas existentes
    const existingAlerts = document.querySelectorAll('.custom-alert');
    existingAlerts.forEach(alert => alert.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `custom-alert alert alert-${type} alert-dismissible fade show`;
    alertDiv.style.position = 'fixed';
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.style.minWidth = '300px';
    alertDiv.style.borderRadius = '0.75rem';
    alertDiv.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.15)';
    
    alertDiv.innerHTML = `
        <strong>${type === 'warning' ? '⚠️' : 'ℹ️'}</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remover después de 5 segundos
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Inicialización de validación del formulario
function initializeFormValidation() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!validateForm()) {
                e.preventDefault();
                return false;
            }
            
            // Mostrar loading
            showLoadingState();
        });
    }
}

// Estado de carga
function showLoadingState() {
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('button[type="submit"]');
    
    if (form) {
        form.classList.add('form-loading');
    }
    
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i><span class="fw-bold">Procesando...</span>';
    }
}

// Efectos visuales mejorados para los checkboxes
function initializeCheckboxEffects() {
    const checkboxContainers = document.querySelectorAll('.form-check');
    checkboxContainers.forEach(container => {
        const checkbox = container.querySelector('.form-check-input');
        if (checkbox) {
            checkbox.addEventListener('change', function() {
                if (this.checked) {
                    container.classList.add('checked');
                } else {
                    container.classList.remove('checked');
                }
                updateButtonStates();
            });
        }
    });
}

// Inicializar el tipo de análisis
function initializeAnalysisType() {
    toggleAnalysisType();
    const analysisTypeSelect = document.getElementById('analysis_type');
    if (analysisTypeSelect) {
        analysisTypeSelect.addEventListener('change', toggleAnalysisType);
    }
}

// Mostrar/ocultar configuraciones según el tipo de análisis seleccionado
function toggleAnalysisType() {
    const analysisType = document.getElementById('analysis_type').value;
    const basketConfig = document.getElementById('basket_config');
    const transactionConfig = document.getElementById('transaction_config');
    if (analysisType === 'basket') {
        if (basketConfig) basketConfig.style.display = '';
        if (transactionConfig) transactionConfig.style.display = 'none';
    } else if (analysisType === 'transaction') {
        if (basketConfig) basketConfig.style.display = 'none';
        if (transactionConfig) transactionConfig.style.display = '';
    }
}


// Mostrar resultados de reglas de asociación
function showAssociationResults(results) {
    const resultsContainer = document.getElementById('association-results');
    if (!resultsContainer) return;
    
    // Limpiar resultados anteriores
    resultsContainer.innerHTML = '';
    
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = '<p class="text-center text-muted py-3">No se encontraron reglas de asociación.</p>';
        return;
    }
    
    // Crear tabla de resultados
    const table = document.createElement('table');
    table.className = 'table table-bordered table-striped';
    
    // Encabezados de la tabla
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Regla</th>
            <th>Confianza</th>
            <th>Soporte</th>
            <th>Lift</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Cuerpo de la tabla
    const tbody = document.createElement('tbody');
    results.forEach(rule => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${rule.rule}</td>
            <td>${rule.confidence}</td>
            <td>${rule.support}</td>
            <td>${rule.lift}</td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    resultsContainer.appendChild(table);
}

// Inicialización de resultados simulados
function initializeSimulatedResults() {
    // Simular resultados de análisis
    const simulatedResults = [
        { rule: 'A → B', confidence: 0.8, support: 0.3, lift: 1.2 },
        { rule: 'B → C', confidence: 0.7, support: 0.25, lift: 1.1 },
        { rule: 'A → C', confidence: 0.6, support: 0.2, lift: 1.3 },
    ];
    
    // Mostrar resultados simulados
    showAssociationResults(simulatedResults);
}
