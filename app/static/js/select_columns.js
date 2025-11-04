// JavaScript para select_columns.html con checkboxes

document.addEventListener('DOMContentLoaded', function() {
    const targetSelect = document.getElementById('target_column');
    const predictorCheckboxes = document.querySelectorAll('.predictor-checkbox');
    const selectAllBtn = document.getElementById('selectAll');
    const deselectAllBtn = document.getElementById('deselectAll');
    const selectedCountBadge = document.getElementById('selected-count');
    const processBtn = document.getElementById('processBtn');
    const selectionSummary = document.getElementById('selection-summary');
    const targetSummary = document.getElementById('target-summary');
    const predictorsSummary = document.getElementById('predictors-summary');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    // Función para actualizar el contador de columnas seleccionadas
    function updateSelectedCount() {
        const selectedCheckboxes = document.querySelectorAll('.predictor-checkbox:checked');
        const count = selectedCheckboxes.length;
        selectedCountBadge.textContent = count + ' columna' + (count !== 1 ? 's' : '') + ' seleccionada' + (count !== 1 ? 's' : '');
        
        // Actualizar resumen si existe
        if (predictorsSummary) {
            predictorsSummary.textContent = count + ' seleccionadas';
        }
        
        // Habilitar/deshabilitar botón de procesar
        const targetSelected = targetSelect.value;
        if (targetSelected && count > 0) {
            if (processBtn) processBtn.disabled = false;
            if (selectionSummary) selectionSummary.style.display = 'block';
        } else {
            if (processBtn) processBtn.disabled = true;
            if (selectionSummary) selectionSummary.style.display = 'none';
        }
        
        checkConflicts();
    }
    
    // Función para actualizar información del target
    function updateTargetInfo() {
        const selectedTarget = targetSelect.value;
        if (targetSummary) {
            targetSummary.textContent = selectedTarget || 'No seleccionada';
        }
        
        // Analizar compatibilidad del target si está seleccionado
        if (selectedTarget) {
            analyzeTargetCompatibility(selectedTarget);
        } else {
            hideTargetAnalysis();
        }
        
        updateSelectedCount();
    }
    
    // Función para analizar la compatibilidad del target con diferentes modelos
    function analyzeTargetCompatibility(targetColumn) {
        fetch('/analyze_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_column: targetColumn
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayTargetAnalysis(data);
            } else {
                showAnalysisError(data.error);
            }
        })
        .catch(error => {
            console.error('Error analyzing target:', error);
            showAnalysisError('Error al analizar la variable target');
        });
    }
    
    // Función para mostrar el análisis del target
    function displayTargetAnalysis(data) {
        // Mostrar información básica del target
        const targetInfo = document.getElementById('target-info');
        const targetDetails = document.getElementById('target-details');
        
        if (targetInfo && targetDetails) {
            targetDetails.innerHTML = `
                <div><strong>Tipo:</strong> ${data.target_info.type}</div>
                <div><strong>Valores únicos:</strong> ${data.target_info.unique_values}</div>
                ${data.target_info.missing_values > 0 ? 
                  `<div class="text-warning"><strong>Valores faltantes:</strong> ${data.target_info.missing_values}</div>` : 
                  ''}
                <div><strong>Valores ejemplo:</strong> ${data.target_info.sample_values.join(', ')}</div>
            `;
            targetInfo.style.display = 'block';
        }
        
        // Mostrar compatibilidad de modelos
        const modelCompatibility = document.getElementById('model-compatibility');
        if (modelCompatibility) {
            modelCompatibility.style.display = 'block';
            
            // Compatibilidad con clasificación
            const classificationDiv = document.getElementById('classification-compatibility');
            const classificationAlert = document.getElementById('classification-alert');
            const classificationMessage = document.getElementById('classification-message');
            
            if (classificationDiv && classificationAlert && classificationMessage) {
                const classData = data.model_compatibility.classification;
                classificationAlert.className = `alert py-2 px-3 ${classData.class}`;
                classificationMessage.innerHTML = classData.message;
                classificationDiv.style.display = 'block';
            }
            
            // Compatibilidad con regresión
            const regressionDiv = document.getElementById('regression-compatibility');
            const regressionAlert = document.getElementById('regression-alert');
            const regressionMessage = document.getElementById('regression-message');
            
            if (regressionDiv && regressionAlert && regressionMessage) {
                const regData = data.model_compatibility.regression;
                regressionAlert.className = `alert py-2 px-3 ${regData.class}`;
                regressionMessage.innerHTML = regData.message;
                regressionDiv.style.display = 'block';
            }
            
            // Mostrar recomendación principal si no es 'none'
            const recommendationDiv = document.getElementById('primary-recommendation');
            const recommendationText = document.getElementById('recommendation-text');
            
            if (recommendationDiv && recommendationText && data.recommendation.primary !== 'none') {
                recommendationText.textContent = data.recommendation.reason;
                recommendationDiv.style.display = 'block';
            } else if (recommendationDiv) {
                recommendationDiv.style.display = 'none';
            }
        }
    }
    
    // Función para mostrar error en el análisis
    function showAnalysisError(errorMessage) {
        const targetInfo = document.getElementById('target-info');
        const targetDetails = document.getElementById('target-details');
        
        if (targetInfo && targetDetails) {
            targetDetails.innerHTML = `
                <div class="text-danger">
                    Error: ${errorMessage}
                </div>
            `;
            targetInfo.style.display = 'block';
        }
        
        hideModelCompatibility();
    }
    
    // Función para ocultar el análisis del target
    function hideTargetAnalysis() {
        const targetInfo = document.getElementById('target-info');
        if (targetInfo) {
            targetInfo.style.display = 'none';
        }
        hideModelCompatibility();
    }
    
    // Función para ocultar la compatibilidad de modelos
    function hideModelCompatibility() {
        const modelCompatibility = document.getElementById('model-compatibility');
        if (modelCompatibility) {
            modelCompatibility.style.display = 'none';
        }
    }
    
    // Función para verificar conflictos entre target y predictores
    function checkConflicts() {
        const targetValue = targetSelect.value;
        const selectedPredictors = Array.from(document.querySelectorAll('.predictor-checkbox:checked')).map(cb => cb.value);
        
        // Remover clases de error
        targetSelect.classList.remove('is-invalid');
        
        // Si hay conflicto, deseleccionar automáticamente el target de los predictores
        if (targetValue && selectedPredictors.includes(targetValue)) {
            // Deseleccionar automáticamente la columna target de los predictores
            const targetCheckbox = document.querySelector(`.predictor-checkbox[value="${targetValue}"]`);
            if (targetCheckbox) {
                targetCheckbox.checked = false;
                const predictorItem = targetCheckbox.closest('.predictor-item');
                if (predictorItem) {
                    predictorItem.classList.remove('selecting');
                }
            }
            // Actualizar contador después de la corrección automática
            updateSelectedCount();
        }
    }
    
    // Función para agregar efecto visual al seleccionar
    function addSelectingEffect(checkbox) {
        const predictorItem = checkbox.closest('.predictor-item');
        if (predictorItem) {
            predictorItem.classList.add('selecting');
            setTimeout(() => {
                predictorItem.classList.remove('selecting');
            }, 100);
        }
    }
    
    // Event listeners para los checkboxes
    predictorCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            addSelectingEffect(this);
            updateSelectedCount();
        });
    });
    
    // Event listeners para los botones
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const targetValue = targetSelect ? targetSelect.value : null;
            
            predictorCheckboxes.forEach(checkbox => {
                // Seleccionar todos los checkboxes EXCEPTO el que es igual al target
                if (checkbox.value !== targetValue) {
                    checkbox.checked = true;
                    addSelectingEffect(checkbox);
                } else {
                    checkbox.checked = false;
                }
            });
            
            setTimeout(() => {
                updateSelectedCount();
            }, 150);
        });
    }
    
    if (deselectAllBtn) {
        deselectAllBtn.addEventListener('click', function(e) {
            e.preventDefault();
            predictorCheckboxes.forEach(checkbox => {
                checkbox.checked = false;
                addSelectingEffect(checkbox);
            });
            
            setTimeout(() => {
                updateSelectedCount();
            }, 150);
        });
    }
    
    // Event listener para cambios en el target
    targetSelect.addEventListener('change', updateTargetInfo);
    
    // Efecto de carga al enviar el formulario
    const form = document.getElementById('columnForm');
    if (form) {
        form.addEventListener('submit', function() {
            if (processBtn) {
                processBtn.disabled = true;
                if (loadingSpinner) loadingSpinner.style.display = 'inline-block';
                processBtn.innerHTML = 'Procesando... <span class="spinner-border spinner-border-sm ms-2"></span>';
            }
        });
    }
    
    // Inicializar
    updateTargetInfo();
    updateSelectedCount();
});
