document.addEventListener('DOMContentLoaded', function() {
    const modelRadios = document.querySelectorAll('.model-radio');
    const modelCards = document.querySelectorAll('.model-card');
    const trainButton = document.getElementById('trainButton');
    
    // Manejar selección de modelo
    modelRadios.forEach(function(radio, index) {
        radio.addEventListener('change', function() {
            // Remover selección de todas las tarjetas
            modelCards.forEach(function(card) {
                card.classList.remove('selected');
            });
            
            // Ocultar todos los parámetros
            document.querySelectorAll('.model-params').forEach(function(params) {
                params.style.display = 'none';
            });
            
            if (this.checked) {
                // Marcar la tarjeta seleccionada
                modelCards[index].classList.add('selected');
                
                // Mostrar parámetros del modelo seleccionado
                const modelType = this.value;
                const paramsDiv = document.getElementById('params_' + modelType);
                if (paramsDiv) {
                    paramsDiv.style.display = 'block';
                }
                
                // Habilitar botón de entrenar
                trainButton.disabled = false;
            }
        });
    });
    
    // Manejar click en las tarjetas
    modelCards.forEach(function(card, index) {
        card.addEventListener('click', function() {
            modelRadios[index].checked = true;
            modelRadios[index].dispatchEvent(new Event('change'));
        });
    });
    
    // Validación del formulario
    document.getElementById('trainModelForm').addEventListener('submit', function(e) {
        const selectedModel = document.querySelector('input[name="model_type"]:checked');
        if (!selectedModel) {
            e.preventDefault();
            alert('Por favor selecciona un tipo de modelo');
            return false;
        }
        
        // Mostrar loading con overlay
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = '<div class="loading-spinner"></div>';
        document.body.appendChild(loadingOverlay);
        
        // Cambiar texto del botón
        trainButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Entrenando...';
        trainButton.disabled = true;
    });
    
    // Añadir efectos de hover mejorados
    modelCards.forEach(function(card) {
        card.addEventListener('mouseenter', function() {
            if (!this.classList.contains('selected')) {
                this.style.transform = 'translateY(-3px)';
            }
        });
        
        card.addEventListener('mouseleave', function() {
            if (!this.classList.contains('selected')) {
                this.style.transform = 'translateY(0)';
            }
        });
    });
});
