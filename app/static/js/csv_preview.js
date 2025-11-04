document.addEventListener('DOMContentLoaded', function() {
    const csvFileInput = document.getElementById('csvFileInput');
    const previewSection = document.getElementById('previewSection');
    const previewHeaders = document.getElementById('previewHeaders');
    const previewRows = document.getElementById('previewRows');
    const csvForm = document.getElementById('csvForm');

    // Configuracion de validacion
    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
    const ALLOWED_EXTENSIONS = ['csv'];
    const MIN_COLUMNS = 2;
    const MIN_ROWS = 1;

    if (csvFileInput) {
        csvFileInput.addEventListener('change', handleFileSelect);
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) {
            hidePreview();
            return;
        }

        // Validar tipo de archivo
        if (!validateFileType(file)) {
            showError('Tipo de archivo no valido. Solo se permiten archivos CSV.');
            clearFileInput();
            return;
        }

        // Validar tamano de archivo
        if (!validateFileSize(file)) {
            showError('Archivo demasiado grande. Tamano maximo: 16MB.');
            clearFileInput();
            return;
        }

        // Leer y validar contenido
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const content = e.target.result;
                if (!validateFileContent(content)) {
                    showError('Contenido de archivo invalido. Verifique que sea un CSV valido.');
                    clearFileInput();
                    return;
                }
                const previewData = parseCSV(content);
                if (!validateCSVStructure(previewData)) {
                    showError('Estructura CSV invalida. Se requieren al menos 2 columnas y 1 fila de datos.');
                    clearFileInput();
                    return;
                }
                renderPreview(previewData);
                hideError();
            } catch (error) {
                showError('Error al procesar el archivo. Verifique que sea un CSV valido.');
                clearFileInput();
            }
        };
        reader.onerror = function() {
            showError('Error al leer el archivo.');
            clearFileInput();
        };
        reader.readAsText(file);
    }

    function parseCSV(text) {
        // Maneja diferentes delimitadores y saltos de línea
        const lines = text.split(/\r\n|\n/).filter(line => line.trim() !== '');
        const delimiter = detectDelimiter(lines[0]);
        const headers = lines[0].split(delimiter);
        const rows = lines.slice(1, 6).map(line => line.split(delimiter)); // Primeras 5 filas
        
        return { headers, rows };
    }

    function detectDelimiter(line) {
        const delimiters = [',', ';', '\t', '|'];
        const counts = delimiters.map(d => (line.match(new RegExp(`\\${d}`, 'g')) || []).length);
        const maxCount = Math.max(...counts);
        return maxCount > 0 ? delimiters[counts.indexOf(maxCount)] : ',';
    }

    function renderPreview(data) {
        // Limpiar tabla
        previewHeaders.innerHTML = '';
        previewRows.innerHTML = '';

        // Añadir headers
        const headerRow = document.createElement('tr');
        data.headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header.trim();
            headerRow.appendChild(th);
        });
        previewHeaders.appendChild(headerRow);

        // Añadir filas
        data.rows.forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell.trim();
                tr.appendChild(td);
            });
            previewRows.appendChild(tr);
        });

        // Mostrar sección
        previewSection.classList.remove('d-none');
    }

    // Validaciones frontend
    function validateFileType(file) {
        const extension = file.name.split('.').pop().toLowerCase();
        return ALLOWED_EXTENSIONS.includes(extension);
    }

    function validateFileSize(file) {
        return file.size <= MAX_FILE_SIZE;
    }

    function validateFileContent(content) {
        if (!content || content.trim().length === 0) return false;
        const lines = content.split(/\r\n|\n/).filter(line => line.trim() !== '');
        return lines.length >= 2; // Al menos header + 1 fila
    }

    function validateCSVStructure(data) {
        return data.headers.length >= MIN_COLUMNS && data.rows.length >= MIN_ROWS;
    }

    function showError(message) {
        let errorDiv = document.getElementById('csvErrorMessage');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'csvErrorMessage';
            errorDiv.className = 'alert alert-danger mt-2';
            csvFileInput.parentNode.insertBefore(errorDiv, csvFileInput.nextSibling);
        }
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }

    function hideError() {
        const errorDiv = document.getElementById('csvErrorMessage');
        if (errorDiv) {
            errorDiv.style.display = 'none';
        }
    }

    function clearFileInput() {
        csvFileInput.value = '';
        hidePreview();
    }

    function hidePreview() {
        if (previewSection) {
            previewSection.classList.add('d-none');
        }
    }

    // Validacion mejorada antes de enviar
    if (csvForm) {
        csvForm.addEventListener('submit', function(e) {
            if (!csvFileInput.files.length) {
                e.preventDefault();
                showError('Por favor selecciona un archivo CSV');
                return false;
            }
            
            const file = csvFileInput.files[0];
            if (!validateFileType(file) || !validateFileSize(file)) {
                e.preventDefault();
                showError('Archivo no valido. Verifique tipo y tamano.');
                return false;
            }
            
            hideError();
        });
    }
});