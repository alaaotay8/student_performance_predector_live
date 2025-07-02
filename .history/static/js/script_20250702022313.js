// API Configuration - Dynamic URL for both local and production
const API_BASE_URL = window.location.origin;
// DOM Elements
const form = document.getElementById('predictionForm');
const resultsContainer = document.getElementById('resultsContainer');
const resultCard = document.getElementById('resultCard');
const loadingOverlay = document.getElementById('loadingOverlay');
const submitBtn = document.getElementById('submitBtn');

// Slider synchronization
const studyTimeInput = document.getElementById('studyTime');
const studyTimeSlider = document.getElementById('studyTimeSlider');
const absencesInput = document.getElementById('absences');
const absencesSlider = document.getElementById('absencesSlider');

// Initialize sliders
function initializeSliders() {
    // Study time slider
    studyTimeSlider.addEventListener('input', function() {
        studyTimeInput.value = this.value;
        updateSliderValue('studyTime', this.value);
    });
    
    studyTimeInput.addEventListener('input', function() {
        studyTimeSlider.value = this.value;
        updateSliderValue('studyTime', this.value);
    });
    
    // Absences slider
    absencesSlider.addEventListener('input', function() {
        absencesInput.value = this.value;
        updateSliderValue('absences', this.value);
    });
    
    absencesInput.addEventListener('input', function() {
        absencesSlider.value = this.value;
        updateSliderValue('absences', this.value);
    });
    
    // Initialize slider value displays
    updateSliderValue('studyTime', studyTimeSlider.value);
    updateSliderValue('absences', absencesSlider.value);
}

// Handle checkbox values
function getCheckboxValue(name) {
    const checkbox = document.querySelector(`input[name="${name}"]`);
    return checkbox && checkbox.checked ? 1 : 0;
}

// Show loading overlay
function showLoading() {
    loadingOverlay.classList.add('show');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.classList.remove('show');
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-magic"></i> Predict Performance';
}

// Display results
function displayResults(data) {
    const gradeColors = {
        'Excellent (90-100%)': '#28a745',
        'Very Good (80-89%)': '#17a2b8',
        'Good (70-79%)': '#ffc107',
        'Acceptable (60-69%)': '#fd7e14',
        'Fail (Below 60%)': '#dc3545'
    };
    
    const gradeIcons = {
        'Excellent (90-100%)': 'fas fa-trophy',
        'Very Good (80-89%)': 'fas fa-medal',
        'Good (70-79%)': 'fas fa-thumbs-up',
        'Acceptable (60-69%)': 'fas fa-meh',
        'Fail (Below 60%)': 'fas fa-times-circle'
    };
    
    const grade = data.predicted_grade;
    const confidence = data.confidence;
    const probabilities = data.prediction_probabilities;
    const model = data.model_used;
    
    resultCard.innerHTML = `
        <div class="grade-result" style="color: ${gradeColors[grade] || '#fff'}">
            <i class="${gradeIcons[grade] || 'fas fa-chart-line'}"></i>
            ${grade}
        </div>
        <div class="confidence-result">
            Confidence: ${confidence}
        </div>
        
        <div class="probabilities">
            ${Object.entries(probabilities).map(([gradeClass, probability]) => `
                <div class="probability-item">
                    <div class="probability-label">${gradeClass}</div>
                    <div class="probability-value">${probability}</div>
                </div>
            `).join('')}
        </div>
        
        <div class="model-info">
            <i class="fas fa-robot"></i> Model: ${model}
        </div>
    `;
    
    resultsContainer.classList.add('show');
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    resultCard.innerHTML = `
        <div class="error-result">
            <i class="fas fa-exclamation-triangle"></i>
            <h3>Prediction Error</h3>
            <p>${message}</p>
            <div class="error-details">
                <p>Please check your inputs and try again.</p>
                <p>Make sure the API server is running on ${API_BASE_URL}</p>
            </div>
        </div>
    `;
    
    resultsContainer.classList.add('show');
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Validate form data
function validateForm(formData) {
    const errors = [];
    
    // Check age range
    const age = parseInt(formData.get('Age'));
    if (age < 15 || age > 18) {
        errors.push('Age must be between 15 and 18 years');
    }
    
    // Check study time range
    const studyTime = parseFloat(formData.get('StudyTimeWeekly'));
    if (studyTime < 0 || studyTime > 20) {
        errors.push('Study time must be between 0 and 20 hours');
    }
    
    // Check absences range
    const absences = parseInt(formData.get('Absences'));
    if (absences < 0 || absences > 30) {
        errors.push('Absences must be between 0 and 30');
    }
    
    return errors;
}

// Handle form submission
async function handleFormSubmit(event) {
    if (event) {
        event.preventDefault();
    }
    
    showLoading();
    
    try {
        // Collect form data
        const formData = new FormData(form);
        
        // Add missing form fields that might not be in FormData
        if (!formData.get('Extracurricular')) {
            formData.set('Extracurricular', getCheckboxValue('Extracurricular'));
        }
        if (!formData.get('Sports')) {
            formData.set('Sports', getCheckboxValue('Sports'));
        }
        if (!formData.get('Music')) {
            formData.set('Music', getCheckboxValue('Music'));
        }
        if (!formData.get('Volunteering')) {
            formData.set('Volunteering', getCheckboxValue('Volunteering'));
        }
        
        // Validate form
        const validationErrors = validateForm(formData);
        if (validationErrors.length > 0) {
            throw new Error(validationErrors.join(', '));
        }
        
        // Prepare API parameters
        const params = new URLSearchParams({
            Age: formData.get('Age'),
            Gender: formData.get('Gender'),
            Ethnicity: formData.get('Ethnicity'),
            ParentalEducation: formData.get('ParentalEducation'),
            StudyTimeWeekly: formData.get('StudyTimeWeekly'),
            Absences: formData.get('Absences'),
            Tutoring: formData.get('Tutoring'),
            ParentalSupport: formData.get('ParentalSupport'),
            Extracurricular: getCheckboxValue('Extracurricular'),
            Sports: getCheckboxValue('Sports'),
            Music: getCheckboxValue('Music'),
            Volunteering: getCheckboxValue('Volunteering')
        });
        
        console.log('API Parameters:', Object.fromEntries(params));
        console.log('API URL:', `${API_BASE_URL}/predict?${params}`);
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/predict?${params}`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            mode: 'cors',
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
            const errorText = await response.text().catch(() => 'Unable to read error details');
            console.error('Error response:', errorText);
            
            let errorData;
            try {
                errorData = JSON.parse(errorText);
            } catch {
                errorData = { detail: errorText };
            }
            
            throw new Error(errorData.detail || `HTTP error! status: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('API Response:', data);
        displayResults(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'An unexpected error occurred while making the prediction.');
    } finally {
        hideLoading();
    }
}

// Add sample data function
function loadSampleData() {
    // Sample data for testing - using correct element IDs from HTML
    document.getElementById('age').value = '17';
    document.getElementById('gender').value = '1';
    document.getElementById('ethnicity').value = '2';
    document.getElementById('parentalEducation').value = '3';
    document.getElementById('studyTime').value = '12.5';
    document.getElementById('studyTimeSlider').value = '12.5';
    document.getElementById('absences').value = '3';
    document.getElementById('absencesSlider').value = '3';
    document.querySelector('input[name="Tutoring"][value="1"]').checked = true;
    document.getElementById('parentalSupport').value = '3';
    document.querySelector('input[name="Extracurricular"]').checked = true;
    document.querySelector('input[name="Sports"]').checked = false;
    document.querySelector('input[name="Music"]').checked = true;
    document.querySelector('input[name="Volunteering"]').checked = true;
    
    // Update slider displays
    updateSliderValue('studyTime', '12.5');
    updateSliderValue('absences', '3');
    
    // Show success notification
    showNotification('Sample data loaded successfully!', 'success');
}

// Add form reset function
function resetForm() {
    form.reset();
    resultsContainer.classList.remove('show');
    
    // Reset sliders
    studyTimeSlider.value = '10';
    absencesSlider.value = '5';
    studyTimeInput.value = '';
    absencesInput.value = '';
}

// Clear Form function (called by Clear Form button)
function clearForm() {
    // Add animation to form sections
    const formSections = document.querySelectorAll('.form-section');
    formSections.forEach((section, index) => {
        setTimeout(() => {
            section.style.transform = 'scale(0.95)';
            section.style.opacity = '0.7';
            
            setTimeout(() => {
                section.style.transform = 'scale(1)';
                section.style.opacity = '1';
            }, 100);
        }, index * 50);
    });
    
    // Reset the form
    resetForm();
    
    // Show notification
    showNotification('Form cleared successfully!', 'success');
}

// Randomize Data function (called by Randomize button)
function randomizeData() {
    // Add animation to form sections
    const formSections = document.querySelectorAll('.form-section');
    formSections.forEach((section, index) => {
        setTimeout(() => {
            section.style.transform = 'rotateY(10deg)';
            section.style.opacity = '0.8';
            
            setTimeout(() => {
                section.style.transform = 'rotateY(0deg)';
                section.style.opacity = '1';
            }, 200);
        }, index * 100);
    });
    
    // Generate random data
    setTimeout(() => {
        // Random age (15-18)
        const randomAge = Math.floor(Math.random() * 4) + 15;
        document.getElementById('age').value = randomAge.toString();
        
        // Random gender (0 or 1)
        const randomGender = Math.floor(Math.random() * 2);
        document.getElementById('gender').value = randomGender.toString();
        
        // Random ethnicity (0-3)
        const randomEthnicity = Math.floor(Math.random() * 4);
        document.getElementById('ethnicity').value = randomEthnicity.toString();
        
        // Random parental education (0-4)
        const randomParentalEducation = Math.floor(Math.random() * 5);
        document.getElementById('parentalEducation').value = randomParentalEducation.toString();
        
        // Random study time (0-20 hours)
        const randomStudyTime = (Math.random() * 20).toFixed(1);
        document.getElementById('studyTime').value = randomStudyTime;
        document.getElementById('studyTimeSlider').value = randomStudyTime;
        updateSliderValue('studyTime', randomStudyTime);
        
        // Random absences (0-30)
        const randomAbsences = Math.floor(Math.random() * 31);
        document.getElementById('absences').value = randomAbsences.toString();
        document.getElementById('absencesSlider').value = randomAbsences.toString();
        updateSliderValue('absences', randomAbsences);
        
        // Random tutoring (0 or 1)
        const randomTutoring = Math.floor(Math.random() * 2);
        document.querySelector(`input[name="Tutoring"][value="${randomTutoring}"]`).checked = true;
        
        // Random parental support (0-4)
        const randomParentalSupport = Math.floor(Math.random() * 5);
        document.getElementById('parentalSupport').value = randomParentalSupport.toString();
        
        // Random activities (50% chance each)
        document.querySelector('input[name="Extracurricular"]').checked = Math.random() > 0.5;
        document.querySelector('input[name="Sports"]').checked = Math.random() > 0.5;
        document.querySelector('input[name="Music"]').checked = Math.random() > 0.5;
        document.querySelector('input[name="Volunteering"]').checked = Math.random() > 0.5;
        
        // Trigger change events for better UX
        triggerChangeEvents();
        
        // Show notification
        showNotification('Random data generated!', 'info');
    }, 500);
}

// Helper function to update slider value displays
function updateSliderValue(type, value) {
    const valueElement = document.getElementById(`${type}Value`);
    if (valueElement) {
        if (type === 'studyTime') {
            valueElement.textContent = `${value}h`;
        } else {
            valueElement.textContent = value;
        }
    }
}

// Helper function to trigger change events for better animations
function triggerChangeEvents() {
    const formElements = form.querySelectorAll('input, select');
    formElements.forEach(element => {
        element.dispatchEvent(new Event('change', { bubbles: true }));
    });
}

// Show notification function
function showNotification(message, type = 'info') {
    // Remove existing notification if any
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add to document
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// Initialize the application
function initializeApp() {
    console.log('Initializing Student Performance Predictor...');
    
    // Check if all required elements exist
    const requiredElements = [
        'predictionForm',
        'resultsContainer', 
        'resultCard',
        'loadingOverlay',
        'submitBtn',
        'studyTime',
        'studyTimeSlider',
        'absences',
        'absencesSlider'
    ];
    
    const missingElements = [];
    requiredElements.forEach(id => {
        if (!document.getElementById(id)) {
            missingElements.push(id);
        }
    });
    
    if (missingElements.length > 0) {
        console.error('Missing elements:', missingElements);
    } else {
        console.log('All required elements found!');
    }
    
    initializeSliders();
    
    // Add form submit handler as backup
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            handleFormSubmit(e);
        });
        console.log('Form submit handler added as backup');
    }
    
    // Make functions globally available
    window.handleFormSubmit = handleFormSubmit;
    window.loadSampleData = loadSampleData;
    window.clearForm = clearForm;
    window.showHelp = showHelp;
    window.closeHelp = closeHelp;
    
    // Test button functions
    window.testLoadSample = () => {
        console.log('Testing loadSampleData...');
        loadSampleData();
    };
    
    window.testClearForm = () => {
        console.log('Testing clearForm...');
        clearForm();
    };
    
    window.testPredict = () => {
        console.log('Testing prediction...');
        loadSampleData();
        setTimeout(() => {
            handleFormSubmit();
        }, 500);
    };
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Ctrl+Enter to submit form
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            handleFormSubmit();
        }
        
        // Ctrl+R to reset form
        if (event.ctrlKey && event.key === 'r') {
            event.preventDefault();
            clearForm();
        }
        
        // Ctrl+S to load sample data (for testing)
        if (event.ctrlKey && event.key === 's') {
            event.preventDefault();
            loadSampleData();
        }
    });
    
    console.log('Student Performance Predictor initialized!');
    console.log('Keyboard shortcuts:');
    console.log('- Ctrl+Enter: Submit form');
    console.log('- Ctrl+R: Reset form');
    console.log('- Ctrl+S: Load sample data');
    console.log('Test functions available:');
    console.log('- testLoadSample() - Test the Load Sample button');
    console.log('- testClearForm() - Test the Clear Form button');
    console.log('- testPredict() - Test full prediction flow');
}

// Additional CSS for notifications and error styling
const helpButtonCSS = `
    .error-result {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        border-radius: 15px;
    }
    
    .error-result i {
        font-size: 3rem;
        margin-bottom: 20px;
        opacity: 0.8;
    }
    
    .error-result h3 {
        font-size: 1.5rem;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .error-result p {
        margin-bottom: 10px;
        opacity: 0.9;
    }
    
    .error-details {
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.3);
    }
    
    .error-details p {
        font-size: 0.9rem;
        opacity: 0.8;
    }
`;

// Inject additional CSS
const style = document.createElement('style');
style.textContent = helpButtonCSS;
document.head.appendChild(style);

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);

// Help Modal Functions
function showHelp() {
    const helpModal = document.getElementById('helpModal');
    if (helpModal) {
        helpModal.classList.add('show');
        document.body.style.overflow = 'hidden';
        
        // Add click outside to close
        helpModal.addEventListener('click', function(e) {
            if (e.target === helpModal) {
                closeHelp();
            }
        });
        
        // Add escape key to close
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && helpModal.classList.contains('show')) {
                closeHelp();
            }
        });
    }
}

function closeHelp() {
    const helpModal = document.getElementById('helpModal');
    if (helpModal) {
        helpModal.classList.remove('show');
        document.body.style.overflow = '';
    }
}

// Enhanced clear form function
function clearForm() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.reset();
        
        // Reset sliders to default values
        const studyTimeSlider = document.getElementById('studyTimeSlider');
        const absencesSlider = document.getElementById('absencesSlider');
        const studyTimeInput = document.getElementById('studyTime');
        const absencesInput = document.getElementById('absences');
        
        if (studyTimeSlider && studyTimeInput) {
            studyTimeSlider.value = 10;
            studyTimeInput.value = '';
            updateSliderValue('studyTime', 10);
        }
        
        if (absencesSlider && absencesInput) {
            absencesSlider.value = 5;
            absencesInput.value = '';
            updateSliderValue('absences', 5);
        }
        
        // Hide results
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.classList.remove('show');
        }
        
        // Show success message
        showNotification('Form cleared successfully!', 'success');
    }
}

// Show notification function
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notif => notif.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        transform: translateX(400px);
        transition: transform 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, 3000);
}

// API Health Check and Debugging
async function checkAPIHealth() {
    try {
        console.log(`Checking API health at: ${API_BASE_URL}/health`);
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        
        if (data.model_status && !data.model_status.loaded) {
            showNotification('⚠️ Model not loaded on server. Predictions may not work.', 'warning');
        } else if (data.model_status && data.model_status.loaded) {
            showNotification('✅ Server and model are ready!', 'success');
        }
        
        return data;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showNotification('❌ Cannot connect to server. Please refresh the page.', 'error');
        return null;
    }
}

// Debug function to test API connectivity
async function testAPIConnection() {
    console.log('Testing API connection...');
    console.log('API Base URL:', API_BASE_URL);
    
    try {
        // Test health endpoint
        const healthResponse = await fetch(`${API_BASE_URL}/health`);
        console.log('Health endpoint status:', healthResponse.status);
        
        if (healthResponse.ok) {
            const healthData = await healthResponse.json();
            console.log('Health data:', healthData);
        }
        
        // Test a simple predict call with sample data
        const testParams = new URLSearchParams({
            Age: 17,
            Gender: 1,
            Ethnicity: 2,
            ParentalEducation: 3,
            StudyTimeWeekly: 12.5,
            Absences: 3,
            Tutoring: 1,
            ParentalSupport: 3,
            Extracurricular: 1,
            Sports: 0,
            Music: 1,
            Volunteering: 1
        });
        
        console.log('Testing predict endpoint with sample data...');
        const predictResponse = await fetch(`${API_BASE_URL}/predict?${testParams}`);
        console.log('Predict endpoint status:', predictResponse.status);
        
        if (predictResponse.ok) {
            const predictData = await predictResponse.json();
            console.log('Predict test successful:', predictData);
        } else {
            const errorText = await predictResponse.text();
            console.error('Predict test failed:', errorText);
        }
        
    } catch (error) {
        console.error('API Connection Test Failed:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Starting initialization...');
    
    // Initialize the app
    initializeApp();
    
    // Check API health after a short delay
    setTimeout(checkAPIHealth, 1000);
    
    console.log('Initialization complete!');
});
