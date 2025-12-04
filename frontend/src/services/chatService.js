const API_URL = 'http://localhost:8080';

export const sendMessage = async (message) => {
    try {
        console.log('[FRONTEND] Sending message:', message);
        
        // Send message to backend - backend returns HTML directly
        const formData = new FormData();
        formData.append('msg', message);
        
        console.log('[FRONTEND] Making fetch request to:', `${API_URL}/get`);
        
        const response = await fetch(`${API_URL}/get`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'text/html'
            }
        });
        
        console.log('[FRONTEND] Response status:', response.status);
        console.log('[FRONTEND] Response headers:', response.headers);
        
        if (!response.ok) {
            // Try to get error message from response
            let errorMessage = response.statusText;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || errorMessage;
            } catch {
                // Response wasn't JSON, use statusText
            }
            throw new Error(`HTTP ${response.status}: ${errorMessage}`);
        }
        
        // Backend returns HTML string directly
        const htmlResponse = await response.text();
        
        console.log('[FRONTEND] Received response:', htmlResponse.substring(0, 100));
        
        // Return the HTML response
        return htmlResponse;
    } catch (error) {
        console.error('[FRONTEND] Error in sendMessage:', error);
        console.error('[FRONTEND] Error type:', error.name);
        console.error('[FRONTEND] Error message:', error.message);
        throw error;
    }
};

// Helper function to check backend health
export const checkBackendHealth = async () => {
    try {
        console.log('[FRONTEND] Checking backend health...');
        const response = await fetch(`${API_URL}/health`, {
            method: 'GET'
        });
        
        if (!response.ok) {
            throw new Error(`Backend health check failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('[FRONTEND] Backend health check passed:', data);
        return data;
    } catch (error) {
        console.error('[FRONTEND] Backend health check failed:', error);
        throw error;
    }
};  