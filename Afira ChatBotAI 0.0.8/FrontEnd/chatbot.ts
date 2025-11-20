export interface Message {
    sender: 'bot' | 'user';
    text: string;
}

const API_URL = 'http://localhost:5000';

let isServerReady = false;
let userId: string | null = null;

function getUserId(): string {
    if (!userId) {
        userId = localStorage.getItem('afira_user_id');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substring(2, 15) + 
                     Math.random().toString(36).substring(2, 15);
            localStorage.setItem('afira_user_id', userId);
        }
    }
    return userId;
}

export async function checkServerHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_URL}/health`, {
            method: 'GET',
        });
        const data = await response.json();
        isServerReady = data.status === 'ok' && data.model_loaded;
        return isServerReady;
    } catch (error) {
        console.error('Server health check failed:', error);
        isServerReady = false;
        return false;
    }
}

export function isReady(): boolean {
    return isServerReady;
}

export async function loadChatData(): Promise<void> {
    console.log('Checking if ML server is ready...');
    const ready = await checkServerHealth();
    
    if (ready) {
        console.log('ML Server is ready!');
    } else {
        console.warn('ML Server not available. Make sure Flask is running!');
        throw new Error('ML Server not available');
    }
}

export function getGreeting(): string {
    return "Hello! I'm Afira, your AI assistant powered by machine learning. Ask me anything!";
}

export async function processUserMessage(message: string): Promise<string> {
    if (!isServerReady) {
        return "Server is not ready. Please make sure the Flask API is running on port 5000.";
    }
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                user_id: getUserId()
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.confidence) {
            console.log(`Intent: ${data.intent} (${(data.confidence * 100).toFixed(1)}%)`);
        } else {
            console.log(`Intent: ${data.intent}`);
        }
        
        if (data.collecting_data === true) {
            console.log(`Collecting data: ${data.progress || 'in progress'}`);
        }
        
        if (data.prediction) {
            console.log(`Prediction completed:`, data.prediction);
        }
        
        return data.response;
        
    } catch (error) {
        console.error('Prediction error:', error);
        return "Sorry, I couldn't process your message. Please check if the server is running.";
    }
}

export async function resetConversation(): Promise<void> {
    console.log("Resetting conversation and session...");
    
    try {
        const response = await fetch(`${API_URL}/reset_session`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                user_id: getUserId() 
            }),
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log(data.message);
        }
    } catch (error) {
        console.error('Error resetting session:', error);
    }
}

export function getCurrentUserId(): string {
    return getUserId();
}