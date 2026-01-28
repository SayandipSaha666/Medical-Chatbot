// src/services/api.js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

// Generic API request function with authentication
const apiRequest = async (endpoint, options = {}) => {
  const token = localStorage.getItem('token');
  
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, config);

  // Handle unauthorized responses
  if (response.status === 401) {
    localStorage.removeItem('token');
    window.location.href = '/login';
    return;
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

// Authentication API functions
export const authAPI = {
  login: async (email, password) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);

    const response = await fetch(`${API_BASE_URL}/api/login`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Login failed');
    }

    return response.json();
  },

  register: async (userData) => {
    const response = await fetch(`${API_BASE_URL}/api/users/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Registration failed');
    }

    return response.json();
  },

  getUserProfile: async () => {
    return apiRequest('/api/users/me');
  },

  updateUserProfile: async (userData) => {
    return apiRequest('/api/users/me', {
      method: 'PATCH',
      body: JSON.stringify(userData),
    });
  },
};

// Chat API functions
export const chatAPI = {
  createChat: async (chatData) => {
    return apiRequest('/api/chats', {
      method: 'POST',
      body: JSON.stringify(chatData),
    });
  },

  getChats: async () => {
    return apiRequest('/api/chats');
  },

  getChat: async (chatId) => {
    return apiRequest(`/api/chats/${chatId}`);
  },

  updateChat: async (chatId, chatData) => {
    return apiRequest(`/api/chats/${chatId}`, {
      method: 'PATCH',
      body: JSON.stringify(chatData),
    });
  },

  deleteChat: async (chatId) => {
    return apiRequest(`/api/chats/${chatId}`, {
      method: 'DELETE',
    });
  },
};

// Message API functions
export const messageAPI = {
  getMessages: async (chatId) => {
    return apiRequest(`/api/chats/${chatId}/messages`);
  },

  sendMessage: async (chatId, messageData) => {
    return apiRequest(`/api/chats/${chatId}/messages`, {
      method: 'POST',
      body: JSON.stringify(messageData),
    });
  },

  streamMessage: async (chatId, messageData) => {
    const token = localStorage.getItem('token');
    
    const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(messageData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.body;
  },
};

// Health check
export const healthCheck = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
};