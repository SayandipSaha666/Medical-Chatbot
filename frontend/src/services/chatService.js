// src/services/chatService.jsx
const API_URL = import.meta.env.VITE_API_BASE_URL;

export const sendMessage = async ({ chatId, message }) => {
  console.log("[FRONTEND] Sending message:", message);
  console.log("Calling API:", `${API_URL}/api/chats/${chatId}/messages`);

  // Get token from localStorage
  const token = localStorage.getItem('token');

  const response = await fetch(
    `${API_URL}/api/chats/${chatId}/messages`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`, // JWT Token
      },
      body: JSON.stringify({
        content: message,
      }),
    }
  );

  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      errorMsg = err.detail || JSON.stringify(err);
    } catch {}
    throw new Error(errorMsg);
  }

  // Backend returns MessageOut JSON
  return await response.json();
};

export const checkBackendHealth = async () => {
  try {
    console.log("Checking backend health...", API_URL);
    const response = await fetch(`${API_URL}/health`);
    console.log("Backend health check response:", response);
    if (!response.ok) {
      throw new Error(`Backend health check failed: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.log("Backend health check error:", error);
    throw error;
  }
};
