// src/services/chatService.jsx
const API_URL = import.meta.env.VITE_API_BASE_URL;

export const sendMessage = async ({ chatId, message, token }) => {
  console.log("[FRONTEND] Sending message:", message);
    console.log("Clerk token:", token);

  const response = await fetch(
    `${API_URL}/api/chats/${chatId}/messages`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`, // ✅ Clerk JWT
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

  // ✅ Backend returns MessageOut JSON
  return await response.json();
};

export const checkBackendHealth = async () => {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) {
    throw new Error(`Backend health check failed: ${response.statusText}`);
  }
  return await response.json();
};
