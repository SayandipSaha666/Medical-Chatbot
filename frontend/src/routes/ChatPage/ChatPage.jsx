import React, { useState, useEffect, useRef } from 'react';
import NewPrompt from '../../components/NewPrompt/NewPrompt';
import { sendMessage, checkBackendHealth } from '../../services/chatService';

function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const endref = useRef(null);

  // Check backend health on component mount
  useEffect(() => {
    const initializeBackend = async () => {
      try {
        console.log('[CHATPAGE] Checking backend health...');
        await checkBackendHealth();
        console.log('[CHATPAGE] Backend is ready');
        setBackendReady(true);
        setBackendError(null);
      } catch (error) {
        console.error('[CHATPAGE] Backend health check failed:', error);
        setBackendError(error.message);
        setBackendReady(false);
      }
    };

    initializeBackend();
  }, []);

  useEffect(() => {
    endref.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleNewMessage = async (text) => {
    if (!text.trim()) return;

    if (!backendReady) {
      setMessages(prev => [...prev, { 
        text: "Backend is not ready. Please check your server connection.",
        isUser: false,
        isError: true
      }]);
      return;
    }

    try {
      console.log('[CHATPAGE] Handling new message:', text);
      setIsLoading(true);
      
      // Add user message immediately 
      setMessages(prev => [...prev, { text, isUser: true }]);

      // Get AI response
      console.log('[CHATPAGE] Calling sendMessage service...');
      const response = await sendMessage(text);
      
      console.log('[CHATPAGE] Got response from service');
      
      // Response is HTML string from backend
      // We'll render it as raw HTML
      setMessages(prev => [...prev, { 
        text: response,
        isUser: false,
        isHTML: true  // Flag to indicate this is HTML
      }]);
    } catch (error) {
      console.error('[CHATPAGE] Error getting response:', error);
      
      // Provide detailed error message
      let errorMessage = error.message;
      if (error.message.includes('Failed to fetch')) {
        errorMessage = "Network error: Cannot reach backend. Make sure the server is running on http://localhost:8080";
      } else if (error.message.includes('HTTP')) {
        errorMessage = `Server error: ${error.message}`;
      }
      
      setMessages(prev => [...prev, { 
        text: `Sorry, I encountered an error: ${errorMessage}`,
        isUser: false,
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='h-full flex flex-col items-center relative pb-24'>
      {backendError && (
        <div className="w-full bg-red-900 text-white p-4 text-center">
          Backend Connection Error: {backendError}
        </div>
      )}
      
      <div className="flex-1 overflow-scroll w-full flex justify-center scroll-container">
        <div className="w-[50%] flex flex-col gap-5">
          {messages.map((msg, index) => (
            <div 
              key={index} 
              className={`message ${msg.isUser ? 'user' : ''} ${msg.isError ? 'error' : ''}`}
              style={{ whiteSpace: msg.isHTML ? 'normal' : 'pre-wrap' }}
            >
              {msg.isHTML ? (
                <div dangerouslySetInnerHTML={{ __html: msg.text }} />
              ) : (
                msg.text
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message">
              <div className="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></div>
            </div>
          )}
          <div ref={endref}/>
        </div>
        <div className="fixed bottom-0 left-100 right-0 flex items-center justify-center pb-6 bg-gradient-to-t from-[#1a1825] to-transparent pt-6">
          <NewPrompt onSubmit={handleNewMessage} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
}
  

export default ChatPage