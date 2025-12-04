import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import ChatPage from './routes/ChatPage/ChatPage';
import DashboardPage from './routes/DashboardPage/DashboardPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/dashboard/chat/:chatId" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
