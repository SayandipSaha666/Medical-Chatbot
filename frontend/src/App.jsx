import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import ChatPage from './routes/ChatPage/ChatPage';
import DashboardPage from './routes/DashboardPage/DashboardPage';
import LoginPage from './routes/SignInpage/SignInPage';
import SignupPage from './routes/SignUpPage/SignUpPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/dashboard/chats/:id" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
