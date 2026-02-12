import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import ChatPage from './routes/ChatPage/ChatPage';
import DashboardPage from './routes/DashboardPage/DashboardPage';
import LoginPage from './routes/SignInpage/SignInPage';
import SignupPage from './routes/SignUpPage/SignUpPage';
import DashBoardLayout from './Layouts/DashBoardLayout/DashBoardLayout';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/dashboard" element={<DashBoardLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="chats/:id" element={<ChatPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
