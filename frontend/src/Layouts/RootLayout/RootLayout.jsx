import React from 'react'
import {Link, useNavigate, Outlet} from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

function RootLayout() {
  const { user, token, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
        <div className="py-4 px-16 h-screen flex flex-col">
            <header className='flex align-center justify-between'>
                <Link to="/" className='flex items-center font-bold gap-2'>
                    <img src="/logo.png" alt="MedGPT" className='w-15 rounded-full'/>
                    <span className='font-bold text-2xl'>MedGPT</span>
                </Link>
                <div className='font-bold text-2xl'>
                  {token ? (
                    <div className="flex items-center gap-4">
                      <span className="text-white">{user?.name || user?.email}</span>
                      <button
                        onClick={handleLogout}
                        className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md"
                      >
                        Logout
                      </button>
                    </div>
                  ) : (
                    <div className="flex gap-2">
                      <Link to="/login" className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
                        Login
                      </Link>
                      <Link to="/signup" className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md">
                        Sign Up
                      </Link>
                    </div>
                  )}
                </div>
            </header>
            <main className='flex-1 overflow-hidden'>
                <Outlet/>
            </main>
        </div>
  )
}

export default RootLayout