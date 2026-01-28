import React from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { useEffect } from 'react'
import ChatList from '../../components/ChatList/ChatList'

function DashBoardLayout() {
    const { token, loading } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        if (!loading && !token) {
            navigate('/login');
        }
    }, [token, loading, navigate]);

    if (loading) {
        return "Loading...";
    }

    return (
        <div className='flex gap-12 pt-1 h-full'>
            <div className="flex-1"><ChatList/></div>
            <div className="flex-4 bg-[#212121]"><Outlet/></div>
        </div>
    )
}

export default DashBoardLayout