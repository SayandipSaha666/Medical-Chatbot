import React from 'react'
import { Outlet,Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '@clerk/clerk-react'
import { useEffect } from 'react'
import ChatList from '../../components/ChatList/ChatList'
function DashBoardLayout() {
    const {userId, isLoaded} = useAuth()
    const navigate = useNavigate()

    useEffect(()=>{
        if(isLoaded && !userId){
            navigate('/sign-in')
        }
    },[isLoaded,userId,navigate])

    if(!isLoaded){
        return "Loading..."
    }

  return (
    <div className='flex gap-12 pt-1 h-full'>
        <div className="flex-1"><ChatList/></div>
        <div className="flex-4 bg-[#212121]"><Outlet/></div>
    </div>
  )
}

export default DashBoardLayout