import React from 'react'
import './ChatList.css'
import { Link } from 'react-router-dom'
function ChatList() {
  return (
    <div className='flex flex-col h-full'>
      <span className='font-semibold text-xl mb-3'>DASHBOARD</span>
      <Link to='/dashboard'>Create a new chat</Link>
      <Link to='/'>Explore to MedGPT</Link>
      <Link to='/'>Contact</Link>
      <hr className='h-1 bg-[#ddd] opacity-10 rounded-2xl mt-5 ml-0 '/>
      <span className='font-semibold text-xl mb-3'>Recent Chats</span>
      <div className='flex flex-col overflow-y-scroll max-h-135 space-y-2 px-3 scroll-container'>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link> 
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
        <Link to='/'>My Chat Title</Link>
      </div>
      <hr />
      <div className="mt-auto flex items-center gap-3 text-2xl ">
        <Link to='/'>
          <img src="/logo.png" alt="MedGPT" className='w-15 rounded-full' />
        </Link>
        <div className="flex flex-col text-sm font-medium">
          <span>Upgrade to MedGPT Pro</span>
          <span>Get unlimited access to all features</span>
        </div>
      </div>
    </div>
  )
}

export default ChatList