import React from 'react'
import { Link } from 'react-router-dom'

function HomePage() {
  return (
    <div className="flex items-center h-full gap-25">
      <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center">
        <h1 className="text-[128px] bg-linear-to-r from-[#217bfe] to-[#e55571] bg-clip-text text-transparent">MedGPT</h1>
        <h2 className='font-semibold text-3xl'>Ask any medical related queries</h2>
        <h3 className='w-[70%] font-medium text-gray-400'>
          Your AI-powered medical assistant. Get instant, reliable answers
          to your health questions — from symptoms and medications to
          wellness tips — all in a secure, private conversation.
        </h3>
        <Link to='/dashboard' className='py-4 px-6 bg-[#217bfe] text-white rounded-full text-2xl mt-5 hover:bg-white hover:text-[#217bfe] transition duration-300'>Get Started</Link>
      </div>
      <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center">
        <div className="flex items-center justify-center bg-[#140e2d] rounded-full w-[50%] h-[50%] relative overflow-hidden">
          <div className="absolute inset-0 bg-linear-to-br from-[#217bfe]/20 to-[#e55571]/20 animate-pulse rounded-full"></div>
          <img src="/img.webp" alt="bot" className="w-full h-full object-contain rounded-2xl relative z-10" />
        </div>
      </div>
    </div>
  )
}

export default HomePage