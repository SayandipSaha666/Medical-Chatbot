import React from 'react'
import { Link } from 'react-router-dom'
function HomePage() {
  return (
    <div className="flex items-center h-full gap-25">
      <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center">
        <h1 className="text-[128px] bg-linear-to-r from-[#217bfe] to-[#e55571] bg-clip-text text-transparent">MedGPT</h1>
        <h2 className='font-semibold text-3xl'>Ask any medical related queries</h2>
        <h3 className=' w-[70%] font-medium'>Lorem ipsum dolor sit amet consectetur adipisicing elit.
          Quidem iusto quisquam error explicabo laudantium facere. 
          Perspiciatis numquam consectetur hic libero!
        </h3> 
        <Link to='/dashboard' className='py-4 px-6 bg-[#217bfe] text-white rounded-full text-2xl mt-5 hover:bg-white hover:text-[#217bfe] transition duration-300'>Get Started</Link>
      </div>
      <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center">
        <div className="flex items-center justify-center bg-[#140e2d] rounded-full w-[50%] h-[50%] relative">
          <div className="">
            <div className="bg">

            </div>
          </div>
          <img src="/img.webp" alt="bot" className="w-full h-full object-contain rounded-2xl" />
        </div>
      </div>
    </div>
  )
}

export default HomePage

/*
  <div className='HomePage'>
    <Link to="/dashboard">Dashboard</Link>
  </div>
*/