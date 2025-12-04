import React from 'react'
const handleSubmit = () => {

}
function DashboardPage() {
  return (
    <div className="h-full flex flex-col items-center">
      <div className="flex-1 flex flex-col items-center justify-center w-[50%] gap-12">
        <div className="flex items-center gap-5 opacity-20">
          <img src="/logo.png" alt="MedGPT" className='w-40 rounded-full'/>
          <h1 className='text-[128px] bg-linear-to-r from-[#217bfe] to-[#e55571] bg-clip-text text-transparent'>MedGPT</h1>
        </div>
      </div>
      <div className="mt-auto w-[50%] bg-[#2c2937] rounded-2xl flex">
        <form onSubmit={handleSubmit} className='w-full h-full flex items-center justify-between gap-5 mb-3'>
          <input type="text" name="text" placeholder="Ask me anything..." className='flex-1 p-5 bg-transparent text-[#ececec] border-none outline-none w-full'/>
          <button className='bg-[#605e68] rounded-full border-none cursor-pointer p-3 flex items-center justify-center mr-5 '>
            <img src="/arrow.png" alt="submit" className='w-8'/>
          </button>
        </form>
      </div>
    </div>
  )
}

export default DashboardPage