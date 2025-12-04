import React, { useState } from 'react';

function NewPrompt({ onSubmit, isLoading }) {
  const [inputText, setInputText] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    
    onSubmit(inputText);
    setInputText(''); // Clear input after submission
  };

  return (
    <div className="w-[50%] mt-auto bg-[#2c2937] rounded-2xl flex fixed bottom-4">
      <form onSubmit={handleSubmit} className='w-full h-full flex items-center justify-between gap-5 mb-3'>
        <input 
          type="text" 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask me anything..." 
          className='flex-1 p-5 bg-transparent text-[#ececec] border-none outline-none w-full'
          disabled={isLoading}
        />
        <button 
          type="submit"
          disabled={isLoading}
          className={`bg-[#605e68] rounded-full border-none cursor-pointer p-3 flex items-center justify-center mr-5 p-4'}`}
        >
          <img src="/arrow.png" alt="submit" className='w-8'/>
        </button>
      </form>
    </div>
  );
}

export default NewPrompt