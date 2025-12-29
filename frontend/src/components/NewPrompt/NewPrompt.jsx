import React, { useState } from "react";
import Upload from "../upload/Upload";

function NewPrompt({ chatId, onSubmit, isLoading }) {
  const [inputText, setInputText] = useState("");

  const [img, setImg] = useState({
    isLoading: false,
    dbData: null,
    aiData: null,
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    onSubmit(inputText, img.dbData); // pass image info if needed
    setInputText("");
    setImg({ isLoading: false, dbData: null, aiData: null });
  };

  return (
    <>
      {img.isLoading && <div>Uploading image...</div>}

      {img.dbData?.image_url && (
        <img
          src={img.dbData.image_url}
          alt="uploaded"
          style={{ maxWidth: "380px", borderRadius: "10px" }}
        />
      )}

      <div className="w-[50%] mt-auto bg-[#2c2937] rounded-2xl flex fixed bottom-4">
        <form
          onSubmit={handleSubmit}
          className="w-full h-full flex items-center justify-between gap-5 mb-3"
        >
          <Upload chatId={chatId} setImg={setImg} />

          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask me anything..."
            className="flex-1 p-5 bg-transparent text-[#ececec] border-none outline-none w-full"
            disabled={isLoading}
          />

          <button
            type="submit"
            disabled={isLoading}
            className="bg-[#605e68] rounded-full border-none cursor-pointer p-4 mr-5"
          >
            <img src="/arrow.png" alt="submit" className="w-8" />
          </button>
        </form>
      </div>
    </>
  );
}

export default NewPrompt;
