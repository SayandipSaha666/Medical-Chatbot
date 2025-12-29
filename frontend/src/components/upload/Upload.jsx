import { useAuth } from "@clerk/clerk-react";

const Upload = ({ chatId, setImg }) => {
  const { getToken } = useAuth();

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImg((prev) => ({ ...prev, isLoading: true }));

    try {
      const token = await getToken();
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/chats/${chatId}/images`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (!res.ok) {
        throw new Error("Image upload failed");
      }

      const data = await res.json();

      setImg({
        isLoading: false,
        dbData: data, // contains image_url, thumbnail_url
        aiData: null,
      });
    } catch (err) {
      console.error(err);
      setImg((prev) => ({ ...prev, isLoading: false }));
    }
  };

  return (
    <label style={{ cursor: "pointer" }}>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        hidden
      />
      <img src="/attachment.png" alt="upload" className="w-8 h-8 mx-2"/>
    </label>
  );
};

export default Upload;
