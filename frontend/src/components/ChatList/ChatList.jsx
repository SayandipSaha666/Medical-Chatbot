import { Link } from "react-router-dom";
import "./ChatList.css";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/clerk-react";

const ChatList = () => {
  const { getToken } = useAuth();

  const { isPending, error, data } = useQuery({
    queryKey: ["userChats"],
    queryFn: async () => {
      const token = await getToken();

      const res = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/chats`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!res.ok) {
        throw new Error("Failed to fetch chats");
      }

      return res.json();
    },
  });

  return (
    <div className="chatList">
      <span className="title">DASHBOARD</span>

      <Link to="/dashboard">➕ Create a new Chat</Link>
      <Link to="/">Explore MedGPT</Link>
      <Link to="/">Contact</Link>

      <hr />

      <span className="title">RECENT CHATS</span>

      <div className="list">
        {isPending && <span>Loading...</span>}

        {error && <span>Something went wrong!</span>}

        {!isPending && !error && data?.length === 0 && (
          <span>No chats yet</span>
        )}

        {data?.map((chat) => (
          <Link
            to={`/dashboard/chat/${chat.id}`}
            key={chat.id}
            className="chatItem"
          >
            {chat.title || "Untitled Chat"}
          </Link>
        ))}
      </div>

      <hr />

      <div className="upgrade">
        <img src="/logo.png" alt="logo" />
        <div className="texts">
          <span>Upgrade to MedGPT Pro</span>
          <span>Get unlimited access to all features</span>
        </div>
      </div>
    </div>
  );
};

export default ChatList;
