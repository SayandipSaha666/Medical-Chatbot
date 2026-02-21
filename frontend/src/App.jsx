import { useRoutes } from "react-router-dom";

import HomePage from "./routes/HomePage/HomePage.jsx";
import ChatPage from "./routes/ChatPage/ChatPage.jsx";
import DashboardPage from "./routes/DashboardPage/DashboardPage.jsx";
import SignInPage from "./routes/SignInpage/SignInPage.jsx";
import SignUpPage from "./routes/SignUpPage/SignUpPage.jsx";

import RootLayout from "./Layouts/RootLayout/RootLayout.jsx";
import DashBoardLayout from "./Layouts/DashBoardLayout/DashBoardLayout.jsx";
import './App.css';
import { Analytics } from "@vercel/analytics/react"

function CustomRoutes() {
  const routes = useRoutes([
    {
      path: "/",
      element: <RootLayout />,
      children: [
        {
          path: "/",
          element: <HomePage />,
        },
        {
          path: "/login",
          element: <SignInPage />,
        },
        {
          path: "/signup",
          element: <SignUpPage />,
        },
        {
          element: <DashBoardLayout />,
          children: [
            {
              path: "/dashboard",
              element: <DashboardPage />,
            },
            {
              path: "/dashboard/chats/:id",
              element: <ChatPage />,
            },
          ],
        },
      ],
    },
  ]);

  return routes;
}

function App() {

  return (
    <>
      <Analytics />
      <CustomRoutes />
    </>

  )
}

export default App;
