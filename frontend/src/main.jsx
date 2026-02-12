import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "./context/AuthContext.jsx";

import HomePage from "./routes/HomePage/HomePage.jsx";
import ChatPage from "./routes/ChatPage/ChatPage.jsx";
import DashboardPage from "./routes/DashboardPage/DashboardPage.jsx";
import SignInPage from "./routes/SignInpage/SignInPage.jsx";
import SignUpPage from "./routes/SignUpPage/SignUpPage.jsx";
import App from './App.jsx'
import RootLayout from "./Layouts/RootLayout/RootLayout.jsx";
import DashBoardLayout from "./Layouts/DashBoardLayout/DashBoardLayout.jsx";

import "./index.css";

/*  React Query  */
const queryClient = new QueryClient();

const router = createBrowserRouter([
  {
    element: <RootLayout />,
    children: [
      {
        path: "/",
        element: <SignInPage />,
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
            path: "/dashboard/chats/:id", // MATCHES ChatList links
            element: <ChatPage />,
          },
        ],
      },
    ],
  },
]);

/* -------------------- Render -------------------- */
createRoot(document.getElementById("root")).render(
  <StrictMode>
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </AuthProvider>
  </StrictMode>
);
