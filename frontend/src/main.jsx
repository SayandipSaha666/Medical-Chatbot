import { StrictMode } from 'react'
import { createRoot, ReactDOM } from 'react-dom/client'
import { createBrowserRouter,RouterProvider } from 'react-router-dom'
import HomePage from './routes/HomePage/HomePage.jsx'
import ChatPage from './routes/ChatPage/ChatPage.jsx'
import DashboardPage from './routes/DashboardPage/DashboardPage.jsx'
import SignInPage from './routes/SignInpage/SignInPage.jsx'
import SignUpPage from './routes/SignUpPage/SignUpPage.jsx'
import './index.css'
import App from './App.jsx'
import RootLayout from './Layouts/RootLayout/RootLayout.jsx'
import DashBoardLayout from './Layouts/DashBoardLayout/DashBoardLayout.jsx'

const router = createBrowserRouter([
  {
    element: <RootLayout/>,
    children: [
      {
        path: '/',
        element: <HomePage/>
      },
      {
        path: '/sign-in/*',
        element: <SignInPage/>
      },
      {
        path: '/sign-up/*',
        element: <SignUpPage/>
      }
      ,{
        element:<DashBoardLayout/>,
        children: [
          {
            path: '/dashboard',
            element: <DashboardPage/>
          },
          {
            path: '/dashboard/chat/:id',
            element: <ChatPage/>
          }
        ]
      }
    ]
  }
])

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router}/>
  </StrictMode>,
)
