import React from 'react'
import {Link,Outlet} from 'react-router-dom'
import { SignedOut, SignedIn } from '@clerk/clerk-react'
import { SignInButton, UserButton } from '@clerk/clerk-react'
// Import your Publishable Key
const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

if (!PUBLISHABLE_KEY) {
  throw new Error('Missing Publishable Key')
}

function RootLayout() {
  return (
        <div className="py-4 px-16 h-screen flex flex-col">
            <header className='flex align-center justify-between'>
                <Link to="/" className='flex items-center font-bold gap-2'>
                    <img src="/logo.png" alt="MedGPT" className='w-15 rounded-full'/>
                    <span className='font-bold text-2xl'>MedGPT</span>
                </Link>
                <div className='font-bold text-2xl'>
                    <SignedOut>
                        <SignInButton />
                    </SignedOut>
                    <SignedIn>
                        <UserButton />
                    </SignedIn>
                </div>
            </header>
            <main className='flex-1 overflow-hidden'>
                <Outlet/>
            </main>
        </div>
  )
}

export default RootLayout