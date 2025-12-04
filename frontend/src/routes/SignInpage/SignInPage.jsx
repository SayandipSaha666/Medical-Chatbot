import { SignIn } from '@clerk/clerk-react'
import React from 'react'

function SignInPage() {
  return (
    <div className='flex h-full items-center justify-center'><SignIn path='/sign-in' signUpUrl='/sign-up' forceRedirectUrl='/dashboard'/></div>
  )
}

export default SignInPage