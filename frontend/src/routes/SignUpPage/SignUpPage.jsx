import { SignUp } from '@clerk/clerk-react'
import React from 'react'

function SignUpPage() {
  return (
    <div className='flex h-full items-center justify-center'><SignUp path='/sign-up' signInUrl='/sign-in'/></div>
  )
}

export default SignUpPage