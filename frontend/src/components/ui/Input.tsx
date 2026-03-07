import React, { useEffect, useState } from 'react'
import { Eye, EyeClosed } from "lucide-react";


type InputProps = {
    id?: string,
    name?: string,
    className?: string,
    type?: string,
    value?: string,
    placeholder: string,
    onChange?: (e: string) => void
    required?: boolean,
    disabled?: Boolean,
}
const Input = ({ id, name, className, type = "text", placeholder, value, onChange, required = false, disabled = false }: InputProps) => {

    const uid = id || `input-${(Math.random() * 100).toFixed(1)}`

    // handle input
    const [input, setInput] = useState<string>(value || "");
    const [inputType, setInputType] = useState<string>(type);
    const [showPassword, setShowPassword] = useState(false);

    useEffect(() => {
        if (value !== undefined && value !== input)
            setInput(value);
    }, [value]);

    // handle input change 
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value;
        setInput(val);
        onChange?.(val);
    };

    useEffect(() => {
        if (type === "password")
            setInputType(showPassword ? "text" : "password")
    }, [showPassword]);

    return (
        <div className='relative'>
            <input
                type={inputType}
                id={id}
                name={name}
                onChange={handleChange}
                value={input}
                required={required}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder={placeholder}
                disabled={disabled as boolean}
            />
            {type === "password" &&
                <label htmlFor={id} className="cursor-pointer absolute right-3 top-1/2 -translate-y-1/2" onClick={() => setShowPassword(!showPassword)}>{!showPassword ? <EyeClosed size={17} /> : <Eye size={17} />}</label>
            }
        </div>
    )
}

export default Input
