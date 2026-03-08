import React from "react"
import type { Metadata } from 'next'

import './globals.css'

export const metadata: Metadata = {
  title: 'Set Solver - Find Sets in Your Cards',
  description: 'Upload a photo of Set cards and instantly find all valid sets using AI vision',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}
