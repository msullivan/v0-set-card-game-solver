import React from "react"
import type { Metadata } from 'next'
import Script from 'next/script'

import './globals.css'

const OPENCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js"

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
      <body className="font-sans antialiased">
        <Script src={OPENCV_URL} strategy="afterInteractive" />
        {children}
      </body>
    </html>
  )
}
