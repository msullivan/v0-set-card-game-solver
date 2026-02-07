"use client"

import React from "react"

import { useCallback, useState } from "react"
import { Upload, X, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ImageUploadProps {
  onImageSelect: (imageData: string) => void
  disabled?: boolean
}

export function ImageUpload({ onImageSelect, disabled }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const MAX_BYTES = 3.5 * 1024 * 1024

  const downscaleImage = useCallback(
    (dataUrl: string): Promise<string> => {
      return new Promise((resolve) => {
        const img = new Image()
        img.onload = () => {
          const canvas = document.createElement("canvas")
          // Scale down by 50% each iteration until under limit
          let { width, height } = img
          let result = dataUrl
          const tryScale = (scale: number) => {
            canvas.width = Math.round(width * scale)
            canvas.height = Math.round(height * scale)
            const ctx = canvas.getContext("2d")!
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
            return canvas.toDataURL("image/jpeg", 0.85)
          }
          let scale = 1
          while (result.length > MAX_BYTES && scale > 0.1) {
            scale *= 0.5
            result = tryScale(scale)
          }
          resolve(result)
        }
        img.src = dataUrl
      })
    },
    []
  )

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        return
      }

      const reader = new FileReader()
      reader.onloadend = async () => {
        let result = reader.result as string
        if (result.length > MAX_BYTES) {
          result = await downscaleImage(result)
        }
        setPreview(result)
        onImageSelect(result)
      }
      reader.readAsDataURL(file)
    },
    [onImageSelect, downscaleImage]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const clearImage = useCallback(() => {
    setPreview(null)
  }, [])

  if (preview) {
    return (
      <div className="relative rounded-xl overflow-hidden border-2 border-border bg-card">
        <img
          src={preview || "/placeholder.svg"}
          alt="Uploaded Set cards"
          className="w-full h-auto max-h-[500px] object-contain"
        />
        <Button
          variant="destructive"
          size="icon"
          className="absolute top-3 right-3 rounded-full shadow-lg"
          onClick={clearImage}
          disabled={disabled}
        >
          <X className="h-4 w-4" />
          <span className="sr-only">Remove image</span>
        </Button>
      </div>
    )
  }

  return (
    <label
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={cn(
        "flex flex-col items-center justify-center gap-4 p-12 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200",
        isDragging
          ? "border-primary bg-accent scale-[1.02]"
          : "border-border hover:border-primary/50 hover:bg-accent/50",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <input
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleInputChange}
        disabled={disabled}
      />
      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-primary/10">
        {isDragging ? (
          <ImageIcon className="w-8 h-8 text-primary" />
        ) : (
          <Upload className="w-8 h-8 text-primary" />
        )}
      </div>
      <div className="text-center">
        <p className="text-lg font-medium text-foreground">
          {isDragging ? "Drop your image here" : "Upload a photo of Set cards"}
        </p>
        <p className="text-sm text-muted-foreground mt-1">
          Drag and drop or click to browse
        </p>
      </div>
      <p className="text-xs text-muted-foreground">
        Supports JPG, PNG, HEIC up to 10MB
      </p>
    </label>
  )
}
