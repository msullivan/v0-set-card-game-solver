"use client"

import { useState } from "react"
import { ImageUpload } from "@/components/image-upload"
import { SetResults } from "@/components/set-results"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Loader2, Sparkles, RefreshCw, HelpCircle } from "lucide-react"
import { type SetCard, type ValidSet } from "@/lib/set-game"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

interface Timing {
  cvInit: number
  cvProcess: number
  ai: number
  sets: number
}

interface AnalysisResult {
  cards: SetCard[]
  validSets: ValidSet[]
  confidence: "high" | "medium" | "low"
  notes?: string
  timing?: Timing
}

export default function SetSolverPage() {
  const [imageData, setImageData] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleImageSelect = (data: string) => {
    setImageData(data)
    setResult(null)
    setError(null)
  }

  const analyzeImage = async () => {
    if (!imageData) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || "Failed to analyze image")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const reset = () => {
    setImageData(null)
    setResult(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">Set Solver</h1>
              <p className="text-sm text-muted-foreground">Find sets with AI and some vibecoded CV</p>
            </div>
          </div>
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon">
                <HelpCircle className="w-5 h-5" />
                <span className="sr-only">How to play Set</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>How to Play Set</DialogTitle>
                <DialogDescription>
                  Set is a card game where players find groups of three cards that form a valid set.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 text-sm text-foreground">
                <p>
                  Each card has four attributes: <strong>color</strong> (red, green, purple), <strong>shape</strong> (diamond, oval, squiggle), <strong>shading</strong> (solid, striped, empty), and <strong>number</strong> (1, 2, or 3).
                </p>
                <p>
                  A valid set consists of three cards where each attribute is either <strong>all the same</strong> or <strong>all different</strong> across the three cards.
                </p>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-medium mb-2">Example of a valid set:</p>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>All different colors (red, green, purple)</li>
                    <li>All same shape (diamonds)</li>
                    <li>All different shading (solid, striped, empty)</li>
                    <li>All same number (2)</li>
                  </ul>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        {/* Upload Section */}
        <section className="space-y-4">
          <ImageUpload onImageSelect={handleImageSelect} disabled={isAnalyzing} />

          {imageData && !result && (
            <div className="flex items-center gap-3">
              <Button
                onClick={analyzeImage}
                disabled={isAnalyzing}
                className="flex-1 sm:flex-none"
                size="lg"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing cards...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Find Sets
                  </>
                )}
              </Button>
              {!isAnalyzing && (
                <Button variant="outline" onClick={reset} size="lg">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Clear
                </Button>
              )}
            </div>
          )}

          {error && (
            <Card className="border-destructive">
              <CardContent className="py-4">
                <p className="text-destructive text-sm">{error}</p>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Results Section */}
        {result && (
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-foreground">Results</h2>
              <Button variant="outline" onClick={reset}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Analyze New Photo
              </Button>
            </div>
            {result.timing && (
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                {result.timing.cvInit > 100 && (
                  <span>OpenCV init: {(result.timing.cvInit / 1000).toFixed(1)}s</span>
                )}
                <span>CV detection: {(result.timing.cvProcess / 1000).toFixed(1)}s</span>
                <span>AI analysis: {(result.timing.ai / 1000).toFixed(1)}s</span>
                <span>Set finding: {result.timing.sets < 1 ? "<1ms" : `${result.timing.sets}ms`}</span>
                <span className="font-medium">
                  Total: {((result.timing.cvInit + result.timing.cvProcess + result.timing.ai + result.timing.sets) / 1000).toFixed(1)}s
                </span>
              </div>
            )}
            <SetResults
              cards={result.cards}
              validSets={result.validSets}
              confidence={result.confidence}
              notes={result.notes}
            />
          </section>
        )}

        {/* Empty State */}
        {!imageData && (
          <Card className="border-dashed">
            <CardContent className="py-12 text-center">
              <div className="w-16 h-16 mx-auto rounded-full bg-muted flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-medium text-foreground mb-2">
                Ready to find some sets?
              </h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Upload a photo of your Set cards and our AI will identify all the cards
                and find every valid set for you.
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center text-sm text-muted-foreground">
          <p>
            Take a clear, well-lit photo of your Set cards for best results.
            Works with 12 or more cards laid out in a grid.
          </p>
        </div>
      </footer>
    </main>
  )
}
