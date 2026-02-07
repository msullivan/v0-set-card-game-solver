# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Set Solver is a Next.js web app that uses Claude Vision API to analyze photos of Set game cards and find all valid sets. Users upload a photo, the app identifies each card's attributes (color, shape, shading, number) via AI, then calculates valid Set combinations.

## Commands

```bash
pnpm install          # Install dependencies (pnpm 10.28.2 required)
npm run dev           # Dev server with Turbo (http://localhost:3000)
npm run build         # Production build (ignores TS errors via next.config.mjs)
npm run lint          # ESLint via Next.js
npm run start         # Production server
```

No unit test framework is configured. Manual testing uses sample images in `test-images/`.

## Architecture

**Data flow:** Image upload (client) → POST `/api/analyze` → CV card detection → parallel AI analysis per card → `findAllSets()` game logic → results displayed

### Two-pass analysis pipeline

The app uses a two-pass approach for higher accuracy than single-pass AI analysis:

1. **Pass 1 — CV card detection** (`lib/detect-cards.ts`): Uses opencv.js (WASM) to locate and crop individual cards from the photo. Pipeline: grayscale → background subtraction (large Gaussian blur) to normalize lighting/shadows → threshold → border clearing → erosion → contour detection filtered by area, aspect ratio, and rectangularity. Processes at 1000px max dimension for speed, crops at full resolution. Falls back to 90-degree rotation if no cards are detected.

2. **Pass 2 — AI card identification** (`app/api/analyze/route.ts`): Sends each cropped card image to Claude Sonnet 4 in parallel via `generateObject` with a Zod schema. Each card is identified independently (color, shape, shading, number), eliminating counting errors from the single-pass approach.

### Key layers

- **`app/page.tsx`** — Main client component. All UI state lives here via `useState` (image data, analysis results, errors). No external state management.
- **`app/api/analyze/route.ts`** — API route. Decodes uploaded image, runs CV detection, sends each crop to Claude Sonnet 4 in parallel, assembles results and finds valid sets.
- **`lib/detect-cards.ts`** — CV card detection using opencv.js + sharp. Singleton WASM initialization. Exports `detectCards(imageBuffer)` returning an array of JPEG crop buffers.
- **`lib/set-game.ts`** — Pure game logic: `isValidSet()`, `findAllSets()` (brute-force O(n³)), type definitions (`SetCard`, `ValidSet`, `CardColor`, `CardShape`, `CardShading`, `CardNumber`).
- **`components/set-card-display.tsx`** — SVG rendering of cards with diamond/oval/squiggle shapes and solid/striped/empty shading patterns.
- **`components/ui/`** — shadcn/ui components (Radix UI + Tailwind). Configured via `components.json`.

### AI Integration

- Model: `anthropic/claude-sonnet-4-20250514`
- Uses `generateObject` from `ai` package (Vercel AI SDK v6) with Zod schema validation
- Per-card prompt focuses on identifying 4 attributes of a single card (simpler than the old whole-image prompt)

## Environment Variables

- `AI_GATEWAY_API_KEY` — Anthropic API key (set in `.env.local`)

## Tech Stack

- Next.js 16 (App Router), React 19, TypeScript (strict mode)
- pnpm 10.28.2 (package manager)
- Tailwind CSS 3 with class-based dark mode, shadcn/ui components
- Vercel AI SDK v6 + Zod for structured AI responses
- opencv.js (@techstark/opencv-js) for CV card detection (WASM)
- sharp for image processing (resize, crop, rotate)
- Path alias: `@/*` maps to project root
