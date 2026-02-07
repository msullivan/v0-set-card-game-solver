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

**Data flow:** Image upload (client) → POST `/api/analyze` → Claude Vision API (`generateObject`) → `findAllSets()` game logic → results displayed

### Key layers

- **`app/page.tsx`** — Main client component. All UI state lives here via `useState` (image data, analysis results, errors). No external state management.
- **`app/api/analyze/route.ts`** — API route that sends base64 image to Claude Sonnet 4 via Vercel AI SDK v6. Uses Zod schemas for structured output. Calls `findAllSets()` on the AI-detected cards before returning.
- **`lib/set-game.ts`** — Pure game logic: `isValidSet()`, `findAllSets()` (brute-force O(n³)), type definitions (`SetCard`, `ValidSet`, `CardColor`, `CardShape`, `CardShading`, `CardNumber`).
- **`components/set-card-display.tsx`** — SVG rendering of cards with diamond/oval/squiggle shapes and solid/striped/empty shading patterns.
- **`components/ui/`** — shadcn/ui components (Radix UI + Tailwind). Configured via `components.json`.

### AI Integration

- Model: `anthropic/claude-sonnet-4-20250514`
- Uses `generateObject` from `ai` package (Vercel AI SDK v6) with Zod schema validation
- The prompt in the route handler includes detailed instructions about common card identification mistakes (especially shading confusion)

## Environment Variables

- `AI_GATEWAY_API_KEY` — Anthropic API key (set in `.env.local`)

## Tech Stack

- Next.js 16 (App Router), React 19, TypeScript (strict mode)
- pnpm 10.28.2 (package manager)
- Tailwind CSS 3 with class-based dark mode, shadcn/ui components
- Vercel AI SDK v6 + Zod for structured AI responses
- Path alias: `@/*` maps to project root
