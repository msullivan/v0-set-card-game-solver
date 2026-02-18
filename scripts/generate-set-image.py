#!/usr/bin/env python3
"""Generate photorealistic Set card images using Gemini Pro via Vercel AI Gateway."""

import argparse
import base64
import json
import random
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

COLORS = ["red", "green", "purple"]
SHAPES = ["diamond", "oval", "squiggle"]
SHADINGS = ["solid", "striped", "outline"]
NUMBERS = [1, 2, 3]


def generate_cards(n: int = 12) -> list[dict]:
    """Generate n random unique Set cards."""
    all_cards = [
        {"color": c, "shape": s, "shading": sh, "number": n}
        for c in COLORS for s in SHAPES for sh in SHADINGS for n in NUMBERS
    ]
    return random.sample(all_cards, n)


def format_card_list(cards: list[dict]) -> str:
    lines = []
    for i, card in enumerate(cards, 1):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        lines.append(f"  Row {row}, Col {col}: {card['number']} {card['color']} {card['shading']} {card['shape']}(s)")
    return "\n".join(lines)


def build_prompt(cards: list[dict]) -> str:
    card_list = format_card_list(cards)
    return f"""Here is a reference photo of real Set game cards. Study the three shapes carefully:

1. DIAMOND: a four-sided rhombus shape, oriented horizontally (wider than tall)
2. OVAL: a rounded rectangle / stadium shape, oriented horizontally
3. SQUIGGLE: a fat blobby bean/slug shape with smooth organic curves — NOT an S or a 2. Look at the squiggle cards in the reference for examples.

All three shadings exist: solid (100% filled with color), striped (horizontal lines inside the shape), and outline (just the colored border, white/blank inside — 0% fill).

Generate a new photorealistic overhead photograph of exactly these 12 Set cards arranged in a 4 columns x 3 rows grid on a wooden table. ALL cards must be in portrait orientation (taller than wide), with shapes stacked vertically on each card. The squiggle must closely match the blobby organic shape from the reference photo.

The cards must be exactly (left to right, top to bottom):
{card_list}"""


def load_api_key() -> str:
    env_path = PROJECT_DIR / ".env.local"
    if not env_path.exists():
        print(f"Error: {env_path} not found", file=sys.stderr)
        sys.exit(1)
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if key.strip() == "AI_GATEWAY_API_KEY":
                return value.strip().strip("'\"")
    print("Error: AI_GATEWAY_API_KEY not found in .env.local", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate Set card images with Gemini Pro")
    parser.add_argument("reference", help="Path to a reference image of real Set cards")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("-n", "--num-cards", type=int, default=12, help="Number of cards (default: 12)")
    args = parser.parse_args()

    api_key = load_api_key()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: {ref_path} not found", file=sys.stderr)
        sys.exit(1)

    cards = generate_cards(args.num_cards)
    prompt = build_prompt(cards)

    print("Generated cards:")
    print(format_card_list(cards))
    print()

    ref_base64 = base64.b64encode(ref_path.read_bytes()).decode()

    body = {
        "model": "google/gemini-3-pro-image",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{ref_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    print("Generating Set card image with Gemini Pro...")
    request = urllib.request.Request(
        "https://ai-gateway.vercel.sh/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    resp = urllib.request.urlopen(request, timeout=120)
    result = json.loads(resp.read())
    msg = result.get("choices", [{}])[0].get("message", {})

    images = msg.get("images", [])
    if not images:
        print("No image returned. Response:", json.dumps(result, indent=2)[:1000], file=sys.stderr)
        sys.exit(1)

    img_data = images[0]["image_url"]["url"]
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    output_path = Path(args.output)
    output_path.write_bytes(base64.b64decode(img_data))
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    main()
