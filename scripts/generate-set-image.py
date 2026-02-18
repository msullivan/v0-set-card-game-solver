#!/usr/bin/env python3
"""Generate photorealistic Set card images using Gemini Pro via Vercel AI Gateway."""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

PROMPT = """Here is a reference photo of real Set game cards. Study the three shapes carefully:

1. DIAMOND: a four-sided rhombus shape, oriented horizontally (wider than tall)
2. OVAL: a rounded rectangle / stadium shape, oriented horizontally
3. SQUIGGLE: a fat blobby bean/slug shape with smooth organic curves — NOT an S or a 2. Look at the squiggle cards in the reference for examples.

All three shadings exist: solid (filled in), striped (horizontal lines through the shape), and empty (outline only).

Generate a new photorealistic overhead photograph of 12 Set cards arranged in a neat 4 columns x 3 rows grid on a wooden table. ALL cards must be in portrait orientation (taller than wide), with shapes stacked vertically on each card. Include a variety of colors (red, green, purple), all three shapes (diamond, oval, squiggle as shown in reference), all three shadings, and different numbers (1, 2, or 3 per card). The squiggle must closely match the blobby organic shape from the reference photo."""


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
    parser.add_argument("--prompt", default=PROMPT, help="Custom prompt (default: built-in Set card prompt)")
    args = parser.parse_args()

    api_key = load_api_key()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: {ref_path} not found", file=sys.stderr)
        sys.exit(1)

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
                    {"type": "text", "text": args.prompt},
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
