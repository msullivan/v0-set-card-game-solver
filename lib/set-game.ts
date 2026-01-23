// Set card game types and logic

export type CardColor = "red" | "green" | "purple"
export type CardShape = "diamond" | "oval" | "squiggle"
export type CardShading = "solid" | "striped" | "empty"
export type CardNumber = 1 | 2 | 3

export interface SetCard {
  id: string
  color: CardColor
  shape: CardShape
  shading: CardShading
  number: CardNumber
  position?: { x: number; y: number } // Position in the image
}

export interface ValidSet {
  cards: [SetCard, SetCard, SetCard]
  reason: string
}

// Check if three cards form a valid Set
export function isValidSet(
  card1: SetCard,
  card2: SetCard,
  card3: SetCard
): boolean {
  const checkAttribute = (a: any, b: any, c: any): boolean => {
    const allSame = a === b && b === c;
    const allDifferent = a !== b && b !== c && a !== c;
    return allSame || allDifferent;
  };

  return (
    checkAttribute(card1.color, card2.color, card3.color) &&
    checkAttribute(card1.shape, card2.shape, card3.shape) &&
    checkAttribute(card1.shading, card2.shading, card3.shading) &&
    checkAttribute(card1.number, card2.number, card3.number)
  );
}

// Generate explanation for why cards form a set
export function getSetReason(
  card1: SetCard,
  card2: SetCard,
  card3: SetCard
): string {
  const parts: string[] = [];

  const describeAttribute = (name: string, a: any, b: any, c: any): string => {
    if (a === b && b === c) {
      return `all ${a} ${name}`;
    }
    return `different ${name} (${a}, ${b}, ${c})`;
  };

  parts.push(describeAttribute("color", card1.color, card2.color, card3.color));
  parts.push(describeAttribute("shape", card1.shape, card2.shape, card3.shape));
  parts.push(describeAttribute("shading", card1.shading, card2.shading, card3.shading));
  parts.push(describeAttribute("number", card1.number, card2.number, card3.number));

  return parts.join(", ");
}

// Find all valid sets from a list of cards
export function findAllSets(cards: SetCard[]): ValidSet[] {
  const validSets: ValidSet[] = [];

  for (let i = 0; i < cards.length - 2; i++) {
    for (let j = i + 1; j < cards.length - 1; j++) {
      for (let k = j + 1; k < cards.length; k++) {
        if (isValidSet(cards[i], cards[j], cards[k])) {
          validSets.push({
            cards: [cards[i], cards[j], cards[k]],
            reason: getSetReason(cards[i], cards[j], cards[k]),
          });
        }
      }
    }
  }

  return validSets;
}

// Format card for display
export function formatCard(card: SetCard): string {
  return `${card.number} ${card.color} ${card.shading} ${card.shape}${card.number > 1 ? "s" : ""}`;
}
