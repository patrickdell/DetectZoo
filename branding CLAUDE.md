# Globe and Mail — Editorial Projects

This file is read automatically by Claude Code for all projects in this directory and subdirectories. It defines the shared design system for internal tools and editorial projects built for The Globe and Mail.

---

## Design system

### Fonts

Always load from the Globe CDN. These are the three licensed typefaces:

```css
@font-face {
  font-family: 'GM Sans';
  src: url('https://www.theglobeandmail.com/files/dev/www/cache-long/fonts/GMsans-Web-Regular.woff2?v=3') format('woff2');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'GM Sans Label';
  src: url('https://www.theglobeandmail.com/files/dev/www/cache-long/fonts/GMsans-Web-Label.woff2?v=3') format('woff2');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Pratt';
  src: url('https://www.theglobeandmail.com/files/dev/www/cache-long/fonts/Pratt-Nova.woff2?v=3') format('woff2');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Pratt';
  src: url('https://www.theglobeandmail.com/files/dev/www/cache-long/fonts/Pratt-Bold.woff2?v=3') format('woff2');
  font-weight: 700; font-style: normal; font-display: swap;
}
```

| Variable | Stack | Use |
|---|---|---|
| `--serif` | Pratt, Georgia, Times New Roman, serif | Headlines, section titles, mastheads |
| `--sans` | GM Sans, Helvetica Neue, Arial, sans-serif | Body text, descriptions, prose |
| `--label` | GM Sans Label, GM Sans, Helvetica Neue, Arial, sans-serif | UI labels, tags, nav, buttons, eyebrows — always uppercase + letter-spaced |

### Colour palette

```css
:root {
  --black:          #000000;
  --near-black:     #1a1a1a;   /* body text, primary UI */
  --dark-grey:      #333333;
  --mid-grey:       #595959;   /* secondary text */
  --light-grey:     #767676;   /* tertiary text, labels */
  --rule-grey:      #d4d4d4;   /* dividers, card borders */
  --bg-grey:        #f2f2f2;   /* page backgrounds, hover states */
  --white:          #ffffff;

  --globe-red:      #b21000;   /* primary accent — CTAs, active states, highlights */
  --globe-red-dark: #8a0d00;   /* hover state on globe-red elements */

  --serif: 'Pratt', Georgia, 'Times New Roman', serif;
  --sans:  'GM Sans', 'Helvetica Neue', Arial, sans-serif;
  --label: 'GM Sans Label', 'GM Sans', 'Helvetica Neue', Arial, sans-serif;
}
```

Derived tints used in components:
- Globe-red callout bg: `#faf5f5`
- Globe-red nav active bg: `#fdf5f5`
- Secondary/neutral section bg: `#f5f7fa`
- Card bg: `#fafafa`

---

## Accessibility

All elements must meet **WCAG 2.1 AAA** standards. Key requirements:

- **Contrast — text:** minimum 7:1 for normal text, 4.5:1 for large text (18pt+ or 14pt+ bold)
- **Contrast — UI components and focus indicators:** minimum 3:1 against adjacent colours
- **Focus visible:** always present, never suppressed — use `outline: 3px solid var(--globe-red); outline-offset: 2px`
- **Keyboard navigation:** all interactive elements reachable and operable by keyboard alone
- **Touch targets:** minimum 44×44px
- **Motion:** wrap animations in `@media (prefers-reduced-motion: no-preference)` — default to no animation
- **Semantic HTML:** use correct elements (`<nav>`, `<main>`, `<header>`, `<button>`, `<section>`, etc.) — never use divs for interactive elements
- **ARIA:** only add ARIA attributes where native semantics are insufficient; always include `aria-label` on icon-only buttons
- **Colour alone:** never use colour as the only way to convey information

When in doubt, test against the Globe palette: `--near-black` (#1a1a1a) on `--white` (#ffffff) gives ~16:1. `--globe-red` (#b21000) on white gives ~5.9:1 — sufficient for large/bold text at AAA, but use `--near-black` for body text.

---

## Aesthetic

- **Newspaper/editorial** — flat design, no drop shadows, structure through rule lines and borders
- **Typography-driven hierarchy** — weight, size, tracking, and case do the work, not colour fills
- Label text: always `text-transform: uppercase`, `letter-spacing: 0.10–0.15em`, small (`0.69–0.8rem`)
- Accent via left-border (`3px solid var(--globe-red)`) rather than background blocks
- Body text: `font-size: 0.9375rem`, `line-height: 1.55–1.6`
- Always set `-webkit-font-smoothing: antialiased` on body
- Square corners throughout — no border-radius

---

## Layout conventions

- Page wrapper max-width: **1100px**, centred, with `1px solid var(--rule-grey)` left/right borders
- Masthead: `border-bottom: 4px solid var(--black)` + a 1px rule 3px below (double-line effect via `::after`)
- Sidebar width: **196px**, sticky, full viewport height
- Content area padding: `0 2.5rem 6rem`
- Mobile breakpoint: **680px** — sidebar goes horizontal/scrollable, grid collapses to single column

---

## UI components

### Buttons
- Primary: `background: var(--globe-red)`, white text, label font, uppercase, padding `0.32rem 0.85rem`
- Hover: `var(--globe-red-dark)`
- Success/copied state: `#2e7d32`
- No border-radius

### Cards
- Border: `1px solid var(--rule-grey)`, background `#fafafa`
- Header: white bg, flex space-between, padding `0.6rem 1rem`, `border-bottom: 1px solid var(--rule-grey)`
- Label column: 88px wide, label font, uppercase, globe-red

### Callouts
- Globe-red accent: `border-left: 3px solid var(--globe-red)`, bg `#faf5f5`, eyebrow in globe-red
- Neutral/secondary: `border-left: 3px solid var(--near-black)`, bg `#f5f7fa`

### Tags / pills
- Default: `border: 1px solid var(--rule-grey)`, bg `var(--bg-grey)`, label font, no border-radius
- Dark variant: `border: 1px solid var(--near-black)`, bg `var(--near-black)`, white text

### Sidebar navigation
- Active: `border-left: 3px solid var(--globe-red)`, text `var(--globe-red)`, bg `#fdf5f5`
- Hover: bg `var(--bg-grey)`, text `var(--near-black)`
- Font: label, uppercase, `0.775rem`, `letter-spacing: 0.04em`

### Focus
- `outline: 3px solid var(--globe-red); outline-offset: 2px`

---

## Project-specific tokens

Not universal — only use within the project that defines them:
- `--gold: #c9a84c` and `--masthead-bg: #0f1923` — USMCA monitor dark masthead

---

## Source files

- `USMCA monitor/frontend/src/globe.css` — canonical CSS (use as starter for new projects)
- `_Newsroom training/.../ai-prompt-reference-reporters-2026-04-28.html` — reference implementation
