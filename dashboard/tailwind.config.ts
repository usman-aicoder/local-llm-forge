import type { Config } from "tailwindcss"

const config: Config = {
  // Enable class-based dark mode so we can toggle via JS
  darkMode: "class",

  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],

  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
      },

      // ── Theme-aware zinc scale ──────────────────────────────────────────
      // Each value points to a CSS variable that is INVERTED in light mode.
      // This means every existing component using bg-zinc-*/text-zinc-*/
      // border-zinc-* automatically adapts to both themes — zero component
      // changes required.
      colors: {
        zinc: {
          50:  "rgb(var(--zinc-50)  / <alpha-value>)",
          100: "rgb(var(--zinc-100) / <alpha-value>)",
          200: "rgb(var(--zinc-200) / <alpha-value>)",
          300: "rgb(var(--zinc-300) / <alpha-value>)",
          400: "rgb(var(--zinc-400) / <alpha-value>)",
          500: "rgb(var(--zinc-500) / <alpha-value>)",
          600: "rgb(var(--zinc-600) / <alpha-value>)",
          700: "rgb(var(--zinc-700) / <alpha-value>)",
          800: "rgb(var(--zinc-800) / <alpha-value>)",
          900: "rgb(var(--zinc-900) / <alpha-value>)",
          950: "rgb(var(--zinc-950) / <alpha-value>)",
        },
        // white / black also invert so bg-white text-black buttons work in both modes
        white: "rgb(var(--color-white) / <alpha-value>)",
        black: "rgb(var(--color-black) / <alpha-value>)",
      },

      boxShadow: {
        card: "0 1px 3px rgb(0 0 0 / 0.25), 0 1px 2px -1px rgb(0 0 0 / 0.25)",
        "card-md": "0 4px 12px rgb(0 0 0 / 0.20), 0 2px 4px -2px rgb(0 0 0 / 0.20)",
        glow: "0 0 12px rgb(var(--accent) / 0.3)",
      },
    },
  },

  plugins: [],
}

export default config
