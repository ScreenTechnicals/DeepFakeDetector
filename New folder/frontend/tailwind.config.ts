import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: { "2xl": "1400px" },
    },
    extend: {
      fontFamily: {
        display: ['Syne', 'sans-serif'],
        body: ['DM Sans', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      fontSize: {
        'label':      ['11px', { lineHeight: '16px' }],
        'caption':    ['13px', { lineHeight: '20px' }],
        'body-sm':    ['15px', { lineHeight: '24px' }],
        'subheading': ['18px', { lineHeight: '28px' }],
        'section':    ['24px', { lineHeight: '32px' }],
        'page':       ['36px', { lineHeight: '44px' }],
        'hero':       ['56px', { lineHeight: '64px' }],
        'display':    ['80px', { lineHeight: '88px' }],
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
        destructive: { DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
        'df-surface':        'hsl(var(--bg-surface))',
        'df-elevated':       'hsl(var(--bg-elevated))',
        'df-border':         'hsl(var(--bg-border))',
        'df-text':           'hsl(var(--text-primary))',
        'df-text-secondary': 'hsl(var(--text-secondary))',
        'df-text-tertiary':  'hsl(var(--text-tertiary))',
        'df-cyan':           'hsl(var(--accent-cyan))',
        'df-cyan-dim':       'hsl(var(--accent-cyan-dim))',
        'df-blue':           'hsl(var(--accent-blue))',
        'df-red':            'hsl(var(--risk-red))',
        'df-red-dim':        'hsl(var(--risk-red-dim))',
        'df-amber':          'hsl(var(--risk-amber))',
        'df-amber-dim':      'hsl(var(--risk-amber-dim))',
        'df-green':          'hsl(var(--safe-green))',
        'df-green-dim':      'hsl(var(--safe-green-dim))',
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-in": "fade-in 0.3s ease forwards",
        shimmer: "shimmer 2s infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;
