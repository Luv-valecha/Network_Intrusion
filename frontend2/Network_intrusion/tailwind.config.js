export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      animation: {
        "spin-slow": "spin 14s linear infinite",
        "spin-reverse": "spin-reverse 10s linear infinite",
        embers: "embers 3s ease-in-out infinite",
        flame: "flame 1.6s ease-in-out infinite",
      },
      keyframes: {
        "spin-reverse": {
          to: { transform: "rotate(-360deg)" },
        },
        embers: {
          "0%, 100%": { opacity: "0.2" },
          "50%": { opacity: "0.6" },
        },
        flame: {
          "0%, 100%": { transform: "scale(1)", opacity: 0.9 },
          "50%": { transform: "scale(1.2)", opacity: 1 },
        },
      },
    },
  },
  plugins: [],
};