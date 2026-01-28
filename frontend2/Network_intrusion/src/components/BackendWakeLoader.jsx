import { useEffect, useState } from "react";

const BackendWakeLoader = ({ onReady }) => {
  const [status, setStatus] = useState("Booting Intrusion Detection Engine...");
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    const messages = [
      "Booting Intrusion Detection Engine...",
      "Loading Machine Learning Models...",
      "Initializing Secure Backend Services...",
      "Establishing Encrypted Channels...",
      "Calibrating Detection Parameters...",
    ];

    let msgIndex = 0;
    const textInterval = setInterval(() => {
      msgIndex = (msgIndex + 1) % messages.length;
      setStatus(messages[msgIndex]);
      setPulse((p) => !p);
    }, 1800);

    const wakeBackend = async () => {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/`);
        if (res.ok) {
          clearInterval(textInterval);
          setStatus("Backend Online. System Ready.");
          setTimeout(onReady, 900);
        }
      } catch {
        setTimeout(wakeBackend, 2500);
      }
    };

    wakeBackend();

    return () => clearInterval(textInterval);
  }, [onReady]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black overflow-hidden">
      
      {/* Background glow */}
      <div className="absolute inset-0 bg-gradient-to-b from-red-900/30 via-black to-black" />

      {/* Ritual Rings */}
      <div
        className={`relative w-90 h-90 rounded-full border border-red-600/40 
        ${pulse ? "animate-pulse" : ""}`}
      >
        <div className="absolute inset-4 rounded-full border border-orange-500/40 animate-spin-slow" />
        <div className="absolute inset-8 rounded-full border border-yellow-500/30 animate-spin-reverse" />
        <div className="absolute inset-4 rounded-full border border-orange-500/40 animate-spin-slow" />
        <div className="absolute inset-8 rounded-full border border-yellow-500/30 animate-spin-reverse" />
      </div>

      {/* Core */}
      <div className="absolute flex flex-col items-center text-center px-6">
        <div className="text-6xl mb-4 animate-flame">🐉</div>

        <h1 className="text-2xl font-bold text-red-500 tracking-widest">
          INTRUSION ENGINE
        </h1>

        <p className="mt-1 text-xs uppercase tracking-wider text-gray-400">
          Backend services are waking up
        </p>

        <p className="mt-4 text-sm text-orange-400 tracking-wide">
          {status}
        </p>

        <p className="mt-6 text-xs text-gray-500">
          First request may take a few seconds on free hosting
        </p>
      </div>

      {/* Embers */}
      <div className="absolute bottom-0 w-full h-32 bg-gradient-to-t from-red-900/30 to-transparent animate-embers" />
    </div>
  );
};

export default BackendWakeLoader;
