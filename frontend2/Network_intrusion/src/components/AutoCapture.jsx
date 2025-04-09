import React, { useState, useEffect } from 'react';

function AutoCapture() {
  const [serverUrl, setServerUrl] = useState('http://localhost:5000');
  const [duration, setDuration] = useState(5);
  const [data, setData] = useState([]);
  const [visibleIndex, setVisibleIndex] = useState(null);
  const [openOS, setOpenOS] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState(() => {
    const stored = localStorage.getItem('packetHistory');
    return stored ? JSON.parse(stored) : [];
  });

  useEffect(() => {
    localStorage.setItem('packetHistory', JSON.stringify(history));
  }, [history]);

  const startCapture = async () => {
    if (!serverUrl.trim()) {
      alert("Please enter a valid server URL.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch(`${serverUrl}/capture?duration=${duration}`);
      const json = await res.json();

      if (!res.ok || !Array.isArray(json)) {
        alert("Error: " + (json.error || "Invalid response from server."));
        setIsLoading(false);
        return;
      }

      const predictedData = await Promise.all(
        json.map(async (packet) => {
          try {
            const response = await fetch('https://net-intrusion-358654395984.asia-south1.run.app/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(packet),
            });

            const data = await response.json();
            const prediction = data.prediction?.[0] ?? 'Unknown';

            return {
              ...packet,
              label: prediction === 1 || prediction === 'Malicious' ? 'Malicious' : 'Benign',
              prediction,
              timestamp: new Date().toLocaleString(),
            };
          } catch (err) {
            return {
              ...packet,
              label: 'Prediction Failed',
              prediction: null,
              timestamp: new Date().toLocaleString(),
            };
          }
        })
      );

      setData(predictedData);

      // Update history
      const newHistoryEntries = predictedData.map(packet => ({
        id: Date.now() + Math.random(),
        prediction: packet.prediction,
        timestamp: packet.timestamp,
        input: { ...packet }
      }));

      setHistory(prev => [...newHistoryEntries, ...prev]);
    } catch (error) {
      alert("Failed to connect to server. Please check the URL.");
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDetails = (index) => {
    setVisibleIndex(prev => (prev === index ? null : index));
  };

  const toggleOS = (os) => {
    setOpenOS(prev => (prev === os ? null : os));
  };

  return (
    <div className="p-6 text-white">
      <h1 className="text-3xl font-bold mb-6 text-center">🔍 AutoCapture - Live Packet Analyzer</h1>

      {/* Input Section */}
      <div className="mb-6 space-y-4 sm:flex sm:items-center sm:space-x-4">
        <label className="flex flex-col text-lg">
          Server URL:
          <input
            type="text"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            placeholder="http://localhost:5000"
            className="mt-1 border border-gray-500 rounded px-3 py-2 w-full sm:w-96 bg-gray-900"
          />
        </label>

        <label className="flex items-center space-x-2 text-lg mt-5">
          <span>Duration:</span>
          <input
            type="number"
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            min={1}
            className="border border-gray-500 rounded px-2 py-1 w-20 bg-gray-900"
          />
        </label>

        <button
          onClick={startCapture}
          className="bg-blue-600 text-white px-5 py-2 rounded hover:bg-blue-700 transition disabled:opacity-50"
          disabled={isLoading}
        >
          {isLoading ? 'Capturing...' : '🚀 Start Capture'}
        </button>
      </div>

      {/* Results */}
      <div className="border-t border-gray-600 pt-4">
        {data.map((item, index) => (
          <div
            key={index}
            className={`border rounded p-4 mb-4 ${
              item.label === 'Malicious'
                ? 'bg-red-900'
                : item.label === 'Benign'
                ? 'bg-green-900'
                : 'bg-gray-800'
            }`}
          >
            <div className="flex justify-between items-center">
              <strong
                className={`font-bold ${
                  item.label === 'Malicious'
                    ? 'text-red-500'
                    : item.label === 'Benign'
                    ? 'text-green-500'
                    : 'text-gray-400'
                }`}
              >
                {item.label === 'Malicious'
                  ? '🔴 Malicious'
                  : item.label === 'Benign'
                  ? '🟢 Benign'
                  : '⚠ Unknown'}
              </strong>
              <span className="text-sm text-gray-400">{item.timestamp}</span>
              <button
                onClick={() => toggleDetails(index)}
                className="text-blue-400 hover:underline"
              >
                {visibleIndex === index ? 'Hide Details' : 'View Details'}
              </button>
            </div>

            {visibleIndex === index && (
              <div className="mt-4 space-y-2">
                {Object.entries(item).map(([key, value]) =>
                  key === 'timestamp' || key === 'label' || key === 'prediction' ? null : (
                    <div key={key} className="text-sm text-gray-300">
                      <strong>{key}:</strong> {value?.toString()}
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Setup Instructions */}
      <div className="mt-10">
        <h2 className="text-2xl font-semibold text-center mb-4">📦 Setup Instructions</h2>
        <p className="mb-4 text-lg">1. Download the required files:</p>
        <div className="flex flex-col sm:flex-row sm:space-x-4 space-y-2 sm:space-y-0 mb-6">
          <a href="/packetcapture.py" download className="bg-gray-700 px-4 py-2 rounded hover:underline">
            Download packetcapture.py
          </a>
          <a href="/requirements.txt" download className="bg-gray-700 px-4 py-2 rounded hover:underline">
            Download requirements.txt
          </a>
        </div>

        {['Windows', 'macOS', 'Linux'].map(os => (
          <div key={os} className="mb-4">
            <button
              onClick={() => toggleOS(os)}
              className="w-full text-left bg-gray-800 px-4 py-2 rounded hover:bg-gray-700 font-semibold text-lg"
            >
              {openOS === os ? '▼' : '▶'} {os} Setup Instructions
            </button>
            {openOS === os && (
              <div className="bg-gray-900 border border-gray-600 rounded mt-2 p-4 space-y-2">
                {os === 'Windows' && (
                  <>
                    <p>2. Run CMD as Administrator and navigate to the folder where files are downloaded.</p>
                    <p>3. Install dependencies: <code>pip install -r requirements.txt</code></p>
                    <p>4. Download and install Npcap from <a href="https://nmap.org/npcap" className="underline text-blue-400">Official Site</a></p>
                    <p>5. Ensure "WinPcap API-compatible Mode" is selected during installation.</p>
                    <p>6. Start the server: <code>python server.py</code></p>
                  </>
                )}
                {os === 'macOS' && (
                  <>
                    <p>2. Open Terminal and navigate to the downloaded folder.</p>
                    <p>3. Install dependencies: <code>pip3 install -r requirements.txt</code></p>
                    <p>4. Start the server with root permissions: <code>sudo python3 server.py</code></p>
                    <p>5. Optional: Grant Terminal full disk/network access via System Settings → Privacy & Security.</p>
                  </>
                )}
                {os === 'Linux' && (
                  <>
                    <p>2. Open Terminal and go to the download directory.</p>
                    <p>3. Install Python & pip: <code>sudo apt update && sudo apt install python3 python3-pip -y</code></p>
                    <p>4. Install requirements: <code>pip3 install -r requirements.txt</code></p>
                    <p>5. Run the server with root: <code>sudo python3 server.py</code></p>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
        <p className='text-xl font-bold'>After the server is running, enter your local server's URL in the field above</p>
      </div>
    </div>
  );
}

export default AutoCapture;