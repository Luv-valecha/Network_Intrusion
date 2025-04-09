import { useRef, useState } from 'react';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef();

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!file) {
      alert('📂 Please select a CSV file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('https://net-intrusion-358654395984.asia-south1.run.app/predict_csv', {
        method: 'POST',
        body: formData
      });

      const contentType = response.headers.get('content-type');
      if (!response.ok || !contentType.includes('application/json')) {
        const errorText = await response.text();
        throw new Error(`Unexpected response: ${errorText}`);
      }

      const data = await response.json();
      setResult(data);

      const newEntry = {
        id: Date.now(),
        type: 'csv',
        timestamp: new Date().toLocaleString(),
        predictions: data.map(item => item.prediction),
        input: data.map(item => item.input)
      };

      const prev = localStorage.getItem('packetHistory');
      const history = prev ? JSON.parse(prev) : [];
      const updated = [newEntry, ...history];
      localStorage.setItem('packetHistory', JSON.stringify(updated));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 text-white">
      <h2 className="text-2xl font-bold mb-6">📤 Upload CSV for Batch Prediction</h2>

      <div className="space-y-4 bg-gray-900 p-6 rounded-lg shadow-lg max-w-2xl mx-auto">
        <div className="flex items-center justify-between">
          <input
            type="file"
            accept=".csv"
            id="file_input"
            onChange={handleFileSelect}
            ref={fileInputRef}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current.click()}
            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-md text-sm font-medium"
          >
            Select CSV File
          </button>

          <div className="text-sm text-gray-300 ml-4">
            {file ? (
              <span className="text-green-400">📄 {file.name}</span>
            ) : (
              <span>No file selected</span>
            )}
          </div>
        </div>

        <button
          onClick={handlePredict}
          className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-md font-semibold w-full transition"
          disabled={loading}
        >
          {loading ? '🔍 Analyzing...' : '🚀 Predict Packets'}
        </button>

        {error && (
          <div className="text-red-400 bg-red-900/40 px-4 py-2 rounded">
            ❌ Error: {error}
          </div>
        )}

        {result && result.length > 0 ? (
          <div className="mt-6 bg-gray-800 p-4 rounded animate-fade-in">
            <h3 className="text-lg font-semibold mb-2">🧾 Prediction Results</h3>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              {result.map((row, idx) => (
                <li key={idx}>
                  Row {idx + 1}:{' '}
                  {row.prediction === 1 ? (
                    <span className="text-red-400 font-medium">🚨 Malicious Packet</span>
                  ) : (
                    <span className="text-green-400 font-medium">✅ Safe Packet</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          result && (
            <div className="mt-4 text-gray-400 text-sm italic">
              No results returned from server.
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default Upload;
