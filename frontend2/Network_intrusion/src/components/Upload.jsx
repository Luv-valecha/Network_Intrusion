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
      alert('Please select a CSV file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);
    try {
      const response = await fetch('https://net-intrusion-358654395984.asia-south1.run.app/predict_csv', {
        method: 'POST',
        body: formData
      });

      const contentType = response.headers.get("content-type");
      if (!response.ok || !contentType.includes("application/json")) {
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
    <div className="p-6">
      <h2 className="text-2xl font-semibold mb-4 text-white">Upload CSV File</h2>

      <div className="mb-4">
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
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium rounded-lg transition duration-150 ease-in-out"
        >
          Select CSV File
        </button>

        <div className="mt-2 text-sm text-gray-300">
          {file ? (
            <span className="text-green-400">📄 {file.name}</span>
          ) : (
            'No file selected'
          )}
        </div>
      </div>

      <button
        onClick={handlePredict}
        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white disabled:opacity-50"
        disabled={loading}
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {error && (
        <div className="mt-4 text-red-500">
          Error: {error}
        </div>
      )}

      {result && (
        <div className="mt-6 bg-gray-800 p-4 rounded text-white">
          <h3 className="text-lg font-bold mb-2">Results</h3>
          <ul className="list-disc pl-5">
            {result.map((row, idx) => (
              <li key={idx}>
                Row {idx + 1}: {row.prediction === 1 ? '🚨 Malicious Packet' : '✅ Safe Packet'}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Upload;
