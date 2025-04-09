import { useEffect, useState } from 'react';

const History = ({ history: propHistory = [], onClear }) => {
  const [history, setHistory] = useState(propHistory);
  const [selected, setSelected] = useState(null);
  const [showInput, setShowInput] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('packetHistory');
    if (stored) {
      setHistory(JSON.parse(stored));
    }
  }, []);

  useEffect(() => {
    document.body.style.overflow = selected ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [selected]);

  const handleClear = () => {
    localStorage.removeItem('packetHistory');
    setHistory([]);
    if (onClear) onClear();
  };

  return (
    <div className="p-6 text-white">
      <h2 className="text-3xl font-bold mb-6 text-center">Prediction History</h2>

      <div className="bg-gray-900 p-6 rounded-xl shadow-lg max-w-3xl mx-auto max-h-[500px] overflow-y-auto">
        <div className="flex justify-between items-center border-b border-gray-700 pb-4 mb-4">
          <h3 className="text-xl font-semibold">Recent Activity</h3>
          {history.length > 0 && (
            <button
              onClick={handleClear}
              className="text-sm text-red-400 hover:text-red-300 underline"
            >
              Clear All
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-gray-400 text-sm text-center">No history found. Start by making a prediction.</p>
        ) : (
          <ul className="space-y-4">
            {history.map((entry) => (
              <li
                key={entry.id}
                className={`p-4 rounded-lg cursor-pointer hover:brightness-110 transition flex justify-between items-center ${
                  entry.type === 'csv'
                    ? 'bg-yellow-800/30'
                    : entry.prediction === 1
                    ? 'bg-red-800/40'
                    : 'bg-green-800/30'
                }`}
                onClick={() => {
                  setSelected(entry);
                  setShowInput(false);
                }}
              >
                <div>
                  <p
                    className={`text-lg font-semibold ${
                      entry.type === 'csv'
                        ? 'text-yellow-400'
                        : entry.prediction === 1
                        ? 'text-red-400'
                        : 'text-green-400'
                    }`}
                  >
                    {entry.type === 'csv'
                      ? 'CSV Predictions'
                      : entry.prediction === 1
                      ? 'Malicious Packet'
                      : 'Benign Packet'}
                  </p>
                  <p className="text-sm text-gray-400">{entry.timestamp}</p>
                </div>
                <span className="text-blue-400 text-sm hover:underline">View</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center px-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="bg-gray-800 p-6 rounded-xl shadow-2xl w-full max-w-3xl max-h-[85vh] overflow-y-auto text-white relative"
            onClick={(e) => e.stopPropagation()}
          >
            <h4 className="text-2xl font-bold mb-4">
              {selected.type === 'csv' ? 'CSV Prediction Details' : 'Packet Inspection'}
            </h4>

            {selected.type === 'csv' ? (
              <>
                <div className="flex justify-end mb-4">
                  <button
                    className="text-sm text-blue-400 hover:underline"
                    onClick={() => setShowInput((prev) => !prev)}
                  >
                    {showInput ? 'Show Predictions' : ''}
                  </button>
                </div>

                {showInput ? (
                  <div className="space-y-3">
                    {selected.input?.map((row, index) => (
                      <div key={index} className="bg-gray-700 p-4 rounded border border-gray-600">
                        <p className="text-sm font-semibold text-gray-300 mb-2">Row {index + 1}</p>
                        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                          {Object.entries(row).map(([key, value]) => (
                            <li key={key} className="flex justify-between">
                              <span className="text-gray-300">{key}</span>
                              <span className="text-white">{value}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {selected.predictions?.map((pred, index) => (
                      <div
                        key={index}
                        className={`p-4 rounded flex justify-between items-center ${
                          pred === 1 ? 'bg-red-800/30' : 'bg-green-800/30'
                        }`}
                      >
                        <span className="text-sm font-medium">Row {index + 1}</span>
                        <span
                          className={`font-bold ${
                            pred === 1 ? 'text-red-400' : 'text-green-400'
                          }`}
                        >
                          {pred === 1 ? 'Malicious' : 'Benign'}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="space-y-4 text-sm">
                <div>
                  <p className="text-gray-400">Prediction:</p>
                  <p
                    className={`font-bold text-lg ${
                      selected.prediction === 1 ? 'text-red-400' : 'text-green-400'
                    }`}
                  >
                    {selected.prediction === 1 ? 'Malicious' : 'Benign'}
                  </p>
                </div>

                <div>
                  <p className="text-gray-400">Timestamp:</p>
                  <p>{selected.timestamp}</p>
                </div>

                <div>
                  <p className="text-gray-400 mb-2">Input Features:</p>
                  <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {selected.input &&
                      Object.entries(selected.input).map(([key, value]) => (
                        <li key={key} className="flex justify-between border-b border-gray-700 pb-1">
                          <span className="text-gray-300">{key}:</span>
                          <span className="text-white">{value}</span>
                        </li>
                      ))}
                  </ul>
                </div>
              </div>
            )}

            <button
              onClick={() => setSelected(null)}
              className="mt-6 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded w-full"
            >
              Close Details
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;
