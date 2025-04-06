import { useEffect, useState } from 'react';

const History = ({ history: propHistory = [], onClear }) => {
  const isStandalone = false; // <-- Set to true if used standalone (like on its own page)
  const [localHistory, setLocalHistory] = useState([]);
  const [selected, setSelected] = useState(null);
  const [showInput, setShowInput] = useState(false);

  const history = isStandalone ? localHistory : propHistory;

  useEffect(() => {
    if (isStandalone) {
      const stored = localStorage.getItem('packetHistory');
      if (stored) {
        const parsed = JSON.parse(stored);
        setLocalHistory(parsed);
      }
    }
  }, [isStandalone]);

  const handleClear = () => {
    localStorage.removeItem('packetHistory');
    if (onClear) onClear();
    if (isStandalone) setLocalHistory([]);
  };

  return (
    <div className="p-6 text-white">
      {isStandalone && <h2 className="text-2xl font-bold mb-6">Prediction History</h2>}

      <div className="bg-gray-900 p-4 rounded-lg shadow-lg max-h-[500px] overflow-y-auto max-w-2xl mx-auto">
        <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2">
          <h3 className="text-xl font-semibold">History</h3>
          {history.length > 0 && (
            <button
              onClick={handleClear}
              className="text-sm text-red-400 hover:text-red-300 underline"
            >
              Clear History
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-gray-400 text-sm">No history yet.</p>
        ) : (
          <ul className="space-y-3">
            {history.map((entry) => (
              <li
                key={entry.id}
                className="p-3 rounded-lg flex justify-between items-center bg-gray-800/50"
              >
                <span
                  className={`font-bold ${entry.type === 'csv'
                    ? 'text-yellow-400'
                    : entry.prediction === 1
                      ? 'text-red-400'
                      : 'text-green-400'
                    }`}
                >
                  {entry.type === 'csv'
                    ? 'CSV file predicted'
                    : entry.prediction === 1
                      ? 'Malicious'
                      : 'Benign'}
                </span>

                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-400">{entry.timestamp}</span>
                  <button
                    onClick={() => {
                      setSelected(entry);
                      setShowInput(false);
                    }}
                    className="text-blue-400 hover:underline text-sm"
                  >
                    View Details
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Detail Modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-xl shadow-2xl w-full max-w-4xl text-white max-h-[80vh] overflow-y-auto">
            <h4 className="text-xl font-bold mb-4">
              {selected.type === 'csv' ? 'CSV Prediction Details' : 'Packet Details'}
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
                  <div className="space-y-2">
                    {selected.input.map((row, index) => (
                      <div
                        key={index}
                        className="bg-gray-700 p-3 rounded border border-gray-600"
                      >
                        <p className="text-sm text-gray-300 font-semibold mb-1">Row {index + 1}</p>
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
                  <div className="space-y-2">
                    {selected.predictions?.map((pred, index) => (
                      <div
                        key={index}
                        className={`p-3 rounded flex justify-between items-center ${pred === 1 ? 'bg-red-800/30' : 'bg-green-800/30'
                          }`}
                      >
                        <span className="text-sm font-semibold">
                          Row {index + 1}
                        </span>
                        <span
                          className={`font-bold ${pred === 1 ? 'text-red-400' : 'text-green-400'
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
              <div>
                <p className="text-gray-400 mb-2">Input Features:</p>
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {selected.input &&
                    Object.entries(selected.input).map(([key, value]) => (
                      <li
                        key={key}
                        className="flex justify-between border-b border-gray-700 pb-1"
                      >
                        <span className="text-gray-300 pr-4">{key}</span>
                        <span className="text-white">{value}</span>
                      </li>
                    ))}
                </ul>
              </div>
            )}

            <button
              onClick={() => setSelected(null)}
              className="mt-6 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded w-full"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;
