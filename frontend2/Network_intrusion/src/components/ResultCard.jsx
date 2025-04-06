const ResultCard = ({ prediction, confidence, suggestion }) => {
  const isMalicious = prediction === 1;

  return (
    <div className="bg-gray-900 text-white rounded-2xl p-8 shadow-2xl w-full max-w-md">
      <h3 className="text-2xl font-bold mb-6">Prediction Result</h3>

      <div className="flex items-center gap-4 mb-4">
        <span className={`text-4xl ${isMalicious ? 'text-red-500' : 'text-green-500'}`}>
          {isMalicious ? '⚠️' : '✅'}
        </span>
        <span className={`text-3xl font-bold ${isMalicious ? 'text-red-500' : 'text-green-500'}`}>
          {isMalicious ? 'Malicious' : 'Benign'}
        </span>
      </div>

      {/* <p className="text-lg text-gray-300 mb-2">
        <span className="font-semibold">Confidence:</span> {confidence ?? 'N/A'}%
      </p> */}

      <p className="text-lg text-gray-300 mb-6">
        <span className="font-semibold">Suggestion:</span>{' '}
        {suggestion ?? (isMalicious ? 'Drop the packet' : 'Allow the packet')}
      </p>

      {/* Optional: Button or link */}
      {/* <a href="#" className="text-blue-400 text-sm underline hover:text-blue-300">
        View Details
      </a> */}
    </div>
  );
};

export default ResultCard;
