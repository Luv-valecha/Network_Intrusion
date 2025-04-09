import { useState, useEffect } from 'react';
import ResultCard from './ResultCard';
import History from './History';

const PacketForm = () => {
  const [formData, setFormData] = useState({
    service: '',
    flag: '',
    src_bytes: '',
    dst_bytes: '',
    same_srv_rate: '',
    diff_srv_rate: '',
    dst_host_srv_count: '',
    dst_host_same_srv_rate: '',
    dst_host_diff_srv_rate: '',
    dst_host_serror_rate: ''
  });

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(() => {
    const stored = localStorage.getItem('packetHistory');
    return stored ? JSON.parse(stored) : [];
  });

  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    localStorage.setItem('packetHistory', JSON.stringify(history));
  }, [history]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSubmitted(false);

    try {
      const response = await fetch('https://net-intrusion-358654395984.asia-south1.run.app/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await response.json();
      const prediction = data.prediction[0];
      setResult(prediction);

      const newEntry = {
        id: Date.now(),
        prediction,
        timestamp: new Date().toLocaleString(),
        input: { ...formData }
      };

      setHistory(prev => [newEntry, ...prev]);
      setSubmitted(true);
      setFormData({
        service: '',
        flag: '',
        src_bytes: '',
        dst_bytes: '',
        same_srv_rate: '',
        diff_srv_rate: '',
        dst_host_srv_count: '',
        dst_host_same_srv_rate: '',
        dst_host_diff_srv_rate: '',
        dst_host_serror_rate: ''
      });
    } catch (err) {
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    localStorage.removeItem('packetHistory');
    setHistory([]);
  };

  const serviceOptions = [
    'IRC', 'X11', 'Z39_50', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard',
    'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data',
    'gopher', 'hostnames', 'http', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin',
    'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn',
    'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private',
    'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup',
    'systat', 'telnet', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois'
  ];

  const flagOptions = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'];

  return (
    <div className="p-6 text-white">
      <h2 className="text-2xl font-bold mb-6">Single Packet Prediction</h2>

      <div className="flex flex-col lg:flex-row gap-10">
        {/* Left Column - Form */}
        <form
          onSubmit={handleSubmit}
          className="space-y-6 max-w-xl flex-1 bg-gray-900 p-6 rounded-lg shadow-lg"
        >
          <div>
            <label className="block mb-1 font-semibold">Service</label>
            <select
              name="service"
              value={formData.service}
              onChange={handleChange}
              className="w-full bg-gray-800 p-2 rounded"
              required
            >
              <option value="" disabled>Select a service</option>
              {serviceOptions.map(service => (
                <option key={service} value={service}>{service}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block mb-1 font-semibold">Flag</label>
            <select
              name="flag"
              value={formData.flag}
              onChange={handleChange}
              className="w-full bg-gray-800 p-2 rounded"
              required
            >
              <option value="" disabled>Select a flag</option>
              {flagOptions.map(flag => (
                <option key={flag} value={flag}>{flag}</option>
              ))}
            </select>
          </div>

          <hr className="border-gray-700" />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { label: 'Source Bytes', name: 'src_bytes' },
              { label: 'Destination Bytes', name: 'dst_bytes' },
              { label: 'Same Service Rate', name: 'same_srv_rate', step: '0.01' },
              { label: 'Different Service Rate', name: 'diff_srv_rate', step: '0.01' },
              { label: 'Dst Host Service Count', name: 'dst_host_srv_count' },
              { label: 'Dst Host Same Service Rate', name: 'dst_host_same_srv_rate', step: '0.01' },
              { label: 'Dst Host Diff Service Rate', name: 'dst_host_diff_srv_rate', step: '0.01' },
              { label: 'Dst Host Serror Rate', name: 'dst_host_serror_rate', step: '0.01' }
            ].map(({ label, name, step }) => (
              <div key={name}>
                <label className="block mb-1 font-semibold">{label}</label>
                <input
                  type="number"
                  step={step || '1'}
                  name={name}
                  value={formData[name]}
                  onChange={handleChange}
                  placeholder={step ? '0.00' : 'e.g. 123'}
                  className="w-full bg-gray-800 p-2 rounded"
                  required
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded w-full font-semibold"
            disabled={loading}
          >
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {/* Right Column - Result + History */}
        <div className="flex-1 flex flex-col gap-6">
          {result !== null && (
            <div className="flex justify-center">
              <ResultCard
                prediction={result}
                confidence={result === 1 ? 94 : 99}
                suggestion={result === 1 ? 'Drop the packet' : 'Allow the packet'}
              />
            </div>
          )}
          <History history={history} onClear={clearHistory} />
        </div>
      </div>
    </div>
  );
};

export default PacketForm;
