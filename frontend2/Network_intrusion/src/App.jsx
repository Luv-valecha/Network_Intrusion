import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useEffect, useState } from 'react';

import Sidebar from './components/Sidebar';
import PacketForm from './components/PacketForm';
import Upload from './components/Upload';
import HistoryPage from './components/HistoryPage';
import About from './components/About';
import AutoCapture from './components/AutoCapture';
import BackendWakeLoader from './components/BackendWakeLoader';

function App() {
  const [backendReady, setBackendReady] = useState(false);

  return (
    <>
      {!backendReady && (
        <BackendWakeLoader onReady={() => setBackendReady(true)} />
      )}

      {backendReady && (
        <Router>
          <div className="flex h-screen bg-gray-950 text-white">
            <Sidebar />
            <div className="flex-1 overflow-y-auto">
              <Routes>
                <Route path="/" element={<PacketForm />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/autocapture" element={<AutoCapture />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/about" element={<About />} />
              </Routes>
            </div>
          </div>
        </Router>
      )}
    </>
  );
}

export default App;
