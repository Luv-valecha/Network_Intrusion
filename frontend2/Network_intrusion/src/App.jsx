import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import PacketForm from './components/PacketForm';
import Upload from './components/Upload';
import HistoryPage from './components/HistoryPage';
import About from './components/About';

function App() {
  return (
    <Router>
      <div className="flex h-screen bg-gray-950 text-white">
        <Sidebar />
        <div className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<PacketForm />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
