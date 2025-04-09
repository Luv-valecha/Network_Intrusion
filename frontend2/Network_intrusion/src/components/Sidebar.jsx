import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { FaGithub, FaBars, FaTimes } from 'react-icons/fa';

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(true);

  const toggleSidebar = () => setIsOpen(!isOpen);

  const linkStyle =
    'block px-4 py-2 hover:bg-gray-800 rounded transition text-sm font-medium';

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div
        className={`bg-gray-900 text-white border-r border-gray-800 transition-all duration-300 ease-in-out
        ${isOpen ? 'w-60 p-4' : 'w-16 p-2 text-center pt-4'} flex flex-col justify-between`}
      >
        {/* Top: Title + Nav */}
        <div>
          <div className="mb-6">
            <button
              onClick={toggleSidebar}
              className="text-white hover:text-gray-300 focus:outline-none mb-2"
            >
              {isOpen ? <FaTimes size={20} /> : <FaBars size={20} />}
            </button>
            {isOpen && (
              <h1 className="text-xl font-bold">
                🧠 Network Intrusion
              </h1>
            )}
          </div>
          <nav className="space-y-2">
            <NavLink to="/" className={linkStyle} title="Single Packet">
              {isOpen ? 'Single Packet' : '📦'}
            </NavLink>
            <NavLink to="/upload" className={linkStyle} title="Upload CSV">
              {isOpen ? 'Upload CSV' : '📁'}
            </NavLink>
            <NavLink to="/autocapture" className={linkStyle} title="Auto Capture">
              {isOpen ? 'Auto Capture' : '📡'}
            </NavLink>
            <NavLink to="/history" className={linkStyle} title="History">
              {isOpen ? 'History' : '📜'}
            </NavLink>
            <NavLink to="/about" className={linkStyle} title="About">
              {isOpen ? 'About' : 'ℹ️'}
            </NavLink>
          </nav>
        </div>

        {/* Bottom: GitHub */}
        <div className="mt-6">
          <a
            href="https://github.com/Luv-valecha/Network_intrusion"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-2 bg-gray-800 text-white rounded hover:bg-gray-700 transition text-sm"
          >
            <FaGithub size={16} />
            {isOpen && <span>Visit GitHub</span>}
          </a>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
