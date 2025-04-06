import { NavLink } from 'react-router-dom';
import { FaGithub } from 'react-icons/fa'; // install via `npm install react-icons`

const Sidebar = () => {
  const linkStyle =
    'block px-4 py-2 hover:bg-gray-800 rounded transition text-sm font-medium';

  return (
    <div className="w-60 bg-gray-900 p-4 border-r border-gray-800 flex flex-col justify-between h-full">
      <div>
        <h1 className="text-xl font-bold mb-6">🧠 Network Intrusion</h1>
        <nav className="space-y-2">
          <NavLink to="/" className={linkStyle}>
            Single Packet
          </NavLink>
          <NavLink to="/upload" className={linkStyle}>
            Upload CSV
          </NavLink>
          <NavLink to="/history" className={linkStyle}>
            History
          </NavLink>
          <NavLink to="/about" className={linkStyle}>
            About
          </NavLink>
        </nav>
      </div>

      {/* Custom GitHub button */}
      <div className="mt-6">
        <a
          href="https://github.com/Luv-valecha/Network_intrusion"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-3 py-2 bg-gray-800 text-white rounded hover:bg-gray-700 transition text-sm"
        >
          <FaGithub size={16} />
          Visit GitHub
        </a>
      </div>
    </div>
  );
};

export default Sidebar;
