import React from 'react';
import { NavLink } from 'react-router-dom';

function About() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      {/* Header Section */}
      <header className="text-center mb-12">
        <h1 className="text-5xl font-extrabold text-white mb-4">Network Intrusion Detection System (IDS)</h1>
        <p className="text-lg text-white max-w-2xl mx-auto">
          Our Network Intrusion Detection System (IDS) utilizes machine learning to detect and prevent cyber threats in real-time. Deployed on Google Cloud, it ensures high scalability and security for your network.
        </p>
      </header>
      <div className="w-full flex justify-end gap-4 mb-8 pr-4">
        <a
          href="https://luv-valecha.github.io/Network_Intrusion/"
          target="_blank"
          rel="noopener noreferrer"
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-xl transition duration-300 shadow-md"
        >
          Project Page
        </a>
        <a
          href="https://1drv.ms/b/c/9b83ad8f9c9cabcd/ERT6hCDqpmFFl1KAcR-fw1MBQldSltRD6SmwVu6B48JDXw?e=Az8iIp"
          target="_blank"
          rel="noopener noreferrer"
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-xl transition duration-300 shadow-md"
        >
          Project Report
        </a>
      </div>

      {/* Project Overview Section */}
      <section className="mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6">Project Overview</h2>
        <p className="text-lg text-white">
          This IDS project is designed to detect malicious network activities using various machine learning algorithms. The system classifies network traffic as either normal or anomalous, protecting against potential cyber attacks in real-time.
        </p>
      </section>

      {/* Key Features Section */}
      <section className="bg-gray-800 p-8 rounded-lg shadow-md mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6">Key Features</h2>
        <ul className="list-disc pl-5 text-white space-y-3">
          <li>Real-time Intrusion Detection and Mitigation</li>
          <li>Scalable Cloud Infrastructure with Google Cloud</li>
          <li>Low False Positive Rate for Accurate Threat Detection</li>
          <li>Automated Threat Alerts and Monitoring</li>
        </ul>
      </section>

      <section className="mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6">Website Features</h2>
        <ul className="list-disc pl-5 text-white space-y-3">
          <li>
            <NavLink to="/" className="hover:underline" title="Single Packet">
              Single Packet Detection:
            </NavLink>{' '}
            Allows the user to manually input network packet details and classify them.
          </li>
          <li>
            <NavLink to="/upload" className="hover:underline" title="Single Packet">
              CSV File Detection:
            </NavLink>{' '}
            Users can upload a CSV file containing multiple network packets for batch classification.
          </li>
          <li>
            <NavLink to="/autocapture" className="hover:underline" title="Single Packet">
              Auto Packet Capturing:
            </NavLink>{' '}
            Enables real-time classification of incoming network packets. Users must set up a local packet sniffing server—setup instructions are provided in that section.
          </li>
        </ul>
      </section>

      {/* Algorithms Used Section */}
      <section className="mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6">Algorithms Used</h2>
        <ul className="list-disc pl-5 text-white space-y-3">
          <li>Logistic Regression</li>
          <li>K-Nearest Neighbors (KNN)</li>
          <li>Decision Tree</li>
          <li>Random Forest</li>
          <li>Support Vector Machines (SVM)</li>
          <li>AdaBoost</li>
          <li>XGBoost</li>
          <li>ANN</li>
        </ul>
      </section>

      {/* Deployment on Google Cloud Section */}
      <section className="bg-gray-800 p-8 rounded-lg shadow-md mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6">Deployment on Google Cloud</h2>
        <p className="text-lg text-white">
          Our system's backend API is deployed on Google Cloud, ensuring scalability, security, and high availability. The cloud infrastructure allows for seamless scaling to handle varying levels of network traffic, while providing real-time threat detection.
        </p>
      </section>

      <section className="bg-gray-800 p-8 rounded-lg shadow-md mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6 text-center">Mentor</h2>

        <div className="flex justify-center items-center">
          <div className="flex items-center">
            <img
              className="w-38 h-38 rounded-full mr-6"
              src="https://th.bing.com/th/id/R.51068bb63aa31b9f0ab82115c4eb62db?rik=TJLQHyDxSTy4IQ&riu=http%3a%2f%2fiitj.ac.in%2fdept_faculty_pic%2fmishra.jpg&ehk=woMXDzoGb8FMiFzQliojL5aELFWvU7BI7f8T7ADciWQ%3d&risl=&pid=ImgRaw&r=0"
              alt="Prof. Anand Mishra"
            />
            <div>
              <h3 className="text-xl font-semibold text-white mb-2">Prof. Anand Mishra</h3>
              <h3 className="text-xl font-semibold text-white mb-2">Assistant Professor , IIT Jodhpur </h3>
              <p className="text-white">
                We would like to express our heartfelt gratitude to Prof. Anand Mishra, Assistant Professor at IIT Jodhpur, for his invaluable guidance and mentorship throughout the development of this project. His insights, encouragement, and support were instrumental in shaping our approach and helping us overcome challenges along the way. This project would not have reached its current level of depth and quality without his expert supervision. We are truly thankful for the opportunity to learn under his mentorship.
              </p>
            </div>
          </div>
        </div>

        {/* New Block for Mentor Details (Below the Picture) */}
        <div className="mt-6 p-4 rounded-lg">
          {/* <h4 className="text-xl font-semibold text-white mb-2">Details</h4> */}
          {/* <p className="text-white"> at the Department of Computer Science and Engineering at the Indian Institute of Technology Jodhpur.</p> */}
        </div>
      </section>

      {/* Meet the Team Section */}
      {/* Meet the Team Section */}
      <section className="mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6 text-center">Meet the Team</h2>

        <div className="flex flex-wrap justify-center gap-8 py-10 px-4">
          {[
            { name: "Luv Valecha", id: "B23CS1093", linkedin: "#", github: "#", pic: "pic.jpg" },
            { name: "Shiv Jee Yadav", id: "B23EE1095", linkedin: "#", github: "#", pic: "https://i.pinimg.com/originals/7a/1c/ba/7a1cba6cbdb9cf66793fcbf7b929b92b.jpg" },
            { name: "Ritik Nagar", id: "B23EE1061", linkedin: "#", github: "#", pic: "temp_pic.jpg" },
            { name: "Pratyush Chauhan", id: "B23CM1030", linkedin: "#", github: "#", pic: "temp_pic.jpg" },
            { name: "Dheeraj Kumar", id: "B23CS1016", linkedin: "#", github: "#", pic: "temp_pic.jpg" },
            { name: "Dhruv Sharma", id: "B23EE1086", linkedin: "#", github: "#", pic: "f_society.jpg" }
          ].map((member, index) => (
            <div
              key={index}
              className="bg-gradient-to-br from-gray-800 to-gray-900 text-white p-8 rounded-2xl shadow-lg w-full sm:w-3/4 md:w-[28rem] transition-transform hover:scale-105"
            >
              <div className="flex flex-col items-center">
                <img
                  src={member.pic}
                  alt={member.name}
                  className="w-28 h-28 rounded-full border-4 border-gray-700 object-cover mb-6 shadow-md"
                />
                <h3 className="text-2xl font-bold mb-1">{member.name}</h3>
                <p className="text-sm text-gray-400 mb-4">{member.id}</p>
                <div className="flex gap-6 mt-2">
                  <a
                    href={member.linkedin}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-blue-400 transition"
                    title="LinkedIn"
                  >
                    <i className="fab fa-linkedin text-2xl"></i>
                  </a>
                  <a
                    href={member.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-gray-400 transition"
                    title="GitHub"
                  >
                    <i className="fab fa-github text-2xl"></i>
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>


      {/* Footer Section */}
      <footer className="text-center py-6 bg-gray-800 text-white">
        <p className="text-lg">Thank you for exploring our project. We are committed to providing a cutting-edge solution for network intrusion detection, enhancing your network security.</p>
      </footer>
    </div>
  );
}

export default About;
