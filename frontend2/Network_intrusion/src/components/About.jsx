import React from 'react';

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
        </ul>
      </section>

      {/* Deployment on Google Cloud Section */}
      <section className="bg-gray-800 p-8 rounded-lg shadow-md mb-16">
        <h2 className="text-3xl font-semibold text-white mb-6">Deployment on Google Cloud</h2>
        <p className="text-lg text-white">
          Our system is deployed on Google Cloud, ensuring scalability, security, and high availability. The cloud infrastructure allows for seamless scaling to handle varying levels of network traffic, while providing real-time threat detection.
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
        <h3 className="text-xl font-semibold text-white mb-2">Assistant Professor , I.I.T.Jodhpur </h3>
        <p className="text-white">
          
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

      {/* Meet the Team Section (3 Members per Row) */}
      <section className="mb-16">
  <h2 className="text-3xl font-semibold text-white mb-6">Meet the Team</h2>

  <div className="flex flex-wrap justify-between">
    {/* Team Member 1 */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full sm:w-1/2 lg:w-1/3 mb-6">
      <img
        className="w-24 h-24 rounded-full mx-auto mb-4"
        src="https://th.bing.com/th/id/OIP.-mDSgRGbUyo9jUJKbWI3nAAAAA?rs=1&pid=ImgDetMain"
        alt="Luv Valecha"
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Luv Valecha</h3>
        <p className="text-white">Student ID: B23CS1093</p>
      </div>
    </div>

    {/* Team Member 2 */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full sm:w-1/2 lg:w-1/3 mb-6">
      <img
        className="w-24 h-24 rounded-full mx-auto mb-4"
        src="https://th.bing.com/th/id/OIP.-mDSgRGbUyo9jUJKbWI3nAAAAA?rs=1&pid=ImgDetMain"
        alt="Shiv Jee Yadav"
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Shiv Jee Yadav</h3>
        <p className="text-white">Student ID: B23EE1095</p>
      </div>
    </div>

    {/* Team Member 3 */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full sm:w-1/2 lg:w-1/3 mb-6">
      <img
        className="w-24 h-24 rounded-full mx-auto mb-4"
        src="https://th.bing.com/th/id/OIP.-mDSgRGbUyo9jUJKbWI3nAAAAA?rs=1&pid=ImgDetMain"
        alt="Ritik Nagar"
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Ritik Nagar</h3>
        <p className="text-white">Student ID: B23EE1061</p>
      </div>
    </div>

    {/* Team Member 4 */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full sm:w-1/2 lg:w-1/3 mb-6">
      <img
        className="w-24 h-24 rounded-full mx-auto mb-4"
        src="https://th.bing.com/th/id/OIP.-mDSgRGbUyo9jUJKbWI3nAAAAA?rs=1&pid=ImgDetMain"
        alt="Pratyush Chauhan"
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Pratyush Chauhan</h3>
        <p className="text-white">Student ID: B23CM1030</p>
      </div>
    </div>

    {/* Team Member 5 */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full sm:w-1/2 lg:w-1/3 mb-6">
      <img
        className="w-24 h-24 rounded-full mx-auto mb-4"
        src="https://th.bing.com/th/id/OIP.-mDSgRGbUyo9jUJKbWI3nAAAAA?rs=1&pid=ImgDetMain"
        alt="Dheeraj Kumar"
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Dheeraj Kumar</h3>
        <p className="text-white">Student ID: B23CS1016</p>
      </div>
    </div>

    {/* Team Member 6 */}
    <div className="bg-gray-800 p-6 rounded-lg shadow-md w-full sm:w-1/2 lg:w-1/3 mb-6">
      <img
        className="w-24 h-24 rounded-full mx-auto mb-4"
        src="https://th.bing.com/th/id/OIP.-mDSgRGbUyo9jUJKbWI3nAAAAA?rs=1&pid=ImgDetMain"
        alt="Dhruv Sharma"
      />
      <div className="text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Dhruv Sharma</h3>
        <p className="text-white">Student ID: B23EE1086</p>
      </div>
    </div>
  </div>
</section>

      {/* Mentor Information Section */}
     

      {/* Footer Section */}
      <footer className="text-center py-6 bg-gray-800 text-white">
        <p className="text-lg">Thank you for exploring our project. We are committed to providing a cutting-edge solution for network intrusion detection, enhancing your network security.</p>
      </footer>
    </div>
  );
}

export default About;
