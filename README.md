
# Network Intrusion Detection

The Network Intrusion Detection System (NIDS) is designed to identify unauthorized access or anomalies within a network. By leveraging machine learning algorithms, this system analyzes network traffic to detect potential threats and intrusions effectively.


## Features


- Machine Learning Models: Utilizes advanced algorithms for accurate anomaly detection.

- User-friendly Interface: Provides a frontend for easy interaction and visualization of detected threats.


## Installation

1. Clone the Repository:

```bash
git clone https://github.com/Luv-valecha/Network_Intrusion.git

```
2. Navigate to the Project Directory:
```bash
cd Network_Intrusion

```
3. Install Required Dependencies:

Ensure that all necessary Python packages are installed by running:
```bash
pip install -r requirements.txt

```
```bash
cd Network_Intrusion

```
    
## Usage
Follow these steps to set up and run the Network Intrusion Detection System:
1. Data Preparation:
- Extract Data Files:
    On Windows, execute the following command in the Command Prompt to extract   the dataset:
    
    ```bash
    Expand-Archive -Path ".\API\data\raw\archive.zip" -DestinationPath ".\API\data\raw"
    ```
2. Preprocess the Data:
    Run the preprocessing script to clean and prepare the data for analysis:
    ```bash
    python preprocess.py

    ```
3. Run the server 
    ```bash
    python run.py
    ```
4. Launch the Frontend Application:

    Navigate to the frontend directory and start the application to visualize and interact with the system:
    ```bash
    cd frontend
    # then run the index.html
    ```
## Contributing

Contributions are always welcome!

You can fork the repository the create a branch and start working.
once you finish create a pull request. 

## Authors

- [Dheeraj Kumar](https://www.github.com/dheeraj-kumar)
- [Luv valecha](https://www.github.com/dheeraj-kumar)
- [Shiv jee yadav](https://www.github.com/dheeraj-kumar)
- [Dhruv sharma](https://www.github.com/dheeraj-kumar)
- [Ritik Nagar](https://www.github.com/dheeraj-kumar)
- [Pratyush(pransu) chauhan](https://www.github.com/dheeraj-kumar)


## License



This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/) . You are free to use, modify, and distribute this software in accordance with the terms of this license.

