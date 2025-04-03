document.addEventListener('DOMContentLoaded', () => {
    async function makePrediction() {
        const data = {
            service: document.getElementById('service').value,
            flag: document.getElementById('flag').value,
            src_bytes: parseFloat(document.getElementById('src_bytes').value),
            dst_bytes: parseFloat(document.getElementById('dst_bytes').value),
            same_srv_rate: parseFloat(document.getElementById('same_srv_rate').value),
            diff_srv_rate: parseFloat(document.getElementById('diff_srv_rate').value),
            dst_host_srv_count: parseInt(document.getElementById('dst_host_srv_count').value),
            dst_host_same_srv_rate: parseFloat(document.getElementById('dst_host_same_srv_rate').value),
            dst_host_diff_srv_rate: parseFloat(document.getElementById('dst_host_diff_srv_rate').value),
            dst_host_serror_rate: parseFloat(document.getElementById('dst_host_serror_rate').value)
        };

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const resultDiv = document.querySelector('#singlePacketUI #result'); // Target result div inside singlePacketUI
            if (response.ok) {
                const result = await response.json();
                resultDiv.style.display = 'block';
                if (Object.keys(result).length === 0) {
                    resultDiv.innerHTML = '<span style="color: white;">Error: API returned no data.</span>';
                } else {
                    if (result.prediction[0] === 0) {
                        resultDiv.innerHTML = '<span style="color: green;">Safe Packet</span>'; // Green text
                    } else {
                        resultDiv.innerHTML = '<span style="color: red;">Malicious Packet</span>'; // Red text
                    }
                }
            } else {
                const error = await response.json();
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<span style="color: white;">Error: ${error.error}</span>`;
            }
        } catch (error) {
            const resultDiv = document.querySelector('#singlePacketUI #result'); // Target result div inside singlePacketUI
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `<span style="color: white;">Error: ${error.message}</span>`;
        }
    }

    function toggleUI(uiId) {
        const uiElement = document.getElementById(uiId);
        const container = document.querySelector('.container');
        const isCurrentlyVisible = uiElement.style.display === 'block';

        // Hide all UIs
        document.getElementById('singlePacketUI').style.display = 'none';
        document.getElementById('csvUploadUI').style.display = 'none';
        document.getElementById('autoCaptureUI').style.display = 'none';

        // Toggle the clicked UI
        uiElement.style.display = isCurrentlyVisible ? 'none' : 'block';

        // Add or remove the 'hidden' class based on visibility
        const anyVisible = ['singlePacketUI', 'csvUploadUI', 'autoCaptureUI'].some(
            id => document.getElementById(id).style.display === 'block'
        );
        container.classList.toggle('hidden', !anyVisible);
    }

    function showSinglePacketUI() {
        toggleUI('singlePacketUI');
    }

    function showCsvUploadUI() {
        toggleUI('csvUploadUI');
    }

    function showAutoCaptureUI() {
        toggleUI('autoCaptureUI');
    }

    async function predictFromCsv(event) { 

        event.preventDefault(); // Prevent the form from submitting normally

        console.log("🚀 predictFromCsv function started!");
        const fileInput = document.getElementById('csvFile');
        if (!fileInput.files.length) {
            alert('Please upload a CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            console.log("test1");
            const response = await fetch('http://127.0.0.1:5000/predict_csv', {
                method: 'POST',
                body: formData
            });
            console.log("test2");

            const csvResultDiv = document.getElementById('csvResult');
            csvResultDiv.style.display = 'block'; // Ensure the result div remains visible
                if (response.ok) {
                    const result = await response.json();
                    console.log("✅ Response received:", result);

                    // Format the results into a table
                    let tableHtml = '<table><tr><th>Row</th><th>Prediction</th></tr>';
                    result.forEach((row, index) => {
                        tableHtml += `<tr><td>${index + 1}</td><td>${row.prediction === 0 ? 'Safe Packet' : 'Malicious Packet'}</td></tr>`;
                    });
                    tableHtml += '</table>';

                    csvResultDiv.innerHTML = tableHtml;
                } else {
                    const error = await response.json();
                    csvResultDiv.innerHTML = `<span style="color: white;">Error: ${error.error}</span>`;
                }
            return false;
        } catch (error) {
            const csvResultDiv = document.getElementById('csvResult');
            csvResultDiv.style.display = 'block'; // Ensure the result div remains visible
            csvResultDiv.innerHTML = `<span style="color: white;">Error: ${error.message}</span>`;
        }
        console.log("ended");
    }


    // document.addEventListener("DOMContentLoaded", function () {
    //     document.getElementById("csvForm").addEventListener("submit", predictFromCsv);
    // });
    document.getElementById("csvForm").addEventListener("submit", predictFromCsv);


    // Attach functions to the global window object
    window.showSinglePacketUI = showSinglePacketUI;
    window.showCsvUploadUI = showCsvUploadUI;
    window.showAutoCaptureUI = showAutoCaptureUI;
    window.makePrediction = makePrediction;
    window.predictFromCsv = predictFromCsv;
});