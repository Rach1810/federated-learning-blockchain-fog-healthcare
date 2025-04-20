# 🏥 Secure Healthcare: Federated Learning in a Blockchain-IoT Fog

This project simulates a privacy-preserving, decentralized healthcare system using:
- **Federated Learning (FL)** on wearable devices
- **Fog Computing** for distributed local training
- **Private Blockchain** for logging and validation of model updates

Based on the research paper:  
**Federated Learning and Blockchain-Enabled Fog-IoT Platform for Wearables in Predictive Healthcare**

## 🧠 Architecture Overview
- IoT Wearables → Simulated clients
- Fog Nodes → Perform local training on HAR dataset
- FL Server → Aggregates models via accuracy-weighted FedAvg
- Blockchain → Logs each client's accuracy and hash securely

## 📁 Folder Structure
```
secure_healthcare_project/
├── data_loader.py
├── client.py
├── server.py
├── blockchain.py
├── model.py
├── simulate.py
├── requirements.txt
├── README.md
├── run.sh
└── UCI HAR Dataset/
```

## 📦 Setup Instructions
### ✅ Create Virtual Environment (Recommended)
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 📥 Dataset Setup
Download and extract the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) inside `secure_healthcare_project/`.

## ▶️ Run Simulation
```bash
python simulate.py
```

## 📊 Output
- Global accuracy progression
- Blockchain log of client accuracy + hashes
- Best model evaluation on test set

## ✍️ Authors
**Prathamesh Jadhav**
**Rachana Misal**  
**Ayush Pande**  
M.Tech CSE, NIT Karnataka  
Guided by: Prof. Sourav Kanti Addya
Course: BlockChain Design and Architecture
