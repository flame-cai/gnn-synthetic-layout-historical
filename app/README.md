## **Semi-Autonomous Mode (Human-in-the-Loop)**
This mode allows users to manually correct and refine the GNN-predicted layouts using an intuitive web-based interface. Users can adjust text-line connections, label text boxes, and modify node placements to ensure high-quality layout annotations.
![GNN Layout UI Demo](./tutorial.gif)


## Installation and Setup Instructions
#### 1 Install Conda Environment
    ```bash
    cd app
    conda env create -f environment.yaml
    conda activate gnn_layout
    ```

#### 2 Start Backend Server
    ```bash
    cd app
    conda activate gnn_layout
    python app.py
    ```
    The server runs on `http://localhost:5000`.

#### 3 Start Frontend
    ```bash
    cd app/my-app
    npm install
    npm run dev
    ```
    Access the UI at `http://localhost:5173`.
