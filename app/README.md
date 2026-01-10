## **Semi-Autonomous Mode (Human-in-the-Loop)**

To run the application with a user interface for verification and correction:


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
