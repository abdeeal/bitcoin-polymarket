import json
import os

def update_notebook_01():
    path = "01_eda.ipynb"
    if not os.path.exists(path):
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "Synthetic demo data" in source:
                new_source = [
                    "# Load from extracted CSV\n",
                    "df = pd.read_csv('../data/bitcoin_180d.csv')\n",
                    "df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')\n",
                    "df['label'] = (df['close'].shift(-5) > df['close']).astype('Int64')\n",
                    "df.set_index('timestamp', inplace=True)\n",
                    "\n",
                    "print(f'Dataset: {len(df):,} rows, {df.index.min()} → {df.index.max()}')\n",
                    "print(f'Missing: {df.isnull().sum().sum()}')\n",
                    "df.head()\n"
                ]
                cell["source"] = new_source
                break
                
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

def update_notebook_02():
    path = "02_feature_importance.ipynb"
    if not os.path.exists(path):
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "np.random.seed(42)" in source and "feature_names" in source:
                new_source = [
                    "import os\n",
                    "import json\n",
                    "\n",
                    "# Load extracted features from data/processed\n",
                    "data_dir = os.path.abspath(os.path.join('..', 'data', 'processed'))\n",
                    "X = np.load(os.path.join(data_dir, 'X_180d.npy'))\n",
                    "close_prices = np.load(os.path.join(data_dir, 'close_180d.npy'))\n",
                    "\n",
                    "with open(os.path.join(data_dir, 'feature_names.json'), 'r') as f:\n",
                    "    feature_names = json.load(f)\n",
                    "\n",
                    "# Compute future looking labels (1 if UP, 0 if DOWN)\n",
                    "horizon = 5\n",
                    "y = np.full(len(close_prices), np.nan, dtype=np.float64)\n",
                    "for i in range(len(close_prices) - horizon):\n",
                    "    y[i] = 1.0 if close_prices[i + horizon] > close_prices[i] else 0.0\n",
                    "\n",
                    "valid_idx = ~np.isnan(y)\n",
                    "X = X[valid_idx]\n",
                    "y = y[valid_idx].astype(int)\n",
                    "\n",
                    "print(f'X: {X.shape}, label rate: {y.mean():.3f}')\n"
                ]
                cell["source"] = new_source
                break
                
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_notebook_01()
    update_notebook_02()
    print("Notebooks 01 and 02 updated successfully!")
