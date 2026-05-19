import nbformat
import os

def update():
    nb_path = 'notebooks/03_walk_forward_results.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Check if the save cell already exists
    has_save_cell = any("np.save(os.path.join(data_dir, 'oos_probs.npy'), all_val_probs)" in cell.source for cell in nb.cells)
    
    if not has_save_cell:
        new_cell = nbformat.v4.new_code_cell(source="""import os
import numpy as np

data_dir = os.path.abspath(os.path.join('..', 'data', 'processed'))
np.save(os.path.join(data_dir, 'oos_probs.npy'), all_val_probs)
np.save(os.path.join(data_dir, 'oos_y.npy'), all_y_val)
print("Saved OOS probabilities and labels to data/processed/")
""")
        nb.cells.append(new_cell)
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Added save cell to 03_walk_forward_results.ipynb")
    else:
        print("Save cell already exists in 03_walk_forward_results.ipynb")

if __name__ == '__main__':
    update()
