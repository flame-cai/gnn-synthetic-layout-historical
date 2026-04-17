import os
import shutil
import pandas as pd
from typing import Dict, List

# Note: All paths are relative to the project root
def gen_training_folder(pgs: List[int], in_dir: str, out_dir: str, doc_type: str, keep_old: bool = False) -> int:
    """Generate training/validation/test folder with renamed images.
    
    Args:
        pgs: List of page numbers
        in_dir: Input directory path
        out_dir: Output directory path 
        doc_type: Document type
        keep_old: Whether to keep existing output directory
        
    Returns:
        Number of files processed
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not keep_old:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    counter = 1
    
    for i in pgs:
        folder_name = os.path.join(in_dir, doc_type, 'images', f'pg{i:04d}')
        
        if not os.path.exists(folder_name):
            print(f"Warning: Folder {folder_name} does not exist, skipping...")
            continue
            
        print(f"Processing: {folder_name}")
        try:
            for file in sorted(os.listdir(folder_name)):
                if file.lower().endswith(('.jpg', '.jpeg')):
                    file_path = os.path.join(folder_name, file)
                    out_file = os.path.join(out_dir, f'{counter}.jpg')
                    shutil.copy2(file_path, out_file)  # copy2 preserves metadata
                    counter += 1
        except (OSError, shutil.Error) as e:
            print(f"Error processing {folder_name}: {str(e)}")
            continue

    print(f"Successfully copied and renamed {counter - 1} files.")
    return counter - 1

def get_annotations(folder_data: Dict) -> None:
    """Generate annotation files from folder data.
    
    Args:
        folder_data: Dictionary containing folder structure
    """
    for m in folder_data:
        for f in folder_data[m]:
            doc = m
            path = os.path.join('/annotations', doc, 'text')
            
            if not os.path.exists(path):
                print(f"Warning: Annotation path {path} does not exist, skipping...")
                continue

            df_list = []
            for num_page in folder_data[m][f]:
                filename = f'pg{num_page:04d}_annotated.txt'
                file_path = os.path.join(path, filename)
                
                try:
                    df = pd.read_table(
                        file_path,
                        lineterminator='\n',
                        header=None,
                        encoding='utf-8'
                    ).replace(to_replace='\r', value='', regex=True)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

            if not df_list:
                print(f"Warning: No valid annotations found for {f}")
                continue

            df = pd.concat(df_list, ignore_index=True)
            df.insert(0, None, [f'{i}.jpg' for i in range(1, df.shape[0] + 1)])
            
            output_dir = os.path.join('/recognition/line_images', f)
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                df.to_csv(
                    os.path.join(output_dir, 'labels.txt'),
                    sep='\t',
                    header=False,
                    index=False
                )
                print(f'Created labels.txt in folder: {f}')
            except Exception as e:
                print(f"Error writing labels for {f}: {str(e)}")

def generate_data(folder_data: Dict) -> None:
    """Generate training data from folder structure.
    
    Args:
        folder_data: Dictionary containing folder structure
    """
    in_dir = '/line-segmentation/output_images/line_images'
    
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory {in_dir} does not exist")

    for m in folder_data:
        for f in folder_data[m]:
            out_dir = os.path.join('/recognition/line_images', f)
            gen_training_folder(folder_data[m][f], in_dir, out_dir, doc_type=m)

    get_annotations(folder_data)

if __name__ == "__main__":
    """
    Folder structure:
    - manuscript_name: manuscript 1 (in annotations folder)
    - folder_type: train, test, val
    - number of pages: 30, 35, 38
    - folder_name = {manuscript_name}_{folder_type}_{number_of_pages}
    - folder_name gets saved in /recognition/line_images/
    - in the dictionary, folder_name : [list of pages]
    """
    folder_data = {
        'manuscript1': {
            'train_manuscript1_30': [i for i in range(2,51,2)] + [i for i in range(17, 27,2)],
            'train_manuscript1_35': [i for i in range(2,51,2)] + [i for i in range(17, 37,2)],
            'train_manuscript1_38': [i for i in range(2,51,2)] + [i for i in range(17, 43,2)],
            'test_manuscript1_30': [i for i in range(9,18,2)],
            'val_manuscript1_30': [i for i in range(3,8,2)],
            'test_manuscript1_35': [i for i in range(9,18,2)],
            'val_manuscript1_35': [i for i in range(3,8,2)],
            'test_manuscript1_38': [i for i in range(9,18,2)],
            'val_manuscript1_38': [i for i in range(3,8,2)]
        }
    }
    
    generate_data(folder_data)
