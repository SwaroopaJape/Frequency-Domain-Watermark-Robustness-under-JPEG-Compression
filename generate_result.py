import sys
import subprocess
from pathlib import Path
import cv2
from skimage import data 

# download famouse grayscale test images to create dataset
def prepare_dataset(output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing dataset in '{output_dir.name}/'...")
    
    images = {
        "camera": data.camera(),
        "coins": data.coins(),
        "moon": data.moon(),
        "brick": data.brick(),
    }
    
    for name, img_array in images.items():
        save_path = output_dir / f"{name}.png"
        cv2.imwrite(str(save_path), img_array)
        print(f"  Saved {name}.png")
        
    print("Dataset ready.\n")

# runs analyze.py on all the images in test_dataset and storing o/p
def run_batch_analysis(dataset_dir: Path, output_csv: Path):
    image_paths = list(dataset_dir.glob("*.png"))

    print(f"Starting batch analysis on {len(image_paths)} images...")
    with open(output_csv, "w") as f:
        header_written = False
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Analyzing {img_path.name}...")
            
            # run analyse.py as subprocess
            result = subprocess.run(
                [sys.executable, "analyze.py", str(img_path)],
                capture_output=True,
                text=True
            )

            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
                
            # parse the comma seperated o/p
            output_lines = result.stdout.strip().split('\n')
            
            if not output_lines or output_lines == ['']:
                continue
                
            # get the header
            header = output_lines[0]
            data_rows = output_lines[1:]
            
            if not header_written:
                f.write(header + '\n')
                header_written = True
                
            # write the data
            for row in data_rows:
                f.write(row + '\n')

    print(f"\nSuccess! All results saved to {output_csv.name}")

def main():
    dataset_dir = Path("test_dataset")
    output_csv = Path("results.csv")
    
    prepare_dataset(dataset_dir)
    run_batch_analysis(dataset_dir, output_csv)

if __name__ == "__main__":
    main()