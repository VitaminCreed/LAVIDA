import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

import os
import json
from pathlib import Path
from typing import List, Dict

def create_video_metadata(dataset_root: str, mode: str, output_json_path: str) -> None:
    jpeg_dir = Path(dataset_root) / mode / "JPEGImages"
    video_paths = [str(folder) for folder in jpeg_dir.iterdir() if folder.is_dir()]
    
    video_pts = []
    video_fps = []
    video_name = []
    
    for video_path in video_paths:
        frame_count = len(list(Path(video_path).glob("*.jpg")))
        video_name.append(os.path.basename(video_path))
        video_pts.append(list(range(frame_count))) 
        video_fps.append(1)  
    
    output_data = {
        "video_name": video_name,
        "video_pts": video_pts,
        "video_fps": video_fps
    }

    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Meta data is saved to: {output_json_path}")

if __name__ == "__main__":
    dataset_root = "<DATA_PATH>/Ref-YoutubeVOS"  
    output_json = "./RefVos_valid_metadata.json"  
    create_video_metadata(dataset_root, 'valid', output_json)