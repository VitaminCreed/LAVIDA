import torch

def collate_fn(
    batch, clip_tokenizer=None, processor=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    images = []
    pixel_values = []
    pixel_values_videos = []
    video_paths = []
    frame_idxs = []
    gt_label = []
    ano_mask = []
    message_list = []
    ano_types = []
    ano_labels = []

    for item in batch:
        images.append(item["images"])
        if item["pixel_values"] is not None:
            pixel_values.append(item["pixel_values"])
        if item["pixel_values_videos"] is not None:
            pixel_values_videos.append(item["pixel_values_videos"])
        video_paths.append(item["video_path"])
        frame_idxs.append(item["frame_idxs"])
        gt_label.append(item["gt_label"])
        ano_mask.append(item["gt_mask"]['anomaly_masks'])
        message_list.append(item["message"])
        ano_types.append(clip_tokenizer(item["all_anomaly_types"], padding=True, return_tensors="pt"))
        ano_labels.append(torch.tensor(item["anomaly_labels"]).view(1, -1) if item["anomaly_labels"] is not None else None)

    ano_mask = torch.cat(ano_mask)
    gt_label = torch.cat(gt_label)
    
    if pixel_values == []:
        pixel_values = None
    if pixel_values_videos == []:
        pixel_values_videos = None
    
    text = processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
    
        
    conversations = processor(
        text=text,
        images=pixel_values,
        videos=pixel_values_videos,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    input_ids = conversations['input_ids']   # [1, 11653]
    attention_masks = conversations['attention_mask']
    
    if pixel_values:
        pixel_values = conversations['pixel_values']
        image_grid_thw = conversations['image_grid_thw']
    else:
        pixel_values = None
        image_grid_thw = None
        
    if pixel_values_videos:
        pixel_values_videos = conversations['pixel_values_videos']
        video_grid_thw = conversations['video_grid_thw']
    else:
        pixel_values_videos = None
        video_grid_thw = None
    
    input_ids_lists = input_ids.tolist()
    assert len(message_list) == len(input_ids_lists) 
    labels = generate_labels(input_ids_lists)


    return {
        "video_path": video_paths,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "pixel_values": pixel_values, 
        "pixel_values_videos": pixel_values_videos,  
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw, 
        "images": images, 
        "frame_idx": frame_idxs,
        "target_masks": ano_mask,
        "target_labels": gt_label,
        "labels": labels,
        "anomaly_types": ano_types,
        "anomaly_labels": ano_labels,
    }
    
    
def find_assistant_content_sublist_indexes(l):
    '''
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end
    return list(zip(start_indexes, end_indexes))
                
def generate_labels(input_ids_lists):
    
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return labels_ids