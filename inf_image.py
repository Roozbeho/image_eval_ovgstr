import argparse
import os
from typing import Dict, List, Tuple
import torch
from PIL import Image
import json

from OvSGTR.util.slconfig import SLConfig
from collections import defaultdict
from OvSGTR.util.visualizer import COCOVisualizer
import numpy as np
from GroundingDINO.groundingdino.util.inference import load_image
from OvSGTR.main import build_model_main

@torch.inference_mode()
def main():
    # Define paths and prompts
    parser = argparse.ArgumentParser(description="Run Scene Graph Generation with GroundingDINO")
    parser.add_argument("--cfg_path", type=str, default="OvSGTR/config/GroundingDINO_SwinB_ovdr.py")
    parser.add_argument("--weights_path", type=str, default="OvSGTR/GroundingDINO/weights/vg-ovdr-swinb-mega-best.pth")
    parser.add_argument("--image_path", type=str, default="data/image.png")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional path to file with object prompts")
    parser.add_argument("--rel_prompt_file", type=str, default=None, help="Optional path to file with relation prompts")
    parser.add_argument("--score_threshold", type=float, default=0.2, help="Score threshold for filtering object detections")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5, help="IoU threshold for NMS in postprocessing")
    args = parser.parse_args()

    device = torch.device(args.device)

    # prompt = "man. shirt. skateboard. wheel. pant. helmet"
    # rel_prompt = "background. on. has. in. wearing. holding. standing on. next to. under. above. behind. in front of. riding. carrying. looking at. covering. sitting on. attached to. walking on. eating. leaning on. parked on. biting. kicking. playing. drinking from. flying in. tied to. pulling. pushing. smiling at. talking on. covering with. parked in. wears. has part. using. surrounded by. consist of. flying over. riding on. woven from. standing in. flowing into. resting on. grazing on. jumping over. pouring into. holding by. hanging from. screaming at."

    # Load prompts
    if args.prompt_file and args.rel_prompt_file:
        with open(args.prompt_file, "r") as f:
            categories = [line.strip() for line in f if line.strip()]
        with open(args.rel_prompt_file, "r") as f:
            rel_categories = [line.strip() for line in f if line.strip()]
        prompt = ". ".join(categories)
        rel_prompt = ". ".join(rel_categories) + "."
    else:
        with open("data/vg_data/stanford_filtered/VG-SGG-dicts.json", "r") as f:
            m_json = json.load(f)
        categories = list(m_json['label_to_idx'].keys())
        rel_categories = list(m_json['idx_to_predicate'].values())
        prompt = ". ".join(categories)
        rel_prompt = ". ".join(rel_categories) + "."


    categories = [p.strip() for p in prompt.split('.') if p.strip()]
    name2classes = {v: k for k, v in enumerate(categories)}

    rel_categories = [rel.strip() for rel in rel_prompt.split('.') if rel.strip()]
    name2predicates = {v: k for k, v in enumerate(rel_categories)}
    name2predicates['background'] = 0 # or set first idx to __background__


    cfg = SLConfig.fromfile(args.cfg_path)
    cfg.device = device
    cfg.do_sgg = True
    cfg.sg_ovd_mode = True
    cfg.sg_ovr_mode = True
    cfg.name2predicates = name2predicates
    model, _, postprocessors = build_model_main(cfg)
    model.to(device)

    # Load model weights
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Image preprocessing
    array_image, transformed_image = load_image(args.image_path)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    

    # Model inference
    with torch.no_grad():
        outputs = model(transformed_image, captions=[prompt], rel_captions=[rel_prompt])


    # Configure postprocessor
    postprocessor = postprocessors['bbox']
    postprocessor.rln_proj = getattr(model, "rln_proj", None)
    postprocessor.rln_classifier = getattr(model, "rln_classifier", None)
    postprocessor.rln_freq_bias = getattr(model, "rln_freq_bias", None)
    postprocessor.name2predicates = name2predicates
    postprocessor.name2classes = name2classes
    postprocessor.score_threshold = args.score_threshold
    postprocessor.nms_iou_threshold = args.nms_iou_threshold

    c, h, w = transformed_image[0].shape
    target_sizes = torch.tensor([[h, w]], device=device)
    det = postprocessor(outputs, target_sizes, test=True)[0]

    print('detected bounding box')
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        print(f"Label: {categories[label.item()]}, Score: {score:.2f}")

    # Extract and process scene graph
    temp_graph: Dict = det['graph']
    pred_classes = temp_graph['pred_boxes_class']
    pred_scores = temp_graph['pred_boxes_score']
    pairs = temp_graph['all_node_pairs']
    logits = temp_graph['all_relation']

    logits[:, 0] = 0  # background/no-relation class
    rel_score, rel_idx = logits.max(dim=1)
    idx2rel = {v: k for k, v in name2predicates.items()}


    relations_dict = defaultdict(list)
    for (s, o), score, ridx in zip(pairs.tolist(), rel_score.tolist(), rel_idx.tolist()):
        if s != o: # ignore sub to sub relations

            subj_lbl = categories[pred_classes[s].item()]
            obj_lbl = categories[pred_classes[o].item()]
            rel_name = idx2rel[ridx]

            relations_dict[(subj_lbl, rel_name, obj_lbl)].append(score)
        
    best_relations: List[Tuple[str, str, float, str]] = [
        (subj, rel, max(scores), obj)
        for (subj, rel, obj), scores in relations_dict.items()
        if subj != obj
    ]
    best_relations.sort(key=lambda x: x[2], reverse=True)

    output_file = "logs/relations.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for subj, rel, score, obj in best_relations:
            f.write(f"{subj} --{rel}---{score:.4f}--> {obj}" + "\n")

    
    relations_scene: List[List[str, str, str]] = []
    for subj, rel, obj in list(relations_dict.keys()):
        relations_scene.append([subj, obj, rel])

    vis_class = COCOVisualizer()
    tgt = {
        'boxes': det['boxes'].to('cpu') / torch.Tensor([w, h, w, h]),
        'size': torch.Tensor(Image.fromarray(array_image).size),
        'image_id': 1,
        'box_label': [categories[l.item()] for l in det['labels']]
    }
    vis_class.visualize(
        transformed_image.squeeze().to('cpu'),
        tgt, 
        savedir="logs",
        relations=relations_scene
    )

if __name__ == "__main__":
    main()