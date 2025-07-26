
##  Scene Graph Generation on Custom Images using OvSGTR

This project allows you to generate **scene graphs** from **custom single images** using the [OvSGTR](https://github.com/gpt4vision/OvSGTR) model, which combines **GroundingDINO** with open-vocabulary relationship detection.

---

### Clone the Repository and Instalation


```bash
git clone https://github.com/Roozbeho/image_eval_ovgstr.git

```

You need to install the required Python packages and set up GroundingDINO:

```bash
bash custom_install.sh
```

**Important:**
Inside `custom_install.sh`, be sure to modify the CUDA version to match your system. For example, if your system uses CUDA 12.1, ensure that lines like this are correct:

```bash
# Example: install torch==2.3.0+cu121
```

After installation, you can run the scene graph generator like this:

```bash
python inf_image.py
```

This runs on `data/image.png` using the default model and label vocabulary.

---

### Command-Line Arguments

To customize input files and parameters:

```bash
python run_sgg.py \
  --cfg_path config/GroundingDINO_SwinB_ovdr.py \
  --weights_path GroundingDINO/weights/vg-ovdr-swinb-mega-best.pth \
  --image_path data/image.png \
  --device cuda \
  --prompt_file data/custom_prompts/objects.txt \
  --rel_prompt_file data/custom_prompts/relations.txt
  --score_threshold 0.3 \
  --nms_iou_threshold 0.5
```

| Argument            | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| `--cfg_path`        | Path to model config(default: GroundingDINO_SwinB_ovdr) file                                            |
| `--weights_path`    | Path to model checkpoint `.pth` file (default: vg-ovdr-swinb-mega-best.pth)                              |
| `--image_path`      | Path to input image                                                  |
| `--device`          | `cuda` or `cpu`  (default: cuda)                                                     |
| `--prompt_file`     |  (Optional) Path to a text file with object classes (one per line)  |
| `--rel_prompt_file` | (Optional) Path to a text file with relation labels (one per line) |
| `--score_threshold`   | Confidence score threshold for detected objects (default: 0.2)|
| `--nms_iou_threshold`   | IoU threshold for non-max suppression (default: 0.5) |

If `--prompt_file` or `--rel_prompt_file` is not provided, the default Visual Genome vocabulary is used (`VG-SGG-dicts.json`)

`VG-SGG-dicts.json` downloaded in custom_install bash.

for default prompt and rel_prompt change commented `prompt & rel_prompt `lines in `inf_image.py` file.

