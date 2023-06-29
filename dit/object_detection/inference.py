import argparse
import cv2
import os
import numpy as np
from ditod import add_vit_config
import torch
import shutil
from operator import itemgetter
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


def sort_index(instance):
    try:
        bboxes = instance.pred_boxes.tensor

        res = []
        new_points = [0]
        new_point = 0
        for i in range(len(bboxes.numpy())):
            if i != new_point:
                continue

            ref = bboxes[i][0]
            nearest_bboxes = [[index, float(bboxes[index][1])] for index in range(len(bboxes)) if abs(float(bboxes[index][0] - ref)) <= 100]
            sorted_list = sorted(nearest_bboxes, key=itemgetter(1))
            res.extend([x[0] for x in sorted_list])
            # res.extend([x + new_point for x in sorted(range(len(nearest_bboxes)), key=lambda k: nearest_bboxes[k][1])])
            
            new_point = len(nearest_bboxes) + new_points[-1]
            new_points.append(new_point)

            if new_point == len(bboxes):
                break
        return res

    except Exception as e:
        raise e


def sort_tensor(tensor, sort_mask):
    try:
        return tensor[sort_mask]

    except Exception as e:
        raise e


def sort(instance):
    try:
        sort_mask = sort_index(instance)
        # sort pred_boxes
        instance.pred_boxes.tensor = sort_tensor(instance.pred_boxes.tensor, sort_mask)
        # sort score
        instance.scores = sort_tensor(instance.scores, sort_mask)
        # sort pred_classes
        instance.pred_classes = sort_tensor(instance.pred_classes, sort_mask)
        # sort pred_masks
        instance.pred_masks = sort_tensor(instance.pred_masks, sort_mask)
        return instance

    except Exception as e:
        raise e


def get_remove_mask(instance, conf):
    try:
        scores = instance.scores
        out_mask = []
        for idx, score in enumerate(scores):
            if score >= conf:
                out_mask.append(idx)
        return out_mask

    except Exception as e:
        raise e


def filter_tensor(tensor, mask):
    try:
        return tensor[mask]

    except Exception as e:
        raise e


def remove_box_lower_than(instance, conf):
    try:
        mask = get_remove_mask(instance, conf)
        # sort pred_boxes
        instance.pred_boxes.tensor = filter_tensor(instance.pred_boxes.tensor, mask)
        # sort score
        instance.scores = filter_tensor(instance.scores, mask)
        # sort pred_classes
        instance.pred_classes = filter_tensor(instance.pred_classes, mask)
        # sort pred_masks
        instance.pred_masks = filter_tensor(instance.pred_masks, mask)

        # # Check if a bbox is in other bbox
        # centres = [[int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)] for bbox in instance.pred_boxes.tensor.numpy()]
        # mask = []
        # bboxes = instance.pred_boxes.tensor.numpy()
        # for index in range(len(bboxes)):
        #     for index_point in range(len(centres)):
        #         if index != index_point and centres[index_point][0] > bboxes[index][0] and centres[index_point][0] < bboxes[index][2] and \
        #         centres[index_point][1] > bboxes[index][1] and centres[index_point][1] < bboxes[index][3] and \
        #         instance.scores.numpy()[index] > instance.scores.numpy()[index_point]:
        #             mask.append(index_point)
        # mask = [x for x in range(len(centres)) if x not in mask]

        # # sort pred_boxes
        # instance.pred_boxes.tensor = filter_tensor(instance.pred_boxes.tensor, mask)
        # # sort score
        # instance.scores = filter_tensor(instance.scores, mask)
        # # sort pred_classes
        # instance.pred_classes = filter_tensor(instance.pred_classes, mask)
        # # sort pred_masks
        # instance.pred_masks = filter_tensor(instance.pred_masks, mask)

        return instance

    except Exception as e:
        raise e


def add_padding(input_img, padding, color):
    try:
        old_image_height, old_image_width, channels = input_img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = old_image_width + 2 * padding
        new_image_height = old_image_height + 2 * padding
        result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # copy img image into center of result image
        result[y_center:y_center + old_image_height,
        x_center:x_center + old_image_width] = input_img
        return result

    except Exception as e:
        raise e
    

def crop_and_save_image(idx, input_img, bbox, filename, class_list, classes, padding=25):
    try:
        height, width, channels = input_img.shape
        x1 = int(bbox[0])
        x2 = int(bbox[2])
        y1 = int(bbox[1])
        y2 = int(bbox[3])
        cropped_image = input_img[y1:y2, x1:x2]
        padded_image = add_padding(cropped_image, padding, color=(255, 255, 255))
        return padded_image

    except Exception as e:
        raise e


def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_path",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file_path",
        help="Path of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if os.path.exists(args.output_file_path):
      shutil.rmtree(args.output_file_path)
    os.mkdir(args.output_file_path)

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    img = cv2.imread(args.image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text","title","list","table","figure"])

    output = predictor(img)["instances"]
    remove_box_lower_than(output, 0.30)
    output = sort(output)

    boxes = output.to("cpu").pred_boxes if output.to("cpu").has("pred_boxes") else None
    scores = output.to("cpu").scores if output.to("cpu").has("scores") else None
    classes = output.to("cpu").pred_classes.tolist() if output.to("cpu").has("pred_classes") else None
    class_list = ["text", "title", "list", "table", "figure"]

    result = []
    cont = 1
    for idx, box in enumerate(boxes):
        if class_list[classes[idx]] in ["figure"]:
            padded_image = crop_and_save_image(idx, img, box, os.path.join(args.output_file_path, str(idx) + ".jpg"), class_list, classes, 25)
            if np.any(padded_image):
                # print("save figure!!!")
                cv2.imwrite(args.output_file_path + str(cont) + ".jpg", padded_image)
                cont += 1
        #     else:
        #         print("figure image not detected")
        # else:
        #     print("detected other object that is not a figure")

    # v = Visualizer(img[:, :, ::-1],
    #                 md,
    #                 scale=1.0,
    #                 instance_mode=ColorMode.SEGMENTATION)
    # result = v.draw_instance_predictions(output.to("cpu"))
    # result_image = result.get_image()[:, :, ::-1]

    # # step 6: save
    # cv2.imwrite(args.output_file_path + "debug/debug.jpg", result_image)

if __name__ == '__main__':
    main()

