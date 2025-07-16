import argparse
import numpy as np
import csv
import cv2
import glob
import yaml


def find_contours(img):
    """
    Looking for contours in image.

    Args:
        img (np.array): An array from image.

    Returns:
        tuple: A tuple of picked bounding boxes after suppression.
    """
    blurred = cv2.GaussianBlur(img, (15,15), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY_INV, 199, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def process_contours(main_gray, min_area, scale_factor=0.8):
    """
    Process contours in image.

    Args:
        main_gray (np.array): An array from grayscale image.
        min_area (int): A number for filtering contours area.
        scale_factor (float): A number for scaling main image area

    Returns:
        np.array: An array of boxes around every single drawing on main drawing.
    """
    # Pre-process the image to find large contours
    main_area = main_gray.shape[0] * main_gray.shape[1]
    contours = find_contours(main_gray)

    # Filter for small contours
    elevation_boxes = []
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if (cnt_area > min_area) and (cnt_area < main_area * scale_factor):
            x, y, w, h = cv2.boundingRect(cnt)
            elevation_boxes.append((x, y, w, h))

    # Apply non-max suppression
    elevation_boxes = np.array(elevation_boxes)
    picked_boxes = non_max_suppression(elevation_boxes, 0.3)
    return picked_boxes


def multi_scale_matching(template, match_area, confidence_threshold=0.7, prepared_box=None, canny=False):
    """
    Process prepared boxes and finding appropriate template.

    Args:
        template (np.array): A search template .
        match_area (np.array): An area where we look for templates.
        confidence_threshold (float): A number for filtering matching results.
        prepared_box (tuple): A tuple of boxes coordinates.
        canny (bool): Use Canny method or not.

    Returns:
        list: A list of matched boxes.
    """
    boxes = []

    if prepared_box:
        px, py, pw, ph = prepared_box
        match_area = match_area[py:py + ph, px:px + pw]


    for h_scale in np.linspace(0.8, 1.4, 20)[::-1]:
        h_scaled = int(template.shape[0] * h_scale)
        for w_scale in np.linspace(0.8, 1.4, 20)[::-1]:
            w_scaled = int(template.shape[1] * w_scale)

            # Ensure scaled template is not larger than the main image
            if w_scaled > match_area.shape[1] or h_scaled > match_area.shape[0]:
                continue

            resized_template = cv2.resize(template, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
            if canny:
                # Use Canny edge detection for more robust shape matching
                template_edges = cv2.Canny(resized_template, 150, 200, apertureSize=5)
                main_edges = cv2.Canny(match_area, 150, 200, apertureSize=5)
                result = cv2.matchTemplate(main_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            else:
                result = cv2.matchTemplate(match_area, resized_template, cv2.TM_CCOEFF_NORMED)

            # Find all locations where the match is above the threshold
            loc = np.where(result >= confidence_threshold)
            for pt in zip(*loc[::-1]):  # Switch x and y
                if prepared_box:
                    boxes.append((pt[0]+px, pt[1]+py, w_scaled, h_scaled))
                else:
                    boxes.append((pt[0], pt[1], w_scaled, h_scaled))
    return boxes


def prepare_boxes(elevation_boxes, height_boxes):
    """
    Prepare upper-half boxes for single drawing which have wall cabinets.

    Args:
        elevation_boxes (np.array): The boxes around every single drawing on main drawing.
        height_boxes (np.array): The boxes founded for wall height template.

    Returns:
        np.array: The boxes which ready for searching templates.
    """
    r_side = None
    l_side = None
    result_boxes = []

    for (ex, ey, ew, eh) in elevation_boxes:
        eval_box_center_x = ex + ew / 2
        eval_box_center_y = ey + eh / 2
        for (hx, hy, hw, hh) in height_boxes:
            # Check if cabinet is inside this elevation
            if hx > ex and hx + hw < ex + ew and hy > ey and hy + hh < ey + eh:
                height_box_center_x = hx + hw / 2
                height_box_center_y = hy + hh / 2
                # Check if it's in the upper half
                if eval_box_center_y > height_box_center_y:
                    # Check if it's in the right/left half
                    if height_box_center_x > eval_box_center_x:
                        r_side = (hx, hy)
                    if height_box_center_x < eval_box_center_x:
                        l_side = (hx, hy)

        # prepare result boxes
        if r_side and l_side:
            box = l_side + ((r_side[0] - l_side[0]), hh)
            result_boxes.append(box)
        elif r_side and not l_side:
            box = (ex, hy, (r_side[0] - ex), hh)
            result_boxes.append(box)
        elif l_side and not r_side:
            box = l_side + (((ex+ew) - l_side[0]), hh)
            result_boxes.append(box)

        r_side = None
        l_side = None

    return result_boxes


def save_results(main_image, output_path, all_detections):
    """
    Prepare results files and save them to disk.

    Args:
        main_image (np.array): The original image.
        output_path (str): The path to save the marked image
        all_detections(np.array): A list of detected cabinet boxes.

    Returns:
        Nothing
    """
    # Generate Output Files
    print("Generating output files: marked_drawing.png and cabinet_coordinates.csv")
    # Draw boxes on the original image
    output_image = main_image.copy()
    for (x, y, w, h) in all_detections:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle

    # Save the marked-up image
    cv2.imwrite(f'{output_path}/marked_drawing.png', output_image)

    # Save coordinates to CSV
    with open(f'{output_path}/cabinet_coordinates.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'width', 'height'])
        for (x, y, w, h) in all_detections:
            writer.writerow([x, y, w, h])


def non_max_suppression(boxes, overlap_thresh):
    """
    Applies Non-Maximum Suppression to a list of bounding boxes.

    Args:
        boxes (np.array): An array of (x, y, w, h) bounding boxes.
        overlap_thresh (float): The threshold for overlapping boxes.

    Returns:
        list: A list of picked bounding boxes after suppression.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to (x1, y1, x2, y2) format
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='Path to the config file', required=True)
    args = parser.parse_args()


    # load config
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Loading images...")
    # Load the main image and convert it to grayscale
    main_image = cv2.imread(config['image_path'])
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    min_area = int(main_gray.shape[0] * main_gray.shape[1] * 0.01)
    elevation_boxes = process_contours(main_gray, min_area, config['scale_factor'])
    print(f"Found {len(elevation_boxes)} potential elevation areas.")

    # Looking for the wall cabinet height
    confidence_height_threshold = 0.7
    height_template = cv2.imread(config['height_template_path'], 0)
    height_boxes = multi_scale_matching(height_template, main_gray, confidence_height_threshold)
    height_boxes = np.array(height_boxes)
    height_boxes = non_max_suppression(height_boxes, 0.3)

    # Preparing boxes for template searching
    prepared_boxes = prepare_boxes(elevation_boxes, height_boxes)

    # Template Matching
    print("Performing template matching for wall cabinets...")
    all_detections = []
    confidence_threshold = 0.5  # Adjust this threshold (0.0 to 1.0)

    for prepared_box in prepared_boxes:
        for template_path in glob.glob(config['template_dir']+'/*'):
            template = cv2.imread(template_path, 0)
            if template is None:
                print(f"Warning: Could not load template image at {template_path}. Skipping.")
                continue
            cabinet_boxes = multi_scale_matching(template, main_gray, confidence_threshold, prepared_box, True)
            all_detections.extend(cabinet_boxes)

    # Applying Non-Maximum Suppression to remove duplicate detections
    all_detections = np.array(all_detections)
    all_detections = non_max_suppression(all_detections, 0.1)
    print(f"Kept {len(all_detections)} detections after NMS.")

    # Generate output files and save them
    save_results(main_image, config['output_path'], all_detections)


