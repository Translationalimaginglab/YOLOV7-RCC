import os
import glob
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def segment_intersection(xb, xe, yb, ye):
    return max(xb, ye) + max(xe, yb) - max(xb, yb) - max(xe, ye)

class Box:
    def __init__(self, centerX, centerY, width, height, label=0):
        self.x = centerX
        self.y = centerY
        self.width = width
        self.height = height

        self.label = label  # YOLO format
        self.visited = False  # For make_tumors_from_boxes
        self.z = 0  # For Z indexing tumors
        self.score = 0.0  # For threshold score

    @staticmethod
    def from_center(centerX, centerY, width, height, label=0):
        return Box(centerX, centerY, width, height, label=label)

    @staticmethod
    def from_corners(lowerX, lowerY, upperX, upperY, label=0):
        width = abs(upperX - lowerX)
        height = abs(upperY - lowerY)
        centerX = (lowerX + upperX) / 2
        centerY = (lowerY + upperY) / 2

        return Box(centerX, centerY, upperX, upperY, label=label)

    @staticmethod
    def parse(line):
        tokens = [float(token) for token in line.strip().split(" ") if len(token) > 0]
        assert len(tokens) == 5 or len(tokens) == 6

        box = Box(tokens[1], tokens[2], tokens[3], tokens[4], label=int(tokens[0]))

        if len(tokens) == 6:
            box.score = tokens[5]

        return box

    def copy(self):
        new_box = Box(self.x, self.y, self.width, self.height, label=self.label)
        new_box.z = self.z
        new_box.score = self.score
        new_box.visited = self.visited
        
        return new_box

    def corners(self):
        lowerX = self.x - self.width/2
        lowerY = self.y - self.height/2
        upperX = lowerX + self.width
        upperY = lowerY + self.height
        return (lowerX, lowerY), (upperX, upperY)

    def center(self):
        return (self.x, self.y)

    def is_inside(self, point):
        lower, upper = self.corners()
        return point[0] >= lower[0] and point[0] <= upper[0] and point[1] >= lower[1] and point[1] <= upper[1]

    def area(self):
        return self.width * self.height

    def intersection(self, other):
        lower, upper = self.corners()
        otherLower, otherUpper = other.corners()

        return segment_intersection(lower[0], upper[0], otherLower[0], otherUpper[0]) * segment_intersection(lower[1], upper[1], otherLower[1], otherUpper[1])

    def iou(self, other):
        intersection = self.intersection(other)
        union = self.area() + other.area() - intersection

        return intersection/union if intersection > 0 else float(union <= 0)

    # How much self overlaps other (self > other = 1)
    def overlap(self, other):
        intersection = self.intersection(other)
        otherArea = other.area()

        if otherArea <= 0:
            return self.is_inside(other.center())

        return intersection/otherArea

    def symmetric_overlap(self, other):
        intersection = self.intersection(other)
        otherArea = min(self.area(), other.area())

        if otherArea <= 0:
            return self.is_inside(other.center()) or other.is_inside(self.center())

        return intersection/otherArea
        #return max(self.overlap(other), other.overlap(self))

class Tumor:
    def __init__(self, boxes):
        boxes_by_z = dict()

        for box in boxes:
            #assert box.z not in boxes_by_z
            #boxes_by_z[box.z] = box
            boxes_by_z.setdefault(box.z, []).append(box)

        self.boxes_by_z = boxes_by_z

    def z_set(self):
        return set(self.boxes_by_z.keys())

    def iou(self, other):
        zRange = self.z_set()
        zRange = zRange.intersection(other.z_set())

        maxIou = 0.0

        for z in zRange:
            boxes = self[z]
            otherBoxes = other[z]

            for box in boxes:
                for otherBox in otherBoxes:
                    maxIou = max(box.iou(otherBox), maxIou)

        #for z in zRange:
        #    box = self[z]
        #    otherBox = other[z]
        #    maxIou = max(box.iou(otherBox), maxIou)

        return maxIou
    def unique_labels(self):
        labels = set()
        for boxes in self.boxes_by_z.values():
            for box in boxes:
                labels.add(box.label)
        return labels        

    def majority_label(self):
        label_counts = dict()
        
        for boxes in self.boxes_by_z.values():
            for box in boxes:
                label_counts[box.label] = label_counts.setdefault(box.label, 0) + 1


        return max(label_counts.items(), key=lambda x : x[1])[0]

    def __getitem__(self, z):
        return self.boxes_by_z[z]

def is_detected(gtTumor, detectedTumors, threshold=0.1):
    for tumor in detectedTumors:
        if tumor.iou(gtTumor) >= threshold:
            return True

    return False

def is_false_positive(detectedTumor, gtTumors, threshold=0.1):
    for tumor in gtTumors:
        if detectedTumor.iou(tumor) >= threshold:
            return False

    return True

def make_tumors_from_boxes2(all_boxes, threshold=0.1):
    boxesByZ = dict()

    for box in all_boxes:
        box.visited = False # XXX: Important, reset this!
        boxesByZ.setdefault(box.z, []).append(box)

    all_boxes = sorted(all_boxes, key=lambda b : b.area(), reverse=True)

    tumors = []

    for box in all_boxes:
        if box.visited:
            continue

        box.visited = True
        tumorBoxes = [ box ]
        i = 0

        while i < len(tumorBoxes):
            box = tumorBoxes[i]
            i += 1

            for z in [ box.z - 1, box.z + 1]:
                if z not in boxesByZ:
                    continue

                for otherBox in boxesByZ[z]:
                    if not otherBox.visited and box.symmetric_overlap(otherBox) >= threshold:
                        #print(box.symmetric_overlap(otherBox))
                        otherBox.visited=True
                        tumorBoxes.append(otherBox)

        tumors.append(Tumor(tumorBoxes))

    return tumors

def make_tumors_from_boxes(all_boxes, threshold=0.2):
    boxesByZ = dict()

    for box in all_boxes:
        box.visited = False # XXX: Important, reset this!
        boxesByZ.setdefault(box.z, []).append(box)

    tumors = []

    all_boxes = sorted(all_boxes, key=lambda b : b.z)

    for box in all_boxes:
        if box.visited:
            continue

        tumorBoxes = [ box ]
        z = box.z + 1 # Only need to search forward since we sorted
        box.visited = True

        while z in boxesByZ:
            overlappingBoxes = []

            for otherBox in boxesByZ[z]:
                if not otherBox.visited and box.symmetric_overlap(otherBox) > threshold:
                    overlappingBoxes.append(otherBox)

            #print(len(overlappingBoxes))
            if len(overlappingBoxes) == 0:
                break

            #overlappingBoxes.sort(key = lambda b : box.iou(b), reverse=True)
            box = max(overlappingBoxes, key=lambda b : box.iou(b)) # Pick the most similar box
            tumorBoxes.append(box)
            box.visited = True

            z += 1 # Next slice!

        tumors.append(Tumor(tumorBoxes))

    return tumors

def calculate_thresholds(allDetBoxes, decimals=3):
    allScores = []
    for boxes in allDetBoxes.values():
        scores = [ box.score for box in boxes ]
        allScores += scores
    
    allScores = np.unique(np.round(allScores, decimals=decimals))
    assert len(allScores) >= 1

    return 0.5*(allScores[:-1] + allScores[1:])

def prune_boxes(allDetBoxes, score_threshold):
    before_count = 0
    after_count = 0
    for acc, boxes in allDetBoxes.items():
        before_count += len(boxes)

        boxes = [ box for box in boxes if box.score >= score_threshold ]

        after_count += len(boxes)

        allDetBoxes[acc] = boxes

    return before_count, after_count

def load_all_boxes(detect_root, ground_truth_root):
    allDetBoxes = dict()
    allGtBoxes = dict()

    for imageDir in glob.glob(os.path.join(ground_truth_root, '0040-Subject-*',"0040-*")):

        if not os.path.isdir(imageDir):
            continue

        acc = os.path.basename(imageDir)

        detectDir = os.path.join(detect_root, acc)
        if not os.path.exists(detectDir):
            continue

        gtBoxes = []

        for boxFile in glob.glob(os.path.join(imageDir, "normalized_aligned*.txt")):
            
            tmp = re.search("[0-9]+\..*txt$", os.path.basename(boxFile))
            tmp = re.search("^[0-9]+", tmp.group(0))

            z = int(tmp.group(0))

            with open(boxFile, mode="rt", newline="") as f:
                boxes = [ Box.parse(line) for line in f if len(line.strip()) > 0 ]

            for box in boxes:
                box.z = z

            gtBoxes += boxes

        detBoxes = []

        for boxFile in glob.glob(os.path.join(detectDir, "normalized_aligned*.txt")):
            tmp = re.search("[0-9]+\..*txt$", os.path.basename(boxFile))
            tmp = re.search("^[0-9]+", tmp.group(0))

            z = int(tmp.group(0))

            with open(boxFile, mode="rt", newline="") as f:
                boxes = [ Box.parse(line) for line in f if len(line.strip()) > 0 ]

            for box in boxes:
                box.z = z

            detBoxes += boxes
                        


        print(f"Info: Loaded {acc} ...")
        print(f"Info: gt boxes = {len(gtBoxes)}, detected boxes = {len(detBoxes)}")
        allGtBoxes[acc] = gtBoxes
        allDetBoxes[acc] = detBoxes

    return allDetBoxes, allGtBoxes

def evaluate(detBoxes, gtBoxes, overlap_threshold=0.1, iou_threshold=0.1):
    detTumors = make_tumors_from_boxes2(detBoxes, threshold=overlap_threshold)
    gtTumors = make_tumors_from_boxes2(gtBoxes, threshold=overlap_threshold)

    #print(len(gtTumors))

    #exit()

    tpCount=0
    fnCount=0
    fpCount=0

    for gtTumor in gtTumors:
        if is_detected(gtTumor, detTumors, threshold=iou_threshold):
            tpCount += 1
        else:
            fnCount += 1
    for detTumor in detTumors:
        if is_false_positive(detTumor, gtTumors, threshold=iou_threshold):
            fpCount += 1

    #print(f"Info: Precision = {tpCount / (tpCount + fpCount):.3f}, Recall = {tpCount / (tpCount + fnCount):.3f}")

    return tpCount, fnCount, fpCount


def find_text_files(root_folder):
    text_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".txt"):
                text_files.append(os.path.join(root, file))
    return text_files
    
def relabel_boxes_by_majority(boxes, overlap_threshold=0.1):
    tumors = make_tumors_from_boxes2(boxes, threshold=overlap_threshold)

    new_boxes = [] # XXX: Not in correspondence

    for tumor in tumors:
        label = tumor.majority_label()
        
        for tmp_boxes in tumor.boxes_by_z.values():
            for box in tmp_boxes:
                box = box.copy()
                box.label = label
                new_boxes.append(box)
    
    assert len(new_boxes) == len(boxes)
    
    return new_boxes

def calculate_performance(allDetBoxes, allGtBoxes, score_threshold):
    total_tp = 0
    total_fn = 0
    total_fp = 0
    
    for acc in allDetBoxes:
        allDetBoxes[acc] = relabel_boxes_by_majority(allDetBoxes[acc])
    
    all_labels = set()
    for boxes in allGtBoxes.values():
        for box in boxes:
            all_labels.add(box.label)

    f1_scores = []
    average_precisions = []
    precisions_list = []  # Add this line
    recalls_list = []  # Add this line

    for label in all_labels:
        label_total_tp = 0
        label_total_fn = 0
        label_total_fp = 0

        for acc in allDetBoxes:
            detBoxes = [box for box in allDetBoxes[acc] if box.label == label]
            gtBoxes = [box for box in allGtBoxes[acc] if box.label == label]

            tp, fn, fp = evaluate(detBoxes, gtBoxes)

            label_total_tp += tp
            label_total_fn += fn
            label_total_fp += fp

        if (label_total_tp + label_total_fp) == 0:
            precision = 0
        else:
            precision = label_total_tp / (label_total_tp + label_total_fp)

        if (label_total_tp + label_total_fn) == 0:
            recall = 0
        else:
            recall = label_total_tp / (label_total_tp + label_total_fn)

        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1_score)
        precisions_list.append(precision)  # Add this line
        recalls_list.append(recall)  # Add this line

        ap = (recall * precision)
        average_precisions.append(ap)

        print(f"Label {label}: Precision = {precision:.3f}, Recall = {recall:.3f}, F1 score = {f1_score:.3f}, Average Precision = {ap:.3f}")

        total_tp += label_total_tp
        total_fn += label_total_fn
        total_fp += label_total_fp

    if (total_tp + total_fp) == 0:
        overall_precision = 0
    else:
        overall_precision = total_tp / (total_tp + total_fp)

    if (total_tp + total_fn) == 0:
        overall_recall = 0
    else:
        overall_recall = total_tp / (total_tp + total_fn)

    if (overall_precision + overall_recall) == 0:
        overall_f1_score = 0
    else:
        overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    
    overall_map = sum(average_precisions) / len(average_precisions) if len(average_precisions) > 0 else 0


    print(f"Overall precision = {overall_precision:.3f}, Overall recall = {overall_recall:.3f}, Overall F1 score = {overall_f1_score:.3f}, Overall mAP (0.5) = {overall_map:.3f}")

    return overall_precision, overall_recall, overall_f1_score, overall_map, f1_scores, average_precisions, precisions_list, recalls_list, all_labels



def plot_metrics(thresholds, metrics, title, xlabel='Threshold', ylabel='Value', output_folder='plots', best_precision=None, best_recall=None):
    plt.figure()
    for label, values in metrics.items():
        print(f"Debug: {label} - thresholds shape: {np.shape(thresholds)}, values shape: {np.shape(values)}")

        plt.plot(thresholds, values, label=label)

    if best_precision is not None and 'Precision' in title:
        plt.axhline(best_precision, color='r', linestyle='--', label=f'Best Precision: {best_precision:.3f}')
    
    if best_recall is not None and 'Recall' in title:
        plt.axhline(best_recall, color='g', linestyle='--', label=f'Best Recall: {best_recall:.3f}')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot to a file
    filename = os.path.join(output_folder, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename)



def main():
    for i in range (10): 

        #detect_root = "/data/AMPrj/yolov7-main/runs/test/detectnum1013/labels/"
        detect_root = f"/data/AMPrj/yolov7-main/runs/test/big{i+1}/labels/"
        print (detect_root)
        ground_truth_root = "/data/AMPrj/Yazdianp/YOLOCutDelDTumor/Images/"
        print (ground_truth_root)

        allDetBoxes, allGtBoxes = load_all_boxes(detect_root, ground_truth_root)
        score_thresholds = calculate_thresholds(allDetBoxes)

        print(f"number of thresholds = {len(score_thresholds)}")

        allPerf = []

        score_thresholds = calculate_thresholds(allDetBoxes)
        precisions = []
        recalls = []
        f1_scores = []

        for threshold in score_thresholds:
            # Prune boxes based on the current score threshold
            prunedDetBoxes = {acc: [box for box in boxes if box.score >= threshold] for acc, boxes in allDetBoxes.items()}
            precision, recall, overall_f1_score, _, label_f1_scores, average_precisions, precisions_list, recalls_list, all_labels = calculate_performance(prunedDetBoxes, allGtBoxes, threshold)

            perf = precision, recall, overall_f1_score, _, label_f1_scores, average_precisions, precisions_list, recalls_list, all_labels
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(overall_f1_score)
            with open(f'allPerf_{i+1}.txt', 'a') as f:
                f.write(f'{perf}\n')
        print(f"Debug: Length of precisions = {len(precisions)}")
        print(f"Debug: Length of recalls = {len(recalls)}")
        print(f"Debug: Length of f1_scores = {len(f1_scores)}")

        # Plot precision-recall
        plt.figure()
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve {i+1}')
    
        # Save the precision-recall curve plot
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/Precision-Recall_Curve {i+1}.png')
    
        # Plot F1-confidence
        plot_metrics(score_thresholds, {'F1 Score': f1_scores}, f'F1 Score {i+1}', xlabel='Confidence Threshold')

        best_f1_index = f1_scores.index(max(f1_scores))

        # Retrieve the corresponding confidence threshold, precision, and recall
        best_threshold = score_thresholds[best_f1_index]
        best_precision = precisions[best_f1_index]
        best_recall = recalls[best_f1_index]

        plot_metrics(score_thresholds, {'Precision': precisions, 'Recall': recalls, 'F1_score': f1_scores}, f'Metrics{i+1}')
        # Plot recall-confidence and precision-confidence
        plot_metrics(score_thresholds, {'Recall': recalls}, f'Recall {i+1}', xlabel='Confidence Threshold', best_recall=best_recall)
        plot_metrics(score_thresholds, {'Precision': precisions}, f'Precision {i+1}', xlabel='Confidence Threshold', best_precision=best_precision)

        # Save the best performance to a text file
        if not os.path.exists('best_performance'):
            os.makedirs('best_performance')
        with open(f'best_performance/best_performance_{i+1}.txt', 'w') as f:
            f.write(f'Best F1 Score: {max(f1_scores)}\n')
            f.write(f'Confidence Threshold: {best_threshold}\n')
            f.write(f'Precision: {best_precision}\n')
            f.write(f'Recall: {best_recall}\n')



if __name__ == "__main__":
    main()


