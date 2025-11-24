import imageio
from ultralytics import YOLO
import cv2

dataset = cv2.VideoCapture("dataset/05.mp4")
model = YOLO("yolo11x-pose.pt")
draw_pose_flag = True
draw_boxes_flag = True
display_status_counts_flag = True

success, first_frame = dataset.read()
dataset.set(cv2.CAP_PROP_POS_FRAMES, 0)
h, w = first_frame.shape[:2]
aspect_ratio = w / h
new_h = int(aspect_ratio * 640)
new_w = 640
writer = imageio.get_writer("output/yolov11_pose_estimator_detected.mp4", fps=30, codec='libx264', quality=8)
save_demo_frame = True
frame_count = 0

def draw_pose(image, keypoints_xy, keypoints_conf, draw_pose_flag):

    joint_labels = {
    5: "L Shoulder", 
    6: "R Shoulder", 
    7: "L Elbow", 
    8: "R Elbow", 
    9: "L Wrist", 
    10: "R Wrist", 
    11: "L Hip", 
    12: "R Hip", 
    13: "L Knee", 
    14: "R Knee", 
    15: "L Ankle", 
    16: "R Ankle"
    }
    skeleton = [
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
    ]
    person_status_details = []


    for kpts, conf in zip(keypoints_xy, keypoints_conf):
        kpts = kpts.cpu().numpy()
        conf = conf.cpu().numpy()
        person_is_disabled = False
        missing_limbs = []


        for i, (x, y) in enumerate(kpts):

            if conf[i] > 0.7 and i in joint_labels:

                if draw_pose_flag:
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                    label = joint_labels[i]
                    text_pos = (int(x) + 7, int(y) - 5)
                    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if conf[i] < 0.7 and i in joint_labels:
                person_is_disabled = True
                missing_limbs.append(joint_labels[i])

        for (start, end) in skeleton:
            if conf[start] > 0.7 and conf[end] > 0.7 and draw_pose_flag:
                pt1 = (int(kpts[start][0]), int(kpts[start][1]))
                pt2 = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        person_status_details.append((person_is_disabled, missing_limbs))
    return person_status_details

def draw_boxes(image, boxes, person_status_details):

    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        is_disabled, missing_limbs = person_status_details[box_idx]
        
        if is_disabled:
            missing_text = f"Missing: {', '.join(missing_limbs)}"
            
            cv2.putText(image, missing_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            label = f"Disabled"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            label = f"Healthy"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def display_status_counts(image, healthy_count, disabled_count):
    
    healthy_text = f"Healthy People: {healthy_count}"
    disabled_text = f"Disabled People: {disabled_count}"
    
    max_width = max(len(healthy_text), len(disabled_text)) * 14
    cv2.rectangle(image, (5, 5), (max_width, 70), (0, 0, 0), -1)
    
    cv2.putText(image, healthy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, disabled_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

while True:
    ret, frame = dataset.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (new_h,new_w))
    results = model(frame, classes=[0])
    keypoints_xy = results[0].keypoints.xy
    keypoints_conf = results[0].keypoints.conf
    person_status_details = []
    
    disabled_person_flag = False
    person_status_details = draw_pose(frame, keypoints_xy, keypoints_conf, draw_pose_flag)

    if draw_boxes_flag == True:
        draw_boxes(frame, results[0].boxes, person_status_details)
    
    if display_status_counts_flag == True:

        disabled_people_count = sum(1 for is_disabled, _ in person_status_details if is_disabled)
        healthy_people_count = len(person_status_details) - disabled_people_count
        
        display_status_counts(frame, healthy_people_count, disabled_people_count)

    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow("Input", frame)
    if cv2.waitKey(1) == 13:
        dataset.release()
        writer.close()
        cv2.destroyAllWindows()
        break

dataset.release()
writer.close()
cv2.destroyAllWindows()