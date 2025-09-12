import cv2, os, string, csv, argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

class PlateDetector:
    def __init__(self, model_path, num_classes=2, device="cpu"):
        self.device = torch.device(device)
        self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval().to(self.device)

        # Transform input image
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def load_image(self, img_path, resize=(480, 480)):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if resize is not None:
            img_rgb = cv2.resize(img_rgb, (int(resize[0]), int(resize[1])))

        img_tensor = self.transform(img_rgb).to(self.device)  # (C,H,W), float [0,1]
        return img_tensor, img_rgb

    def predict(self, img_tensor):
        """Trả về raw prediction (tensors trên cpu)"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model([img_tensor])[0]
        # chuyển tất cả tensor về cpu để xử lý tiếp
        pred_cpu = {k: v.cpu() for k, v in pred.items()}
        return pred_cpu

    def apply_nms(self, orig_prediction, iou_thresh=0.1, score_thresh=0.7):
        # Lấy index NMS
        keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

        # Giữ lại boxes có score cao hơn ngưỡng
        keep = keep[orig_prediction['scores'][keep] >= score_thresh]

        final_prediction = {}
        final_prediction['boxes'] = orig_prediction['boxes'][keep]
        final_prediction['scores'] = orig_prediction['scores'][keep]
        final_prediction['labels'] = orig_prediction['labels'][keep]

        return final_prediction

    def crop_boxes(self, img_rgb, prediction):
        """Crop ảnh RGB theo danh sách box (prediction trên CPU). Trả về list numpy arrays (RGB)."""
        crops = []
        h, w = img_rgb.shape[:2]
        for box in prediction['boxes']:
            x1, y1, x2, y2 = box.int().tolist()
            # clamp to image bounds
            x1 = max(0, min(w-1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h-1, y1))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                crops.append(img_rgb[y1:y2, x1:x2])
        return crops

    def plot_predictions(self, img_rgb, prediction):
        img_draw = img_rgb.copy()
        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_draw, f"{int(label)}:{float(score):.2f}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        return img_draw

class PlateNumberExtractor:
    def __init__(self, image_path, model):
        self.image_path = image_path
        self.model = model
        self.img = cv2.imread(image_path)
        self.CLASS_NAMES = [str(i) for i in range(10)] + list(string.ascii_uppercase)
        self.plate_as = None

    def extract_numbers(self, img):
        #Convert gray image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_blur = cv2.blur(img_gray, (3, 3))
        img_canny = cv2.Canny(img_blur, 100, 200)
        img_design = cv2.addWeighted(img_gray, 0.7, img_canny, 0.3, 0)
        img_blur2 = cv2.blur(img_design, (3, 3))
        _, img_th = cv2.threshold(img_blur2, 127, 255, cv2.THRESH_BINARY)

        # Give contour from image
        contour, hierachy = cv2.findContours(img_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #Check size of contour
        area_cnt = [cv2.contourArea(cnt) for cnt in contour]
        area_sort = np.argsort(area_cnt)[::-1]

        #Position of number plate in image (which contour largest)
        np_list = contour[area_sort[0]]
        np_list = np.array(np_list)
        pos_x = []      #Give x
        pos_y = []      #Give y
        for i in range(np_list.shape[0]):
            tamthoi = np_list[i]
            pos_x.append(tamthoi[0][0])
            pos_y.append(tamthoi[0][1])

        pos_x_soft = np.sort(pos_x)
        pos_y_soft = np.sort(pos_y)

        first_reach = pos_x_soft[0]
        second_reach = pos_x_soft[-1]
        third_reach = pos_y_soft[0]
        four_reach = pos_y_soft[-1]

        #Convert and comparison which each point for Perspective Transform
        #Point 1 (Leftest)
        save1 = self.img.shape[1]
        for i in range(len(np_list)):
            tamthoi = np_list[i]
            if tamthoi[0][0] == first_reach:
                choose = tamthoi[0][1]
                if save1 >= choose and (np.abs(choose - third_reach)+np.abs(choose-four_reach)>=50):
                    save1 = choose

        #Point 2 (Rightest)
        save2 = 0
        for i in range(len(np_list)):
            tamthoi = np_list[i]
            if np.abs(tamthoi[0][0] - second_reach) <= 3:
                choose = tamthoi[0][1]
                if save2 <= choose:
                    save2 = choose

        #Point 3 (Highest)
        save3 = 0
        for i in range(len(np_list)):
            tamthoi = np_list[i]
            if np.abs(tamthoi[0][1] - third_reach) <= 3:
                choose = tamthoi[0][0]
                if save3 <= choose:
                    save3 = choose

        #Point 4 (Lowest)
        save4 = self.img.shape[1]
        for i in range(len(np_list)):
            tamthoi = np_list[i]
            if tamthoi[0][1] == four_reach:
                choose = tamthoi[0][0]
                if save4 >= choose:
                    save4 = choose


        # Perspective Transform
        if save1 < save2 and np.abs(save1-third_reach) + np.abs(first_reach-save3) >= 50: in_pts = np.float32([[first_reach, save1], [save3, third_reach], [save4, four_reach],[second_reach, save2]])        # 1-3-4-2
        elif save2 < save1 and np.abs(save1-third_reach) + np.abs(first_reach-save3) >= 50: in_pts = np.float32([[save3, third_reach], [second_reach, save2], [first_reach, save1], [save4, four_reach]])    # 3-2-1-4
        #Special condition when the the object is almost not tilted make 1 similar 3, 2 similar 4 (Give 1 and 2)
        elif np.abs(save1-third_reach) + np.abs(first_reach-save3) <= 50: in_pts = np.float32([[first_reach, save1], [second_reach, save1], [first_reach, save2], [second_reach, save2]])
        out_pts = np.float32([[0, 0], [160, 0],
                                [0, 160], [160, 160]])

        M = cv2.getPerspectiveTransform(in_pts, out_pts)
        result = cv2.warpPerspective(img, M, (160, 160))

        self.plate_as = result

        # tách ký tự
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        _, result_thresholding = cv2.threshold(result_gray, 150, 255, cv2.THRESH_BINARY_INV)
        contour2, hierachy2 = cv2.findContours(result_thresholding.copy(),
                                            cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
        #Give contour of each number
        each_nb = []
        for i in range(len(contour2)):
            if hierachy2[0][i][3] == -1:
                x1, y1, w, h = cv2.boundingRect(contour2[i])

                #Because the number in plate always longer than height/3 and need suitabel with width
                if h >= (result.shape[0]/2.5):
                    #Add position into each number for next step
                    each_nb.append((x1, y1, w, h))
                    # #Check position of each number (don't need in real code)
                    # a = cv2.rectangle(test, (x1, y1), (x1+w, y1+h), color=[255, 0, 0], thickness=2)

        # sắp xếp ký tự
        final = []

        y_center = [y for (_, y, _, _) in each_nb]
        if max(y_center) - min(y_center) < result.shape[0] * 0.3:   # biển 1 tầng
            x_center = np.sort([x for (x, _, _, _) in each_nb])
            for x_val in x_center:
                for (x, y, w, h) in each_nb:
                    if x == x_val:
                        final.append((x, y, w, h))
        else:  # biển 2 tầng
            on_Top = np.sort([x for (x, y, w, h) in each_nb if y < result.shape[0] / 2])
            on_Bot = np.sort([x for (x, y, w, h) in each_nb if y >= result.shape[0] / 2])
            for x_val in on_Top:
                for (x, y, w, h) in each_nb:
                    if y < result.shape[0] / 2 and x == x_val:
                        final.append((x, y, w, h))
            for x_val in on_Bot:
                for (x, y, w, h) in each_nb:
                    if y >= result.shape[0] / 2 and x == x_val:
                        final.append((x, y, w, h))

        char_images = [result[y:y+h, x:x+w] for (x, y, w, h) in final]
        return char_images
    
    def predict_characters(self):
        char_images = self.extract_numbers(self.img)
        results = []

        for char_img in char_images:
            # Tien xu ly anh
            img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(img_gray, (64, 64))
            _, img_th = cv2.threshold(img_resize, 120, 255, cv2.THRESH_BINARY_INV)
            img_ivt = 255 - img_th
            inp = img_ivt / 255.0
            inp_reshaped = inp.reshape((1, 64, 64, 1))

            # ---- Dự đoán bằng model ----
            outp = self.model.predict(inp_reshaped)
            class_id = np.argmax(outp, axis=1)[0]

            results.append(self.CLASS_NAMES[class_id])

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biển số xe detector + recognizer")
    parser.add_argument("--input_dir", type=str, default="./data/plate", help="Thư mục chứa ảnh input")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục lưu kết quả")
    parser.add_argument("--detector_model", type=str, default="./models/fastRCNN_for_plate.pth", help="Path tới model FasterRCNN (.pth)")
    parser.add_argument("--char_model", type=str, default="./models/LandD_last_model.h5", help="Path tới model nhận diện ký tự (.h5)")
    args = parser.parse_args()

    # --- Khởi tạo detector ---
    detector = PlateDetector(
        model_path=args.detector_model,
        num_classes=2,
        device="cpu"
    )

    # --- Load model ký tự ---
    char_model = tf.keras.models.load_model(args.char_model)
    extractor = PlateNumberExtractor("", char_model)

    # --- Tạo output dir ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- CSV lưu kết quả ---
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["STT", "Tên ảnh", "Biển số"])

        # --- Loop qua tất cả ảnh ---
        for idx, fname in enumerate(sorted(os.listdir(args.input_dir))):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(args.input_dir, fname)
            print(f"[INFO] Đang xử lý ảnh: {img_path}")

            # 1. Detect plate
            img_tensor, img_rgb = detector.load_image(img_path)
            raw_pred = detector.predict(img_tensor)
            final_pred = detector.apply_nms(raw_pred, iou_thresh=0.05)

            # 2. Crop biển số (lấy box có score cao nhất)
            crops = detector.crop_boxes(img_rgb, final_pred)
            if len(crops) == 0:
                print("Không tìm thấy biển số trong ảnh.")
                continue
            plate_img = crops[0]

            # 3. Extract + warpPerspective bằng PlateNumberExtractor
            extractor.img = plate_img
            extractor.extract_numbers(plate_img)
            plate_warp = extractor.plate_as

            # 4. Lưu ảnh warpPerspective
            plate_name = f"plate_as_{idx}.png"
            plate_path = os.path.join(args.output_dir, plate_name)
            plt.imsave(plate_path, plate_warp, cmap="gray")

            # 4. Nhận diện ký tự
            extractor.img = plate_img
            chars = extractor.predict_characters()
            plate_number = "".join(chars)

            # 5. Ghi kết quả vào CSV
            writer.writerow([idx, plate_name, plate_number])

            print(f"Biển số đoán: {plate_number}")

    print(f"\n Hoàn tất! Kết quả được lưu tại: {csv_path}")