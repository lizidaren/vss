from groundingdino.util.inference import load_model, load_image, predict, annotate
from PIL import Image
import pickle
import time
import cv2

import timm 
import torch
import torch.nn as nn
from timm.data.transforms_factory import create_transform
import numpy as np

import multiprocessing
import requests

import traceback

LINK = "http://your-camera-image.cn"

def dino_inference(IMAGE_PATH, model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD):
    image_source, image = load_image(IMAGE_PATH)

    # 框、置信度、标签
    # 框为yolo格式：中心x、中心y、长、宽
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    return [boxes,logits,phrases]

# print(logits, phrases)

def decode_pos(x, width, height):
    x = [int(i) for i in [x[0]*width, x[1]*height, x[2]*width, x[3]*height]]
    return [int(x[0]-x[2]*0.5), int(x[1]-x[3]*0.5), int(x[0]+x[2]*0.5), int(x[1]+x[3]*0.5)]

def crop_img(img, bbox_x0, bbox_y0, x1, y1):
    if isinstance(img, str):
        img = cv2.imread(img)
    result = img[bbox_y0:y1, bbox_x0:x1]
    return result

def cos_sim(vector_a, vector_b):
    return nn.functional.cosine_similarity(vector_a,vector_b)

def get_features(img,model):
    pool_layer = nn.AdaptiveAvgPool2d(1)
    return pool_layer(model.forward_features(img))

def postprocess(IMAGE_PATH, base_image, boxes, logits, phrases, transforms, model_rn, THRESHOLD):
    img=cv2.imread(IMAGE_PATH)

    width = img.shape[1]
    height= img.shape[0]

    # import matplotlib.pyplot as plt
    pil_images = []
    for i,x in enumerate(boxes.numpy()):
        pos = decode_pos(x, width, height)
        # cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0,0,255), 10)
        res = crop_img(img, *pos)
        # print(f"置信度{logits[i]}，标签{phrases[i]}")
        result = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(result).convert('RGB')
        # img2.save(f"bottle{i}.png")
        pil_images.append(transforms(img2).unsqueeze(0))

    results = {}

    # 简易版实现
    vector = get_features(torch.cat(pil_images).cuda(),model_rn)
    base_image_vector = get_features(transforms(base_image).unsqueeze(0).cuda(),model_rn)
    sim = cos_sim(vector, base_image_vector)
    result = sim.detach().cpu().numpy()
    result = result.reshape([result.size])
    print("相似度",result)
    
    possible = []
    result_l = list(result)
    max_res = result_l.index(max(result_l))
    print(f"{max_res}, {result_l[max_res]}  > {THRESHOLD}")
    if result_l[max_res] >= THRESHOLD:
        possible.append(max_res)

    print("后处理回传possible", possible)
    return possible

    
def worker(queue, name, base_image_path, prompt, PROCESS_NUM, THRESHOLD = 0.65):
    time.sleep(2*int(name))
    
    base_image = Image.open(base_image_path).convert('RGB')
    model_rn = timm.create_model('resnet34',pretrained=True).eval().cuda()
    transforms = create_transform(
        input_size=(3,224,224),
        is_training=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229,0.224,0.225)
    )
    
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    # IMAGE_PATH = "desk.jpg"
    if "." not in prompt:
        TEXT_PROMPT = prompt + " ."
    else:
        TEXT_PROMPT = prompt
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    while True:
        #print(f"[Worker {name}]", "Get frame")
        frame_name = f"worker{name}_tmp.jpg"
        with open(frame_name, "wb") as f:
            f.write(requests.get(LINK).content)
        
        
        #print(f"[Worker {name}]", "Inference")
        boxes,logits,phrases = dino_inference(frame_name, model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)
        
        #print(f"[Worker {name}]", "Postprocess")
        try:
            result = postprocess(frame_name, base_image, boxes, logits, phrases, transforms, model_rn, THRESHOLD)
        except:
            str1 = traceback.format_exc()
            if "NotIm" in str1 or "empty" in str1:
                continue
            else:
                raise
        #print(f"[Worker {name}]", "Result", result)
        useful_boxes = [boxes[i] for i in result]
        print(f"[Worker {name}]", "Useful", useful_boxes)
        PUSH_LINK = f"https://your-vss-api.cn/ai_push/v1/object?time={int(time.time())}&bbox={pickle.dumps(useful_boxes)}"
        
        #print(requests.get(PUSH_LINK))
        
        #print(f"[Worker {name}]", "Queue Put")
        
        img_tmp = cv2.imread(frame_name)
        width = img_tmp.shape[1]
        height= img_tmp.shape[0]
        for box in useful_boxes:
            pos = decode_pos(box, width, height)
            cv2.rectangle(img_tmp, (pos[0], pos[1]), (pos[2], pos[3]), (0,0,255), 10)
        
        if queue[0].qsize() != PROCESS_NUM:
            queue[0].put(img_tmp)
        
        # 结束监测
        try:
            res = queue[1].get(False)
        except:
            if "Empty" in traceback.format_exc():
                continue
            else:
                raise
        print("signal detected", res, "!")
        if "STOP" in res:
            print("STOPPING!")
            queue[1].put("OK")
            time.sleep(1.5)
            prompt = queue[1].get()
        print("resume")


        
        
if __name__ == "__main__":
    # 进程数量在此修改
    PROCESS_NUM = 3
    
    base_img_path = "base.jpg"
    
    queue = [multiprocessing.Queue(maxsize=PROCESS_NUM), multiprocessing.Queue(maxsize=PROCESS_NUM)]
    
    with open("prompt.txt") as f:
        prompt = f.read()
    
    wait_data = False
    
    processes = []
    for i in range(PROCESS_NUM):
        processes.append(multiprocessing.Process(target = worker, name = f"worker-{i}", args=(queue,str(i),base_img_path, prompt, PROCESS_NUM)))
        processes[-1].start()

    from flask import Flask, make_response, request
    
    app = Flask("gd_deploy")
    
    @app.route("/")
    def index():
        global queue, wait_data
        if wait_data:
            img_encode = cv2.imencode('.jpg', cv2.imread("loading.jpg"))[1]
            data_encode = np.array(img_encode)
            frame_encode = data_encode.tobytes()

            resp = make_response(frame_encode)
            resp.headers['Content-Type'] = "image/jpeg"
            return resp
        try:
            frame = queue[0].get(True, None)
            img_encode = cv2.imencode('.jpg', frame)[1]
            data_encode = np.array(img_encode)
            frame_encode = data_encode.tobytes()

            resp = make_response(frame_encode)
            resp.headers['Content-Type'] = "image/jpeg"
            return resp
        except cv2.error:
            img_encode = cv2.imencode('.jpg', cv2.imread("loading.jpg"))[1]
            data_encode = np.array(img_encode)
            frame_encode = data_encode.tobytes()

            resp = make_response(frame_encode)
            resp.headers['Content-Type'] = "image/jpeg"
            return resp
    
    @app.route("/push")
    def get_push():
        print("Reloading...")
        with open("base.jpg", "wb") as f:
            f.write(requests.get("https://your-vss-api.cn/static/prompt.jpg").content)
        prompt = request.args.get("prompt", None)
        if prompt is None:
            return "Error"
        
        with open("prompt.txt", "w") as f:
            f.write(prompt)

        global processes, queue, base_image_path, wait_data
        wait_data = True
        time.sleep(1)
        for i in range(PROCESS_NUM):
            processes[i].terminate()
        time.sleep(1)
        processes = []
        queue = [multiprocessing.Queue(maxsize=PROCESS_NUM), multiprocessing.Queue(maxsize=PROCESS_NUM)]
        for i in range(PROCESS_NUM):
            processes.append(multiprocessing.Process(target = worker, name = f"worker-{i}", args=(queue,str(i),base_img_path, prompt, PROCESS_NUM)))
            processes[-1].start()
        print("Reload: ok")
        wait_data = False
        return "OK"
            
        

    app.run("0.0.0.0", 6006)