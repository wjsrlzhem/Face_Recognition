import dlib, cv2
import numpy as np


file_dir = 'drive-download-20210102T085952Z-001'
            
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(file_dir+'/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(file_dir+'/models/dlib_face_recognition_resnet_model_v1.dat')

descs = np.load(file_dir+'/img/descs.npy', allow_pickle=True)[()]

def encode_face(img):
  dets = detector(img, 1)

  if len(dets) == 0:
    return np.empty(0)

  for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)

video_path = file_dir+'/img/kang_vi.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  exit()

_, img_bgr = cap.read() # (800, 900, 3)
#print("PRINT !! ~~~~~   img_bgr  = {}".format(img_bgr))
padding_size = 0
resized_width = 900

video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
writer = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

while True:
  ret, img_bgr = cap.read()
  if not ret:
    break
    
  img_bgr = cv2.resize(img_bgr, video_size)
  img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE) # 이미지 돌리기

  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  img_bgr = cv2.copyMakeBorder(img_bgr, top=padding_size, bottom=padding_size, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
  
  dets = detector(img_bgr, 1)

  for k, d in enumerate(dets):
    shape = sp(img_rgb, d)
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

    last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

    for name, saved_desc in descs.items():
      dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)
      #print("PRINT !! ~~~~~  = {}{}".format(name,saved_desc))

      if dist < last_found['dist']:
        last_found = {'name': "dict", 'dist': dist, 'color': (255,255,255)}

    cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
    cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)

  writer.write(img_bgr)

  cv2.imshow('img', img_bgr)
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
writer.release()
