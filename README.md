# AI CCTV

`index.html` = puro web app (camera / video file / demo, zones, counting, graph, validate, REC).
`server/` = optional Python (YOLOv8 + ByteTrack) version, PC-te chole.

## GitHub + Vercel (online korar niyom)
1. github.com -> New repository -> name: `ai-cctv` -> Create.
2. "uploading an existing file" -> `index.html`, `vercel.json`, `README.md`, `.gitignore`, `server` folder upload -> Commit.
3. vercel.com -> Sign up with GitHub -> Add New -> Project -> `ai-cctv` -> Import -> Deploy (kono setting change lagbe na).
4. Phone-e `https://ai-cctv-xxxx.vercel.app` khulo. HTTPS thake bole camera chole.

## Python server
pip install -r server/requirements.txt
python server/cctv_server.py http://PHONE_IP:8080/video   (IP Webcam app)   |  0 (webcam)  |  video.mp4
Phone browser: http://PC_IP:5000
