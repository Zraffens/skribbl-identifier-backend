from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_drawing_bytes

app = FastAPI(title="Skribbl-ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "model": "SkribblCNN-v1"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_drawing_bytes(image_bytes)
    return result


# for live guessing while user is drawing
@app.websocket("/ws/draw")
async def websocket_drawing(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        result = predict_drawing_bytes(data)
        await websocket.send_json(result)