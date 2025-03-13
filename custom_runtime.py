import base64
import json
from mlserver import MLModel, types
from ultralytics import YOLO
import numpy as np
import cv2

class WeedDetectionRuntime(MLModel):

    async def load(self) -> bool:
        model_path = self.settings.parameters.uri
        self.model = YOLO(model_path)
        self.ready = True
        return self.ready

    async def predict(self, payload) -> types.InferenceResponse:
        encoded_image = payload.inputs[0].data[0]
        image_data = base64.b64decode(encoded_image)
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        results = self.model.predict(img)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = [{"annotated_image_base64": img_base64}]

        return types.InferenceResponse(
            model_name=self.name,
            outputs=[
                types.ResponseOutput(
                    name="annotated_image",
                    shape=[1],
                    datatype="BYTES",
                    data=[json.dumps(predictions)]
                )
            ]
        )
