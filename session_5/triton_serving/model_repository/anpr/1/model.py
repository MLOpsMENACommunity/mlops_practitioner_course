"""Server-side ANPR pipeline for Triton's Python backend (BLS).

numpy only — the stock tritonserver image ships numpy and nothing else we need, so this
model loads without building a custom execution environment.

    frame [720,1280,3] uint8 -> letterbox -> detector_nms -> crop -> ocr -> CTC decode -> plates
"""

import numpy as np
import triton_python_backend_utils as pb_utils

DET_W, DET_H, OCR_W, OCR_H = 640, 384, 128, 32
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SCORE_THRESHOLD, MARGIN = 0.35, 0.08


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    scale = min(DET_W / w, DET_H / h)
    nw, nh = round(w * scale), round(h * scale)
    if w == 2 * nw and h == 2 * nh:  # exact factor 2: a box average, identical to PIL's Image.reduce(2)
        small = frame.reshape(nh, 2, nw, 2, 3).mean(axis=(1, 3))
    else:
        small = frame[(np.arange(nh) / scale).astype(int)][:, (np.arange(nw) / scale).astype(int)]
    canvas = np.full((DET_H, DET_W, 3), 114.0, np.float32)
    px, py = (DET_W - nw) // 2, (DET_H - nh) // 2
    canvas[py : py + nh, px : px + nw] = small
    return (canvas.transpose(2, 0, 1)[None] / 255.0).astype(np.float32), scale, px, py


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape
    ys = np.clip((np.arange(out_h) + 0.5) * h / out_h - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * w / out_w - 0.5, 0, w - 1)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bottom = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def crops_for(frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
    gray = frame @ np.array([0.299, 0.587, 0.114], np.float32)
    out = []
    for x0, y0, x1, y1 in dets[:, :4]:
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        a, b = max(0, int(y0 - my)), min(frame.shape[0], int(y1 + my) + 1)
        c, d = max(0, int(x0 - mx)), min(frame.shape[1], int(x1 + mx) + 1)
        patch = gray[a:b, c:d] if b - a >= 4 and d - c >= 4 else np.zeros((4, 4), np.float32)
        out.append(resize_bilinear(patch, OCR_H, OCR_W))
    return (np.stack(out)[:, None] / 255.0).astype(np.float32)


def ctc_decode(logits: np.ndarray) -> list[str]:
    texts = []
    for path in logits.argmax(-1):
        chars = [c for t, c in enumerate(path) if c != 0 and (t == 0 or c != path[t - 1])]
        texts.append("".join(CHARSET[c - 1] for c in chars))
    return texts


def call(model: str, name: str, array: np.ndarray, output: str) -> np.ndarray:
    """One BLS request to another model in this server. Errors surface; they are never swallowed."""
    request = pb_utils.InferenceRequest(model_name=model, requested_output_names=[output], inputs=[pb_utils.Tensor(name, array)])
    response = request.exec()
    if response.has_error():
        raise pb_utils.TritonModelException(response.error().message())
    return pb_utils.get_output_tensor_by_name(response, output).as_numpy()


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            frame = pb_utils.get_input_tensor_by_name(request, "frame").as_numpy().astype(np.float32)
            images, scale, px, py = letterbox(frame)
            dets = call("detector_nms", "images", images, "detections")[0]
            dets = dets[dets[:, 4] >= SCORE_THRESHOLD]
            dets[:, [0, 2]] = (dets[:, [0, 2]] - px) / scale  # letterboxed pixels -> frame pixels
            dets[:, [1, 3]] = (dets[:, [1, 3]] - py) / scale
            texts = ctc_decode(call("ocr", "crops", crops_for(frame, dets), "logits")) if len(dets) else []
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("detections", dets.astype(np.float32)),
                pb_utils.Tensor("plates", np.array([t.encode() for t in texts], dtype=np.object_)),
            ]))  # fmt: skip
        return responses
