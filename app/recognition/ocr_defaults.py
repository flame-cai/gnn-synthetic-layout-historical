from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .dataset import AlignCollate, RawDataset
    from .model import Model
    from .utils import AttnLabelConverter, CTCLabelConverter
except ImportError:  # pragma: no cover - script execution fallback
    from dataset import AlignCollate, RawDataset
    from model import Model
    from utils import AttnLabelConverter, CTCLabelConverter


SANSKRIT_OCR_CHARACTER_SET = (
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ."
    "\u0901\u0902\u0903\u0905\u0905\u0902\u0905\u0903\u0906\u0907\u0908\u0909\u090a\u090b\u090f"
    "\u0910\u0911\u0913\u0914\u0915\u0916\u0917\u0918\u0919\u091a\u091b\u091c\u091d\u091e\u091f"
    "\u0920\u0921\u0922\u0923\u0924\u0925\u0926\u0927\u0928\u0929\u092a\u092b\u092c\u092d\u092e"
    "\u092f\u0930\u0931\u0932\u0933\u0935\u0936\u0937\u0938\u0939\u093c\u093e\u093f\u0940\u0941"
    "\u0942\u0943\u0945\u0947\u0948\u0949\u094b\u094c\u094d\u0950\u0952\u0958\u0959\u095a\u095b"
    "\u095c\u095d\u095e\u0960\u0964\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f"
    "\u0970"
)


def build_ocr_config(saved_model_path: str | Path | None = None, **overrides):
    base = {
        "saved_model": str(saved_model_path) if saved_model_path is not None else "",
        "Transformation": None,
        "FeatureExtraction": "ResNet",
        "SequenceModeling": "BiLSTM",
        "Prediction": "CTC",
        "batch_size": 1,
        "workers": 0,
        "batch_max_length": 250,
        "imgH": 50,
        "imgW": 2000,
        "rgb": False,
        "character": SANSKRIT_OCR_CHARACTER_SET,
        "sensitive": False,
        "PAD": True,
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 512,
        "baiduCTC": False,
    }
    base.update(overrides)

    if base.get("Transformation") == "None":
        base["Transformation"] = None
    if base.get("SequenceModeling") == "None":
        base["SequenceModeling"] = None
    if base.get("Prediction") == "None":
        base["Prediction"] = None
    if base.get("rgb"):
        base["input_channel"] = 3

    converter = build_label_converter(SimpleNamespace(**base))
    base["num_class"] = len(converter.character)
    return SimpleNamespace(**base)


def build_label_converter(opt):
    if "CTC" in str(opt.Prediction):
        return CTCLabelConverter(opt.character)
    return AttnLabelConverter(opt.character)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict):
    stripped = OrderedDict()
    for key, value in state_dict.items():
        stripped[key[7:] if key.startswith("module.") else key] = value
    return stripped


def _add_module_prefix(state_dict):
    prefixed = OrderedDict()
    for key, value in state_dict.items():
        prefixed[key if key.startswith("module.") else f"module.{key}"] = value
    return prefixed


def _unwrap_checkpoint_payload(payload):
    if isinstance(payload, dict):
        for candidate in ("state_dict", "model_state_dict", "model"):
            value = payload.get(candidate)
            if isinstance(value, OrderedDict):
                return value
    if isinstance(payload, OrderedDict):
        return payload
    return payload


def load_state_dict_compat(model, checkpoint_path: str | Path, map_location=None, strict=True):
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    state_dict = _unwrap_checkpoint_payload(checkpoint)
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)!r}")

    state_dict = OrderedDict(state_dict.items())
    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())
    if model_keys and state_keys:
        model_has_module = model_keys[0].startswith("module.")
        state_has_module = state_keys[0].startswith("module.")
        if state_has_module and not model_has_module:
            state_dict = _strip_module_prefix(state_dict)
        elif model_has_module and not state_has_module:
            state_dict = _add_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def create_model(opt, device=None, data_parallel=False):
    device = device or get_device()
    model = Model(opt)
    if data_parallel:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model


def load_inference_model(saved_model_path: str | Path, device=None, **overrides):
    device = device or get_device()
    opt = build_ocr_config(saved_model_path=saved_model_path, **overrides)
    model = create_model(opt, device=device, data_parallel=False)
    load_state_dict_compat(model, opt.saved_model, map_location=device, strict=True)
    model.eval()
    converter = build_label_converter(opt)
    return model, converter, opt, device


def run_line_image_inference_from_loaded_model(image_root: str | Path, model, converter, opt, device):
    dataset = RawDataset(root=str(image_root), opt=opt)
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=align_collate,
        pin_memory=True,
    )

    predictions = []
    with torch.no_grad():
        for image_tensors, image_paths in data_loader:
            image = image_tensors.to(device)
            batch_size = image.size(0)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if "CTC" in str(opt.Prediction):
                preds = model(image, text_for_pred, is_train=False)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_path, pred_text, pred_max_prob in zip(image_paths, preds_str, preds_max_prob):
                if "Attn" in str(opt.Prediction):
                    pred_eos = pred_text.find("[s]")
                    pred_text = pred_text[:pred_eos]
                    pred_max_prob = pred_max_prob[:pred_eos]

                try:
                    confidence = float(pred_max_prob.cumprod(dim=0)[-1])
                except Exception:
                    confidence = 0.0

                predictions.append(
                    {
                        "image_path": img_path,
                        "predicted_label": pred_text,
                        "confidence_score": confidence,
                    }
                )

    return predictions


def run_line_image_inference(image_root: str | Path, saved_model_path: str | Path, device=None, **overrides):
    model, converter, opt, device = load_inference_model(saved_model_path, device=device, **overrides)
    return run_line_image_inference_from_loaded_model(image_root, model, converter, opt, device)
