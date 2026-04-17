from __future__ import annotations

import argparse
import os
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from nltk.metrics.distance import edit_distance

try:
    from .dataset import AlignCollate, hierarchical_dataset
    from .model import Model
    from .ocr_defaults import SANSKRIT_OCR_CHARACTER_SET, get_device, load_state_dict_compat
    from .utils import AttnLabelConverter, Averager, CTCLabelConverter
except ImportError:  # pragma: no cover - script execution fallback
    from dataset import AlignCollate, hierarchical_dataset
    from model import Model
    from ocr_defaults import SANSKRIT_OCR_CHARACTER_SET, get_device, load_state_dict_compat
    from utils import AttnLabelConverter, Averager, CTCLabelConverter


DEVICE = get_device()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    eval_data_list = [
        "IIIT5k_3000",
        "SVT",
        "IC03_860",
        "IC03_867",
        "IC13_857",
        "IC13_1015",
        "IC15_1811",
        "IC15_2077",
        "SVTP",
        "CUTE80",
    ]

    evaluation_batch_size = 1 if calculate_infer_time else opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    result_dir = Path(getattr(opt, "result_dir", Path("result") / opt.exp_name))
    result_dir.mkdir(parents=True, exist_ok=True)
    log = (result_dir / "log_all_evaluation.txt").open("a", encoding="utf-8")
    dashed_line = "-" * 80
    print(dashed_line)
    log.write(dashed_line + "\n")
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        align_collate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_dataset, eval_dataset_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=align_collate_evaluation,
            pin_memory=True,
        )

        _, accuracy_by_best_model, norm_ed_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model,
            criterion,
            evaluation_loader,
            converter,
            opt,
        )
        list_accuracy.append(f"{accuracy_by_best_model:0.3f}")
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_dataset)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_dataset_log)
        print(f"Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ed_by_best_model:0.3f}")
        log.write(f"Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ed_by_best_model:0.3f}\n")
        print(dashed_line)
        log.write(dashed_line + "\n")

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum(np.prod(parameter.size()) for parameter in model.parameters())

    evaluation_log = "accuracy: "
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f"{name}: {accuracy}\t"
    evaluation_log += (
        f"total_accuracy: {total_accuracy:0.3f}\t"
        f"averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}"
    )
    print(evaluation_log)
    log.write(evaluation_log + "\n")
    log.close()


def validation(model, criterion, evaluation_loader, converter, opt):
    n_correct = 0
    norm_ed = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for image_tensors, labels in evaluation_loader:
        batch_size = image_tensors.size(0)
        length_of_data += batch_size
        image = image_tensors.to(DEVICE)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(DEVICE)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(DEVICE)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if "CTC" in str(opt.Prediction):
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, : text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if "Attn" in str(opt.Prediction):
                gt = gt[: gt.find("[s]")]
                pred_eos = pred.find("[s]")
                pred = pred[:pred_eos]
                pred_max_prob = pred_max_prob[:pred_eos]

            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = "0123456789abcdefghijklmnopqrstuvwxyz"
                out_of_alphanumeric_case_insensitve = f"[^{alphanumeric_case_insensitve}]"
                pred = re.sub(out_of_alphanumeric_case_insensitve, "", pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, "", gt)

            if pred == gt:
                n_correct += 1

            if len(gt) == 0 or len(pred) == 0:
                norm_ed += 0
            elif len(gt) > len(pred):
                norm_ed += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ed += 1 - edit_distance(pred, gt) / len(pred)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except Exception:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ed = norm_ed / float(length_of_data)
    return valid_loss_avg.val(), accuracy, norm_ed, preds_str, confidence_score_list, labels, infer_time, length_of_data


def test(opt):
    if "CTC" in str(opt.Prediction):
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    opt.Transformation = None if opt.Transformation == "None" else opt.Transformation
    opt.SequenceModeling = None if opt.SequenceModeling == "None" else opt.SequenceModeling
    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )
    model = torch.nn.DataParallel(model).to(DEVICE)

    print(f"loading pretrained model from {opt.saved_model}")
    load_state_dict_compat(model, opt.saved_model, map_location=DEVICE, strict=True)
    opt.exp_name = "_".join(Path(opt.saved_model).parts[1:])

    result_dir = Path(getattr(opt, "result_dir", Path("result") / opt.exp_name))
    result_dir.mkdir(parents=True, exist_ok=True)

    if "CTC" in str(opt.Prediction):
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(DEVICE)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log_path = result_dir / "log_evaluation.txt"
            with log_path.open("a", encoding="utf-8") as log:
                align_collate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
                eval_dataset, eval_dataset_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
                evaluation_loader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=int(opt.workers),
                    collate_fn=align_collate_evaluation,
                    pin_memory=True,
                )
                _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                    model,
                    criterion,
                    evaluation_loader,
                    converter,
                    opt,
                )
                log.write(eval_dataset_log)
                print(f"{accuracy_by_best_model:0.3f}")
                log.write(f"{accuracy_by_best_model:0.3f}\n")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", required=True, help="path to evaluation dataset")
    parser.add_argument("--benchmark_all_eval", action="store_true", help="evaluate 10 benchmark evaluation datasets")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument("--saved_model", required=True, help="path to saved_model to evaluation")
    parser.add_argument("--batch_max_length", type=int, default=25, help="maximum-label-length")
    parser.add_argument("--imgH", type=int, default=32, help="the height of the input image")
    parser.add_argument("--imgW", type=int, default=100, help="the width of the input image")
    parser.add_argument("--rgb", action="store_true", help="use rgb input")
    parser.add_argument("--character", type=str, default=SANSKRIT_OCR_CHARACTER_SET, help="character label")
    parser.add_argument("--sensitive", action="store_true", help="for sensitive character mode")
    parser.add_argument("--PAD", action="store_true", help="whether to keep ratio then pad for image resize")
    parser.add_argument("--data_filtering_off", action="store_true", help="for data_filtering_off mode")
    parser.add_argument("--baiduCTC", action="store_true", help="for data_filtering_off mode")
    parser.add_argument("--Transformation", type=str, required=True, help="Transformation stage. None|TPS")
    parser.add_argument("--FeatureExtraction", type=str, required=True, help="FeatureExtraction stage. VGG|RCNN|ResNet")
    parser.add_argument("--SequenceModeling", type=str, required=True, help="SequenceModeling stage. None|BiLSTM")
    parser.add_argument("--Prediction", type=str, required=True, help="Prediction stage. CTC|Attn")
    parser.add_argument("--num_fiducial", type=int, default=20, help="number of fiducial points of TPS-STN")
    parser.add_argument("--input_channel", type=int, default=1, help="the number of input channel of Feature extractor")
    parser.add_argument("--output_channel", type=int, default=512, help="the number of output channel of Feature extractor")
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the LSTM hidden state")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    opt = parser.parse_args(argv)
    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    test(opt)


if __name__ == "__main__":
    main()
