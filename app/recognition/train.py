from __future__ import annotations

import argparse
import random
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

try:
    from .dataset import AlignCollate, Batch_Balanced_Dataset, hierarchical_dataset
    from .model import Model
    from .ocr_defaults import SANSKRIT_OCR_CHARACTER_SET, get_device, load_state_dict_compat
    from .test import validation
    from .utils import AttnLabelConverter, Averager, CTCLabelConverter, CTCLabelConverterForBaiduWarpctc
except ImportError:  # pragma: no cover - script execution fallback
    from dataset import AlignCollate, Batch_Balanced_Dataset, hierarchical_dataset
    from model import Model
    from ocr_defaults import SANSKRIT_OCR_CHARACTER_SET, get_device, load_state_dict_compat
    from test import validation
    from utils import AttnLabelConverter, Averager, CTCLabelConverter, CTCLabelConverterForBaiduWarpctc


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _experiment_dir(opt):
    if getattr(opt, "experiment_dir", None):
        return Path(opt.experiment_dir)
    return Path("saved_models") / opt.exp_name


def _normalize_stage(value):
    return None if value == "None" else value


def _prepare_options(opt):
    opt.Transformation = _normalize_stage(opt.Transformation)
    opt.SequenceModeling = _normalize_stage(opt.SequenceModeling)
    opt.Prediction = _normalize_stage(opt.Prediction)

    if isinstance(opt.select_data, str):
        opt.select_data = opt.select_data.split("-")
    if isinstance(opt.batch_ratio, str):
        opt.batch_ratio = opt.batch_ratio.split("-")

    if opt.rgb:
        opt.input_channel = 3

    experiment_dir = _experiment_dir(opt)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _build_converter(opt):
    if "CTC" in str(opt.Prediction):
        if opt.baiduCTC:
            return CTCLabelConverterForBaiduWarpctc(opt.character)
        return CTCLabelConverter(opt.character)
    return AttnLabelConverter(opt.character)


def _seed_everything(opt):
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print("------ Use multi-GPU setting ------")
        print("if you stuck too long time with multi-GPU setting, try to set --workers 0")
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu


def train(opt):
    device = get_device()
    experiment_dir = _prepare_options(opt)
    _seed_everything(opt)

    if not opt.data_filtering_off:
        print("Filtering the images containing characters which are not in opt.character")
        print("Filtering the images whose label is longer than opt.batch_max_length")

    train_dataset = Batch_Balanced_Dataset(opt)

    dataset_log_path = experiment_dir / "log_dataset.txt"
    with dataset_log_path.open("a", encoding="utf-8") as log:
        align_collate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=align_collate_valid,
            pin_memory=True,
        )
        log.write(valid_dataset_log)
        print("-" * 80)
        log.write("-" * 80 + "\n")

    converter = _build_converter(opt)
    opt.num_class = len(converter.character)

    model = Model(opt)
    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.kaiming_normal_(param)
        except Exception:
            if "weight" in name:
                param.data.fill_(1)

    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model:
        print(f"loading pretrained model from {opt.saved_model}")
        load_state_dict_compat(model, opt.saved_model, map_location=device, strict=not opt.FT)

    if "CTC" in str(opt.Prediction):
        if opt.baiduCTC:
            from warpctc_pytorch import CTCLoss

            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_avg = Averager()

    filtered_parameters = []
    params_num = []
    for parameter in filter(lambda item: item.requires_grad, model.parameters()):
        filtered_parameters.append(parameter)
        params_num.append(np.prod(parameter.size()))
    print("Trainable params num : ", sum(params_num))

    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    opt_path = experiment_dir / "opt.txt"
    with opt_path.open("a", encoding="utf-8") as opt_file:
        opt_log = "------------ Options -------------\n"
        for key, value in vars(opt).items():
            opt_log += f"{key}: {value}\n"
        opt_log += "---------------------------------------\n"
        print(opt_log)
        opt_file.write(opt_log)

    start_iter = 0
    if opt.saved_model:
        try:
            start_iter = int(Path(opt.saved_model).stem.split("_")[-1])
            print(f"continue to train, start_iter: {start_iter}")
        except Exception:
            start_iter = 0

    start_time = time.time()
    best_accuracy = -1.0
    best_norm_ed = -1.0
    iteration = start_iter

    train_log_path = experiment_dir / "log_train.txt"
    while True:
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if "CTC" in str(opt.Prediction):
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text[:, :-1])
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_avg.add(cost)

        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            with train_log_path.open("a", encoding="utf-8") as log:
                model.eval()
                with torch.no_grad():
                    (
                        valid_loss,
                        current_accuracy,
                        current_norm_ed,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data,
                    ) = validation(model, criterion, valid_loader, converter, opt)
                model.train()

                loss_log = (
                    f"[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, "
                    f"Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}"
                )
                loss_avg.reset()

                current_model_log = (
                    f"{'Current_accuracy':17s}: {current_accuracy:0.3f}, "
                    f"{'Current_norm_ED':17s}: {current_norm_ed:0.2f}"
                )

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), experiment_dir / "best_accuracy.pth")
                if current_norm_ed > best_norm_ed:
                    best_norm_ed = current_norm_ed
                    torch.save(model.state_dict(), experiment_dir / "best_norm_ED.pth")
                best_model_log = (
                    f"{'Best_accuracy':17s}: {best_accuracy:0.3f}, "
                    f"{'Best_norm_ED':17s}: {best_norm_ed:0.2f}"
                )

                loss_model_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
                print(loss_model_log)
                log.write(loss_model_log + "\n")

                dashed_line = "-" * 80
                head = f"{'Ground Truth':25s} | {'Prediction':25s} | Confidence Score & T/F"
                predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if "Attn" in str(opt.Prediction):
                        gt = gt[: gt.find("[s]")]
                        pred = pred[: pred.find("[s]")]

                    predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += dashed_line
                print(predicted_result_log)
                log.write(predicted_result_log + "\n")

        if (iteration + 1) % int(1e5) == 0:
            torch.save(model.state_dict(), experiment_dir / f"iter_{iteration + 1}.pth")

        if (iteration + 1) == opt.num_iter:
            print("end the training")
            return {
                "best_accuracy": best_accuracy,
                "best_norm_ED": best_norm_ed,
                "last_iteration": iteration + 1,
                "experiment_dir": str(experiment_dir.resolve()),
                "best_accuracy_path": str((experiment_dir / "best_accuracy.pth").resolve()),
                "best_norm_ED_path": str((experiment_dir / "best_norm_ED.pth").resolve()),
            }

        iteration += 1


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="Where to store logs and models")
    parser.add_argument("--experiment_dir", help="Optional absolute directory to store this training run.")
    parser.add_argument("--train_data", required=True, help="path to training dataset")
    parser.add_argument("--valid_data", required=True, help="path to validation dataset")
    parser.add_argument("--manualSeed", type=int, default=1111, help="for random seed setting")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument("--num_iter", type=int, default=300000, help="number of iterations to train for")
    parser.add_argument("--valInterval", type=int, default=2000, help="Interval between each validation")
    parser.add_argument("--saved_model", default="", help="path to model to continue training")
    parser.add_argument("--FT", action="store_true", help="whether to do fine-tuning")
    parser.add_argument("--adam", action="store_true", help="Whether to use adam (default is Adadelta)")
    parser.add_argument("--lr", type=float, default=1, help="learning rate, default=1.0 for Adadelta")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9")
    parser.add_argument("--rho", type=float, default=0.95, help="decay rate rho for Adadelta. default=0.95")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for Adadelta. default=1e-8")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping value. default=5")
    parser.add_argument("--baiduCTC", action="store_true", help="for data_filtering_off mode")
    parser.add_argument("--select_data", type=str, default="/", help="select training data")
    parser.add_argument("--batch_ratio", type=str, default="1", help="assign ratio for each selected data in the batch")
    parser.add_argument(
        "--total_data_usage_ratio",
        type=str,
        default="1.0",
        help="total data usage ratio, this ratio is multiplied to total number of data.",
    )
    parser.add_argument("--batch_max_length", type=int, default=25, help="maximum-label-length")
    parser.add_argument("--imgH", type=int, default=32, help="the height of the input image")
    parser.add_argument("--imgW", type=int, default=100, help="the width of the input image")
    parser.add_argument("--rgb", action="store_true", help="use rgb input")
    parser.add_argument("--character", type=str, default=SANSKRIT_OCR_CHARACTER_SET, help="character label")
    parser.add_argument("--sensitive", action="store_true", help="for sensitive character mode")
    parser.add_argument("--PAD", action="store_true", help="whether to keep ratio then pad for image resize")
    parser.add_argument("--data_filtering_off", action="store_true", help="for data_filtering_off mode")
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

    if not opt.exp_name:
        opt.exp_name = f"{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}"
        opt.exp_name += f"-Seed{opt.manualSeed}"

    if opt.sensitive:
        opt.character = string.printable[:-6]

    train(opt)


if __name__ == "__main__":
    main()
