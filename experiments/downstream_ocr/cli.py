from __future__ import annotations

import argparse
from pathlib import Path

from .reproducibility import write_reproducibility_manifest
from .runners import (
    BASE_OCR_CHECKPOINT,
    PRETRAINED_GNN_CONFIG,
    PRETRAINED_GNN_MODEL,
    REPO_ROOT,
    adapt_vlm_json_and_evaluate,
    evaluate_existing_prediction_tree,
    prepare_gt_layout_pages_only,
    run_local_gt_layout_experiment,
    run_methods_experiment,
)
from .splits import DEFAULT_SPLIT_SEED, default_manuscript_paths
from .reporting import write_combined_table_report, write_experiment_report
from .validation import validate_manuscript_pagexml, write_validation_report


DEFAULT_YAJN_ROOT = Path("app") / "input_manuscripts" / "yajn"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Downstream OCR comparison harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser(
        "run-local-gt-layout",
        help="Run local OCR methods that use GT PAGE-XML layout, including optional fine-tuning ablations.",
    )
    local.add_argument("--manuscript-root", default=str(DEFAULT_YAJN_ROOT))
    local.add_argument("--output-root", required=True)
    local.add_argument("--method-id", action="append", dest="method_ids")
    local.add_argument("--write-diagnostics", action="store_true")
    local.add_argument("--fold-id", action="append", dest="fold_ids")
    local.add_argument("--max-test-pages", type=int)
    local.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)

    prepare = subparsers.add_parser(
        "prepare-gt-layout",
        help="Prepare production-style OCR line crops from GT PAGE baselines without running OCR.",
    )
    prepare.add_argument("--manuscript-root", default=str(DEFAULT_YAJN_ROOT))
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--page-id", action="append", dest="page_ids")

    existing = subparsers.add_parser(
        "evaluate-existing",
        help="Evaluate an existing prediction tree organized as <predictions-root>/<fold_id>/<method_id>/<page>.xml.",
    )
    existing.add_argument("--manuscript-root", default=str(DEFAULT_YAJN_ROOT))
    existing.add_argument("--predictions-root", required=True)
    existing.add_argument("--method-id", required=True)
    existing.add_argument("--output-root", required=True)
    existing.add_argument("--write-diagnostics", action="store_true")
    existing.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)

    adapt_json = subparsers.add_parser(
        "adapt-vlm-json",
        help="Convert saved VLM/Gemini JSON outputs into PAGE-XML and evaluate them.",
    )
    adapt_json.add_argument("--manuscript-root", default=str(DEFAULT_YAJN_ROOT))
    adapt_json.add_argument("--json-root", required=True)
    adapt_json.add_argument("--output-root", required=True)
    adapt_json.add_argument("--method-id", default="vlm_e2e")
    adapt_json.add_argument("--write-diagnostics", action="store_true")
    adapt_json.add_argument("--fold-id", action="append", dest="fold_ids")
    adapt_json.add_argument("--max-test-pages", type=int)
    adapt_json.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)

    methods = subparsers.add_parser(
        "run-methods",
        help="Run selected experiment methods across the standard three folds.",
    )
    methods.add_argument(
        "--manuscript-root",
        action="append",
        dest="manuscript_roots",
        help="Manuscript root to run. Repeat to run several manuscripts and build combined tables.",
    )
    methods.add_argument("--output-root", required=True)
    methods.add_argument(
        "--method-id",
        action="append",
        dest="method_ids",
        required=True,
        help="Method id to run. Repeat for multiple methods.",
    )
    methods.add_argument("--write-diagnostics", action="store_true")
    methods.add_argument("--fold-id", action="append", dest="fold_ids")
    methods.add_argument("--max-test-pages", type=int)
    methods.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    methods.add_argument("--gemini-input-usd-per-1m-tokens", type=float)
    methods.add_argument("--gemini-output-usd-per-1m-tokens", type=float)

    validate = subparsers.add_parser(
        "validate-dataset",
        help="Validate strict PAGE-XML geometry before running OCR experiments.",
    )
    validate.add_argument("--manuscript-root", default=str(DEFAULT_YAJN_ROOT))
    validate.add_argument("--output-json")
    validate.add_argument(
        "--repair-geometry",
        action="store_true",
        help="Validate using the experiment repair policy that rasterizes invalid Coords and extracts valid contours.",
    )

    snapshot = subparsers.add_parser(
        "snapshot-env",
        help="Write reproducibility.json with git, dependency, and checkpoint hash metadata.",
    )
    snapshot.add_argument("--output-root", required=True)

    report = subparsers.add_parser(
        "write-report",
        help="Generate summary tables, figures, Gemini usage tables, and a Markdown report for an existing run.",
    )
    report.add_argument("--output-root", required=True)
    report.add_argument("--gemini-input-usd-per-1m-tokens", type=float)
    report.add_argument("--gemini-output-usd-per-1m-tokens", type=float)

    combined_report = subparsers.add_parser(
        "write-combined-report",
        help="Generate the paper tables for multiple existing manuscript run roots at once.",
    )
    combined_report.add_argument(
        "--input-root",
        action="append",
        dest="input_roots",
        required=True,
        help="Existing downstream OCR run root to include. Repeat once per manuscript.",
    )
    combined_report.add_argument("--output-root", required=True)
    combined_report.add_argument("--gemini-input-usd-per-1m-tokens", type=float)
    combined_report.add_argument("--gemini-output-usd-per-1m-tokens", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-local-gt-layout":
        run_local_gt_layout_experiment(
            manuscript_root=args.manuscript_root,
            output_root=args.output_root,
            method_ids=args.method_ids
            or (
                "annotation_tool_gt_layout",
                "annotation_tool_gt_layout_ft_1",
                "annotation_tool_gt_layout_ft_2",
                "annotation_tool_gt_layout_ft_3",
            ),
            write_diagnostics=args.write_diagnostics,
            fold_ids=args.fold_ids,
            max_test_pages=args.max_test_pages,
            split_seed=args.split_seed,
        )
    elif args.command == "prepare-gt-layout":
        prepare_gt_layout_pages_only(
            manuscript_root=args.manuscript_root,
            output_root=args.output_root,
            page_ids=args.page_ids,
        )
    elif args.command == "evaluate-existing":
        evaluate_existing_prediction_tree(
            manuscript_root=args.manuscript_root,
            predictions_root=args.predictions_root,
            method_id=args.method_id,
            output_root=args.output_root,
            write_diagnostics=args.write_diagnostics,
            split_seed=args.split_seed,
        )
    elif args.command == "adapt-vlm-json":
        adapt_vlm_json_and_evaluate(
            manuscript_root=args.manuscript_root,
            json_root=args.json_root,
            output_root=args.output_root,
            method_id=args.method_id,
            write_diagnostics=args.write_diagnostics,
            fold_ids=args.fold_ids,
            max_test_pages=args.max_test_pages,
            split_seed=args.split_seed,
        )
    elif args.command == "run-methods":
        manuscript_roots = args.manuscript_roots or [str(DEFAULT_YAJN_ROOT)]
        if len(manuscript_roots) == 1:
            run_methods_experiment(
                manuscript_root=manuscript_roots[0],
                output_root=args.output_root,
                method_ids=args.method_ids,
                write_diagnostics=args.write_diagnostics,
                fold_ids=args.fold_ids,
                max_test_pages=args.max_test_pages,
                split_seed=args.split_seed,
                input_usd_per_1m_tokens=args.gemini_input_usd_per_1m_tokens,
                output_usd_per_1m_tokens=args.gemini_output_usd_per_1m_tokens,
            )
        else:
            combined_input_roots = []
            parent_output_root = Path(args.output_root)
            for manuscript_root in manuscript_roots:
                paths = default_manuscript_paths(manuscript_root)
                manuscript_output_root = parent_output_root / paths.manuscript_id
                run_methods_experiment(
                    manuscript_root=manuscript_root,
                    output_root=manuscript_output_root,
                    method_ids=args.method_ids,
                    write_diagnostics=args.write_diagnostics,
                    fold_ids=args.fold_ids,
                    max_test_pages=args.max_test_pages,
                    split_seed=args.split_seed,
                    input_usd_per_1m_tokens=args.gemini_input_usd_per_1m_tokens,
                    output_usd_per_1m_tokens=args.gemini_output_usd_per_1m_tokens,
                )
                combined_input_roots.append(manuscript_output_root)
            write_combined_table_report(
                combined_input_roots,
                parent_output_root,
                input_usd_per_1m_tokens=args.gemini_input_usd_per_1m_tokens,
                output_usd_per_1m_tokens=args.gemini_output_usd_per_1m_tokens,
            )
    elif args.command == "validate-dataset":
        report = validate_manuscript_pagexml(args.manuscript_root, repair_geometry=args.repair_geometry)
        if args.output_json:
            write_validation_report(report, args.output_json)
        print(
            {
                "manuscript_id": report["manuscript_id"],
                "page_count": report["page_count"],
                "valid_page_count": report["valid_page_count"],
                "invalid_page_count": report["invalid_page_count"],
                "repair_geometry": report["repair_geometry"],
                "valid": report["valid"],
            }
        )
        if not report["valid"]:
            first_invalid = next(item for item in report["pages"] if not item["valid"])
            print({"first_invalid_page": first_invalid["page_id"], "error": first_invalid["error"]})
    elif args.command == "snapshot-env":
        path = write_reproducibility_manifest(
            args.output_root,
            repo_root=REPO_ROOT,
            artifact_paths=(
                BASE_OCR_CHECKPOINT,
                PRETRAINED_GNN_MODEL,
                PRETRAINED_GNN_CONFIG,
            ),
        )
        print({"reproducibility_json": str(path)})
    elif args.command == "write-report":
        artifacts = write_experiment_report(
            args.output_root,
            input_usd_per_1m_tokens=args.gemini_input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=args.gemini_output_usd_per_1m_tokens,
        )
        print({"report": str(artifacts.markdown_path)})
    elif args.command == "write-combined-report":
        artifacts = write_combined_table_report(
            args.input_roots,
            args.output_root,
            input_usd_per_1m_tokens=args.gemini_input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=args.gemini_output_usd_per_1m_tokens,
        )
        print({"report": str(artifacts.markdown_path)})
    else:  # pragma: no cover
        parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
