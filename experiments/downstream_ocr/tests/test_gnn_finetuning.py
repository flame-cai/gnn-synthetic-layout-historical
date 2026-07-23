from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.downstream_ocr.gnn_finetuning import (
    REPO_ROOT,
    TRAINABLE_SCOPE_FULL_MODEL,
    TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER,
    configure_gnn_trainable_parameters,
    load_gnn_finetuning_recipe,
    prepare_deleted_node_supervision_page,
    process_page,
    select_history_replay_page_ids,
)


class GNNFineTuningTests(unittest.TestCase):
    def test_checked_in_recipe_pins_fifty_augmentations_and_twenty_percent_replay(self):
        recipe = load_gnn_finetuning_recipe(
            REPO_ROOT
            / "experiments"
            / "downstream_ocr"
            / "configs"
            / "gnn_finetuning.yaml"
        )

        self.assertEqual(recipe.augmentations_per_page, 50)
        self.assertEqual(recipe.history_replay_ratio, 0.20)
        self.assertEqual(
            recipe.expected_model_class,
            "gnn_training.training.models.gnn_models.EdgeClassifier",
        )
        self.assertEqual(
            recipe.expected_backbone_class,
            "gnn_training.training.models.gnn_models.SplineConvBackbone",
        )
        self.assertEqual(
            recipe.trainable_scope,
            TRAINABLE_SCOPE_FULL_MODEL,
        )
        self.assertEqual(
            recipe.deleted_node_supervision,
            {
                "enabled": True,
                "match_tolerance_image_pixels": 2.0,
                "noise_textline_label": -1,
            },
        )
        self.assertEqual(
            recipe.preprocessing_config.input_graph.directionality,
            "bidirectional",
        )

    def test_full_model_scope_unfreezes_every_parameter(self):
        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_preprocessor = torch.nn.Linear(3, 2)
                self.convs = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
                )

        class DummyEdgeClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DummyBackbone()
                self.predictor = torch.nn.Linear(4, 2)

        model = DummyEdgeClassifier()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        metadata = configure_gnn_trainable_parameters(
            model,
            TRAINABLE_SCOPE_FULL_MODEL,
        )

        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        self.assertEqual(
            metadata["trainable_parameter_count"],
            metadata["total_parameter_count"],
        )
        self.assertEqual(metadata["frozen_parameter_count"], 0)
        self.assertIsNone(metadata["trainable_final_gnn_layer_index"])

    def test_only_predictor_and_final_message_passing_layer_are_trainable(self):
        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_preprocessor = torch.nn.Linear(3, 2)
                self.convs = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
                )

        class DummyEdgeClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DummyBackbone()
                self.predictor = torch.nn.Linear(4, 2)

        model = DummyEdgeClassifier()
        metadata = configure_gnn_trainable_parameters(
            model,
            TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER,
        )
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        self.assertEqual(
            trainable_names,
            {
                "backbone.convs.1.weight",
                "backbone.convs.1.bias",
                "predictor.weight",
                "predictor.bias",
            },
        )
        self.assertFalse(model.backbone.edge_preprocessor.weight.requires_grad)
        self.assertFalse(model.backbone.convs[0].weight.requires_grad)
        self.assertEqual(
            metadata["trainable_parameter_names"],
            [
                "backbone.convs.1.weight",
                "backbone.convs.1.bias",
                "predictor.weight",
                "predictor.bias",
            ],
        )
        self.assertEqual(metadata["trainable_final_gnn_layer_index"], 1)
        self.assertLess(
            metadata["trainable_parameter_count"],
            metadata["total_parameter_count"],
        )

    def test_history_replay_is_deterministic_and_uses_twenty_percent(self):
        augmented_ids = tuple(f"page_{index}" for index in range(50))

        first = select_history_replay_page_ids(
            augmented_ids,
            replay_ratio=0.20,
            seed=42,
        )
        second = select_history_replay_page_ids(
            reversed(augmented_ids),
            replay_ratio=0.20,
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)
        self.assertTrue(set(first).issubset(augmented_ids))

    def test_deleted_craft_nodes_produce_only_negative_incident_edges(self):
        recipe = load_gnn_finetuning_recipe(
            REPO_ROOT
            / "experiments"
            / "downstream_ocr"
            / "configs"
            / "gnn_finetuning.yaml"
        )
        raw_normalized = np.array(
            [
                [0.05 + index * 0.05, 0.10, 0.0]
                for index in range(10)
            ]
        )
        corrected_normalized = np.concatenate(
            (
                raw_normalized[:9],
                np.array([[0.70, 0.70, 0.0]]),
            ),
            axis=0,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw_dir = root / "raw"
            corrected_dir = root / "corrected"
            output_dir = root / "supervision"
            raw_dir.mkdir()
            corrected_dir.mkdir()
            for source_dir in (raw_dir, corrected_dir):
                (source_dir / "page_dims.txt").write_text(
                    "100 100\n",
                    encoding="utf-8",
                )
            np.savetxt(
                raw_dir / "page_inputs_normalized.txt",
                raw_normalized,
            )
            np.savetxt(
                raw_dir / "page_inputs_unnormalized.txt",
                raw_normalized * np.array([100.0, 100.0, 1.0]),
            )
            np.savetxt(
                corrected_dir / "page_inputs_normalized.txt",
                corrected_normalized,
            )
            np.savetxt(
                corrected_dir / "page_inputs_unnormalized.txt",
                corrected_normalized * np.array([100.0, 100.0, 1.0]),
            )
            np.savetxt(
                corrected_dir / "page_labels_textline.txt",
                np.array([0] * 9 + [1]),
                fmt="%d",
            )

            manifest = prepare_deleted_node_supervision_page(
                page_id="page",
                raw_source_dir=raw_dir,
                corrected_source_dir=corrected_dir,
                output_dir=output_dir,
                recipe=recipe,
            )
            output_points = np.loadtxt(
                output_dir / "page_inputs_normalized.txt",
                ndmin=2,
            )
            output_labels = np.atleast_1d(
                np.loadtxt(
                    output_dir / "page_labels_textline.txt",
                    dtype=int,
                )
            )
            graph = process_page(
                "page",
                output_dir,
                recipe.preprocessing_config,
            )

        self.assertEqual(manifest["recovered_deleted_raw_node_count"], 1)
        self.assertEqual(
            manifest["corrected_unmatched_node_count_left_untouched"],
            1,
        )
        self.assertEqual(len(output_points), 11)
        np.testing.assert_allclose(output_points[9], corrected_normalized[9])
        np.testing.assert_allclose(output_points[10], raw_normalized[9])
        self.assertEqual(output_labels.tolist(), [0] * 9 + [1, -1])
        self.assertIsNotNone(graph)
        deleted_node_index = 10
        incident_mask = (
            (graph.edge_index[0] == deleted_node_index)
            | (graph.edge_index[1] == deleted_node_index)
        )
        self.assertTrue(bool(incident_mask.any()))
        self.assertTrue(bool((graph.edge_y[incident_mask] == 0).all()))


if __name__ == "__main__":
    unittest.main()
