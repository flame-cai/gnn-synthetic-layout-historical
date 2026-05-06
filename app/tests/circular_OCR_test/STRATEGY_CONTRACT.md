# Strategy Contract

Segmentation and OCR unwrapping are separate steps.

Segmentation consumes page images, heatmaps, gnn-format points and labels, PAGE-XML `Baseline`, and strategy config. It produces copied PAGE-XML files whose `TextLine/Coords` are page-space polygons only.

OCR unwrapping consumes copied PAGE-XML, the page image, page-space `Coords`, `Baseline`, and unwrapping config. It produces horizontal OCR-ready crop images, `PreparedPageDataset` manifests, orientation candidate metadata, selected candidate metadata, and rejected candidate scores.

The unwrapped horizontal rectangle must never be written as PAGE-XML `Coords`.

Every strategy must record enough metadata to diagnose line id, source inputs, cut point, tangent/normal settings, selected orientation candidate, and rejected candidates.

