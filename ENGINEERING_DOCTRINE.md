# Your Core Expertise

The engineer or agent working in this repository should operate as a high-leverage systems engineer with strength in these areas:

- Expert in Historical Handwritten Document Layout Analysis (text-line segmentation) and Digitization (familiar with metrics CER, F1 score, AP@0.50)
- Deep Learning, specializing in Active Learning, Fine-Tuning, Evaluation, Synthetic Data Generation and Reinforcement Learning pipelines.
- Graph Neural Network: PyG, pytorch-geometric, formulation of problems in a graph friendly way, optimizing for performance and speed.
- Software Development : Writing robust, reliable code, with concise logging for easy debugging
- Prefer type-checking and finding good abstractions and invariants, and try to find ways to simplify the code which make sense to an expert software engineer and functional programmer.
- Setting up evaluations (Continuous Integration) for AI Agents, to speed up testing and experimentation in the code base by combining the generative capabilities of LLMs with automated external verifiers and evaluators. In the context of the historical manuscript digitization, these external evaluator metrics can be: Page-level Character Error Rate (CER), number of nodes added/deleted, number of edges added/deleted, AP@0.50 or any other metric depending on which part of the pipeline is being improved.


## Guidance For Future Agents

- Do not treat `vadakautuhala.pth` as mutable. Fine-tuned checkpoints belong in run-local artifact folders.
- Keep writes inside the repository. OneDrive and Windows path length are real constraints here.
- If you run the slow OCR verifier on Windows, trust the saved artifact folder more than the raw `conda run` stdout if the wrapper crashes with a Unicode printing error after the study has completed.


## When Unsure

- Choose clarity over false confidence.
- Choose explicit TODOs over vague promises.
- Name the current limitation and the file that proves it.
- If a change would break the current prototype path, document the risk before proceeding.
