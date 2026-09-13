# Session 5 — model optimization on one ANPR pipeline. One target per stage.
#
#   make setup       .venv with the pinned CPU stack + OpenVINO     (once)
#   make profile     data -> train -> s00: the baseline profile in ONE command
#   make all         every stage in order; GPU-only stages record "not_run" on a CPU box
#   make table       regenerate results/journey_table.md from results/results.json
#
# ANPR_PROFILE=ci|quick|full sets data size and epochs (default: quick).
# ANPR_THREADS sets the CPU thread count every runtime uses (default: 4).

SHELL := /bin/bash
PY ?= .venv/bin/python
EDGE_PY ?= .venv-edge/bin/python
export ANPR_PROFILE ?= quick
ART := artifacts/$(ANPR_PROFILE)
DATA := data/synthetic/$(ANPR_PROFILE)/manifest.json

.DEFAULT_GOAL := help
.PHONY: help setup setup-edge setup-gpu doctor data train profile all table cards test lint gate \
        gate-demo pyspy int8-debug triton-up triton-down clean \
        s00 s01 s02 s03 s04 s05 s06a s06b s07 s08 s09 s10 s11

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create .venv with requirements.txt + requirements-openvino.txt
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt -r requirements-openvino.txt

setup-edge: ## Create .venv-edge for onnx2tf + LiteRT (conflicting pins, see requirements-edge.txt)
	python3.12 -m venv .venv-edge
	$(EDGE_PY) -m pip install -r requirements-edge.txt

setup-gpu: ## Linux + NVIDIA: swap onnxruntime for onnxruntime-gpu, add TensorRT 10.16
	$(PY) -m pip uninstall -y onnxruntime
	$(PY) -m pip install -r requirements-gpu.txt

doctor: ## Pre-flight: interpreters, GPU, TensorRT version, Docker, data, checkpoints
	$(PY) -m tools.doctor

data: $(DATA) ## Generate the synthetic dataset for ANPR_PROFILE
$(DATA):
	$(PY) -m src.datasets.synthetic

train: $(ART)/detector_baseline.pt $(ART)/ocr_baseline.pt ## Train the baseline detector + recognizer
$(ART)/detector_baseline.pt: $(DATA)
	$(PY) -m src.train detector --arch resnet34 --name detector_baseline
$(ART)/ocr_baseline.pt: $(DATA)
	$(PY) -m src.train ocr --arch crnn --name ocr_baseline

profile: train s00 ## One command: data, training, and the baseline latency breakdown

s00: train ## Profile first: phase breakdown, torch.profiler, the gate, pre/post-processing fixes
	$(PY) -m src.stages.s00_profile
s01: train ## Baseline row (eager PyTorch FP32)
	$(PY) -m src.stages.s01_baseline
s02: train ## torch.compile
	$(PY) -m src.stages.s02_torch_compile
s03: train ## TorchScript (trace + save + load)
	$(PY) -m src.stages.s03_torchscript
s04: train ## ONNX export + ONNX Runtime execution providers
	$(PY) -m src.stages.s04_onnx_export
# Stage targets run ONLY their own stage: `make s09` must not retrain the s07 students.
# Each script checks the artifacts it needs and names the target that builds them;
# `make all` runs every stage in dependency order.
s05: train ## Pruning: masked vs physically sliced (needs s04)
	$(PY) -m src.stages.s05_pruning
s06a: train ## PTQ: dynamic, static, calibration sets, per-tensor vs per-channel (needs s04)
	$(PY) -m src.stages.s06a_ptq
int8-debug: train ## When INT8 breaks: sensitivity, outlier channels, mixed precision (needs s06a)
	$(PY) -m src.quant_debug
s06b: train ## QAT on the recognizer (needs s06a)
	$(PY) -m src.stages.s06b_qat
s07: train ## Knowledge distillation: detector + recognizer students
	$(PY) -m src.stages.s07_distillation
s08: train ## TensorRT engines FP32 / FP16 / INT8 (NVIDIA GPU; needs s07)
	$(PY) -m src.stages.s08_tensorrt
s09: train ## OpenVINO IR + NNCF INT8, latency vs throughput hints (needs s04, s07)
	$(PY) -m src.stages.s09_openvino
s10: train ## TFLite via onnx2tf, full-integer INT8, LiteRT (runs in .venv-edge; needs s07)
	$(EDGE_PY) -m src.stages.s10_tflite_edge
s11: train ## Triton: model repository, BLS pipeline, dynamic batching (Docker; needs s07)
	$(PY) -m src.stages.s11_triton

pyspy: train ## py-spy flame graph of the baseline pipeline (needs sudo on macOS)
	mkdir -p results/profile
	py-spy record --rate 250 --format flamegraph -o results/profile/pyspy_$(ANPR_PROFILE).svg -- $(PY) -m src.stages.s00_profile --workload 300

all: profile s01 s02 s03 s04 s05 s06a int8-debug s06b s07 s08 s09 s10 s11 table ## Every stage, in order
	@echo "done — results/journey_table.md"

table: ## Regenerate the journey table, benchmark cards, and guide result blocks
	$(PY) -m tools.journey_table
	$(PY) -m tools.benchmark_card
	$(PY) -m tools.render_results

gate: ## CI perf gate: accuracy vs baseline + re-measured p95 vs the row's own record
	$(PY) -m ci.perf_gate --candidate s04_onnx_export:ort-cpu-fp32 --remeasure

gate-demo: ## Show the gate FAILING on an injected +40 ms regression
	-ANPR_INJECT_LATENCY_MS=40 $(PY) -m ci.perf_gate --candidate s04_onnx_export:ort-cpu-fp32 --remeasure

triton-up: ## Start Triton 26.05 on the generated model repository
	$(PY) -m src.stages.s11_triton --up-only
triton-down: ## Stop the Triton container
	-docker rm -f anpr-triton

test: ## Unit tests: harness contract, parity, cost, KD math, docs hygiene
	$(PY) -m pytest -q

lint: ## Ruff
	$(PY) -m ruff check src tools ci tests

clean: ## Remove generated artifacts for ANPR_PROFILE (keeps results/)
	rm -rf $(ART)
