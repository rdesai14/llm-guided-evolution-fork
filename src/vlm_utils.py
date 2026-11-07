"""Show the mutation LLM the circuit, not just one block of code (H2).

LLMGE_CIRCUIT_VIEW picks the arm (default "off", so a normal run is unchanged):
    off     code only, as before
    ascii   the parent circuit's text drawing (qc.draw("text")) goes in front of the prompt
    gemma   Gemma 3 looks at a picture of the parent circuit (server.py's /critique) and
            its notes go in front of the prompt. LLMGE_VLM_MODEL_PATH picks the model

Each parent's drawing, picture and notes are cached in sota/QuantumVQC/renders/, so a
parent mutated several times is only drawn once.

Client side (mutation jobs): circuit_note(). Server side (server.py): GemmaCritic.
"""
import contextlib
import io
import os
import signal

CIRCUIT_VIEW = os.getenv("LLMGE_CIRCUIT_VIEW", "off").lower()
VLM_MODEL_PATH = os.getenv(
    "LLMGE_VLM_MODEL_PATH",
    "/storage/ice-shared/vip-vvk/llm_storage/hub/models--google--gemma-3-12b-it",
)
VLM_MAX_NEW_TOKENS = 200
ASCII_FOLD = 120  # characters per drawing line before it wraps

VLM_PROMPT = (
    "This is a drawing of a variational quantum circuit that classifies the Iris flower "
    "dataset into 3 classes. The x[i] parameters load the 4 input features, the w[i] "
    "parameters are trainable angles, and the class is read from the measurement of "
    "qubits 0 and 1. In at most 3 short bullet points, say what in this circuit looks "
    "wasteful or unlikely to help it classify: for example qubits that never interact "
    "with the measured qubits, gates that cancel each other, rotations right before "
    "measurement that cannot change the result, or too little entanglement. "
    "Only describe the circuit; do not write code."
)


class _Timeout(Exception):
    pass


def build_parent_circuit(network_file, timeout=60):
    """The (unbound) QuantumCircuit a network file builds."""
    import importlib.util
    import qiskit  # noqa: F401  a cold import on PACE takes >60 s; keep it out of the timeout

    def _alarm(*_):
        raise _Timeout(f"building the circuit took over {timeout}s")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        spec = importlib.util.spec_from_file_location("vlm_parent", network_file)
        mod = importlib.util.module_from_spec(spec)          # not __main__, so main() doesn't run
        with contextlib.redirect_stdout(io.StringIO()):      # some evolved blocks print at import
            spec.loader.exec_module(mod)
            return mod.build_circuit()[0]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def render_circuit(network_file, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = build_parent_circuit(network_file).draw("mpl")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def critique_circuit(png_path, timeout=600):
    """Ask the server's Gemma what looks wrong in the picture. None on any failure."""
    import requests
    from llm_utils import get_llm_server_hostname
    from cfg.constants import PORT
    url = f"http://{get_llm_server_hostname()}:{PORT}/critique"
    payload = {"image_path": png_path, "prompt": VLM_PROMPT, "max_new_tokens": VLM_MAX_NEW_TOKENS}
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        print(f"VLM critique failed: {r.status_code} {r.text[:200]}", flush=True)
        return None
    return r.json().get("generated_text", "").strip() or None


def _ascii_note(network_file, cache):
    if not os.path.exists(cache):
        drawing = str(build_parent_circuit(network_file).draw("text", fold=ASCII_FOLD))
        with open(cache, "w") as f:
            f.write(drawing + "\n")
    drawing = open(cache).read().rstrip()
    # indented, not in ``` fences: the code extractor takes the first fence in the reply,
    # so a drawing echoed back inside fences would be taken for the new code
    indented = "\n".join("    " + line for line in drawing.splitlines())
    return ("Here is a text drawing of the whole circuit this code builds. q_0, q_1, ... are "
            "the qubits, x[i] are the input features, w[i] are the trainable angles, and "
            "qubits 0 and 1 are measured to give the class:\n\n"
            f"{indented}\n\n"
            "Use the drawing to see how your change affects the whole circuit. "
            "Do not repeat the drawing in your answer.\n\n")


def _gemma_note(network_file, png, cache):
    if not os.path.exists(cache):
        if not os.path.exists(png):
            render_circuit(network_file, png)
        critique = critique_circuit(png)
        if not critique:
            return ""
        with open(cache, "w") as f:
            f.write(critique + "\n")
    critique = open(cache).read().strip()
    return ("A reviewer looked at a drawing of the whole circuit this code builds and noted:\n"
            f"{critique}\n"
            "Keep these notes in mind when you make your change.\n\n")


def circuit_note(network_file, render_dir):
    """Text to put in front of the mutation prompt for the current arm; "" on any failure."""
    if CIRCUIT_VIEW not in ("ascii", "gemma"):
        return ""
    gene = os.path.splitext(os.path.basename(network_file))[0]
    try:
        os.makedirs(render_dir, exist_ok=True)
        if CIRCUIT_VIEW == "ascii":
            note = _ascii_note(network_file, os.path.join(render_dir, f"{gene}_ascii.txt"))
        else:
            note = _gemma_note(network_file, os.path.join(render_dir, f"{gene}.png"),
                               os.path.join(render_dir, f"{gene}_critique.txt"))
    except Exception as err:                                  # never let this break a mutation
        print(f"Circuit view ({CIRCUIT_VIEW}) skipped for {gene}: {type(err).__name__}: {err}", flush=True)
        return ""
    print(f"Circuit view ({CIRCUIT_VIEW}) added for {gene}", flush=True)
    return note


class GemmaCritic:
    """Gemma 3 image+text model, loaded once in the LLM server process."""

    def __init__(self, model_path=VLM_MODEL_PATH):
        import threading
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        snapshots = os.path.join(model_path, "snapshots")            # HF cache layout
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            model_path = os.path.join(snapshots, sorted(os.listdir(snapshots))[-1])
        print(f"Loading VLM from {model_path}", flush=True)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path)
        # whole model on the last GPU: Llama fills the first one (server.py caps it at 120GiB each)
        device = torch.cuda.device_count() - 1 if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
        self.lock = threading.Lock()                                  # one image at a time
        print("VLM loaded", flush=True)

    def critique(self, image_path, prompt=VLM_PROMPT, max_new_tokens=VLM_MAX_NEW_TOKENS):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        if getattr(self.processor, "chat_template", None):           # instruct model (-it)
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        else:                                                         # base model (-pt): no chat template
            inputs = self.processor(text=f"<start_of_image> {prompt}\n", images=image,
                                    return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.torch.bfloat16)
        n_in = inputs["input_ids"].shape[-1]
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
