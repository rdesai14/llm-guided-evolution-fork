import sys
sys.path.append("src")

import re
import os
import glob
import time
import numpy as np
import transformers
from torch import bfloat16
from utils.privit import *
from cfg.constants import *
from utils.print_utils import box_print

from typing import Optional
#import fire
# from llama import Llama
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer
from google import genai
from google.genai import types



def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()


def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    return '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()


def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature, inference_submission=False):
    """Generates augmented code using Mixtral."""
    box_print("PROMPT TO LLM", print_bbox_len=60, new_line_end=False)
    print(txt2llm)
    
    if inference_submission is False:
        llm_code_generator = submit_mixtral
        qc_func = llm_code_qc
    else:
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_paceice
        elif LLM_MODEL == 'gemini':
            llm_code_generator = submit_gemini_api
        qc_func = llm_code_qc_hf
    
    if apply_quality_control:
        base_code = retrieve_base_code(augment_idx)
        code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
        code_from_llm = qc_func(code_from_llm, base_code, generate_text)
    else:
        code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)
        box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
        print(code_from_llm)
        code_from_llm = clean_code_from_llm(code_from_llm)
    box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    return code_from_llm

def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

# Function to load and split the file
def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(ROOT_DIR, 'templates/llm_quality_control.txt')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    res = generate_text(prompt2llm) # clean txt
    code_from_llm = res[0]["generated_text"]
    code_from_llm = '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    return code_from_llm


def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(ROOT_DIR, f'templates/{fname}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_hf(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm


def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    """
    This function submits a model prompt to mixtral through the HuggingFace Inference API

    Parameters
    ----------
    txt2mixtral : str
        Prompt that will be sent to mixtral
    max_new_tokens : int, optional
       A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
       Which mixtral variant to utilize for inference, by default "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """    
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_llama3_paceice(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                        model_path="/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct", 
                        return_gen=False):
    """Submits a prompt to the Llama-3.3-70B-Instruct model on PACE-ICE.

    This function loads the specified Llama 3 model from a local path, formats
    the input prompt into a chat template, and generates a response. Note that
    the 'max_new_tokens' argument is overridden internally by a random value
    between 2000 and 2400.

    Args:
        txt2llama (str): The input prompt to send to the model.
        max_new_tokens (int, optional): The maximum number of new tokens to
            generate. Defaults to 1024, but is ignored.
        top_p (float, optional): The nucleus sampling probability. Defaults to 0.15.
        temperature (float, optional): The sampling temperature for generation.
            Defaults to 0.1.
        model_path (str, optional): The local file path to the Llama-3.3-70B
            model directory. Defaults to a shared path on the PACE-ICE cluster.
        return_gen (bool, optional): If True, returns a tuple containing the
            result text and None. Defaults to False.

    Returns:
        str or tuple[str, None]: The generated text if 'return_gen' is False.
        Otherwise, a tuple containing the generated text and None.
    """

    max_new_tokens = np.random.randint(6000, 8000)  # Randomize new tokens
    print("Utilizing llama3.3-70B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # Load Model & Tokenizer Locally
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Use bfloat16 for GPUs
        device_map="auto"
    ).eval()  # Set to eval mode

    instructions = [
        {"role": "user", "content": "Provide code in Python\n" + txt2llama}
    ]
    
    # Convert instructions to model format
    prompt = tokenizer.apply_chat_template(instructions, tokenize=False)

    # Tokenize Input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate Output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Decode & Return Result
    # Slice off the prompt tokens. model.generate() returns prompt+completion, so
    # decoding output[0] whole echoes the prompt back - and clean_code_from_llm()
    # takes split("```")[1], the FIRST fence, which is then the original block
    # quoted in the prompt rather than the model's rewrite. That silently made
    # every mutated individual a byte-identical clone of its parent.
    gen_only = output[0][inputs["input_ids"].shape[-1]:]
    result_text = tokenizer.decode(gen_only, skip_special_tokens=True)

    return (result_text, None) if return_gen else result_text


def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", return_gen=False):
    """
    This function submits a model prompt to Llama3 through the HuggingFace Inference API

    Parameters
    ----------
    txt2llama : str
        Prompt that will be sent to Llama3
    max_new_tokens : int, optional
        A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
        Which Llama3 variant to utilize for inference, by default "meta-llama/Meta-Llama-3.1-70B-Instruct"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """    
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2llama,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_gemini_api(txt2gemini, **kwargs):
    """
    This function submits a model prompt to Gemini through its API

    Parameters
    ----------
    txt2gemini : str
        Prompt that will be sent to Gemini

    Returns
    -------
    str
        Model's output from inference
    """    
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[txt2gemini],
        
    )
    return response.text



def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
    
def mutate_prompts(n=5):
    templates = np.random.choice(glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt'), n)
    for i, template in enumerate(templates):
        path, filename = os.path.split(template)
        with open(template, 'r') as file:
            prompt_text = file.read()
        prompt_text = prompt_text.split("```")[0].strip()
        prompt = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
        temp = np.random.uniform(0.01, 0.4)
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_paceice
        output = llm_code_generator(prompt, temperature=temp).strip()
        if "```" in output:
            output = output.split("```")[0]
        output = output + "\n```python\n{}\n```"
        with open(os.path.join(path, "mutant{}.txt".format(i)), 'w') as file:
            file.write(output)
