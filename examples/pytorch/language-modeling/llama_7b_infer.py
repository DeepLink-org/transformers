import torch
try:
    import torch_dipu
except:
    pass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def test_llama_infer(MODEL_PATH, model):
    model = model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = "I believe the meaning of life is"
    answer_reference = '''I believe the meaning of life is to find your gift. The purpose of life is to give it away.
I believe that we are all born with a special talent, and our job in this lifetime is to discover what that talent is and then use it for the benefit of others.
I believe that when'''
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    print("------start LLaMA 7B inference------")
    start = datetime.now()
    # The generate function uses greedy search by default, and thus the inference result of LLaMA 7B does not contain randomness.
    generate_ids = model.generate(
        input_ids, max_length=64, repetition_penalty=1.1)
    answer_output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = datetime.now()
    print("------end LLaMA 7B inference------")
    print(
        f'The inference time of LLaMA 7B on the current device is: {(end - start).total_seconds():.2f} s', flush=True)
    print("prompt: \n", prompt)
    print("The answer_output generated on the current device is: \n", answer_output)
    passed = (answer_output == answer_reference)
    if passed:
        print("Successfully pass the test for the inference of LLaMA 7B!")
    else:
        print("Fail to pass the test for the inference of LLaMA 7B!")
        print("The answer_reference generated on gpu is: \n", answer_reference)
    assert passed, "The inference result of LLaMA 7B on the current device is not the same as the reference result generated on the gpu, please check the operator implementation of LLaMA 7B!"


if __name__ == "__main__":
    # LLaMA 7B inference
    parser = argparse.ArgumentParser(
        description="Script for LLaMA 7B inference")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lustre/share_data/PAT/datasets/llama_7B_hf")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    test_llama_infer(args.model_path, model)
