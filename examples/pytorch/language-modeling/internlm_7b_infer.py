import torch
try:
    import torch_dipu
except:
    pass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def test_internlm_infer(MODEL_PATH, model):
    model = model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    prompt = "来到美丽的大自然，我们发现"
    answer_reference = '''来到美丽的大自然，我们发现很多植物都长得非常奇特。
比如我们今天说的这种树—“猴面包”（Moringa oleifera）, 它是一种生长在非洲的树木品种之一；它的果实可以食用、种子可以用来榨油和制作肥皂等产品……但是你知道吗？其实'''
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    print("------start InternLM 7B inference------")
    start = datetime.now()
    # The generate function uses greedy search by default, and thus the inference result of InternLM 7B does not contain randomness.
    generate_ids = model.generate(
        input_ids, max_length=64, repetition_penalty=1.1)
    answer_output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = datetime.now()
    print("------end InternLM 7B inference------")
    print(
        f'The inference time of InternLM 7B on the current device is: {(end - start).total_seconds():.2f} s', flush=True)
    print("prompt: \n", prompt)
    print("The answer_output generated on the current device is: \n", answer_output)
    passed = (answer_output == answer_reference)
    if passed:
        print("Successfully pass the test for the inference of InternLM 7B!")
    else:
        print("Fail to pass the test for the inference of InternLM 7B!")
        print("The answer_reference generated on gpu is: \n", answer_reference)
    assert passed, "The inference result of InternLM 7B on the current device is not the same as the reference result generated on the gpu, please check the operator implementation of InternLM 7B!"


if __name__ == "__main__":
    # InternLM 7B inference
    parser = argparse.ArgumentParser(
        description="Script for InternLM 7B inference")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lustre/share_data/PAT/datasets/internlm_7B_hf")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
    test_internlm_infer(args.model_path, model)
