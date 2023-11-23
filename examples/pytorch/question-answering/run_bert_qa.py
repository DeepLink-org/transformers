import subprocess
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Script for fine-tuning BERT model for question answering")
    parser.add_argument("--output_dir", type=str, default="./one_iter_data/workdirs_bert", help="Output directory")
    args = parser.parse_args()
    
    current_working_directory = os.path.dirname(os.path.abspath(__file__))
    model_name_or_path = "bert-base-uncased"
    dataset_name = "squad"
    batch_size = 12
    learning_rate = 3e-5
    num_train_epochs = 2
    max_seq_length = 384
    doc_stride = 128
    USE_PIN_MEMORY = "False"

    bert_command = [
        "python", os.path.join(current_working_directory, "run_qa.py"),
        "--model_name_or_path", model_name_or_path,
        "--dataset_name", dataset_name,
        "--do_train",
        "--per_device_train_batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--max_seq_length", str(max_seq_length),
        "--doc_stride", str(doc_stride),
        "--output_dir", args.output_dir,
        "--overwrite_output_dir",
        "--dataloader_pin_memory", USE_PIN_MEMORY,
    ]
    
    subprocess.run(bert_command)

if __name__ == "__main__":
    main()
