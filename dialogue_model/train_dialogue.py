import math
import os.path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import sklearn.model_selection


class GptDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        self.transform = None
        with open(file_path, encoding="utf-8") as f:
            chunks = f.read().split('\n\n')
            for chunk in chunks:
                if len(chunk) > 0:
                    tokenized_chunk = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chunk.strip()))
                    l = len(tokenized_chunk)
                    if l < block_size:
                        self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_chunk))
                    else:
                        while tokenized_chunk:
                            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_chunk[:block_size]))
                            tokenized_chunk = tokenized_chunk[block_size:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.examples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    proj_dir = os.path.expanduser('')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 50

    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    dataset_path = os.path.join(proj_dir, 'tmp', 'chan_dialogues.txt')

    print(f'Loading samples from "{dataset_path}"...')
    train_dataset = GptDialogueDataset(tokenizer=tokenizer,  file_path=dataset_path, block_size=128)
    train_samples, test_samples = sklearn.model_selection.train_test_split(train_dataset, test_size=0.1)
    print(f'{len(train_dataset)} samples in total, {len(train_samples)} in train, {len(test_samples)} in test')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_model_dir = os.path.join(proj_dir, "tmp/rugpt_short_dialogie")

    training_args = TrainingArguments(
        output_dir=output_model_dir,
        overwrite_output_dir=True,
        learning_rate=3e-4,
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        save_total_limit=2,
        warmup_steps=100,
        logging_first_step=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        disable_tqdm=True
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_samples,
        eval_dataset=test_samples,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    loss_start = trainer.evaluate()['eval_loss']
    ppl_start = math.exp(loss_start)
    print(f"\nPerplexity before finetuning = {ppl_start:.2f}\n")

    train_result = trainer.train()

    eval_hist = [(0, 0.0, loss_start)] + sorted([(info['step'], info['epoch'], info['eval_loss']) for info in trainer.state.log_history if 'eval_loss' in info], key=lambda z: z[0])
    with open(os.path.join(proj_dir, 'tmp', 'train_chitchat_rugpt.eval_history.tsv'), 'w') as f:
        f.write('step\tepoch\tloss\tperplexity\n')
        for step, epoch, loss in eval_hist:
            ppl = math.exp(loss)
            f.write(f'{step}\t{epoch}\t{loss}\t{ppl}\n')

    ppl_end = math.exp(trainer.evaluate()['eval_loss'])
    print(f"\nPerplexity after finetuning = {ppl_end:.2f}\n")

    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)