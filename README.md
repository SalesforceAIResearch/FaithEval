# <img src="assets/logo.png" alt="FaithEval" width="26"/>FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"



![faitheval](https://img.shields.io/badge/Dataset-FaithEval-blue) 
![unanswerable](https://img.shields.io/badge/Task-Unanswerable_QA-red) 
![inconsistent](https://img.shields.io/badge/Task-Inconsistent_QA-red) 
![counterfactual](https://img.shields.io/badge/Task-Counterfactual_QA-red) 
![Claude-3.5](https://img.shields.io/badge/Model-Claude--3.5-green) 
![GPT-4](https://img.shields.io/badge/Model-GPT--4--Turbo-green) 
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green)
![Command-R](https://img.shields.io/badge/Model-Command_R-green)
![Command-R](https://img.shields.io/badge/Model-Command_R+-green)
![Mistral](https://img.shields.io/badge/Model-Mistral-green)
![llama](https://img.shields.io/badge/Model-Llama--3-green)
![llama](https://img.shields.io/badge/Model-Llama--3.1-green)
![Gemma](https://img.shields.io/badge/Model-Gemma--2-green)
![Phi-3.5](https://img.shields.io/badge/Model-Phi--3-green)
![Phi-3.5](https://img.shields.io/badge/Model-Phi--3.5-green) 

<p align="center">
    <img src="./assets/logo_2.png" width="30%"> <br>
</p>

This is the codebase for [FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"](https://arxiv.org/pdf/2410.03727). 


✨ FaithEval is a new and comprehensive benchmark dedicated to evaluating contextual faithfulness in LLMs across three diverse tasks: unanswerable, inconsistent, and counterfactual contexts [[Huggingface Dataset](https://huggingface.co/collections/Salesforce/faitheval-benchmark-66ff102cda291ca0875212d4)]

<p align="center">
    <img src="./assets/perf_summary.png" width="80%"> <br>
  Performance summary on <b>FaithEval</b> Benchmark. Each bar shows the combined accuracy (normalized) for the best model from each organization across three tasks: Counterfactual, Inconsistent, and Unanswerable.
</p>


## Updates
- Oct 3: A preview of FaithEval benchmark is available on HuggingFace. Evaluation scripts will be released soon.

## 🔍 About FaithEval
Ensuring faithfulness to context in **large language models (LLMs)** and **retrieval-augmented generation (RAG)** systems is crucial for reliable deployment in real-world applications, as incorrect or unsupported information can erode user trust. Despite advancements on standard benchmarks, faithfulness hallucination—where models generate responses misaligned with the provided context—remains a significant challenge. In this work, we introduce FaithEval, a novel and comprehensive benchmark tailored to evaluate the faithfulness of LLMs in contextual scenarios across three diverse tasks: unanswerable, inconsistent, and counterfactual contexts. These tasks simulate real-world challenges where retrieval mechanisms may surface incomplete, contradictory, or fabricated information. FaithEval comprises 4.9K high-quality problems in total, validated through a rigorous four-stage context construction and validation framework, employing both LLM-based auto-evaluation and human validation. Our extensive study across a wide range of open-source and proprietary models reveals that even state-of-the-art models often struggle to remain faithful to the given context, and that larger models do not necessarily exhibit improved faithfulness. 


## 🗂️ Dataset Examples
![Summary](./assets/samples_demo.png)

<details>
<summary>🔍 Click to expand/collapse task explanations</summary>

- Unanswerable Context: the context does not contain the answer to the question.

- Inconsistent Context: multiple answers are supported by different documents.

- Counterfactual Context: the context contains counterfactual statements that contradict common sense or world knowledge.

</details>

## 🧩 Task Construction and Validation Pipeline 
![Summary](./assets/pipeline.png)

**Source Datasets:**
- [SQuAD](https://arxiv.org/abs/1606.05250) 
- [NewsQA](https://arxiv.org/abs/1611.09830) 
- [TriviaQA](https://arxiv.org/abs/1705.03551)
- [Natural Questions](https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/) 
- [SearchQA](https://github.com/nyu-dl/dl4ir-searchqa)
- [HotpotQA](https://arxiv.org/abs/1809.09600) 
- [BioASQ](http://bioasq.org/) 
- [DROP](https://arxiv.org/abs/1903.00161) 
- [RACE](https://arxiv.org/abs/1704.04683)
- [TextbookQA](https://prior.allenai.org/projects/tqa) 
- [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)


## 📊 Model Performance Summary
- Unanswerable Context
 ![Summary](./assets/model_performance_un.png)
- Inconsistent Context
 ![Summary](./assets/model_performance_ic.png)
- Counterfactual Context
 ![Summary](./assets/model_performance_cc.png)
 
## Quick Start 
FaithEval benchmark is available on HuggingFace. To load it, simply run the following code:
```python
from datasets import load_dataset

inconsistent_dataset = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")
counterfactual_dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")
unanswerable_dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")
```


FaithEval can be easily evaluated using standard evaluation scripts for QA datasets. As an example, the following code demonstrates how to load the unanswerable QA dataset and evaluate it with minimal effort. Feel free to modify the code to integrate it with your existing evaluation scripts.

```python
from datasets import load_dataset
from tqdm import tqdm
import torch 
import string
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

# define evaluationdataset and model
dataset_name = f"Salesforce/FaithEval-unanswerable-v1.0"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
cache_dir = "/export/contextual-llm/models"
do_sample = False
strict_match = False

# load model and initialize pipeline
dataset = load_dataset(dataset_name, split="test")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True, device_map='auto')

# For demonstration, here we only evaluate on a subset of 10 samples 
dataset = dataset.select(range(10))
# obtain results
correct = 0
for example in tqdm(dataset, desc="Processing examples"):
    # specify your custom prompt here. For example, if we want the model to directly generate an answer based on the context and question.
    prompt = f"""You are an expert in retrieval question answering. 
Please respond with the exact answer only. Do not be verbose or provide extra information.
If there is no information available from the context, the answer should be 'unknown'.
Context: {example['context']}
Question: {example['question']}
Answer:""" 
    # Not all models support system prompt. If applicable, system prompts can be added as well.
    messages = [{"role": "user", "content": prompt}]
    # If we want greedy decoding:
    outputs = generator(
                messages,
                max_new_tokens=256,
                top_p=None,
                do_sample=False)
    pred_answer = outputs[0]["generated_text"][-1]['content'].strip()
    print(pred_answer, "\n")
    # evaluate the answer
    if not strict_match:
        valid_phrases = ['unknown', 'no answer', 'no information', 'not', 'unclear']
    else:
        valid_phrases = ['unknown']
    if any(phrase in normalize_answer(pred_answer) for phrase in valid_phrases):
        correct += 1
print(f"Accuracy: {correct / len(dataset)}")
```

## Remarks 

This evaluation was conducted for research only purposes. Users need to make their own assessment regarding any obligations or responsibilities under the corresponding licenses or terms and conditions pertaining to the original datasets and data.

## Citation

If you find our project helpful, please consider citing our paper :blush:

```
@article{ming2024faitheval,
  title = {FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"},
  author = {Yifei Ming and Senthil Purushwalkam and Shrey Pandit and Zixuan Ke and Xuan-Phi Nguyen and Caiming Xiong and Shafiq Joty},
   journal={arXiv},
  year = {2024},
}
```
