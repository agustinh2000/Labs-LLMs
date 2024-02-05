from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig


def make_prompt(example_indices_full, example_index_to_sumarize):
    prompt = ''
    
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt += f"""
    
    Dialogue:
    
    {dialogue}

    What was going on?

    {summary}
    
    """
        dialogue = dataset['test'][example_index_to_sumarize]['dialogue']
        prompt += f"""
    Dialogue:

    {dialogue}

    What was going on?

    """
    return prompt


hugging_face_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(hugging_face_dataset_name)

model_name = 'google/flan-t5-base'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#Convert row text into our vector space that can be processed by the model
tokenizer = AutoTokenizer.from_pretrained(model_name)


dash_line = '-'.join('' for _ in range(0, 50))

example_indices_full = [40]

indices_to_test = [200]

generation_config = GenerationConfig(max_length=50)
# generation_config = GenerationConfig(max_length=50, do_sample=True, temperature=0.7)

for i,index in enumerate(indices_to_test):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    one_shot_prompt = make_prompt(example_indices_full, index)

    # Take a text (generated prompt), tokenize it, convert it into PyTorch tensors,
    # feed it to a pre-trained language model to generate text, and then decode the
    # result to obtain the generated text, skipping special tokens.

    inputs = tokenizer(one_shot_prompt, return_tensors="pt")  # noqa: E501

    output = tokenizer.decode(model.generate(inputs['input_ids'], generation_config=generation_config)[0], skip_special_tokens=True)  # noqa: E501
   
    print(dash_line)
    print(f"Example {i+1}:")
    print(dash_line)
    print("PROMPT:")
    print(one_shot_prompt)
    print(dash_line)
    print("BASELINE HUMAN SUMMARY:")
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print("MODEL GENERATED SUMMARY")
    print(output)