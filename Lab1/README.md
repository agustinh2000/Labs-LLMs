# Lab1

This code performs summary generation for a set of dialogues using a pre-trained seq2seq (sequence-to-sequence) language model from the Transformers library by Hugging Face.

For this purpose, by instantiating a flan-t5-base by means of one-shot inference and using the knkarthick/dialogsum dataset, the corresponding summaries are generated. It was found that when doing few-shot inference, the results did not improve much in comparison with one-shot.
