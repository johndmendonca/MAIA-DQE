# MAIA-DQE

This is the official repository for the MAIA-DQE (Dialogue Quality and Emotion annotations) dataset. 


## Data

The dataset is split into independent subsets and can be found in `data/`. 

Each subset is a list of dialogues identified using an `id`.

A dialogue consists of a list of `turns` and a `dialogue` dictionary that contains `"Dropped conversation"` and `"Task Sucess"` annotations.

Each turn is a dictionary formed by a list of sentences representing the turn from the POV of agent `"text_mt"` and client `"text_src"`.

`""floor"` identifies the direction of the conversation.
* `inbound` indicates the customer is speaking.
* `outbound` indicates the agent is speaking.

Accompanying the turn we have sentence level annotations `"Correctness"`, `"Templated"`, `"Engagement"`, `"Emotion"` (each a list with size equal to the number of sentences) and turn level annotations `"Understanding"`, `"Sensibleness"`, `"Politeness"`, `"IQ"`.

## Benchmarking

Benchmark code is also provided and can be reproduced in `DialogueEvaluation/` and `Emotion/`.

## Citation

If you use this work, please consider citing:

~~~
John Mendonça, Patrícia Pereira, Miguel Menezes, Vera Cabarrão, Helena Moniz, João Paulo Carvalho, Alon Lavie, and Isabel Trancoso. 2023. Dialogue Quality and Emotion Annotations for Customer Support Conversations. In Proceedings of the 3rd Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2023), Singapore. Association for Computational Linguistics.
~~~