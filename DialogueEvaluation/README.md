# Dialogue Evaluation benchmarking

## Data processing

The baseline models follow the [QualityAdapt](https://github.com/johndmendonca/qualityadapt) format:


* The tokenizer receives as input `res` and optionally `ctx` (context is needed to evaluate context dependent metrics such as NSP). 
* `ctx` can be multi-turn, the only limitation relates to `max_length=124`. 
* Who said what is determined by appending the speaker token at the start of the sentence.

~~~
A: Gosh, you took all the word right out of my mouth. Let's go out and get crazy tonight.
B: Let's go to the new club on West Street .
A: I'm afraid I can't.


ctx = "<speaker1>Gosh , you took all the word right out of my mouth . Let's go out and get crazy tonight .</s><s><speaker2>Let's go to the new club on West Street ."
res = "<speaker1>I ' m afraid I can ' t ."
~~~

The already processed data can be found in `data/`. Alternatively, you can process the data yourself using `data_processing.py`.

## Evaluation

The baseline models can be obtained from [DialEvalML](https://github.com/johndmendonca/DialEvalML).

To obtain test predictions simply run `score.py`.

