# False faced: can a simple classifier tell when a model is manipulating its activations to evade probes?

LessWrong [write-up](https://www.lesswrong.com/posts/TpjpTxfswnBaXPbEn/false-faced-can-a-simple-classifier-tell-when-a-model-is).

<img width="2500" height="1062" alt="0_cover" src="https://github.com/user-attachments/assets/d75518bc-a016-4838-8c7c-ff6fc92366ff" />

We test whether simple classifiers can tell when a model is hiding concepts in its activations to evade probes. We find that, for Bayesian classifiers, the probabilities are enough to catch most evading rows on concepts seen in training, but the correlation aspect that the Dependent version adds over the Naive approach does not measurably help. None of the classifiers work on a new concept, because they learn the trigger word that elicits the evasion, rather than the actual evasion.
