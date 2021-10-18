# -*- coding: utf-8 -*-
"""01-gpt2-with-value-head.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oA8OGTrINbn0gEIlzj1trQ5mDJ8MwTWU

# GPT2 with value head
> A GPT2 model with a value head built on the `transformer` library by Hugging Face.

## Why a value head?
Optimisation through PPO requires estimates on the current states value. The value can be estimated by adding a second head to the GPT2 model which outputs a scalar for each output token.

## Detach head
I experimented with detaching the head from the body when optimizing the model. This means that only the head is trained and the gradients are not passed through the body. Although I did not use it in the end it is still possible to detach the head by calling `model.detach_head()`.
"""

# default_exp gpt2

# export

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch

# exports

class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)
            
        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

# exports

class GPT2HeadWithValueModel(GPT2PreTrainedModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
       
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
        
        return outputs

"""## Load a pre-trained language model
Loading a pretrained language model works like loading it with a model from the `transformer` library.
"""

model = GPT2HeadWithValueModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

"""## Forward pass"""

input_txt = "I liked the movie Transformers!" + tokenizer.eos_token
input_ids = tokenizer.encode(input_txt, add_special_tokens=True, return_tensors="pt")
logits, transformer_outputs, values = model(input_ids)

"""## Model outputs

We input a batch of `1` with `7` tokens.
"""

input_ids.shape

"""The logits tensor is of shape `[batch_size, num_input_tokens, vocab_size]`:"""

logits.shape

"""The value tensor is of shape `[batch_size, num_input_tokens]`:"""

values.shape

"""We can greedy decode the next token predictions from the logits:"""

pred_ids = torch.argmax(logits, dim=-1)

for i in range(input_ids.shape[1]):
    current_id = tokenizer.decode(input_ids[:, i])
    next_id = tokenizer.decode(pred_ids[:, i])
    print(current_id, '-->', next_id)

"""## Batched response to queries
To speed up computations it helps to process queries in a batched fashion.
"""

# exports

def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]

"""We have the model respond to two queries in parallel:"""

query_txt_1 = "My most favourite movie is"
query_txt_2 = "My least favourite movie is"
queries_txt = [query_txt_1, query_txt_2]

queries = [tokenizer.encode(query_txt, return_tensors="pt") for query_txt in queries_txt]
print([q.shape for q in queries])
queries = torch.cat(queries)

responses = respond_to_batch(model, queries, txt_len=10)

"""**Note:** This only works because both queries have the same number of tokens. If that is not the case one must pad the tensors before stacking them in `torch.cat(queries)`.

Then we can decode the responses:
"""

for i in range(responses.shape[0]):
    response_txt = tokenizer.decode(responses[i])
    query_txt = queries_txt[i]
    print(query_txt + response_txt)

"""## Why the custom response function?
The models in the `transformer` library come with a very useful and optimised generation function `model.generate()`. In the beginning this function was indeed used to generate text but after lengthy debugging it turned out that PPO was exploiting some aspects that are generally useful for text generation but allowed the model to abuse it and gain extra rewards.

### The model reward
To understand how the model was able to exploit the generation function it is worth looking at the reward function for language modeling with PPO. The reward consists of an arbitrary score (any scalar to indicate whether the model output was good or bad) and the KL-divergence from the untrained model:

$$reward = score - \beta \times KL$$

where $\beta$ is some positive factor. The KL divergence is calculate with:

$$ KL = \mathbb{E}_{x \sim p_{model}} [\log p_{model}(x) - \log p_{refmodel}(x)]$$

Since $x$ is sampled from $p_{model}$ the KL-divergence is always positive. However, if the model found a way to get negative KL-divergence it would achieve a positive reward. This is what happened twice with in the experiment and both times a quirk of the text generation was abused to avoid proper sampling from the probability distribution.

### Case 1: `min_length=None`
When no `min_length` is specified in the `model.generate()` function the model probability distribution is normally sampled until the first `<eos>` token appears. Then the rest of the sequence is padded with a padding token until `max_length` is reached (for GPT2 this is also the `<eos>` token). If that sequence is again passed through the model to evaluate the log-probabilities everything is normal until after the first `<eos>` token, since multiple `<eos>` tokens are very unlikely. The model exploited this by decreasing the probability for the `<eos>` token after the first appearence even further below the probability of the reference model, thus achieving negative KL-divergence. Additionally, it inserted the first `<eos>` earlier and earlier in the sentences to minimize the KL-divergence and thus maximise the reward. This only worked because the sequence after the first `<eos>` token wasn't properly sampled but padded, otherwise the low probabilities would have lead to other tokens with higher probability being sampled.


### Case 2: `min_length=max_length`
I thought this could be easily fixed: just set the `min_length=max_length`. This seemed to work fine for a few experiments until the training failed again due to negative KL-divergences. Finding the problem was harder than before, since it only happened rarely after several training steps. In addition the generated sentences deteriorated quickly to complete gibberish. After some investigation it turned out that the model was again exploiting the sampling function. Up to this point I was not aware that the model was also not allowed to produce an `<eos>` token before `min_length` is reached. In practice this is achieved by setting the next token logit to -infinity:

```
next_token_logits[:, eos_token_id] = -float("inf")
```

This makes sure that after the softmax function the probability for the `<eos>` token is zero, no matter the model output. The model exploited this by maximizing the logit output for that token and thus setting all other logits to increasingly small numbers. Since, I did not apply the same step when evaluating the generated sequence (calculating softmax without the -inf trick) the probabilities for the generated sequences were extremely small and in fact smaller than the probabilities of the reference model. This lead again to negative KL-divergence.

### Conclusion
In both cases $x \sim p_{model}$ in the KL-divergence equation was not satisfied, but this was hidden in the sequence generating function. Reinforcement Learning is very effective in finding and exploiting environment quirks as others have pointed out for other environments such as ATARI games. The solution was to go back to a simpler sequence sampler to avoid this exploits. Alternatively, I could have applied the same tricks and some masking to the model outputs when evaluating the sequences, but I didn't feel confident enough that there would not be other sequence generation tricks the model could abuse.
"""

