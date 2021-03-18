#https://huggingface.co/transformers/_modules/transformers/generation_utils.html
import json
import requests
import time
from transformers import pipeline, set_seed

API_TOKEN = "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"
headers = {"Authorization": "Bearer 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"}
API_URL = "https://api-inference.huggingface.co/models/mrm8488/CodeGPT-small-finetuned-python-token-completion"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

##ST = time.time()
##data = query({"inputs":"from pandas", "parameters":{"use_gpu":True}})
##print(data)
##print("time with gpu", time.time() - ST)
ST = time.time()
data = query({"inputs":"experimental", "parameters": {"num_return_sequences": 4, "num_beams":4, "num_beam_groups":4, "diversity_penalty":0.5}})

print(data)
print("time without gpu", time.time() - ST)


ST = time.time()
data = query({"inputs":"from numpy import", "parameters": {"num_return_sequences": 4, "num_beams":4, "num_beam_groups":4, "diversity_penalty":0.5}})

print(data)
print("time without gpu", time.time() - ST)

ST = time.time()
data = query({"inputs":"from pandas import", "parameters": {"num_return_sequences": 4, "num_beams":4, "num_beam_groups":4, "diversity_penalty":0.5}, "options":{"use_gpu":True}})

print(data)
print(data.headers)
print("time with gpu", time.time() - ST)

generator = pipeline('text-generation', model='mrm8488/CodeGPT-small-finetuned-python-token-completion')
#set_seed(42)
ST = time.time()
#setting max_length=4 gives us more suggestions which is good b/c we can rank them
print(generator("from json import", num_return_sequences=4, num_beams=4, num_beam_groups=4, diversity_penalty=0.5))
print(time.time()-ST)
#print(generator("def query(payload): data = json.dumps(payload) response =", max_length=100, num_return_sequences=5, num_beams=5, output_scores=True))

'''
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
'''
