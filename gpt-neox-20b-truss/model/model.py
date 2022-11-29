from typing import Dict, List

import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.model = None
        self.tokenizer = None

    def load(self):
        load_dir = str(self._data_dir / "weights/")
        offload_dir = str(self._data_dir / "offload_dir/")
        self.model = GPTNeoXForCausalLM.from_pretrained(
            load_dir,
            device_map="auto",
            offload_folder=offload_dir,
        )
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            load_dir,
            device_map="auto",
            offload_folder=offload_dir,
        )

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        response = {}
        prompt = request["prompt"]  # noqa
        max_length = request.get("max_length", 100)
        temperature = request.get("temperature", 0.9)
        do_sample = request.get("do_sample", True)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=do_sample,
            temperature=temperature,
            max_length=max_length,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        # Invoke model and calculate predictions here.
        response["predictions"] = gen_text
        return response
