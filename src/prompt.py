import logging
import torch
import numpy as np

PREPARATION_MESSAGE = "Run self.prepare() first."
NOT_IMPLEMENTED_MESSAGE = "This method must be implemented in a subclass."


class Prompt:

    def __init__(self, tokenizers: list, text_encoders: list, unet, initial_text, n_random_tokens, batch_size: int = 1):
        # Public properties.
        self.initial_text = initial_text
        if initial_text is None:
            self.initial_text = ''
        self.n_random_tokens = n_random_tokens
        self.batch_size = batch_size

        # Protected properties.
        self._logger = logging.getLogger(__name__)
        self._tokenizers = tokenizers
        self._text_encoders = text_encoders
        self._unet = unet
        self._vocab_size = tokenizers[0].vocab_size
        self._special_token_ids = set(tokenizers[0].all_special_ids)
        self._text = None
        self._tokens = None
        self._embeds = None
        self._pooled_embeds = None
        self._lazy_random_tokens = None

    # Public properties ################################################################################################

    @property
    def random_tokens(self):
        if self._lazy_random_tokens is None:
            self._lazy_random_tokens = self._generate_random_tokens()
        return self._lazy_random_tokens

    @property
    def text(self):
        if self._text is None:
            raise ValueError(PREPARATION_MESSAGE)
        return self._text

    @property
    def tokens(self):
        if self._tokens is None:
            raise ValueError(PREPARATION_MESSAGE)
        return self._tokens

    @property
    def embeds(self):
        if self._embeds is None:
            raise ValueError(PREPARATION_MESSAGE)
        return self._embeds

    @property
    def pooled_embeds(self):
        if self._pooled_embeds is None:
            raise ValueError(PREPARATION_MESSAGE)
        return self._pooled_embeds

    # Public methods ###################################################################################################

    def prepare(self):
        """This must be called to do the actual work of tokenizing and encoding the prompt, which might be too much work
        to do in the constructor or lazy loading."""

        if not 0 <= self.n_random_tokens <= 75:
            raise ValueError("n random tokens must be between 0 and 75")

        initial_text_list = self.batch_size * [self.initial_text]  # Replicate the prompt for the batch

        prompt_embeds = None
        pooled_prompt_embeds = None
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        text_inputs_list = []

        # Tokenize and encode the prompt with each of the encoders
        for tokenizer, text_encoder in zip(self._tokenizers, self._text_encoders):
            text_inputs = self._tokenize_prompt(initial_text_list)
            text_inputs_list.append(text_inputs)
            prompt_length = self._prompt_length_in_tokens(text_inputs)

            # Append random tokens to the user prompt if needed
            if self.n_random_tokens > 0:
                text_inputs.input_ids = self._append_random_tokens(text_inputs.input_ids, self.random_tokens, prompt_length, self.n_random_tokens)

            prompt_embeds, pooled_prompt_embeds = self._encode_text(text_encoder, text_inputs.input_ids)
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        self._logger.debug("Prompt embeddings [0] shape: " + str(prompt_embeds_list[0].shape))
        if len(prompt_embeds_list) > 1:
            self._logger.debug("Prompt embeddings [1] shape: " + str(prompt_embeds_list[1].shape))
        self._logger.debug("Pooled prompt embeddings [0] shape: " + str(pooled_prompt_embeds_list[0].shape))
        if len(prompt_embeds_list) > 1:
            self._logger.debug("Pooled prompt embeddings [1] shape: " + str(pooled_prompt_embeds_list[1].shape))

        if prompt_embeds is None:
            raise ValueError("Prompt embeddings must not be None.")

        decoded_prompts = []
        for encoded_prompt in text_inputs_list[0].input_ids:
            decoded_prompt = self._tokenizers[0].decode(encoded_prompt, skip_special_tokens=True)
            decoded_prompts.append(decoded_prompt)

        # Concatenate the prompt embeddings from the two encoders.
        if len(prompt_embeds_list) > 1:
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=self._unet.dtype, device='cuda')
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1).view(bs_embed, seq_len, -1)
        if len(prompt_embeds_list) > 1 and pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed, -1)

        self._embeds = prompt_embeds
        self._pooled_embeds = pooled_prompt_embeds
        self._text = decoded_prompts
        self._tokens = text_inputs_list[0].input_ids

    # Private methods ##################################################################################################

    @staticmethod
    def _get_hidden_state(prompt_embeds):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def _generate_random_token(self):
        while True:
            token_id = torch.randint(low=0, high=self._vocab_size, size=(1,), dtype=torch.int32).item()
            if token_id not in self._special_token_ids:
                return token_id

    def _generate_random_tokens(self):
        random_tokens = torch.zeros((self.batch_size, self.n_random_tokens), dtype=torch.int32)
        for i in range(self.batch_size):
            for j in range(self.n_random_tokens):
                random_tokens[i, j] = self._generate_random_token()
        return random_tokens

    def _tokenize_prompt(self, prompt_text):
        return self._tokenizers[0](
            prompt_text,
            padding="max_length",
            max_length=self._tokenizers[0].model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def _prompt_length_in_tokens(self, text_inputs):
        return np.where(text_inputs.input_ids[0] == self._tokenizers[0].eos_token_id)[0][0] - 1

    def _append_random_tokens(self, input_ids, random_tokens, prompt_length, n_random_tokens):
        if prompt_length + n_random_tokens > 75:
            raise ValueError("Number of user prompt tokens and random tokens must be <= 75")
        for i in range(len(input_ids)):
            input_ids[i][1 + prompt_length:1 + prompt_length + n_random_tokens] = random_tokens[i]
            input_ids[i][1 + prompt_length + n_random_tokens] = self._tokenizers[0].eos_token_id
        return input_ids

    def _encode_text(self, text_encoder, text_input_ids):
        prompt_embeds = text_encoder(text_input_ids.to('cuda'), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = self._get_hidden_state(prompt_embeds)
        return prompt_embeds, pooled_prompt_embeds
