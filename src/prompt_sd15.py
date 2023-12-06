from src.prompt import Prompt


class PromptSD15(Prompt):

    @staticmethod
    def _get_hidden_state(prompt_embeds):
        return prompt_embeds.last_hidden_state
