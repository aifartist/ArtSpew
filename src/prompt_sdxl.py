from src.prompt import Prompt


class PromptSDXL(Prompt):

    @staticmethod
    def _get_hidden_state(prompt_embeds):
        return prompt_embeds.hidden_states[-2]
