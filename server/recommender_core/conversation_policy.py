import logging

from langchain_classic.schema import AIMessage, SystemMessage

from .llm_prompts import (
    build_augmented_prompt,
    build_no_product_reprompt,
    build_search_query_prompt,
)
from .parsing import parse_product_choice
from .reply_types import ProductSearchResult, RecommendationDecision


class ConversationPolicy:
    """LLM-driven policy for deciding and generating the next assistant response."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def build_search_query(self, conversation_history):
        search_query_prompt = build_search_query_prompt()
        llm_search_query = self.chat_model.invoke(
            conversation_history + [SystemMessage(content=search_query_prompt)]
        ).content
        logging.info("LLM's suggested search query: %s", llm_search_query)
        return llm_search_query

    def decide_next_response(self, conversation_history, search_result: ProductSearchResult):
        initial_llm_response = self.choose_product_or_next_action(
            conversation_history=conversation_history,
            source_knowledge=search_result.source_knowledge,
        )

        chosen_pid, chosen_product, dive_deeper = parse_product_choice(
            llm_response=initial_llm_response,
            found_products=search_result.found_products,
        )

        return RecommendationDecision(
            initial_llm_response=initial_llm_response,
            chosen_pid=chosen_pid,
            chosen_product=chosen_product,
            dive_deeper=dive_deeper,
        )

    def choose_product_or_next_action(self, conversation_history, source_knowledge):
        augmented_prompt = build_augmented_prompt(source_knowledge)

        temporary_history = conversation_history + [
            SystemMessage(content=augmented_prompt)
        ]

        logging.info("conversation_history: %s\n\n", temporary_history)
        llm_response = self.chat_model.invoke(temporary_history).content
        logging.info("Initial LLM Response: %s", llm_response)

        return llm_response

    def generate_final_response(self, conversation_history, decision, source_knowledge):
        ai_ans = AIMessage(content=decision.initial_llm_response)

        if decision.chosen_pid:
            reprompt_str = (
                "Let's continue the conversation while recommending the following product "
                "(you don't need to describe every detail of the product, just whatever seems relevant "
                "for the buyer based on this conversation): "
            )
            reprompt_str += decision.chosen_product["llm_str"]
            log_message = "Recommendation LLM Response"
        else:
            reprompt_str, log_message = build_no_product_reprompt(
                dive_deeper=decision.dive_deeper,
                source_knowledge=source_knowledge,
            )

        reprompt = SystemMessage(content=reprompt_str)
        temporary_history = conversation_history + [ai_ans, reprompt]

        final_llm_response = self.chat_model.invoke(temporary_history).content
        logging.info("%s: %s", log_message, final_llm_response)

        return final_llm_response