import logging

from langchain_classic.schema import AIMessage, SystemMessage

from .llm_prompts import (
    build_augmented_prompt,
    build_no_product_reprompt,
    build_search_query_prompt,
)
from .reply_types import (
    ProductSearchResult,
    RecommendationAction,
    RecommendationDecision,
)


class ConversationPolicy:
    """LLM-driven policy for deciding and generating the next assistant response."""

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.action_model = chat_model.with_structured_output(RecommendationAction)

    def build_search_query(self, conversation_history):
        search_query_prompt = build_search_query_prompt()
        llm_search_query = self.chat_model.invoke(
            conversation_history + [SystemMessage(content=search_query_prompt)]
        ).content
        logging.info("LLM's suggested search query: %s", llm_search_query)
        return llm_search_query

    def decide_next_response(self, conversation_history, search_result: ProductSearchResult):
        action = self.choose_product_or_next_action(
            conversation_history=conversation_history,
            source_knowledge=search_result.source_knowledge,
            found_products=search_result.found_products,
        )

        chosen_pid = None
        chosen_product = None
        dive_deeper = action.action == "dive_deeper"

        if action.action == "recommend":
            chosen_pid = action.product_id

            if chosen_pid not in search_result.found_products:
                logging.warning(
                    "Structured LLM decision selected invalid product_id=%s. "
                    "Available product ids: %s",
                    chosen_pid,
                    list(search_result.found_products),
                )
                chosen_pid = None
                chosen_product = None
                dive_deeper = True
            else:
                chosen_product = search_result.found_products[chosen_pid]

        initial_llm_response = self._action_to_debug_text(action)

        return RecommendationDecision(
            initial_llm_response=initial_llm_response,
            chosen_pid=chosen_pid,
            chosen_product=chosen_product,
            dive_deeper=dive_deeper,
        )
    
    def choose_product_or_next_action(
        self,
        conversation_history,
        source_knowledge,
        found_products,
    ):
        augmented_prompt = build_augmented_prompt(source_knowledge)
        structured_prompt = self._build_structured_action_prompt(found_products)

        temporary_history = conversation_history + [
            SystemMessage(content=augmented_prompt),
            SystemMessage(content=structured_prompt),
        ]

        logging.info("conversation_history: %s\n\n", temporary_history)

        action = self.action_model.invoke(temporary_history)

        logging.info("Structured LLM action: %s", action)

        return action

    def generate_final_response(self, conversation_history, decision, source_knowledge):
        ai_ans = AIMessage(content=decision.initial_llm_response)

        if decision.chosen_pid:
            reprompt_str = (
                "Let's continue the conversation while recommending the following product "
                "(you don't need to describe every detail of the product, just whatever seems relevant "
                "for the buyer based on this conversation): "
            )
            reprompt_str += decision.chosen_product.llm_str
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
    
    def _build_structured_action_prompt(self, found_products):
        product_ids = "\n".join(f"- {product_id}" for product_id in found_products)

        return (
            "Choose the next recommendation action using the structured schema.\n\n"
            "Rules:\n"
            "- Use action='recommend' only if one retrieved product is clearly a good fit.\n"
            "- If action='recommend', product_id must exactly match one of the product ids below.\n"
            "- Use action='dive_deeper' if you need more information from the user.\n"
            "- Use action='wrong_track' if the retrieved products are not relevant.\n"
            "- Do not invent product ids.\n\n"
            "Available product ids:\n"
            f"{product_ids}"
        )


    def _action_to_debug_text(self, action: RecommendationAction):
        if action.action == "recommend":
            return f"<{action.product_id}>"

        if action.action == "dive_deeper":
            return "<DIVE DEEPER>"

        return "<WRONG TRACK>"