import logging

from langchain_classic.schema import AIMessage, SystemMessage

from .llm_prompts import (
    build_product_decision_context,
    build_no_product_reprompt,
    build_search_decision_prompt,
)
from .reply_types import (
    ProductSearchResult,
    RecommendationAction,
    RecommendationDecision,
    SearchDecision,
)


class ConversationPolicy:
    """LLM-driven policy for deciding and generating the next assistant response."""

    def __init__(self, chat_model):
        # Keep the raw chat model for free-form response generation. This is used
        # after the structured control decision has already been made.
        self.chat_model = chat_model

        # Wrap the same chat model with Pydantic/LangChain structured-output
        # schemas for control-flow decisions. These wrappers constrain the LLM to
        # return machine-readable objects instead of arbitrary assistant text.
        self.recommendation_action_model = chat_model.with_structured_output(
            RecommendationAction
        )
        self.search_decision_model = chat_model.with_structured_output(SearchDecision)        

    def decide_search_action(self, conversation_history):
        """Decide whether the current turn needs product retrieval."""
        # The prompt tells the LLM to make a narrow routing decision: either build
        # a search query for product retrieval or answer conversationally without
        # touching the vector database.
        prompt = SystemMessage(content=build_search_decision_prompt())

        # The existing conversation provides user intent and prior preferences;
        # the final system prompt supplies the schema-specific decision rules.
        search_decision = self.search_decision_model.invoke(
            conversation_history + [prompt]
        )

        logging.info("Structured search decision: %s", search_decision)

        return search_decision

    def decide_next_response(self, conversation_history, search_result: ProductSearchResult):
        # Ask the LLM to judge the already-retrieved candidates. This method does
        # not perform retrieval; it only interprets the retrieved product set.
        action = self.decide_next_action(
            conversation_history=conversation_history,
            source_knowledge=search_result.source_knowledge,
            found_products=search_result.found_products,
        )

        # Default to no selected product. The structured action may later choose
        # one product, or it may decide the system should ask a follow-up question
        # or recover from irrelevant retrieval results.
        chosen_pid = None
        chosen_product = None

        # Collapse the structured three-way action into the older downstream
        # boolean expected by RecommendationDecision/final-response generation.
        # True means "ask the user for more detail"; False with no chosen product
        # means "wrong track" or another non-recommendation path.
        dive_deeper = (action.action == "dive_deeper")

        if action.action == "recommend":
            chosen_pid = action.product_id

            # The structured schema requires a product_id for recommendations, but
            # it cannot by itself guarantee the ID is one of the retrieved products.
            # This check prevents hallucinated or stale IDs from entering the final
            # recommendation path.
            if chosen_pid not in search_result.found_products:
                logging.warning(
                    "Structured LLM decision selected invalid product_id=%s. "
                    "Available product ids: %s",
                    chosen_pid,
                    list(search_result.found_products),
                )
                # Treat an invalid recommendation as an ambiguity/follow-up case
                # rather than recommending a product that the system cannot ground
                # in the retrieved candidate set.
                chosen_pid = None
                chosen_product = None
                dive_deeper = True
            else:
                # Keep the full ProductCandidate, not just the ID, so later code
                # can use the product text/images when generating the final answer
                # and updating the UI.
                chosen_product = search_result.found_products[chosen_pid]

        # Convert the structured decision into the legacy/debug marker string used
        # by diagnostics and by the reprompting step in generate_final_response().
        initial_llm_response = self._action_to_debug_text(action)

        return RecommendationDecision(
            initial_llm_response=initial_llm_response,
            chosen_pid=chosen_pid,
            chosen_product=chosen_product,
            dive_deeper=dive_deeper,
        )
    
    def decide_next_action(
        self,
        *,
        conversation_history,
        found_products,
        source_knowledge=None,
    ):
        """Choose the next structured recommendation action.

        This method is intentionally narrow so evaluation code can exercise the
        LLM decision layer without running retrieval, Gradio, or final-response
        generation.
        """

        # Build temporary system prompts that are only used for this decision.
        # They are not appended to the persistent conversation history.
        prompt_messages = []

        # source_knowledge is the formatted evidence from the retrieved products:
        # product IDs, names, types, similarity scores, and product details. It
        # gives the LLM grounding material beyond the raw conversation.
        if source_knowledge is not None:
            prompt_messages.append(
                SystemMessage(content=build_product_decision_context(source_knowledge))
            )

        # Add the actual decision instructions and the whitelist of legal product
        # IDs. This is what tells the LLM to choose recommend/dive_deeper/wrong_track.
        prompt_messages.append(
            SystemMessage(content=self._build_structured_action_prompt(found_products))
        )

        # Keep the user's conversation intact, then append decision-only system
        # messages. This lets the LLM evaluate the candidate products in the
        # context of the user's current and prior stated preferences.
        temporary_history = conversation_history + prompt_messages

        logging.info("conversation_history: %s\n\n", temporary_history)

        # Invoke the structured-output model, so the return value should be a
        # RecommendationAction object rather than prose.
        action = self.recommendation_action_model.invoke(temporary_history)

        logging.info("Structured LLM action: %s", action)

        return action


    def generate_final_response(self, conversation_history, decision, source_knowledge):
        # Represent the structured control decision as an assistant message before
        # asking the LLM to produce the final user-facing prose. This preserves the
        # decision as context for the final response generation step.
        ai_ans = AIMessage(content=decision.initial_llm_response)

        if decision.chosen_pid:
            # Recommendation path: reprompt the base chat model with the selected
            # product's full LLM-facing description. The LLM should turn that into
            # a useful recommendation, not dump every product field.
            reprompt_str = (
                "Let's continue the conversation while recommending the following product "
                "(you don't need to describe every detail of the product, just whatever seems relevant "
                "for the buyer based on this conversation): "
            )
            reprompt_str += decision.chosen_product.llm_str
            log_message = "Recommendation LLM Response"
        else:
            # Non-recommendation path: build a prompt that either asks for more
            # preferences (dive_deeper=True) or explains/recover from irrelevant
            # retrieval results (dive_deeper=False).
            reprompt_str, log_message = build_no_product_reprompt(
                dive_deeper=decision.dive_deeper,
                source_knowledge=source_knowledge,
            )

        reprompt = SystemMessage(content=reprompt_str)

        # The final response sees the real conversation, the internal decision
        # marker, and the final reprompt. The temporary messages are not persisted;
        # only the generated final answer is appended by ShopTalkRecommender.
        temporary_history = conversation_history + [ai_ans, reprompt]

        final_llm_response = self.chat_model.invoke(temporary_history).content
        logging.info("%s: %s", log_message, final_llm_response)

        return final_llm_response
    
    def _build_structured_action_prompt(self, found_products):
        # Only these IDs are valid recommendation targets. The product details are
        # supplied separately through source_knowledge; this string is the compact
        # whitelist used by the structured decision prompt.
        product_ids = "\n".join(f"- {product_id}" for product_id in found_products)

        return (
            "Choose the next recommendation action using the structured schema.\n\n"
            "Rules:\n"
            "- Use action='recommend' only if one retrieved product is clearly a good fit based upon the user's request and the product evidence.\n"
            "- If action='recommend', product_id must exactly match one of the product ids below.\n"
            "- Use action='dive_deeper' if the retrieved products are broadly relevant, but the user's needs are not clear enough for confident selection.\n"
            "- Use action='wrong_track' if the retrieved products are not relevant to the user's request.\n"
            "- Do not invent product ids.\n\n"
            "Available product ids:\n"
            f"{product_ids}"
        )


    def _action_to_debug_text(self, action: RecommendationAction):
        # Keep the older marker-string representation for diagnostics and for the
        # final-response reprompt. The structured action remains the authoritative
        # decision, but these tags are easy to read in logs/debug output.
        if action.action == "recommend":
            return f"<{action.product_id}>"

        if action.action == "dive_deeper":
            return "<DIVE DEEPER>"

        return "<WRONG TRACK>"
    
    def generate_response_without_search(self, conversation_history):
        """Generate a user-facing response when no new product search is needed."""
        # This path is used when the search-decision layer says the user's latest
        # turn does not require retrieval. The prompt explicitly prevents the LLM
        # from pretending it searched or recommending unseen products.
        prompt = SystemMessage(
            content=(
                "Respond naturally to the user's latest message without recommending "
                "a new product. Do not claim to have searched the product database. "
                "If the user asks a general product-selection question, answer it briefly. "
                "If more shopping preferences would help, ask for them."
            )
        )

        final_llm_response = self.chat_model.invoke(
            conversation_history + [prompt]
        ).content

        logging.info("No-search LLM response: %s", final_llm_response)

        return final_llm_response
