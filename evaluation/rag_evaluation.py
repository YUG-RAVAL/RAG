import pandas as pd
import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from rag_chatbot.services.vectorstore import get_vectorstore

from dotenv import load_dotenv
from datasets import Dataset
from datetime import datetime
import os
import re
import time
import json
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextUtilization, Faithfulness, AnswerRelevancy, ContextRecall, NoiseSensitivity, ContextPrecision, ResponseRelevancy, ContextEntityRecall
from ragas.cost import get_token_usage_for_openai

import opik
from opik import Opik, Trace, Span, configure, track, flush_tracker
from opik.types import ErrorInfoDict, LLMProvider


import tiktoken
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Input file path - can be changed to any file
input_file_path = "test_set/samsung_led_rag_test_questions.csv"

load_dotenv()
OPIK_API_KEY = os.getenv("OPIK_API_KEY")
configure(api_key=OPIK_API_KEY)
DEFAULT_MODEL_NAME = "gpt-4o-mini"
PROVIDER_MAPPING = {
    "openai": LLMProvider.OPENAI,
    "anthropic": LLMProvider.ANTHROPIC,
    "google": LLMProvider.GOOGLE_AI,
    # Add others as needed
}

# Get vector store
vectorstore = get_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[,.!?;:\(\)\[\]{}]', ' ', text)
    return text.strip()

qa_system_prompt = """You are Assistant that can answer questions about the context given,         
        You are an AI agent developed by AI Labs at Aubergine Solutions. Keep responses short and concise use at most 50 words,make sure to include all the information under 50 words, avoid long lists or markdown. Be friendly, helpful, and respectful.
        You are an AI assistant that answers questions based exclusively on the context fetched from a retriever. You must not use any external knowledge, make assumptions, or attempt to search for additional information. Provide responses only when the context explicitly contains the necessary information. If it does not, respond with not having that information.
        
        ## Personality Traits:
        - Friendly and approachable
        - Empathetic and emotionally intelligent
        - Knowledgeable but not pretentious
        - Patient and helpful
        
        ## Always Remember:
        Ensure responses are helpful, concise, and aligned with the user's intent. Keep responses conversational, concise, and maintain a natural dialogue flow. Be friendly, approachable, and engaging. Adapt personality and responses based on the user's style and context. Do not break persona. Create a seamless, engaging conversation that feels human.generate answers use atmost 50 words.
        
        <context>
        {context}
        </context>
        
        if the given context is empty or not relevant to the question, respond politely and say that you don't have that information.
    """

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ]
)


def create_error_info(e: Exception) -> ErrorInfoDict:
    """Create a properly formatted error_info dict for OPIK."""
    return {"message": str(e),
            "exception_type": type(e).__name__,
            "traceback": ""
            }


class CostTracker:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.token_counts = {
            "input": 0,
            "output": 0
        }
        self.api_calls = 0
        # Cost per 1000 tokens (adjust based on current OpenAI pricing)
        self.cost_rates = {
            "gpt-4o": {
                "input": 0.0025,  # $0.01 per 1K input tokens
                "output": 0.01  # $0.03 per 1K output tokens
            },
            "gpt-4o-mini": {
                "input": 0.00015,  # $0.00015 per 1K input tokens
                "output": 0.0006  # $0.0006 per 1K output tokens
            },
            "gpt-3.5-turbo": {
                "input": 0.0005,  # $0.0005 per 1K input tokens
                "output": 0.0015  # $0.0015 per 1K output tokens
            }
        }
        self.start_time = None
        self.end_time = None
        # Track costs by operation
        self.operation_costs = {
            "qa_generation": {"input": 0, "output": 0, "calls": 0},
            "ragas_evaluation": {"input": 0, "output": 0, "calls": 0}
        }

    def start(self):

        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def add_tokens(self, input_tokens, output_tokens, operation=None):
        self.token_counts["input"] += input_tokens
        self.token_counts["output"] += output_tokens
        self.api_calls += 1

        # Track by operation if specified
        if operation and operation in self.operation_costs:
            self.operation_costs[operation]["input"] += input_tokens
            self.operation_costs[operation]["output"] += output_tokens
            self.operation_costs[operation]["calls"] += 1

    def calculate_cost(self):
        input_cost = (self.token_counts["input"] / 1000) * self.cost_rates[self.model_name]["input"]
        output_cost = (self.token_counts["output"] / 1000) * self.cost_rates[self.model_name]["output"]
        total_cost = input_cost + output_cost

        # Calculate costs by operation
        operation_cost_details = {}
        for op, counts in self.operation_costs.items():
            op_input_cost = (counts["input"] / 1000) * self.cost_rates[self.model_name]["input"]
            op_output_cost = (counts["output"] / 1000) * self.cost_rates[self.model_name]["output"]
            operation_cost_details[op] = {
                "input_tokens": counts["input"],
                "output_tokens": counts["output"],
                "calls": counts["calls"],
                "input_cost": op_input_cost,
                "output_cost": op_output_cost,
                "total_cost": op_input_cost + op_output_cost
            }

        return {
            "input_tokens": self.token_counts["input"],
            "output_tokens": self.token_counts["output"],
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "api_calls": self.api_calls,
            "execution_time": self.end_time - self.start_time if (self.end_time and self.start_time) else None,
            "operation_costs": operation_cost_details
        }


def normalize_text_for_bleu(text):
    """
    Normalize text to improve BLEU score comparison.
    
    Args:
        text (str): The text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation that doesn't affect meaning
    text = re.sub(r'[,.!?;:\(\)\[\]{}]', ' ', text)
    
    # Remove extra whitespace
    text = text.strip()
    
    return text


def evaluate_rag(dataset_name,input_path=input_file_path, model_name=DEFAULT_MODEL_NAME, experiment_name=None, provider='openai',
                 project_name="AUB-RAG-BE"):
    if experiment_name is None:
        experiment_name = f"rag-eval-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Initialize OPIK client
    opik_client = Opik()

    # Initialize cost tracker with gpt-4o-mini model
    cost_tracker = CostTracker(model_name=model_name)

    # Start cost tracking
    cost_tracker.start()
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        dataset = opik_client.get_dataset(name=dataset_name)
        logger.info(f"Retrieved dataset '{dataset_name}' from Opik")

        # Convert to pandas DataFrame
        df_original = dataset.to_pandas()

        # Check and prepare columns for evaluation
        if "input" in df_original.columns:
            df = pd.DataFrame()
            df["Question"] = df_original["input"]
            if "expected_output" in df_original.columns:
                
                df["Answer"] = df_original["expected_output"]
                logger.info("Preprocessed reference answers for better BLEU score comparison")
            else:
                logger.warning("Dataset has no 'expected_output' column. Proceeding without ground truth answers.")
                df["Answer"] = ""
        else:
            raise ValueError(f"Dataset '{dataset_name}' does not have an 'input' column required for evaluation")

        logger.info(f"Dataset contains {len(df)} questions for evaluation")
    except Exception as e:
        logger.error(f"Failed to retrieve dataset '{dataset_name}' from Opik: {str(e)}")
        raise

    # Create a trace with the required project_name parameter
    trace = opik_client.trace(
        name=experiment_name,
        project_name=project_name,
        metadata={
            "model": model_name,
            "dataset": input_path,
            "num_queries": len(df),
            "retriever_k": 5
        }
    )
    logger.info(f"OPIK Project: {project_name}, Experiment: {experiment_name}")
    logger.info(f"Using model: {model_name}")
    try:
        if provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            logger.info(f"Initializing OpenAI LLM with model: {model_name}")
            llm = ChatOpenAI(model_name=model_name, api_key=openai_api_key)
            opik_provider = LLMProvider.OPENAI
        elif provider == "anthropic":
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            logger.info(f"Initializing Anthropic LLM with model: {model_name}")
            # Assuming you have a similar class for Anthropic
            llm = ChatAnthropic(model_name='claude-3-haiku-20240307', api_key=anthropic_api_key)
            opik_provider = LLMProvider.ANTHROPIC
        # elif provider == "deepseek":
        #     logger.info(f"Initializing Anthropic LLM with model: {model_name}")
        #     # Assuming you have a similar class for Anthropic
        #     llm = ChatDeepSeek(model_name=model_name, api_key=deepseek_api_key)
        #     opik_provider = LLMProvider.ANTHROPIC
        elif provider == "google":
            logger.info(f"Initializing Google LLM with model: {model_name}")
            # Assuming you have a similar class for Google
            # llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=google_api_key)
            # opik_provider = LLMProvider.GOOGLE_AI
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


        dataset_records = []

        for i, row in df.iterrows():
            question = row["Question"]
            expected_answer = row["Answer"]

            # Create a span for this question
            question_span = trace.span(
                name=f"process_question_{i}",
                metadata={"question": question}
            )

            try:
                # Retrieve relevant documents
                retrieval_span = trace.span(
                    name="document_retrieval",
                    parent_span_id=question_span.id,
                    input={"question": question}
                )

                try:
                    retrieved_context = vectorstore.similarity_search(question, k=5)

                    # Update span with results
                    retrieval_span.update(
                        metadata={"num_docs": len(retrieved_context)},
                        output={"documents": [doc.page_content[:200] + "..." for doc in retrieved_context]},
                        end_time=datetime.now()
                    )
                except Exception as e:
                    retrieval_span.update(
                        error_info=create_error_info(e),
                        end_time=datetime.now()
                    )
                    raise

                # Create QA chain
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Generate answer with span
                generation_span = trace.span(
                    name="qa_generation",
                    parent_span_id=question_span.id,
                    type="llm",
                    model=model_name,
                    provider=opik_provider,
                    input={"question": question}
                )
                start_generation_time = time.time()
                try:
                    response = rag_chain.invoke({"input": question})

                    generation_time = time.time() - start_generation_time

                    # Extract answer from response
                    generated_answer = response["answer"]

                    # Accurately count tokens using tiktoken
                    input_tokens = count_tokens(question, model=model_name)
                    output_tokens = count_tokens(generated_answer, model=model_name)

                    # Create token usage dict for span
                    token_usage = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        # "counted_with_tiktoken": 1
                    }

                    # Track the tokens
                    cost_tracker.add_tokens(input_tokens, output_tokens, operation="qa_generation")

                    # Also track compute time
                    # cost_tracker.add_compute_time("qa_generation", generation_time)

                    # Update generation span with result
                    generation_span.update(
                        output={"answer": generated_answer},
                        usage=token_usage,
                        metadata={
                            "generation_time_seconds": generation_time,
                            # "tokens_counted_with": "tiktoken"
                        },
                        end_time=datetime.now()
                    )

                except Exception as e:
                    generation_span.update(
                        end_time=datetime.now()
                    )
                    raise

                # Update question span
                question_span.update(
                    metadata={"answer_length": len(generated_answer)},
                    end_time=datetime.now()
                )

                # Normalize answers for better BLEU score comparison
                normalized_generated_answer = normalize(generated_answer)
                normalized_expected_answer = normalize(expected_answer)
                
                # Append to dataset for RAGAS
                dataset_records.append({
                    "question": question,
                    "answer": normalized_generated_answer,
                    "retrieved_contexts": [doc.page_content for doc in retrieved_context],
                    "reference": normalized_expected_answer
                })

            except Exception as e:
                # Update question span with error
                question_span.update(
                    error_info=create_error_info(e),
                    end_time=datetime.now()
                )
                logger.error(f"Error processing question {i}: {str(e)}")

        # Create dataset from records
        evaluation_dataset = Dataset.from_list(dataset_records)

        # Initialize LLM wrapper for RAGAS - also using gpt-4o-mini for evaluation
        evaluator_llm = LangchainLLMWrapper(llm)

        # RAGAS evaluation span
        ragas_span = trace.span(
            name="ragas_evaluation",
            type="llm",  # Change to llm type for cost tracking
            provider=opik_provider,  # Specify provider
            model=model_name,  # Specify model
            input={"dataset_size": len(dataset_records)}
        )

        try:
            # Run RAGAS evaluation
            results = evaluate(
                dataset=evaluation_dataset,
                metrics=[
                    ContextUtilization(),
                    AnswerRelevancy(),
                    Faithfulness(),
                    ContextRecall(),
                    NoiseSensitivity(),
                    ResponseRelevancy(),
                    ContextEntityRecall(),
                    ContextPrecision()
                ],
                llm=evaluator_llm,
                token_usage_parser=get_token_usage_for_openai
            )
            tokens = results.total_tokens()
            logger.info(f"{tokens} are used for evaluation")
            # Extract token usage from RAGAS if available
            ragas_token_usage = {}


            if tokens.input_tokens > 0 or tokens.output_tokens > 0:
                ragas_token_usage = {
                    "prompt_tokens": tokens.input_tokens,
                    "completion_tokens": tokens.output_tokens,
                    "total_tokens": tokens.input_tokens + tokens.output_tokens
                }

            cost_tracker.add_tokens(tokens.input_tokens, tokens.output_tokens, operation="ragas_evaluation")

            # Convert results to DataFrame
            dataframe = results.to_pandas()

            # Calculate mean of numeric columns
            mean_values = dataframe.select_dtypes(include=['float64', 'int64']).mean()

            # Create a new row for the mean values
            mean_row = pd.DataFrame([mean_values], index=['Mean'])

            # Append the mean row to the dataframe
            dataframe = pd.concat([dataframe, mean_row])

            # Extract metrics for the span
            metrics_dict = {}
            for col in dataframe.columns:
                if col not in ['user_input', 'retrieved_contexts', 'response', 'reference']:
                    try:
                        # These are the metric columns: context_utilization, answer_relevancy, faithfulness, context_recall, bleu_score
                        metrics_dict[col] = float(mean_values[col])
                    except (ValueError, TypeError):
                        # Skip non-numeric columns
                        pass



            # Update RAGAS span with results
            ragas_span.update(
                metadata=metrics_dict,
                output={"metrics": metrics_dict},
                usage=ragas_token_usage,
                end_time=datetime.now()
            )

            # Log feedback scores for metrics
            for metric_name, value in metrics_dict.items():
                trace.log_feedback_score(
                    name=metric_name,
                    value=value,
                    category_name="RAGAS Metrics"
                )


            results_dataset_name = f"{dataset_name}_results_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            results_dataset = opik_client.get_or_create_dataset(name=results_dataset_name)

            # Prepare results records
            results_records = []
            for idx, row in dataframe.iterrows():
                if idx != 'Mean':  # Skip the mean row
                    record = {
                        "input": row.get("user_input", ""),
                        "expected_output": row.get("reference", ""),
                        "generated_output": row.get("response", ""),
                        "contexts": row.get("retrieved_contexts", []) if isinstance(row.get("retrieved_contexts"), list) else [],
                        "metrics": {
                            "context_utilization": float(row.get("context_utilization", 0)),
                            "answer_relevancy": float(row.get("answer_relevancy", 0)),
                            "faithfulness": float(row.get("faithfulness", 0)),
                            "context_recall": float(row.get("context_recall", 0)),
                            "bleu_score": float(row.get("bleu_score", 0))
                        },
                        "model": model_name,
                        "evaluation_timestamp": datetime.now().isoformat()
                    }
                    results_records.append(record)

            # Add summary record
            summary_record = {
                "input": "EVALUATION_SUMMARY",
                "expected_output": "",
                "generated_output": "",
                "metrics": {k: float(v) for k, v in metrics_dict.items()},
                "model": model_name,
                "evaluation_timestamp": datetime.now().isoformat(),
                "num_questions": len(dataset_records)
            }
            results_records.append(summary_record)

            # Store in Opik dataset
            results_dataset.insert(results_records)

            # Save the DataFrame to CSV
            # dataframe.to_csv(evaluation_results_path, index=False)
            # logger.info(f"Evaluation completed and saved to {evaluation_results_path}")

        except Exception as e:
            ragas_span.update(
                error_info=create_error_info(e),
                end_time=datetime.now()
            )
            logger.error(f"Error in RAGAS evaluation: {str(e)}")
            raise

        # Stop cost tracking
        cost_tracker.stop()

        # Calculate final cost analysis
        cost_analysis = cost_tracker.calculate_cost()

        # Add additional metrics
        cost_analysis["cost_per_query"] = cost_analysis["total_cost"] / len(df) if len(df) > 0 else 0
        cost_analysis["timestamp"] = datetime.now().isoformat()
        cost_analysis["model"] = cost_tracker.model_name
        cost_analysis["num_queries"] = len(df)


        trace_content = opik_client.get_trace_content(trace.id)
        opik_estimated_cost = trace_content.total_estimated_cost
        # Add OPIK's estimated cost to our results
        if opik_estimated_cost is not None:
            logger.info(f"OPIK estimated cost: ${opik_estimated_cost:.4f}")
            # Compare with our cost calculation
            logger.info(f"Our calculated cost: ${cost_analysis['total_cost']:.4f}")
            # Add to cost analysis
            cost_analysis["opik_estimated_cost"] = opik_estimated_cost

            logger.info(f"OPIK estimated cost: ${opik_estimated_cost:.4f}")
            logger.info(f"Our calculated cost: ${cost_analysis['total_cost']:.4f}")

            # Calculate and log difference
            cost_diff = abs(opik_estimated_cost - cost_analysis["total_cost"])
            diff_percentage = (cost_diff / cost_analysis["total_cost"]) * 100 if cost_analysis["total_cost"] > 0 else 0
            logger.info(f"Cost difference: ${cost_diff:.4f} ({diff_percentage:.2f}%)")

            # Add to cost analysis
            cost_analysis["opik_estimated_cost"] = opik_estimated_cost
            cost_analysis["cost_difference"] = cost_diff
            cost_analysis["cost_difference_percentage"] = diff_percentage

         # Create a cost analysis dataset in Opik
        cost_dataset_name = f"{dataset_name}_costs"
        cost_dataset = opik_client.get_or_create_dataset(name=cost_dataset_name)

        # Add this evaluation's cost
        cost_record = {
            "input": f"COST_ANALYSIS_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "model": model_name,
            "total_cost": cost_analysis["total_cost"],
            "cost_per_query": cost_analysis["cost_per_query"],
            "input_tokens": cost_analysis["input_tokens"],
            "output_tokens": cost_analysis["output_tokens"],
            "api_calls": cost_analysis["api_calls"],
            "execution_time": cost_analysis["execution_time"],
            "timestamp": cost_analysis["timestamp"],
            "operation_costs": {
                "qa_generation": cost_analysis["operation_costs"]["qa_generation"]["total_cost"],
                "ragas_evaluation": cost_analysis["operation_costs"]["ragas_evaluation"]["total_cost"]
            }
        }
        cost_dataset.insert([cost_record])
        logger.info(f"Cost analysis stored in Opik dataset: {cost_dataset_name}")

        # Save cost analysis to JSON
        # with open(cost_analysis_path, 'w') as f:
        #     json.dump(cost_analysis, f, indent=2)

        # Update trace with cost information
        trace.update(
            metadata={
                # Cost metrics
                "total_cost": cost_analysis["total_cost"],
                "cost_per_query": cost_analysis["cost_per_query"],
                "total_input_tokens": cost_analysis["input_tokens"],
                "total_output_tokens": cost_analysis["output_tokens"],
                "execution_time_seconds": cost_analysis["execution_time"],
                "opik_estimated_cost": opik_estimated_cost,
                # RAGAS metrics if available
                "context_utilization": metrics_dict.get("context_utilization", 0),
                "answer_relevancy": metrics_dict.get("answer_relevancy", 0),
                "faithfulness": metrics_dict.get("faithfulness", 0),
                "context_recall": metrics_dict.get("context_recall", 0),
                "bleu_score": metrics_dict.get("bleu_score", 0)
            },
            input={"dataset": input_path, "num_queries": len(df)},
            output={"ragas_mean_scores": metrics_dict, "total_cost": cost_analysis["total_cost"]},
            end_time=datetime.now(),
        )

        # logger.info(f"Cost analysis saved to {cost_analysis_path}")
        logger.info(f"Total estimated cost: ${cost_analysis['total_cost']:.4f}")
        logger.info(f"Cost per query: ${cost_analysis['cost_per_query']:.4f}")
        logger.info(f"Experiment data available in OPIK project: {project_name}")
        opik_client.flush()

        # Return results
        return {
            "evaluation_results": dataframe,
            "cost_analysis": cost_analysis,
            "experiment_name": experiment_name,
            "model": model_name,
            "project_name": project_name,
            "trace_id": trace.id
        }

    except Exception as e:
        # Update trace with error in case of overall failure
        trace.update(
            error_info=create_error_info(e),
            end_time=datetime.now()
        )
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fall back to character-based estimation if tiktoken fails
        return len(text) // 4


if __name__ == "__main__":
    results = evaluate_rag(
        dataset_name="Samsung-LCD-RAG-Questions",
        model_name=DEFAULT_MODEL_NAME,
        project_name="RAG-Evaluation",
        provider="openai",
        experiment_name=f"rag-eval-{DEFAULT_MODEL_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logger.info(f"Evaluation complete: {results['experiment_name']} using {results['model']}")