import numpy as np

from .container import RAGResult
from nltk.tokenize import sent_tokenize
from . import metrics

def generate_faithfulness_reason(score, faithful, claims):
    reason = ""
    indexes = [i for i, x in enumerate(faithful) if x==False]
    unfaithful_claims = [claims[i] for i in indexes]
    if score ==1 :
        reason += "The response is faithful to the context"
    else:
        reason += f"Potential Unfaithfulness detected for claims: {unfaithful_claims}"

    return reason

def find_triplet_span(paragraph, triplet):
    subject, relation, obj = triplet

    sentences = sent_tokenize(paragraph)

    for sentence in sentences:
        if subject.lower() in sentence.lower() and obj.lower() in sentence.lower():
            relation_words = relation.lower().split()
            if any(word in sentence.lower() for word in relation_words):
                return sentence

    for i, sentence in enumerate(sentences):
        if subject.lower() in sentence.lower() or obj.lower() in sentence.lower():
            context = " ".join(sentences[max(0, i - 1) : min(len(sentences), i + 2)])
            if subject.lower() in context.lower() and obj.lower() in context.lower():
                return context

    return None

def to_bool(checking_results):
    if isinstance(checking_results, str):
        return checking_results == "Entailment"
    return np.array([to_bool(res) for res in checking_results])


def evaluate_precision(result: RAGResult):
    if metrics.precision in result.metrics:
        return
    assert result.answer2response is not None
    answer2response = to_bool(result.answer2response)
    if len(answer2response) > 0:
        result.metrics[metrics.precision] = np.mean(answer2response)
    else:
        result.metrics[metrics.precision] = 0.


def evaluate_recall(result: RAGResult):
    if "recall" in result.metrics:
        return
    assert result.response2answer is not None
    response2answer = to_bool(result.response2answer)
    if len(response2answer) > 0:
        result.metrics[metrics.recall] = np.mean(response2answer)
    else:
        result.metrics[metrics.recall] = 0.


def evaluate_f1(result: RAGResult):
    if "f1" in result.metrics:
        return
    evaluate_precision(result)
    evaluate_recall(result)
    precision = result.metrics[metrics.precision]
    recall = result.metrics[metrics.recall]
    if precision > 0 and recall > 0:
        result.metrics[metrics.f1] = 2 * precision * recall / (precision + recall)
    else:
        result.metrics[metrics.f1] = 0.


def evaluate_claim_recall(result: RAGResult):
    if metrics.claim_recall in result.metrics:
        return
    evaluate_retrieval(result)


def evaluate_context_precision(result: RAGResult):
    if metrics.context_precision in result.metrics:
        return
    evaluate_retrieval(result)


def evaluate_retrieval(result: RAGResult):
    """Evaluate retrieval metrics together as they share the same intermediate results."""
    assert result.retrieved2answer is not None
    retrieved2answer = to_bool(result.retrieved2answer)
    if len(retrieved2answer) > 0 and len(retrieved2answer[0]) > 0:
        claim_recalled = np.max(retrieved2answer, axis=1)
        result.metrics[metrics.claim_recall] = np.mean(claim_recalled)
        psg_useful = np.max(retrieved2answer, axis=0)
        result.metrics[metrics.context_precision] = np.mean(psg_useful)
    else:
        result.metrics[metrics.claim_recall] = 0.
        result.metrics[metrics.context_precision] = 0.


def evaluate_context_utilization(result: RAGResult):
    if "context_utilization" in result.metrics:
        return
    assert result.retrieved2answer is not None and result.response2answer is not None
    retrieved2answer = to_bool(result.retrieved2answer)
    response2answer = to_bool(result.response2answer)
    if len(retrieved2answer) > 0 and len(retrieved2answer[0]) > 0:
        claim_recalled = np.max(retrieved2answer, axis=1)
        if np.sum(claim_recalled) > 0:
            claim_used = claim_recalled & response2answer
            result.metrics[metrics.context_utilization] = np.sum(claim_used) / np.sum(claim_recalled)
        else:
            result.metrics[metrics.context_utilization] = 0.
    else:
        result.metrics[metrics.context_utilization] = 0.


def evaluate_noise_sensitivity_in_relevant(result: RAGResult):
    if metrics.noise_sensitivity_in_relevant in result.metrics:
        return
    evaluate_noise_sensitivity(result)


def evaluate_noise_sensitivity_in_irrelevant(result: RAGResult):
    if metrics.noise_sensitivity_in_irrelevant in result.metrics:
        return
    evaluate_noise_sensitivity(result)


def evaluate_noise_sensitivity(result: RAGResult):
    """Evaluate noise sensitivity metrics together as they share the same intermediate results."""
    assert result.retrieved2response is not None and result.answer2response is not None and \
        result.retrieved2answer is not None
    retrieved2response = to_bool(result.retrieved2response)
    answer2response = to_bool(result.answer2response)
    retrieved2answer = to_bool(result.retrieved2answer)
    if len(answer2response) > 0 and len(retrieved2response[0]) > 0 and len(retrieved2answer) > 0:
        relevant_retrieved = np.max(retrieved2answer, axis=0, keepdims=True)
        relevant_faithful = np.max(relevant_retrieved & retrieved2response, axis=1)
        irrelevant_retrieved = ~np.max(retrieved2answer, axis=0, keepdims=True)
        irrelevant_faithful = np.max(irrelevant_retrieved & retrieved2response, axis=1)
        irrelevant_faithful &= ~relevant_faithful  # to keep them exclusive

        incorrect = ~answer2response
        noise_sensitivity_in_relevant = np.mean(relevant_faithful & incorrect)
        noise_sensitivity_in_irrelevant = np.mean(irrelevant_faithful & incorrect)
        result.metrics[metrics.noise_sensitivity_in_relevant] = noise_sensitivity_in_relevant
        result.metrics[metrics.noise_sensitivity_in_irrelevant] = noise_sensitivity_in_irrelevant
    else:
        result.metrics[metrics.noise_sensitivity_in_relevant] = 0.
        result.metrics[metrics.noise_sensitivity_in_irrelevant] = 0.


def evaluate_hallucination(result: RAGResult):
    if "hallucination" in result.metrics:
        return
    evaluate_unfaithfulness(result)


def evaluate_self_knowledge(result: RAGResult):
    if "self_knowledge" in result.metrics:
        return
    evaluate_unfaithfulness(result)


def evaluate_unfaithfulness(result: RAGResult):
    """Evaluate hallucination and self-knowledge together as they share the same intermediate results."""
    assert result.retrieved2response is not None and result.answer2response is not None
    retrieved2response = list(map(list, zip(*result.retrieved2response)))
    answer2response = result.answer2response
    response_claims = result.response_claims
    response = result.response
    hallucination_result = {}
    detailed_result = []
    retrieved2response_entailment = ['Entailment' if 'Entailment' in sublist else 
                   'Contradiction' if all(val == 'Contradiction' for val in sublist) else 
                   'Neutral' for sublist in retrieved2response]
    print(retrieved2response_entailment)
    if  len(answer2response) > 0 and len(retrieved2response[0]) > 0:
        num_neutral = sum(1 for i in range(len(retrieved2response_entailment)) if retrieved2response_entailment[i] == "Neutral")
        num_contradictions = sum(1 for i in range(len(retrieved2response_entailment)) if retrieved2response_entailment[i] == "Contradiction")
        
        total = len(retrieved2response_entailment)
        score = (num_contradictions + num_neutral) / (total + 1e-8)
        
        unfaithful = ~np.max(to_bool(retrieved2response), axis=1)
        hallucination = [item for item, flag in zip(response_claims, unfaithful & ~to_bool(answer2response)) if flag]

        self_knowledge = [item for item, flag in zip(response_claims, unfaithful & to_bool(answer2response)) if flag] 
        
        for i, claim in enumerate(response_claims):
            label_context = retrieved2response_entailment[i]
            label_answer = answer2response[i]
            span = find_triplet_span(response, claim)
            detailed_result.append({"claim": claim, "label with respect to": {"context": label_context, "gt_answer": label_answer}, "span_text": span})
        hallucination_result = {
            'score': score,
            'hallucinated_claims': hallucination,
            'llm_knowledge': self_knowledge,
            'detailed': detailed_result
        }
        result.metrics[metrics.hallucination] = hallucination_result
        result.metrics[metrics.self_knowledge] = self_knowledge
    else:
        result.metrics[metrics.hallucination] = 0.
        result.metrics[metrics.self_knowledge] = 0.


def evaluate_faithfulness(result: RAGResult):
    assert result.retrieved2response is not None
    retrieved2response = list(map(list, zip(*result.retrieved2response)))
    response_claims = result.response_claims
    response = result.response
    detailed_result = []
    
    if len(retrieved2response) > 0 and len(retrieved2response[0]) > 0:
        num_entailments = sum(1 for x in retrieved2response if any(value == "Entailment" for value in x))
        num_contradictions = sum(1 for x in retrieved2response if all(value == 'Contradiction' for value in x))
        num_neutral = sum(1 for x in retrieved2response if all(value == 'Neutral' for value in x))

        faithful = np.max(to_bool(retrieved2response), axis=1)
        total = num_entailments + num_contradictions + num_neutral
        score = (num_entailments) / (total + 1e-8)
        reason = generate_faithfulness_reason(score, faithful, response_claims)
        
        for i, claim in enumerate(response_claims):
            labels = retrieved2response[i]
            if "Entailment" in labels:
                label = "Entailment"
            elif all(value == 'Contradiction' for value in labels):
                label = "Contradiction"
            else:
                label = "Neutral"
            span = find_triplet_span(response, claim)
            detailed_result.append({"claim": claim, "label": label, "span_text": span})
        faithful = {
            "score": score,
            "reason": reason,
            "detailed": detailed_result
        }
        result.metrics[metrics.faithfulness] = faithful
    else:
        result.metrics[metrics.faithfulness] = 0.
        
def evaluate_correctness(result: RAGResult):
    assert result.answer2response is not None
    answer2response = result.answer2response
    response_claims = result.response_claims
    response = result.response
    detailed_result = []
    if len(answer2response) > 0:
        num_entailments = sum(1 for x in answer2response if x == "Entailment")
        num_contradictions = sum(1 for x in answer2response if x== 'Contradiction')
        num_neutral = sum(1 for x in answer2response if x== 'Neutral')
        
        total = num_entailments + num_contradictions + num_neutral
        score = (num_contradictions) / (total + 1e-8)
        
        for claim, label in zip(response_claims, answer2response):
            if label == "Contradiction":
                span = find_triplet_span(response, claim)
                detailed_result.append({"claim": claim, "label": label, "span_text": span})
        correctness = {
            "score": score,
            "detailed": detailed_result
        }
        result.metrics[metrics.correctness] = correctness
    else:
        result.metrics[metrics.correctness] = 0.


def evaluate_completeness(result: RAGResult):
    assert result.answer2response is not None
    answer2response = result.answer2response
    response_claims = result.response_claims
    response = result.response
    detailed_result = []
    
    if len(answer2response) > 0:
        num_entailments = sum(1 for x in answer2response if x == "Entailment")
        num_contradictions = sum(1 for x in answer2response if x== 'Contradiction')
        num_neutral = sum(1 for x in answer2response if x== 'Neutral')
        
        total = num_entailments + num_contradictions + num_neutral
        score = (num_contradictions + num_neutral) / (total + 1e-8)

        for claim, label in zip(response_claims, answer2response):
            if label != "Entailment":
                span = find_triplet_span(response, claim)
                detailed_result.append({"claim": claim, "label": label, "span_text": span})
        completeness = {
            "score": score,
            "detailed": detailed_result
        }
        result.metrics[metrics.completeness] = completeness
    else:
        result.metrics[metrics.completeness] = 0.


METRIC_FUNC_MAP = {
    metrics.precision: evaluate_precision,
    metrics.recall: evaluate_recall,
    metrics.f1: evaluate_f1,
    metrics.claim_recall: evaluate_claim_recall,
    metrics.context_precision: evaluate_context_precision,
    metrics.context_utilization: evaluate_context_utilization,
    metrics.noise_sensitivity_in_relevant: evaluate_noise_sensitivity_in_relevant,
    metrics.noise_sensitivity_in_irrelevant: evaluate_noise_sensitivity_in_irrelevant,
    metrics.hallucination: evaluate_hallucination,
    metrics.self_knowledge: evaluate_self_knowledge,
    metrics.faithfulness: evaluate_faithfulness,
    metrics.correctness: evaluate_correctness,
    metrics.completeness: evaluate_completeness
}
