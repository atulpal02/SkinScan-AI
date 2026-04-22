from .rag_engine import retrieve

def generate_explanation(prediction):
    query = f"What is {prediction} and should I worry?"
    context = retrieve(query)
    return " ".join(context)


def answer_question(prediction, question):
    query = f"{prediction}. {question}"
    context = retrieve(query, k=3)
    return " ".join(context)