
from paperpilot.llm import answer_question

question = "What were the main findings of the study?"

print("\nQuestion:", question)
print("\nPaperPilot's answer:\n")

answer = answer_question(question)

print(answer)
