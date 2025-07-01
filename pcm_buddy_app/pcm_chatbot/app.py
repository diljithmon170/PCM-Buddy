# main.py
from utils import get_answer

if __name__ == "__main__":
    subject = input("Enter subject (Physics / Chemistry / Maths): ").strip().capitalize()
    query = input("Enter your class 10 question: ").strip()

    print("\nGenerating answer...\n")
    answer = get_answer(query, subject)
    print(f"\n🧠 Answer:\n{answer}")
