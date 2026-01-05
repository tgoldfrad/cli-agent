import os
import time
import csv
from dotenv import load_dotenv
import openai
import gradio as gr

# טען סביבה
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "אתה מתרגם הוראות בשפה טבעית לפקודות CLI מתאימות ל-Windows (cmd/powershell).\n"
    "התנהג כך:\n"
    "- החזר רק את פקודת ה-CLI המדוייקת בלבד, בלי הסברים, בלי backticks, בלי טקסט נוסף.\n"
    "- אם הבקשה לא ניתנת להמרה או מסוכנת (כמו פקודות שמוחקות כוננים שלמים), אחזר בדיוק: UNABLE_TO_PARSE\n"
    "- השתמש בתחביר של cmd/powershell של Windows (לדוגמה: dir, del, ipconfig, tasklist וכו').\n"
)

LOG_CSV = os.path.join(os.path.dirname(__file__), "results.csv")


def update_test_cases(input_text: str, agent_output: str):
    """עדכון test_cases.csv עם הפלט של ה-agent והוספת score (תקין/שגוי)"""
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), "test_cases.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    # חפש את השורה המתאימה
    mask = df["input"] == input_text
    if mask.any():
        expected = df.loc[mask, "expected_output"].values[0]
        score = "תקין" if agent_output.strip() == str(expected).strip() else "שגוי"
        df.loc[mask, "score"] = score
        df.loc[mask, "agent_output"] = agent_output
        df.to_csv(csv_path, index=False)


def generate_command(user_text: str) -> str:
    """קריאה ל-LLM עם פרומפט מובנה להמרה לפקודת CLI."""
    if not user_text or not user_text.strip():
        return "UNABLE_TO_PARSE"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"הוראה: {user_text}\n
החזר רק את פקודת ה-CLI המתאימה."},
    ]

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )
        cmd = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        cmd = f"UNABLE_TO_PARSE"

    # רישום פשוט
    try:
        with open(LOG_CSV, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user_text, cmd])
    except Exception:
        pass

    # עדכון test_cases.csv
    update_test_cases(user_text, cmd)

    return cmd


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Prompt Engineering בפעולה — ממיר טקסט לפקודת CLI")

    with gr.Row():
        inp = gr.Textbox(label="הוראה בשפה טבעית", lines=3)
        out = gr.Textbox(label="פקודת CLI (תוצאה)", lines=1)

    btn = gr.Button("המר")
    btn.click(fn=generate_command, inputs=inp, outputs=out)

    gr.Markdown("---\nהתוצאות נרשמות ל־results.csv בספריית הפרויקט.")


if __name__ == '__main__':
    demo.launch()
