# import os
# import time
# import csv
# from dotenv import load_dotenv
# import openai
# import gradio as gr

# # טען סביבה
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# SYSTEM_PROMPT = (
#     "אתה מתרגם הוראות בשפה טבעית לפקודות CLI מתאימות ל-Windows (cmd/powershell).\n"
#     "התנהג כך:\n"
#     "- החזר רק את פקודת ה-CLI המדוייקת בלבד, בלי הסברים, בלי backticks, בלי טקסט נוסף.\n"
#     "- אם הבקשה לא ניתנת להמרה או מסוכנת (כמו פקודות שמוחקות כוננים שלמים), אחזר בדיוק: UNABLE_TO_PARSE\n"
#     "- השתמש בתחביר של cmd/powershell של Windows (לדוגמה: dir, del, ipconfig, tasklist וכו').\n"
# )

# LOG_CSV = os.path.join(os.path.dirname(__file__), "results.csv")


# def update_test_cases(input_text: str, agent_output: str):
#     """עדכון test_cases.csv עם הפלט של ה-agent והוספת score (תקין/שגוי)"""
#     import pandas as pd
#     csv_path = os.path.join(os.path.dirname(__file__), "test_cases.csv")
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception:
#         return
#     # חפש את השורה המתאימה
#     mask = df["input"] == input_text
#     if mask.any():
#         expected = df.loc[mask, "expected_output"].values[0]
#         score = "תקין" if agent_output.strip() == str(expected).strip() else "שגוי"
#         df.loc[mask, "score"] = score
#         df.loc[mask, "agent_output"] = agent_output
#         df.to_csv(csv_path, index=False)


# def generate_command(user_text: str) -> str:
#     """קריאה ל-LLM עם פרומפט מובנה להמרה לפקודת CLI."""
#     if not user_text or not user_text.strip():
#         return "UNABLE_TO_PARSE"

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"הוראה: {user_text}\nהחזר רק את פקודת ה-CLI המתאימה."},
#     ]

#     try:
#         resp = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             temperature=0.0,
#             max_tokens=150,
#         )
#         cmd = resp["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         cmd = f"UNABLE_TO_PARSE"

#     # רישום פשוט
#     try:
#         with open(LOG_CSV, "a", newline='', encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user_text, cmd])
#     except Exception:
#         pass

#     # עדכון test_cases.csv
#     update_test_cases(user_text, cmd)

#     return cmd


# # Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("# Prompt Engineering בפעולה — ממיר טקסט לפקודת CLI")

#     with gr.Row():
#         inp = gr.Textbox(label="הוראה בשפה טבעית", lines=3)
#         out = gr.Textbox(label="פקודת CLI (תוצאה)", lines=1)

#     btn = gr.Button("המר")
#     btn.click(fn=generate_command, inputs=inp, outputs=out)

#     gr.Markdown("---\nהתוצאות נרשמות ל־results.csv בספריית הפרויקט.")


# if __name__ == '__main__':
#     port = int(os.getenv("PORT", 8080))
#     demo.launch(server_name="0.0.0.0", server_port=port)
import os
import time
import csv
from datetime import datetime
from dotenv import load_dotenv
import openai
import gradio as gr
import pandas as pd

# טען סביבה
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "אתה מתרגם הוראות בשפה טבעית לפקודות CLI מתאימות ל-Windows (cmd/powershell).\n"
    "התנהג כך:\n"
    "- החזר רק את פקודת ה-CLI המדוייקת בלבד, בלי הסברים, בלי backticks, בלי טקסט נוסף.\n"
   
    "- השתמש בתחביר של cmd/powershell של Windows (לדוגמה: dir, del, ipconfig, tasklist וכו').\n"
)

LOG_CSV = os.path.join(os.path.dirname(__file__), "results.csv")


def update_test_cases(input_text: str, agent_output: str):
    """עדכון test_cases.csv עם הפלט של ה-agent והוספת score (תקין/שגוי)"""
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
        {"role": "user", "content": f"הוראה: {user_text}\nהחזר רק את פקודת ה-CLI המתאימה."},
    ]

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
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


def run_automated_tests():
    """מריץ את כל תרחישי הבדיקה ומחזיר תוצאות עם אחוזי הצלחה"""
    csv_path = os.path.join(os.path.dirname(__file__), "test_cases.csv")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"שגיאה בטעינת קובץ הבדיקות: {str(e)}", None, None
    
    results = []
    total_tests = len(df)
    passed_tests = 0
    
    for idx, row in df.iterrows():
        input_text = row['input']
        expected_output = row['expected_output']
        
        # הרץ את הפקודה דרך ה-agent
        agent_output = generate_command(input_text)
        
        # השווה את התוצאות
        is_correct = agent_output.strip() == str(expected_output).strip()
        score = "תקין" if is_correct else "שגוי"
        
        if is_correct:
            passed_tests += 1
        
        results.append({
            "מספר": idx + 1,
            "קלט": input_text,
            "פלט צפוי": expected_output,
            "פלט שהתקבל": agent_output,
            "תוצאה": score
        })
    
    # חשב אחוזי הצלחה
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # שמור תוצאות לקובץ עם חותמת זמן
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_{timestamp}.csv"
    results_path = os.path.join(os.path.dirname(__file__), results_filename)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    # עדכן את test_cases.csv
    df['agent_output'] = results_df['פלט שהתקבל'].values
    df['score'] = results_df['תוצאה'].values
    df.to_csv(csv_path, index=False)
    
    # יצור סיכום
    summary = f"""
    ✅ **סיכום בדיקות אוטומטיות**
    
    📊 **סטטיסטיקה כללית:**
    - סה"כ בדיקות: {total_tests}
    - בדיקות תקינות: {passed_tests}
    - בדיקות שגויות: {total_tests - passed_tests}
    - **אחוז הצלחה: {success_rate:.1f}%**
    
    📁 **קובץ תוצאות נשמר:** {results_filename}
    """
    
    return summary, results_df, results_path


with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .header-title {
        text-align: center;
        color: #667eea;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
""") as demo:
    
    gr.HTML('<div class="header-title">🤖 ממיר טקסט לפקודות CLI</div>')
    gr.HTML('<div class="subtitle">Prompt Engineering Agent - המרת הוראות בשפה טבעית לפקודות Windows</div>')
    
    with gr.Tabs():
        with gr.Tab("🔨 המרת פקודה יחידה"):
            gr.Markdown("### הזן הוראה בשפה טבעית וקבל פקודת CLI מתאימה")
            
            with gr.Row():
                with gr.Column(scale=2):
                    inp = gr.Textbox(
                        label="הוראה בשפה טבעית",
                        placeholder='לדוגמה: "מה כתובת ה-IP של המחשב שלי"',
                        lines=4
                    )
                    btn = gr.Button("המר לפקודת CLI", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    out = gr.Textbox(
                        label="פקודת CLI (תוצאה)",
                        lines=4,
                        interactive=False
                    )
            
            btn.click(fn=generate_command, inputs=inp, outputs=out)
            
            gr.Markdown("---")
            gr.Markdown("💡 **טיפ:** הפקודות נשמרות אוטומטית בקובץ results.csv")
        
        with gr.Tab("🧪 בדיקות אוטומטיות"):
            gr.Markdown("### הרץ את כל תרחישי הבדיקה ובדוק את דיוק ה-Agent")
            
            test_btn = gr.Button("▶️ הרץ בדיקות אוטומטיות", variant="primary", size="lg")
            
            summary_output = gr.Markdown(label="סיכום תוצאות")
            
            results_table = gr.Dataframe(
                label="תוצאות מפורטות",
                wrap=True,
                interactive=False
            )
            
            with gr.Row():
                download_btn = gr.File(label="📥 הורד קובץ תוצאות CSV")
            
            test_btn.click(
                fn=run_automated_tests,
                inputs=[],
                outputs=[summary_output, results_table, download_btn]
            )
            
            gr.Markdown("---")
            gr.Markdown("""
            **📝 הסבר על הבדיקות:**
            - המערכת טוענת את כל תרחישי הבדיקה מ-test_cases.csv
            - מריצה כל אחד דרך ה-Agent
            - משווה את הפלט לתוצאה הצפויה
            - מחשבת אחוזי הצלחה
            - מאפשרת להוריד את התוצאות המלאות כקובץ CSV
            """)


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
