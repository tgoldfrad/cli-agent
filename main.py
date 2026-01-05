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
import json

# טען סביבה
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "אתה מתרגם הוראות בשפה טבעית לפקודות CLI מתאימות ל-Windows (cmd/powershell).\n"
    "entialAction כך:\n"
  
    "- אם הבקשה לא ניתנת להמרה או מסוכנת (כמו פקודות שמוחקות כוננים שלמים), אחזר בדיוק: UNABLE_TO_PARSE\n"
    "- השתמש בתחביר של cmd/powershell של Windows (לדוגמה: dir, del, ipconfig, tasklist וכו').\n"
)

LOG_CSV = os.path.join(os.path.dirname(__file__), "results.csv")
HISTORY_JSON = os.path.join(os.path.dirname(__file__), "test_history.json")


def load_history():
    """טוען את ההיסטוריה של הבדיקות"""
    if os.path.exists(HISTORY_JSON):
        try:
            with open(HISTORY_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    """שומר את ההיסטוריה של הבדיקות"""
    try:
        with open(HISTORY_JSON, 'w', encoding='utf-8') as f:
            json.dump(history, indent=2, fp=f, ensure_ascii=False)
    except Exception as e:
        print(f"שגיאה בשמירת היסטוריה: {e}")


def reset_history():
    """מאפס את ההיסטוריה"""
    save_history([])
    return "ההיסטוריה אופסה בהצלחה!", None


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


def generate_command(user_text: str, custom_prompt: str = None) -> str:
    """קריאה ל-LLM עם פרומפט מובנה להמרה לפקודת CLI."""
    if not user_text or not user_text.strip():
        return "UNABLE_TO_PARSE"

    prompt_to_use = custom_prompt if custom_prompt else SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": prompt_to_use},
        {"role": "user", "content": f"הוראה: {user_text}\nהחזר רק את פקודת ה-CLI המתאימה."},
    ]

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )
        cmd = resp.choices[0].message.content.strip()
        
        if "\`\`\`" in cmd:
            lines = cmd.split("\n")
            cleaned_lines = []
            for line in lines:
                if not line.strip().startswith("\`\`\`"):
                    cleaned_lines.append(line)
            cmd = "\n".join(cleaned_lines).strip()
            
    except Exception as e:
        print(f"[v0] Error calling OpenAI API: {e}")
        cmd = "UNABLE_TO_PARSE"

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


def run_automated_tests(custom_prompt: str, complexity_level: str):
    """מריץ את כל תרחישי הבדיקה לפי רמת מורכבות ומחזיר תוצאות עם אחוזי הצלחה"""
    csv_path = os.path.join(os.path.dirname(__file__), "test_cases.csv")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"שגיאה בטעינת קובץ הבדיקות: {str(e)}", None, None
    
    if complexity_level == "פשוטות":
        # בדיקות 1-5 (פקודות בסיסיות)
        df = df.iloc[0:5]
    elif complexity_level == "בינוניות":
        # בדיקות 6-10 (פקודות עם פרמטרים)
        df = df.iloc[5:10]
    elif complexity_level == "מורכבות":
        # בדיקות 11-15 (פקודות מתקדמות)
        df = df.iloc[10:15]
    # אחרת (הכל) - מריץ את כל הבדיקות
    
    prompt_to_use = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else SYSTEM_PROMPT
    
    results = []
    total_tests = len(df)
    passed_tests = 0
    
    for idx, row in df.iterrows():
        input_text = row['input']
        expected_output = row['expected_output']
        
        agent_output = generate_command(input_text, prompt_to_use)
        
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
    
    history = load_history()
    test_run = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": prompt_to_use,
        "complexity_level": complexity_level,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": round(success_rate, 2),
        "results": results
    }
    history.append(test_run)
    save_history(history)
    
    # שמור תוצאות לקובץ עם חותמת זמן
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_{timestamp}.csv"
    results_path = os.path.join(os.path.dirname(__file__), results_filename)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    # יצור סיכום
    summary = f"""
    ### סיכום בדיקות אוטומטיות
    
    **רמת מורכבות:** {complexity_level}  
    **סה"כ בדיקות:** {total_tests}  
    **בדיקות תקינות:** {passed_tests}  
    **בדיקות שגויות:** {total_tests - passed_tests}  
    **אחוז הצלחה:** {success_rate:.1f}%  
    
    **קובץ תוצאות נשמר:** {results_filename}
    """
    
    return summary, results_df, results_path


def download_full_history():
    """יוצר קובץ CSV עם כל ההיסטוריה"""
    history = load_history()
    
    if not history:
        return None
    
    # בנה רשימה שטוחה של כל התוצאות
    flat_data = []
    for run in history:
        for result in run['results']:
            flat_data.append({
                "תאריך ושעה": run['timestamp'],
                "System Prompt": run['system_prompt'][:100] + "..." if len(run['system_prompt']) > 100 else run['system_prompt'],
                "רמת מורכבות": run['complexity_level'],
                "אחוז הצלחה כללי": f"{run['success_rate']}%",
                "מספר בדיקה": result['מספר'],
                "קלט": result['קלט'],
                "פלט צפוי": result['פלט צפוי'],
                "פלט שהתקבל": result['פלט שהתקבל'],
                "תוצאה": result['תוצאה']
            })
    
    # צור DataFrame ושמור
    df = pd.DataFrame(flat_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"full_history_{timestamp}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    return filepath


def show_history_summary():
    """מציג סיכום של כל ההיסטוריה"""
    history = load_history()
    
    if not history:
        return "אין עדיין היסטוריה של בדיקות.", None
    
    summary_data = []
    for idx, run in enumerate(history, 1):
        summary_data.append({
            "ריצה #": idx,
            "תאריך ושעה": run['timestamp'],
            "רמת מורכבות": run['complexity_level'],
            "סה"כ בדיקות": run['total_tests'],
            "בדיקות תקינות": run['passed_tests'],
            "אחוז הצלחה": f"{run['success_rate']}%",
            "System Prompt (100 תווים ראשונים)": run['system_prompt'][:100] + "..."
        })
    
    df = pd.DataFrame(summary_data)
    
    summary_text = f"""
    ### סיכום היסטוריה
    
    **סה"כ ריצות:** {len(history)}  
    **אחוז הצלחה ממוצע:** {sum(r['success_rate'] for r in history) / len(history):.1f}%
    """
    
    return summary_text, df


with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {
        max-width: 1400px !important;
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
    .prompt-box {
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        background: #f8f9ff;
    }
""") as demo:
    
    gr.HTML('<div class="header-title">🤖 ממיר טקסט לפקודות CLI</div>')
    gr.HTML('<div class="subtitle">Prompt Engineering Agent - המרת הוראות בשפה טבעית לפקודות Windows</div>')
    
    with gr.Tabs():
        with gr.Tab("⚙️ הגדרות System Prompt"):
            gr.Markdown("### ערוך את ה-System Prompt לפי צורכיך")
            gr.Markdown("System Prompt קובע איך ה-Agent מתרגם הוראות לפקודות CLI. נסה גרסאות שונות ובדוק איזו עובדת הכי טוב!")
            
            system_prompt_input = gr.Textbox(
                label="System Prompt",
                value=SYSTEM_PROMPT,
                lines=10,
                placeholder="הכנס את ה-System Prompt המותאם שלך כאן...",
                elem_classes="prompt-box"
            )
            
            gr.Markdown("---")
            gr.Markdown("💡 **טיפ:** אחרי שתשנה את ה-System Prompt, עבור לטאב 'בדיקות אוטומטיות' כדי לבדוק את השפעת השינוי")
        
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
            
            btn.click(fn=lambda text, prompt: generate_command(text, prompt), 
                     inputs=[inp, system_prompt_input], 
                     outputs=out)
            
            gr.Markdown("---")
            gr.Markdown("💡 **טיפ:** הפקודות נשמרות אוטומטית בקובץ results.csv")
        
        with gr.Tab("🧪 בדיקות אוטומטיות"):
            gr.Markdown("### הרץ בדיקות לפי רמת מורכבות ועקוב אחרי ביצועים")
            
            with gr.Row():
                with gr.Column():
                    complexity_selector = gr.Radio(
                        choices=["פשוטות", "בינוניות", "מורכבות", "הכל"],
                        value="הכל",
                        label="רמת מורכבות הפקודות",
                        info="בחר איזה סוג פקודות לבדוק"
                    )
                    
                    test_btn = gr.Button("▶️ הרץ בדיקות אוטומטיות", variant="primary", size="lg")
            
            summary_output = gr.Markdown(label="סיכום תוצאות")
            
            results_table = gr.Dataframe(
                label="תוצאות מפורטות",
                wrap=True,
                interactive=False
            )
            
            with gr.Row():
                download_btn = gr.File(label="📥 הורד קובץ תוצאות ריצה נוכחית")
            
            test_btn.click(
                fn=run_automated_tests,
                inputs=[system_prompt_input, complexity_selector],
                outputs=[summary_output, results_table, download_btn]
            )
            
            gr.Markdown("---")
            gr.Markdown("""
            **📝 הסבר על רמות מורכבות:**
            - **פשוטות:** בדיקות 1-5 (פקודות בסיסיות כמו ipconfig, tasklist)
            - **בינוניות:** בדיקות 6-10 (פקודות עם פרמטרים כמו copy, ren)
            - **מורכבות:** בדיקות 11-15 (פקודות מתקדמות עם pipes וסינונים)
            - **הכל:** מריץ את כל 15 הבדיקות
            """)
        
        with gr.Tab("📊 היסטוריית בדיקות"):
            gr.Markdown("### עקוב אחרי כל הריצות וראה איך ה-System Prompt משפיע על התוצאות")
            
            with gr.Row():
                show_history_btn = gr.Button("🔍 הצג היסטוריה", variant="secondary")
                download_history_btn = gr.Button("📥 הורד היסטוריה מלאה (CSV)", variant="primary")
                reset_history_btn = gr.Button("🗑️ אפס היסטוריה", variant="stop")
            
            history_summary = gr.Markdown(label="סיכום היסטוריה")
            history_table = gr.Dataframe(
                label="כל הריצות",
                wrap=True,
                interactive=False
            )
            
            history_download = gr.File(label="קובץ היסטוריה מלאה")
            
            show_history_btn.click(
                fn=show_history_summary,
                inputs=[],
                outputs=[history_summary, history_table]
            )
            
            download_history_btn.click(
                fn=download_full_history,
                inputs=[],
                outputs=history_download
            )
            
            reset_history_btn.click(
                fn=reset_history,
                inputs=[],
                outputs=[history_summary, history_table]
            )
            
            gr.Markdown("---")
            gr.Markdown("""
            **📈 איך להשתמש בהיסטוריה:**
            1. הרץ בדיקות עם system prompts שונים
            2. השווה את אחוזי הצלחה בין גרסאות
            3. הורד את ההיסטוריה המלאה לניתוח מעמיק
            4. כשמוצא system prompt שעובד טוב - שמור אותו!
            5. אפס את ההיסטוריה כשרוצה להתחיל ניסוי חדש
            """)


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
