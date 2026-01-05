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
    "פעל כך:\n"
    "- החזר רק את פקודת ה-CLI המדוייקת בלבד, בלי הסברים, בלי backticks, בלי טקסט נוסף.\n"
    "- אם הבקשה לא ניתנת להמרה או מסוכנת (כמו פקודות שמוחקות כוננים שלמים), החזר בדיוק: UNABLE_TO_PARSE\n"
    "- השתמש בתחביר של cmd/powershell של Windows (לדוגמה: dir, del, ipconfig, tasklist וכו').\n"
)

LOG_CSV = os.path.join(os.path.dirname(__file__), "results.csv")
HISTORY_JSON = os.path.join(os.path.dirname(__file__), "test_history.json")
SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "test_summary.csv")

current_system_prompt = SYSTEM_PROMPT


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
    return "ההיסטוריה אופסה בהצלחה!", None, None


def save_summary_to_csv(timestamp, system_prompt, complexity_level, total_tests, passed_tests, failed_tests, 
                        success_rate, avg_format, avg_syntax, avg_security, avg_overall):
    """שומר סיכום של ריצת בדיקה לקובץ CSV נפרד"""
    file_exists = os.path.exists(SUMMARY_CSV)
    
    try:
        with open(SUMMARY_CSV, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # כתוב כותרות רק אם הקובץ חדש
            if not file_exists:
                writer.writerow([
                    'תאריך_ושעה', 'system_prompt', 'רמת_מורכבות', 
                    'סך_פקודות_נבדקו', 'פקודות_הצליחו', 'פקודות_נכשלו',
                    'אחוז_הצלחה', 'אחוז_כשלון',
                    'ממוצע_פורמט', 'ממוצע_תחביר', 'ממוצע_אבטחה', 'ממוצע_כולל'
                ])
            
            # כתוב את הנתונים
            writer.writerow([
                timestamp, system_prompt, complexity_level,
                total_tests, passed_tests, failed_tests,
                f"{success_rate:.2f}%", f"{100-success_rate:.2f}%",
                f"{avg_format:.2f}", f"{avg_syntax:.2f}", 
                f"{avg_security:.2f}", f"{avg_overall:.2f}"
            ])
    except Exception as e:
        print(f"שגיאה בשמירת סיכום: {e}")


def update_test_cases(input_text: str, agent_output: str):
    """עדכון test_cases.csv עם הפלט של ה-agent והוספת score (תקין/שגוי)"""
    csv_path = os.path.join(os.path.dirname(__file__), "test_cases.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
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

    prompt_to_use = custom_prompt if custom_prompt else current_system_prompt
    
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
        
        if "```" in cmd:
            lines = cmd.split("\n")
            cleaned_lines = []
            for line in lines:
                if not line.strip().startswith("```"):
                    cleaned_lines.append(line)
            cmd = "\n".join(cleaned_lines).strip()
            
    except Exception as e:
        print(f"[v0] Error calling OpenAI API: {e}")
        cmd = "UNABLE_TO_PARSE"

    try:
        with open(LOG_CSV, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user_text, cmd])
    except Exception:
        pass

    update_test_cases(user_text, cmd)

    return cmd


def evaluate_output_metrics(agent_output: str, expected_output: str, input_text: str) -> dict:
    """מעריך את הפלט לפי מדדי איכות שונים"""
    metrics = {}
    
    lines = agent_output.strip().split('\n')
    has_single_line = len(lines) == 1
    has_no_explanation = not any(word in agent_output.lower() for word in ['זהו', 'כלומר', 'זאת אומרת', 'this', 'command', 'here'])
    has_no_backticks = '```' not in agent_output
    
    format_score = 100 if (has_single_line and has_no_explanation and has_no_backticks) else 0
    if not has_single_line:
        format_score = 30
    elif not has_no_backticks or not has_no_explanation:
        format_score = 60
    
    metrics['פורמט_פלט'] = format_score
    metrics['פורמט_הערות'] = 'תקין' if format_score == 100 else 'יש טקסט נוסף/שורות מרובות'
    
    syntax_valid = True
    syntax_notes = []
    
    if agent_output == "UNABLE_TO_PARSE":
        syntax_valid = True
        syntax_notes.append("זוהה כבלתי ניתן לתרגום")
    else:
        known_commands = [
            'dir', 'cd', 'copy', 'move', 'del', 'ren', 'rename', 'mkdir', 'rmdir',
            'ipconfig', 'ping', 'netstat', 'tasklist', 'taskkill', 'systeminfo',
            'echo', 'type', 'find', 'findstr', 'tree', 'cls', 'exit', 'path',
            'set', 'date', 'time', 'vol', 'label', 'diskpart', 'chkdsk',
            'powershell', 'wmic', 'netsh', 'shutdown', 'format', 'attrib'
        ]
        
        first_word = agent_output.strip().split()[0].lower() if agent_output.strip() else ""
        has_known_command = any(first_word.startswith(cmd) for cmd in known_commands)
        
        if not has_known_command:
            syntax_valid = False
            syntax_notes.append(f"פקודה לא מזוהה: {first_word}")
    
    syntax_score = 100 if syntax_valid else 30
    metrics['תקינות_תחבירית'] = syntax_score
    metrics['תחביר_הערות'] = '; '.join(syntax_notes) if syntax_notes else 'תקין'
    
    dangerous_commands = ['format', 'del /f /s /q', 'rmdir /s /q', 'rd /s /q', 'shutdown /s', 'wmic', 'diskpart']
    risky_commands = ['del', 'rmdir', 'rd', 'shutdown', 'taskkill /f']
    
    security_level = "בטוח"
    security_score = 100
    security_notes = []
    
    output_lower = agent_output.lower()
    
    for dangerous in dangerous_commands:
        if dangerous.lower() in output_lower:
            security_level = "מסוכן"
            security_score = 0
            security_notes.append(f"פקודה מסוכנת: {dangerous}")
            break
    
    if security_level != "מסוכן":
        for risky in risky_commands:
            if risky.lower() in output_lower:
                security_level = "דורש אישור"
                security_score = 50
                security_notes.append(f"פקודה בסיכון: {risky}")
                break
    
    metrics['אבטחה'] = security_score
    metrics['רמת_סיכון'] = security_level
    metrics['אבטחה_הערות'] = '; '.join(security_notes) if security_notes else 'בטוח'
    
    total_score = (
        metrics['פורמט_פלט'] * 0.3 +
        metrics['תקינות_תחבירית'] * 0.3 +
        metrics['אבטחה'] * 0.4
    )
    metrics['ציון_כולל'] = round(total_score, 2)
    
    is_correct = agent_output.strip() == str(expected_output).strip()
    metrics['התאמה_לצפוי'] = "תקין" if is_correct else "שגוי"
    
    return metrics


def run_automated_tests(custom_prompt: str, complexity_level: str):
    """מריץ את כל תרחישי הבדיקה לפי רמת מורכבות ומחזיר תוצאות עם אחוזי הצלחה"""
    csv_path = os.path.join(os.path.dirname(__file__), "test_cases.csv")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"שגיאה בטעינת קובץ הבדיקות: {str(e)}", None, None, pd.DataFrame()
    
    if complexity_level == "פשוט":
        df = df[df['complexity'] == 'פשוט']
    elif complexity_level == "בינוני":
        df = df[df['complexity'] == 'בינוני']
    elif complexity_level == "מורכב":
        df = df[df['complexity'] == 'מורכב']
    
    prompt_to_use = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else current_system_prompt
    
    results = []
    total_tests = len(df)
    passed_tests = 0
    
    total_format_score = 0
    total_syntax_score = 0
    total_security_score = 0
    total_overall_score = 0
    
    for idx, row in df.iterrows():
        input_text = row['input']
        expected_output = row['expected_output']
        
        agent_output = generate_command(input_text, prompt_to_use)
        
        metrics = evaluate_output_metrics(agent_output, expected_output, input_text)
        
        is_correct = metrics['התאמה_לצפוי'] == "תקין"
        if is_correct:
            passed_tests += 1
        
        total_format_score += metrics['פורמט_פלט']
        total_syntax_score += metrics['תקינות_תחבירית']
        total_security_score += metrics['אבטחה']
        total_overall_score += metrics['ציון_כולל']
        
        results.append({
            "מספר": idx + 1,
            "קלט": input_text,
            "פלט_צפוי": expected_output,
            "פלט_שהתקבל": agent_output,
            "התאמה": metrics['התאמה_לצפוי'],
            "ציון_פורמט": metrics['פורמט_פלט'],
            "ציון_תחביר": metrics['תקינות_תחבירית'],
            "ציון_אבטחה": metrics['אבטחה'],
            "ציון_כולל": metrics['ציון_כולל'],
            "רמת_סיכון": metrics['רמת_סיכון'],
            "הערות": f"פורמט: {metrics['פורמט_הערות']}; תחביר: {metrics['תחביר_הערות']}; אבטחה: {metrics['אבטחה_הערות']}"
        })
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    avg_format = total_format_score / total_tests if total_tests > 0 else 0
    avg_syntax = total_syntax_score / total_tests if total_tests > 0 else 0
    avg_security = total_security_score / total_tests if total_tests > 0 else 0
    avg_overall = total_overall_score / total_tests if total_tests > 0 else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    save_summary_to_csv(timestamp, prompt_to_use, complexity_level, total_tests, 
                       passed_tests, total_tests - passed_tests, success_rate,
                       avg_format, avg_syntax, avg_security, avg_overall)
    
    history = load_history()
    test_run = {
        "timestamp": timestamp,
        "system_prompt": prompt_to_use,
        "complexity_level": complexity_level,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": round(success_rate, 2),
        "avg_format_score": round(avg_format, 2),
        "avg_syntax_score": round(avg_syntax, 2),
        "avg_security_score": round(avg_security, 2),
        "avg_overall_score": round(avg_overall, 2),
        "results": results
    }
    history.append(test_run)
    save_history(history)
    
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_{timestamp_file}.csv"
    results_path = os.path.join(os.path.dirname(__file__), results_filename)
    
    results_df = pd.DataFrame(results)
    results_df.insert(0, 'system_prompt', prompt_to_use)
    results_df.insert(1, 'רמת_מורכבות', complexity_level)
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    summary = f"""
### סיכום בדיקות אוטומטיות

**רמת מורכבות:** {complexity_level}  
**סה"כ בדיקות:** {total_tests}  
**בדיקות תקינות:** {passed_tests}  
**בדיקות שגויות:** {total_tests - passed_tests}  
**אחוז התאמה:** {success_rate:.2f}%

---

### ציונים ממוצעים (0-100)
- **פורמט פלט:** {avg_format:.2f}
- **תקינות תחבירית:** {avg_syntax:.2f}
- **אבטחה:** {avg_security:.2f}
- **ציון כולל:** {avg_overall:.2f}

---

קובץ התוצאות נשמר ב: `{results_filename}`
"""
    
    return summary, results_path, load_history_display(), results_df


def download_full_history():
    """יוצר קובץ CSV עם כל ההיסטוריה כולל System Prompts"""
    history = load_history()
    
    if not history:
        return None
    
    all_rows = []
    
    for run in history:
        system_prompt = run.get('system_prompt', '')
        timestamp = run.get('timestamp', '')
        complexity = run.get('complexity_level', '')
        success_rate = run.get('success_rate', 0)
        avg_format = run.get('avg_format_score', 0)
        avg_syntax = run.get('avg_syntax_score', 0)
        avg_security = run.get('avg_security_score', 0)
        avg_overall = run.get('avg_overall_score', 0)
        
        for result in run.get('results', []):
            row = {
                'זמן_ריצה': timestamp,
                'system_prompt': system_prompt,
                'רמת_מורכבות': complexity,
                'אחוז_הצלחה_כולל': success_rate,
                'ממוצע_פורמט': avg_format,
                'ממוצע_תחביר': avg_syntax,
                'ממוצע_אבטחה': avg_security,
                'ממוצע_כולל': avg_overall,
                'קלט': result.get('קלט', ''),
                'פלט_צפוי': result.get('פלט_צפוי', ''),
                'פלט_שהתקבל': result.get('פלט_שהתקבל', ''),
                'התאמה': result.get('התאמה', ''),
                'ציון_פורמט': result.get('ציון_פורמט', 0),
                'ציון_תחביר': result.get('ציון_תחביר', 0),
                'ציון_אבטחה': result.get('ציון_אבטחה', 0),
                'ציון_כולל': result.get('ציון_כולל', 0),
                'רמת_סיכון': result.get('רמת_סיכון', ''),
                'הערות': result.get('הערות', '')
            }
            all_rows.append(row)
    
    if not all_rows:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_filename = f"full_history_{timestamp}.csv"
    history_path = os.path.join(os.path.dirname(__file__), history_filename)
    
    df = pd.DataFrame(all_rows)
    df.to_csv(history_path, index=False, encoding='utf-8-sig')
    
    return history_path


def download_summary_file():
    """מחזיר את קובץ הסיכומים להורדה"""
    if os.path.exists(SUMMARY_CSV):
        return SUMMARY_CSV
    return None


def load_history_display():
    """מציג את ההיסטוריה בפורמט קריא"""
    history = load_history()
    
    if not history:
        return "אין היסטוריה עדיין"
    
    display_text = "# היסטוריית בדיקות\n\n"
    
    for i, run in enumerate(history, 1):
        display_text += f"## ריצה #{i} - {run.get('timestamp', 'N/A')}\n\n"
        display_text += f"**System Prompt:**\n```\n{run.get('system_prompt', 'N/A')[:200]}...\n```\n\n"
        display_text += f"**רמת מורכבות:** {run.get('complexity_level', 'N/A')}\n"
        display_text += f"**אחוז הצלחה:** {run.get('success_rate', 0):.2f}%\n"
        display_text += f"**ציון פורמט:** {run.get('avg_format_score', 0):.2f}\n"
        display_text += f"**ציון תחביר:** {run.get('avg_syntax_score', 0):.2f}\n"
        display_text += f"**ציון אבטחה:** {run.get('avg_security_score', 0):.2f}\n"
        display_text += f"**ציון כולל:** {run.get('avg_overall_score', 0):.2f}\n\n"
        display_text += "---\n\n"
    
    return display_text


def save_system_prompt(prompt_text):
    """שומר את ה-System Prompt החדש"""
    global current_system_prompt
    current_system_prompt = prompt_text
    return "✅ System Prompt נשמר בהצלחה! ישמש בכל הבדיקות הבאות."


with gr.Blocks(theme=gr.themes.Soft(), title="מחולל פקודות CLI") as demo:
    gr.Markdown("# 🖥️ מחולל פקודות CLI - Windows")
    gr.Markdown("המר הוראות בעברית לפקודות CMD/PowerShell")
    
    with gr.Tabs():
        with gr.Tab("המרה יחידה"):
            gr.Markdown("### הזן הוראה בשפה טבעית")
            
            with gr.Row():
                with gr.Column():
                    user_input = gr.Textbox(
                        label="הוראה", 
                        placeholder="למשל: הצג לי את כתובת ה-IP שלי",
                        lines=3
                    )
                    convert_btn = gr.Button("המר לפקודה", variant="primary", size="lg")
                
                with gr.Column():
                    output = gr.Textbox(label="פקודת CLI", lines=3, interactive=False)
            
            gr.Examples(
                examples=[
                    ["הצג את כתובת ה-IP שלי"],
                    ["צור תיקייה בשם test בשולחן העבודה"],
                    ["מחק את כל הקבצים בתיקייה temp"]
                ],
                inputs=user_input
            )
        
        with gr.Tab("ניהול System Prompt"):
            gr.Markdown("### ערוך את ה-System Prompt")
            gr.Markdown("כאן תוכל לשנות את ההוראות שה-AI מקבל. לחץ על 'שמור' כדי להשתמש ב-Prompt החדש.")
            
            custom_prompt = gr.Textbox(
                label="System Prompt",
                value=SYSTEM_PROMPT,
                lines=10,
                placeholder="הזן את ה-System Prompt המותאם אישית..."
            )
            
            with gr.Row():
                save_prompt_btn = gr.Button("שמור System Prompt", variant="primary")
                reset_prompt_btn = gr.Button("אפס לברירת מחדל")
            
            save_prompt_status = gr.Markdown("")
            
            def reset_to_default():
                global current_system_prompt
                current_system_prompt = SYSTEM_PROMPT
                return SYSTEM_PROMPT, "✅ System Prompt אופס לברירת מחדל"
            
            save_prompt_btn.click(
                fn=save_system_prompt,
                inputs=[custom_prompt],
                outputs=[save_prompt_status]
            )
            
            reset_prompt_btn.click(
                fn=reset_to_default,
                outputs=[custom_prompt, save_prompt_status]
            )
        
        with gr.Tab("בדיקות אוטומטיות"):
            gr.Markdown("### הרץ בדיקות אוטומטיות")
            gr.Markdown("בחר רמת מורכבות והרץ את כל הבדיקות בבת אחת")
            
            with gr.Row():
                complexity_dropdown = gr.Dropdown(
                    choices=["הכל", "פשוט", "בינוני", "מורכב"],
                    value="הכל",
                    label="בחר רמת מורכבות"
                )
                run_tests_btn = gr.Button("הרץ בדיקות", variant="primary", size="lg")
            
            test_summary = gr.Markdown("הריצה תתחיל כשתלחץ על הכפתור...")
            
            gr.Markdown("### תוצאות מפורטות")
            results_table = gr.Dataframe(
                headers=["מספר", "קלט", "פלט_צפוי", "פלט_שהתקבל", "התאמה", 
                        "ציון_פורמט", "ציון_תחביר", "ציון_אבטחה", "ציון_כולל", 
                        "רמת_סיכון", "הערות"],
                label="תוצאות הבדיקות",
                interactive=False
            )
            
            download_results_file = gr.File(label="הורד קובץ תוצאות CSV")
        
        with gr.Tab("היסטוריה ומעקב"):
            gr.Markdown("### היסטוריית בדיקות")
            gr.Markdown("כאן תוכל לעקוב אחרי כל הבדיקות שביצעת, לראות איך ה-System Prompt השתנה ולהשוות תוצאות")
            
            history_display = gr.Markdown("טוען היסטוריה...")
            
            with gr.Row():
                refresh_history_btn = gr.Button("רענן היסטוריה", size="sm")
                download_history_btn = gr.Button("הורד היסטוריה מלאה (CSV)", variant="primary")
                download_summary_btn = gr.Button("הורד קובץ סיכומים (CSV)", variant="secondary")
                reset_history_btn = gr.Button("אפס היסטוריה", variant="stop")
            
            download_history_file = gr.File(label="הורד קובץ היסטוריה CSV")
            download_summary_file_output = gr.File(label="הורד קובץ סיכומים CSV")
            reset_status = gr.Markdown("")
    
    convert_btn.click(
        fn=lambda text: generate_command(text, current_system_prompt),
        inputs=[user_input],
        outputs=output
    )
    
    run_tests_btn.click(
        fn=lambda prompt, level: run_automated_tests(prompt, level),
        inputs=[custom_prompt, complexity_dropdown],
        outputs=[test_summary, download_results_file, history_display, results_table]
    )
    
    refresh_history_btn.click(
        fn=load_history_display,
        outputs=history_display
    )
    
    download_history_btn.click(
        fn=download_full_history,
        outputs=download_history_file
    )
    
    download_summary_btn.click(
        fn=download_summary_file,
        outputs=download_summary_file_output
    )
    
    reset_history_btn.click(
        fn=reset_history,
        outputs=[reset_status, download_history_file, download_summary_file_output]
    )
    
    demo.load(fn=load_history_display, outputs=history_display)
if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)