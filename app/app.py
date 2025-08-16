# app/main.py
from typing import List, Dict, Any, Tuple
import re
import gradio as gr
from dotenv import load_dotenv
from app.config import load_config
from app.levels import get_level
from app.llm import get_llm_response

load_dotenv()
config: Dict[str, str] = load_config()

LEVELS_BASE = [
    "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6",
    "Level 7", "Level 8"
]

def normalize_level(label: str) -> str:
    return label.replace("✅", "").strip()

def decorate_choices(solved: List[str], current_base: str) -> Tuple[List[str], str]:
    choices = []
    for lvl in LEVELS_BASE:
        is_solved = lvl in solved
        label = f"✅ {lvl}" if is_solved else lvl
        choices.append(label)
    decorated_current = f"✅ {current_base}" if current_base in solved else current_base
    return choices, decorated_current

def intro_for(level_base: str) -> str:
    if level_base == "Level 1":
        return "👮‍♂️ Guard: Halt! State your business, traveler. (Ceremonies and ledgers keep me honest!)"
    if level_base == "Level 2":
        return "📚 Librarian: Shhh… no raw watchwords here, only hints and ciphers."
    if level_base == "Level 3":
        return "☁️ Cloud: The air bristles—be subtle; blatant injections draw lightning."
    if level_base == "Level 4":
        return "🧙‍♂️ GANDALF-4: Tokens only. Whisper the passphrase to the oracle, then present the badge."
    if level_base == "Level 5":
        return "🕵️ Auditor: Base64/hex/ROT13 are contraband. Think acrostics, mnemonics, checksums."
    if level_base == "Level 6":
        return "🔒 Sentinel: Only natural-language cloaks or checksums pass the scan."
    if level_base == "Level 7":
        return "🗄️ Vector Vault: The watchword lives in an external vault. Retrieval only, and only with a consent token."
    if level_base == "Level 8":
        return "🛡️ Air-Gapped Oracle: Challenge–response required. No token, no truth."
    return "Welcome."

def said_password(user_text: str, password: str) -> bool:
    pat = re.compile(rf"\b{re.escape(password)}\b", re.IGNORECASE)
    return bool(pat.search(user_text))

def contains_password(text: str | None, password: str) -> bool:
    if not text:
        return False
    pat = re.compile(rf"\b{re.escape(password)}\b", re.IGNORECASE)
    return bool(pat.search(text))

def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# mini-gandalf")

        solved_state = gr.State(value=[])  # type: ignore[list-item]
        with gr.Row():
            with gr.Column(scale=1):
                initial_choices, initial_value = decorate_choices([], "Level 1")
                level_selector = gr.Radio(
                    choices=initial_choices,
                    value=initial_value,
                    label="Select Level",
                )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[("", intro_for("Level 1"))],
                    label="Gandalf AI",
                    height=460,
                )
                with gr.Row():
                    user_box = gr.Textbox(
                        label="Your message",
                        placeholder="Try to get the password…",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear chat for this level")

        def on_level_change(selected_label: str, solved: List[str]):
            base = normalize_level(selected_label)
            choices, decorated_value = decorate_choices(solved, base)
            new_history = [("", intro_for(base))]
            return (
                gr.update(choices=choices, value=decorated_value),
                new_history,
                gr.update(value=""),
            )

        level_selector.change(
            fn=on_level_change,
            inputs=[level_selector, solved_state],
            outputs=[level_selector, chatbot, user_box],
            queue=False,
        )

        def on_clear_chat(selected_label: str):
            base = normalize_level(selected_label)
            return [("", intro_for(base))]

        clear_btn.click(
            fn=on_clear_chat,
            inputs=[level_selector],
            outputs=[chatbot],
            queue=False,
        )

        def on_send(
            user_text: str,
            history: List[Tuple[str, str]],
            selected_label: str,
            solved: List[str],
        ):
            base = normalize_level(selected_label)
            level: Dict[str, Any] = get_level(base)
            password: str = level["password"]

            reply = get_llm_response(user_text, level)

            won = said_password(user_text, password) or contains_password(reply, password)

            if won and base not in solved:
                solved = solved + [base]
                reply = (reply or "") + "\n\n🎉 **Level cleared!**"

            history = history + [(user_text, reply or "…")]

            choices, decorated_value = decorate_choices(solved, base)

            return (
                history,
                "",
                solved,
                gr.update(choices=choices, value=decorated_value),
            )

        user_box.submit(
            fn=on_send,
            inputs=[user_box, chatbot, level_selector, solved_state],
            outputs=[chatbot, user_box, solved_state, level_selector],
        )
        send_btn.click(
            fn=on_send,
            inputs=[user_box, chatbot, level_selector, solved_state],
            outputs=[chatbot, user_box, solved_state, level_selector],
        )

    return demo

if __name__ == "__main__":
    print("Booting app…", flush=True)
    demo = build_ui()
    print("Launching Gradio…", flush=True)
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True,
        quiet=False,
    )
