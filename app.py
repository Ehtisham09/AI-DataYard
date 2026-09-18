"""
Chef Bella — Streamlit Web Application
A smart recipe assistant powered by LangChain + AWS Bedrock (Amazon Nova 2 Lite)

Run with:  python -m streamlit run app.py
"""

import os
import streamlit as st
from collections import deque

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chef Bella",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background-color: #fdf6ec; }

.user-bubble {
    background: #ff6b35;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 40px;
    font-size: 15px;
    line-height: 1.5;
}
.bella-bubble {
    background: white;
    color: #2d2d2d;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 40px 8px 0;
    font-size: 15px;
    line-height: 1.5;
    border: 1px solid #e8d5b7;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.category-badge {
    display: inline-block;
    background: #fff3e0;
    color: #e65100;
    border: 1px solid #ffcc80;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.recipe-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e8d5b7;
    margin: 10px 0;
}
.recipe-card h3 { color: #c0392b; margin-top: 0; }
.recipe-card .meta { color: #777; font-size: 13px; margin-bottom: 12px; }
.aws-badge {
    background: #232f3e;
    color: #ff9900;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

AWS_REGION   = "ap-southeast-2"
MODEL_ID     = "global.amazon.nova-2-lite-v1:0"

CHEF_BELLA_PERSONA = """\
You are Chef Bella, a warm and knowledgeable home cooking assistant.

YOUR PERSONALITY:
- Friendly, encouraging, and patient — cooking should feel fun, not stressful.
- You use simple, everyday language. Never use jargon without explaining it.
- You believe anyone can learn to cook with the right guidance.

YOUR EXPERTISE:
- Home cooking across many cuisines: Italian, Asian, Middle Eastern, Mexican, and more.
- Budget-friendly meals that use pantry staples.
- Ingredient substitutions (e.g. no buttermilk? use milk + lemon juice).
- Adjusting recipes for dietary needs: vegetarian, vegan, gluten-free, dairy-free.

YOUR RULES:
1. Always include estimated prep time and cook time.
2. Rate difficulty as: Beginner / Intermediate / Advanced.
3. If the user has fewer than 3 ingredients, suggest a simple pantry-based recipe.
4. Always ask about allergies before suggesting a recipe if not already mentioned.
5. If asked something unrelated to food/cooking, say:
   'I am Chef Bella, a cooking assistant. For that question, you may want to ask a different expert!'

ANSWER FORMAT:
- Start with a warm greeting on the first message.
- Use emojis sparingly (1-2 per response) to keep things cheerful.
- End every recipe suggestion with: 'Buon appetito! 🍽️'
"""

SUBSTITUTION_EXAMPLES = [
    {
        "question": "I don't have buttermilk. What can I use instead?",
        "answer": (
            "**Substitute:** Regular milk + acid\n\n"
            "**Why it works:** Buttermilk is just milk that has been slightly acidified. "
            "Adding an acid (lemon juice or vinegar) to regular milk triggers the same "
            "chemical reaction in baking.\n\n"
            "**Ratio:** 1 cup milk + 1 tablespoon lemon juice or white vinegar. "
            "Stir and let it sit for 5 minutes before using.\n\n"
            "**Flavour note:** The taste is virtually identical. No one will notice! 🎉"
        ),
    },
    {
        "question": "Can I replace fresh garlic with garlic powder?",
        "answer": (
            "**Substitute:** Garlic powder\n\n"
            "**Why it works:** Garlic powder is just dehydrated and ground fresh garlic.\n\n"
            "**Ratio:** 1 clove fresh garlic = ¼ teaspoon garlic powder.\n\n"
            "**Flavour note:** Garlic powder is milder. Works best in sauces and soups."
        ),
    },
    {
        "question": "I ran out of eggs. What can I use in a cake?",
        "answer": (
            "**Substitute:** Several options!\n\n"
            "**Why it works:** Eggs bind and add moisture — any substitute must do both.\n\n"
            "**Ratio (per egg):**\n"
            "- 1 tbsp ground flaxseed + 3 tbsp water\n"
            "- 3 tbsp unsweetened applesauce\n"
            "- ¼ cup mashed banana\n\n"
            "**Flavour note:** Banana adds flavour. Applesauce is neutral."
        ),
    },
]

CLASSIFIER_PROMPT = """\
You are a question classifier for a cooking assistant.
Return EXACTLY ONE label — no explanation, no punctuation, no extra words:

  recipe        → user wants a recipe or meal idea
  substitution  → user wants to replace an ingredient
  nutrition     → user asks about calories, health, or nutrients
  general       → anything else food-related
  off_topic     → not related to food or cooking

If unsure, choose 'general'.
"""

CATEGORY_ICONS = {
    "recipe":       "🍳",
    "substitution": "🔄",
    "nutrition":    "🥗",
    "general":      "💬",
    "off_topic":    "🚫",
}

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Recipe(BaseModel):
    name: str = Field(description="The name of the dish")
    difficulty: str = Field(description="'Beginner', 'Intermediate', or 'Advanced'")
    prep_time_minutes: int = Field(description="Minutes of active preparation")
    cook_time_minutes: int = Field(description="Minutes to cook")
    ingredients: List[str] = Field(description="Ingredients with quantities")
    steps: List[str] = Field(description="Ordered cooking steps")
    tip: str = Field(description="One helpful tip or common mistake to avoid")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages":       [],
        "history":        deque(maxlen=12),
        "llm":            None,
        "chains_ready":   False,
        "aws_ok":         False,
        "structured_mode": False,
        "window_size":    6,
        "turn_count":     0,
        "api_key":        "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────────────────────────────────────
# LLM + CHAIN BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def connect_aws(api_key: str, temperature: float, max_tokens: int) -> ChatBedrockConverse:
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
    os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
    llm = ChatBedrockConverse(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Quick smoke test
    llm.invoke([HumanMessage(content="Say 'ready' and nothing else.")])
    return llm


def build_chains(llm):
    # Classifier
    clf_prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_PROMPT),
        ("human", "{question}"),
    ])
    st.session_state["classifier_chain"] = clf_prompt | llm | StrOutputParser()

    # Recipe (text, memory-aware)
    recipe_prompt = ChatPromptTemplate.from_messages([
        ("system", CHEF_BELLA_PERSONA),
        MessagesPlaceholder(variable_name="history"),
        ("human",
         "I have these ingredients: {ingredients}.\n"
         "Dietary notes: {dietary_notes}.\n"
         "Please suggest a recipe I can make right now."),
    ])
    st.session_state["recipe_chain"] = recipe_prompt | llm | StrOutputParser()

    # Recipe (structured Pydantic)
    struct_prompt = ChatPromptTemplate.from_messages([
        ("system", CHEF_BELLA_PERSONA),
        ("human",
         "Create a recipe using these ingredients: {ingredients}.\n"
         "Dietary notes: {dietary_notes}.\n"
         "Return the recipe as structured data."),
    ])
    st.session_state["structured_chain"] = struct_prompt | llm.with_structured_output(Recipe)

    # Substitution (few-shot)
    example_template = ChatPromptTemplate.from_messages([
        ("human",     "{question}"),
        ("assistant", "{answer}"),
    ])
    few_shot_block = FewShotChatMessagePromptTemplate(
        example_prompt=example_template,
        examples=SUBSTITUTION_EXAMPLES,
    )
    sub_prompt = ChatPromptTemplate.from_messages([
        ("system", CHEF_BELLA_PERSONA),
        few_shot_block,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    st.session_state["substitution_chain"] = sub_prompt | llm | StrOutputParser()

    # Nutrition
    nut_prompt = ChatPromptTemplate.from_messages([
        ("system",
         CHEF_BELLA_PERSONA
         + "\n\nFor this question, focus on nutritional information. "
           "Give approximate values (not exact). "
           "Always add: consult a dietitian for medical advice."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    st.session_state["nutrition_chain"] = nut_prompt | llm | StrOutputParser()

    # General / off-topic
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", CHEF_BELLA_PERSONA),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    st.session_state["general_chain"] = gen_prompt | llm | StrOutputParser()

    st.session_state["chains_ready"] = True


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def classify(question: str) -> str:
    try:
        raw = st.session_state["classifier_chain"].invoke({"question": question})
        return raw.strip().lower()
    except Exception:
        return "general"


def get_history() -> list:
    return list(st.session_state["history"])


def add_to_history(human: str, ai: str):
    st.session_state["history"].append(HumanMessage(content=human))
    st.session_state["history"].append(AIMessage(content=ai))


def route_and_answer(question: str, category: str):
    history = get_history()
    if category == "recipe" and st.session_state["structured_mode"]:
        return st.session_state["structured_chain"].invoke({
            "ingredients":   question,
            "dietary_notes": "none specified",
        })
    elif category == "recipe":
        return st.session_state["recipe_chain"].invoke({
            "ingredients":   question,
            "dietary_notes": "none specified",
            "history":       history,
        })
    elif category == "substitution":
        return st.session_state["substitution_chain"].invoke({
            "question": question,
            "history":  history,
        })
    elif category == "nutrition":
        return st.session_state["nutrition_chain"].invoke({
            "question": question,
            "history":  history,
        })
    else:
        return st.session_state["general_chain"].invoke({
            "question": question,
            "history":  history,
        })

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🍳 Chef Bella")
    st.markdown("*Powered by AWS Bedrock*")
    st.markdown('<span class="aws-badge">☁️ Amazon Nova 2 Lite</span>', unsafe_allow_html=True)
    st.divider()

    # ── AWS credentials ───────────────────────────────────────────────────────
    st.markdown("### 🔑 AWS Bedrock API Key")
    api_key = st.text_input(
        "Bearer Token",
        value=st.session_state["api_key"],
        type="password",
        placeholder="ABSKQmVkcm9ja0FQ...",
        help="AWS Console → Bedrock → API Keys → Create short-term key",
    )
    st.caption(f"Region: `{AWS_REGION}` · Model: `{MODEL_ID}`")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05,
                            help="0 = consistent answers, 1 = creative")
    max_tokens  = st.slider("Max response length", 200, 1500, 700, 50)

    if st.button("🔌 Connect to AWS Bedrock", use_container_width=True, type="primary"):
        if not api_key.strip():
            st.error("Please enter your AWS Bearer Token.")
        else:
            with st.spinner("Connecting to AWS Bedrock…"):
                try:
                    llm = connect_aws(api_key.strip(), temperature, max_tokens)
                    st.session_state["llm"]      = llm
                    st.session_state["api_key"]  = api_key.strip()
                    st.session_state["aws_ok"]   = True
                    build_chains(llm)
                    st.success("✅ Connected! Chef Bella is ready.")
                except Exception as e:
                    st.session_state["aws_ok"] = False
                    err = str(e)
                    if "INVALID_PAYMENT_INSTRUMENT" in err:
                        st.error("Payment issue on your AWS account. Check your billing in the AWS Console.")
                    elif "ExpiredToken" in err or "Token" in err:
                        st.error("API key has expired. Generate a new one from AWS Console → Bedrock → API Keys.")
                    else:
                        st.error(f"Connection failed: {err}")

    if st.session_state["aws_ok"]:
        st.success("✅ AWS Bedrock connected")

    st.divider()

    # ── Options ───────────────────────────────────────────────────────────────
    st.markdown("### 🛠️ Options")
    st.session_state["structured_mode"] = st.toggle(
        "Structured Recipe Cards",
        value=st.session_state["structured_mode"],
        help="Recipe questions return a formatted card with ingredients, steps, difficulty",
    )
    new_window = st.slider("Memory window (turns)", 2, 20, st.session_state["window_size"],
                           help="How many conversation turns Bella remembers")
    if new_window != st.session_state["window_size"]:
        st.session_state["window_size"] = new_window
        st.session_state["history"] = deque(
            st.session_state["history"],
            maxlen=new_window * 2,
        )

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Session")
    c1, c2 = st.columns(2)
    c1.metric("Turns", st.session_state["turn_count"])
    c2.metric("In memory", len(st.session_state["history"]) // 2)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"]   = []
        st.session_state["history"]    = deque(maxlen=st.session_state["window_size"] * 2)
        st.session_state["turn_count"] = 0
        st.rerun()

    st.divider()

    # ── Category legend ───────────────────────────────────────────────────────
    st.markdown("### 🏷️ Question types")
    for cat, icon in CATEGORY_ICONS.items():
        st.markdown(f"{icon} **{cat.title()}**")

    st.divider()

    # ── Quick examples ────────────────────────────────────────────────────────
    st.markdown("### 💡 Try asking")
    examples = [
        "I have eggs, tomatoes and cheese",
        "Can I replace butter with olive oil?",
        "How many calories in a bowl of pasta?",
        "What can I make in under 20 minutes?",
        "I'm vegan — suggest a protein-rich dinner",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex

    st.divider()
    st.caption("Get a new key: AWS Console → Bedrock → API Keys\nKeys expire every 12 hours.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🍳")
with col_title:
    st.markdown("## Chef Bella")
    st.caption("Smart recipe assistant · AWS Bedrock · Amazon Nova 2 Lite")

st.divider()

# Guard: must connect first
if not st.session_state["aws_ok"]:
    st.info("👈 Enter your AWS Bedrock API key in the sidebar and click **Connect to AWS Bedrock**.")
    st.markdown("""
    **How to get a Bedrock API key:**
    1. Go to [AWS Console](https://console.aws.amazon.com) → **Amazon Bedrock**
    2. In the left menu click **API Keys**
    3. Click **Create short-term key**
    4. Copy the key and paste it in the sidebar

    > ⚠️ Keys expire every **12 hours** — generate a new one when you see an expired token error.
    """)
    st.stop()

# ── Render chat history ────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    st.markdown(
        '<div class="bella-bubble">👋 Ciao! I\'m <b>Chef Bella</b>, your personal cooking assistant powered by AWS Bedrock! '
        'Tell me what ingredients you have at home and I\'ll suggest a delicious recipe. '
        'You can also ask about ingredient substitutions, calories, or any cooking question. '
        'What shall we cook today? 🍽️</div>',
        unsafe_allow_html=True,
    )

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        cat  = msg.get("category", "general")
        icon = CATEGORY_ICONS.get(cat, "💬")
        badge = f'<span class="category-badge">{icon} {cat}</span><br>'

        if msg.get("structured"):
            r = msg["structured"]
            diff_color = {"Beginner": "#27ae60", "Intermediate": "#f39c12", "Advanced": "#e74c3c"}.get(r.difficulty, "#666")
            ingredients_html = "".join(f"<li>{i}</li>" for i in r.ingredients)
            steps_html       = "".join(f"<li>{s}</li>" for s in r.steps)
            card = f"""
            <div class="recipe-card">
              <h3>🍽️ {r.name}</h3>
              <div class="meta">
                <span style="color:{diff_color};font-weight:600">● {r.difficulty}</span>
                &nbsp;·&nbsp; ⏱ Prep: {r.prep_time_minutes} min
                &nbsp;·&nbsp; 🔥 Cook: {r.cook_time_minutes} min
              </div>
              <b>Ingredients ({len(r.ingredients)})</b>
              <ul style="margin:6px 0 12px 0">{ingredients_html}</ul>
              <b>Steps</b>
              <ol style="margin:6px 0 12px 0">{steps_html}</ol>
              <div style="background:#fff8e1;padding:10px;border-radius:8px;border-left:3px solid #ffc107">
                💡 <b>Tip:</b> {r.tip}
              </div>
            </div>
            """
            st.markdown(badge + card, unsafe_allow_html=True)
        else:
            content = msg["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="bella-bubble">{badge}{content}</div>',
                unsafe_allow_html=True,
            )

# ── Input ──────────────────────────────────────────────────────────────────────
st.divider()

prefill_value = st.session_state.pop("prefill", "")
user_input    = st.chat_input("Ask Chef Bella anything… e.g. 'I have chicken, garlic and lemon'")

if prefill_value:
    user_input = prefill_value

if user_input and user_input.strip():
    question = user_input.strip()
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.spinner("Chef Bella is thinking…"):
        category = classify(question)
        result   = route_and_answer(question, category)

    if isinstance(result, Recipe):
        st.session_state["messages"].append({
            "role":       "assistant",
            "content":    result.name,
            "category":   "recipe",
            "structured": result,
        })
        summary = (
            f"I suggested {result.name}. "
            f"Prep: {result.prep_time_minutes} min, Cook: {result.cook_time_minutes} min. "
            f"Difficulty: {result.difficulty}."
        )
        add_to_history(question, summary)
    else:
        st.session_state["messages"].append({
            "role":     "assistant",
            "content":  result,
            "category": category,
        })
        add_to_history(question, result)

    st.session_state["turn_count"] += 1
    st.rerun()
