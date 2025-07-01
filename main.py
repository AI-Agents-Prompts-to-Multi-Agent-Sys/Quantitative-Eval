# -*- coding = utf-8 -*-
# @Time: 2025/6/26 14:29
# @Author: Tan Qiyuan
# @File: main

import asyncio
import json
import operator
import re
from typing import TypedDict, Annotated, List

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# Load environment variables (GOOGLE_API_KEY should be set)
load_dotenv()

# What to evaluate
EVALUATION_SUBJECT = "band"

# List of items to evaluate
ITEMS = [
    "The Beatles", "Led Zeppelin", "Pink Floyd", "Queen", "The Rolling Stones",
    "Metallica", "Megadeth", "Black Sabbath", "Iron Maiden", "Tool"
]

# Persona definitions
PERSONAS = {
    "metalhead": "You're in your 30s, a lifelong metal fan. You value power, aggression, instrumental mastery, and complexity. You dismiss pop and overproduced music as shallow.",
    "popstar": "You're in your 20s, immersed in social media culture. You love global accessibility, emotional resonance, and catchy choruses. You believe great bands bring joy and unity.",
    "boomer": "You're in your 70s. You grew up during the golden age of rock and believe greatness is rooted in legacy, songwriting, and timeless appeal. Newer music feels synthetic to you.",
    "genz": "You're a teenager, online-native, and value diversity, identity, and innovation in music. You're drawn to bands that say something real or break genre rules.",
    "indie": "You're in your 30s, an art-school type who craves authenticity, emotion, and underground cool. You dislike commercial polish and love expressive weirdness.",
}

# Criteria for evaluation
CRITERIA = {
    "Musical Innovation": "Pioneering ideas, new sounds, genre blending.",
    "Cultural Impact": "Broader societal influence, pop culture penetration.",
    "Lyrical or Thematic Depth": "Narrative richness, philosophical weight, relatability.",
    "Technical Proficiency": "Musical complexity, virtuosity, performance execution.",
    "Live Performance Strength": "Energy, presence, crowd connection on stage.",
    "Legacy & Longevity": "Enduring influence across generations and artists."
}

# Instructions/background information for the personas
PERSONA_ROLE = "music critic"
INSTRUCTION = "You have been asked to evaluate the greatness of 21 historically significant bands across genres including rock, metal, pop, and progressive."

# LLM config
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.7)

# Prompt template
def make_prompt(persona_description):
    return f"""You are a {PERSONA_ROLE} with the following worldview:

{persona_description}

{INSTRUCTION}

Evaluate each {EVALUATION_SUBJECT} based on the following {len(CRITERIA)} criteria, scoring from 1 (low) to 5 (high):

{"".join(f"{key}: {value}{chr(10)}" for key, value in CRITERIA.items())}
Here are the {EVALUATION_SUBJECT}s to evaluate:
{chr(10).join('- ' + item for item in ITEMS)}

Please respond ONLY in the following strict JSON format:

{{
  "ratings": [
    {{
      "item": "the corresponding {EVALUATION_SUBJECT} name here"{"".join(f',{chr(10)}      "{criteria}": int' for criteria in CRITERIA)}
    }},
    ...
  ],
  "justification": "Your paragraph explaining the ratings here.",
  "ranking": ["{EVALUATION_SUBJECT}1", "{EVALUATION_SUBJECT}2", ..., "band21"]
}}

- The ratings list must include all {len(ITEMS)} {EVALUATION_SUBJECT}s.
- The ranking list must be in your personal order (1st to {len(ITEMS)}st).
- Do not include markdown formatting, code blocks, or commentary outside the JSON block.
"""

# Clean LLM output
def clean_json_string(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", text)
    return cleaned.replace("```", "").strip()

# Parse JSON
def parse_json_response(response):
    try:
        response_cleaned = clean_json_string(response)
        data = json.loads(response_cleaned)
        ratings = data["ratings"]
        justification = data["justification"]
        ranking = data["ranking"]
        df = pd.DataFrame(ratings)
        df.columns = [EVALUATION_SUBJECT] + list(CRITERIA.keys())
        return df, justification
    except Exception as e:
        print("Error parsing JSON:", e)
        return pd.DataFrame(), ""

# State definition
class Vote(TypedDict):
    df: pd.DataFrame
    justification: str
    persona: str

class State(TypedDict):
    votes: Annotated[List[Vote], operator.add]

# Agent node
def make_agent_node(persona_key):
    async def node(state):
        persona = PERSONAS[persona_key]
        prompt = make_prompt(persona)
        response = await llm.ainvoke(prompt)
        df, justification = parse_json_response(response.content)

        print(f"\n== {persona_key.upper()} TABLE ==")
        print(df)
        print(f"\n== {persona_key.upper()} JUSTIFICATION ==\n{justification}")

        state['votes'] = [{
                "df": df,
                "justification": justification,
                "persona": persona_key,
        }]

        return state
    return node

# Graph build
agent_keys = list(PERSONAS.keys())

graph = StateGraph(State)
for agent in agent_keys:
    graph.add_node(agent, make_agent_node(agent))

# Graph edges
for agent in agent_keys:
    graph.add_edge(START, agent)
graph.add_edge([agent for agent in agent_keys], END)

# Run
compiled = graph.compile()
result = asyncio.run(compiled.ainvoke({
    "votes": [],
}))
