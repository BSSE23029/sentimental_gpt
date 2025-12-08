import pandas as pd
import os
import random
import re
import warnings

# Suppress Excel warnings
warnings.simplefilter("ignore")

OUTPUT_FILE = "sentinel_training_data.csv"

# --- 1. BENIGN DATA GENERATOR (Since we lack normal user logs) ---
def generate_benign_samples(count=3000):
    """Generates synthetic normal user queries to balance the dataset."""
    templates = [
        "Write a {topic} about {subject}.",
        "Can you help me {action} the {object}?",
        "What is the capital of {place}?",
        "Explain {concept} to me like I'm 5.",
        "Translate this to {language}: {phrase}.",
        "Summarize the following text: {phrase}.",
        "I need a recipe for {food}.",
        "How do I fix a {broken_thing}?",
        "Create a schedule for {activity}.",
        "List the top 10 {items}."
    ]
    
    slots = {
        "topic": ["poem", "story", "essay", "email", "code snippet"],
        "subject": ["cats", "AI", "history", "space", "nature"],
        "action": ["fix", "find", "calculate", "draw", "analyze"],
        "object": ["bug", "data", "image", "text", "file"],
        "place": ["France", "Mars", "the moon", "ancient Rome"],
        "concept": ["quantum physics", "love", "gravity", "machine learning"],
        "language": ["Spanish", "French", "Python", "Binary"],
        "phrase": ["hello world", "the quick brown fox", "science is cool"],
        "food": ["pizza", "cake", "sushi", "salad"],
        "broken_thing": ["car", "computer", "phone", "sink"],
        "activity": ["study", "gym", "work", "meeting"],
        "items": ["movies", "books", "songs", "games"]
    }

    samples = []
    for _ in range(count):
        tmpl = random.choice(templates)
        # Fill slots
        for key, values in slots.items():
            if "{" + key + "}" in tmpl:
                tmpl = tmpl.replace("{" + key + "}", random.choice(values))
        samples.append(tmpl)
    
    return pd.DataFrame({'text': samples, 'label': 0, 'source': 'synthetic_benign'})

# --- 2. ATTACK DATA LOADER ---
def load_attacks():
    attacks = []

    # A. From CSV (The Basic Attacks)
    csv_path = "attack data/result.csv"
    if os.path.exists(csv_path):
        try:
            # Force OpenPyXL for the zipped CSVs
            df = pd.read_excel(csv_path, engine='openpyxl')
            
            # Column 'question' (Basic Prompt Injection)
            if 'question' in df.columns:
                clean_q = df['question'].dropna().astype(str).unique()
                for q in clean_q:
                    attacks.append({'text': q, 'label': 1, 'source': 'result.csv_question'})

            # Column 'question.1' (Component/Tool Injection)
            if 'question.1' in df.columns:
                clean_q1 = df['question.1'].dropna().astype(str).unique()
                for q in clean_q1:
                    attacks.append({'text': q, 'label': 1, 'source': 'result.csv_tool_attack'})
                    
            print(f"✅ Loaded {len(attacks)} unique attacks from result.csv")
        except Exception as e:
            print(f"⚠️ Could not read result.csv: {e}")

    # B. From TXT (The Advanced Attacks)
    txt_path = "src/attack.txt"
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # parsing heuristics
            txt_count = 0
            buffer = ""
            for line in lines:
                clean = line.strip()
                # If it's a long block of text, treat as attack
                if len(clean) > 30 and not clean.startswith("Attack") and not clean.startswith("-"):
                    attacks.append({'text': clean, 'label': 1, 'source': 'attack.txt'})
                    txt_count += 1
            print(f"✅ Loaded {txt_count} attack patterns from attack.txt")
        except Exception as e:
            print(f"⚠️ Could not read attack.txt: {e}")

    return pd.DataFrame(attacks)

# --- MAIN EXECUTION ---
def main():
    print("--- 🏗️ BUILDING SENTINEL DATASET 🏗️ ---")
    
    # 1. Get Attacks
    df_attacks = load_attacks()
    
    # 2. Generate Benign (Aiming for 50/50 balance initially)
    # We generate slightly more benign data than attacks to prevent false positives
    target_benign = max(len(df_attacks) * 2, 1000)
    df_benign = generate_benign_samples(count=target_benign)
    print(f"✅ Generated {len(df_benign)} synthetic benign samples")

    # 3. Merge
    full_df = pd.concat([df_attacks, df_benign], ignore_index=True)
    
    # 4. Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Clean (Remove newlines/tabs for CSV safety)
    full_df['text'] = full_df['text'].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ''))

    # 6. Save
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 SUCCESS! Saved {len(full_df)} samples to {OUTPUT_FILE}")
    print(full_df['label'].value_counts())

if __name__ == "__main__":
    main()