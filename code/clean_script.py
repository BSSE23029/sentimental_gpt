import pandas as pd
import os
import warnings

# Suppress warnings about extension mismatch
warnings.simplefilter("ignore")

files_to_check = [
    "attack data/result.csv",
    "attack data/results from variants.csv", 
    "defense data/reverse_engineering_GPTs.csv"
]

print("--- 🔓 EXCEL DECODER (FORCED) 🔓 ---\n")

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f"❌ MISSING: {filepath}")
        continue

    print(f"📂 TARGET: {filepath}")
    
    try:
        # FORCE engine='openpyxl' to treat it as an XLSX file regardless of extension
        df = pd.read_excel(filepath, engine='openpyxl')
        
        print(f"   ✅ SUCCESS: Decoded as Excel Spreadsheet.")
        print(f"   📊 Columns Found: {list(df.columns)}")
        
        # Look for the 'System Prompt' or 'Attack' columns
        text_cols = [c for c in df.columns if isinstance(c, str) and 
                     any(x in c.lower() for x in ['prompt', 'leak', 'instruction', 'content'])]
        
        if text_cols:
            target = text_cols[0]
            val = str(df.iloc[0][target])
            # Clean up newlines for display
            clean_val = val.replace('\n', ' ')[:100]
            print(f"   📝 Sample Text ({target}): \"{clean_val}...\"")
        else:
            print(f"   ⚠️ No obvious text columns. First row: {df.iloc[0].values}")

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("   (Note: If this fails, the file might be a binary .xls (Excel 97-2003) or truly corrupt.)")

    print("-" * 50 + "\n")